import time
import torch
import numpy as np
from dataclasses import dataclass
from transformers import AutoProcessor
from megatron.energon import DefaultTaskEncoder, stateless, CrudeSample, Cooker, TextSample, basic_sample_keys, InterleavedSample

from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.base_dataset import Sample

class VQABaseTaskEncoder(DefaultTaskEncoder):
    """Base class for VQA task encoders providing visualization metadata."""
    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct", max_length: int = 2048, **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.max_length = max_length
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        
        self.CAT_MAP = {
            "Padding": 0,
            "User Text": 1,
            "Assistant Text": 2,
            "Image": 3,
            "Special": 4,
            "Boundary": 5
        }
        self.COLORS = ["#ecf0f1", "#3498db", "#9b59b6", "#2ecc71", "#e74c3c", "#000000"]
        
        self.special_ids = set(self.tokenizer.all_special_ids)
        vision_tokens = ["<|image_pad|>", "<|vision_pad|>", "<|vision_start|>", "<|vision_end|>"]
        self.vision_ids = {self.tokenizer.convert_tokens_to_ids(vt) for vt in vision_tokens if self.tokenizer.convert_tokens_to_ids(vt) is not None}
        
        im_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_start_id = im_start if im_start is not None else -100

    def categorize(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, cu_seqlens=None) -> np.ndarray:
        """Maps token IDs to category integers for visualization. Safely handles 1D and 2D batches."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            if cu_seqlens is not None:
                cu_seqlens = cu_seqlens.unsqueeze(0) if cu_seqlens.dim() == 1 else cu_seqlens

        batch_size, seq_len = input_ids.shape
        cat_matrix = np.zeros((batch_size, seq_len), dtype=int)
        
        for i in range(batch_size):
            current_role = self.CAT_MAP["User Text"]
            
            if cu_seqlens is not None:
                seq_b = cu_seqlens[i]
                if isinstance(seq_b, torch.Tensor):
                    seq_b = seq_b.tolist()
                boundaries = set(seq_b)
            else:
                boundaries = set()

            for j in range(seq_len):
                token_id = input_ids[i, j].item()
                is_valid = attention_mask[i, j].item()
                
                if not is_valid:
                    cat_matrix[i, j] = self.CAT_MAP["Padding"]
                    continue

                if j in boundaries and j > 0:
                    cat_matrix[i, j] = self.CAT_MAP["Boundary"]
                    current_role = self.CAT_MAP["User Text"]
                    continue

                if token_id == self.im_start_id and j + 1 < seq_len:
                    next_token = input_ids[i, j+1].item()
                    next_str = self.tokenizer.decode([next_token]).strip().lower()
                    if "user" in next_str:
                        current_role = self.CAT_MAP["User Text"]
                    elif "assistant" in next_str:
                        current_role = self.CAT_MAP["Assistant Text"]

                if token_id in self.vision_ids:
                    cat_matrix[i, j] = self.CAT_MAP["Image"]
                elif token_id in self.special_ids:
                    cat_matrix[i, j] = self.CAT_MAP["Special"]
                else:
                    cat_matrix[i, j] = current_role
                    
        return cat_matrix


@edataclass
class EnergonSample(Sample):
    image: torch.Tensor
    messages: list


@stateless
def cooker_captioning(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
    role_map = {'human': 'user', 'gpt': 'assistant', 'user': 'user', 'assistant': 'assistant'}
    
    messages = []
    if not add_system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": ""}]})
    image_added = False
    
    for turn in sample['json']['conversations']:
        raw_role = turn.get('from', turn.get('role', 'user'))
        role = role_map.get(str(raw_role).lower(), 'user')
        text_val = turn.get('value', turn.get('content', ''))
        
        content = []
        
        if "<image>" in text_val or (role == 'user' and not image_added):
            content.append({"type": "image"})
            text_val = text_val.replace("<image>", "").strip()
            image_added = True
            
        if text_val:
            content.append({"type": "text", "text": text_val})
            
        if not content:
            content.append({"type": "text", "text": ""})
            
        messages.append({"role": role, "content": content})
    
    image = sample['jpg']

    return EnergonSample(
        **basic_sample_keys(sample),
        image=image,
        messages=messages,
    )

@dataclass
class EncodedSample:
    __key__: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    length: int
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor


class DataPackingEncoder(VQABaseTaskEncoder):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self._batch_type = None

    cookers = [
        # subflavors can be used to distinguish datasets when using a Metadataset
        Cooker(cooker_captioning),
    ]

    # transform the RAW data, tokenize a single sample
    @stateless(restore_seeds=True)
    def encode_sample(self, sample: EnergonSample) -> EncodedSample:
        text = self.processor.apply_chat_template(sample.messages, tokenize=False, add_generation_prompt=False)
        n_images = text.count("<|image_pad|>")
        inputs = self.processor(text=[text], images=[sample.image]* n_images, padding=False, return_tensors="pt")

        return EncodedSample(
            __key__=sample.__key__,
            input_ids=inputs["input_ids"][0],
            attention_mask=inputs["attention_mask"][0],
            length=len(inputs["input_ids"][0]),
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw")
        )

    def select_samples_to_pack(self, samples: list[EncodedSample]) -> list[list[EncodedSample]]:
        samples.sort(key=lambda x: x.length, reverse=True)
        groups = []
        while samples:
            current_group = [samples.pop(0)]
            current_len = current_group[0].length
            i = 0
            while i < len(samples):
                if current_len + samples[i].length <= self.max_length:
                    sample = samples.pop(i)
                    current_group.append(sample)
                    current_len += sample.length
                else:
                    i += 1
            groups.append(current_group)
        return groups

    # collate the batch into a single sample
    @stateless
    def pack_selected_samples(self, samples: list[EncodedSample]) -> dict:
        packed_input_ids = torch.cat([s.input_ids for s in samples])
        packed_attention_mask = torch.cat([s.attention_mask for s in samples])
        cu_seqlens = torch.tensor([0] + list(np.cumsum([s.length for s in samples])), dtype=torch.int32)
        
        pad_len = self.max_length - packed_input_ids.size(0)
        if pad_len > 0:
            packed_input_ids = torch.cat([packed_input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
            packed_attention_mask = torch.cat([packed_attention_mask, torch.zeros((pad_len,), dtype=torch.long)])
        
        batch_out = {
            "input_ids": packed_input_ids,
            "attention_mask": packed_attention_mask,
            "cu_seqlens": cu_seqlens,
        }

        valid_pixel_values = [s.pixel_values for s in samples if s.pixel_values is not None]
        if valid_pixel_values:
            batch_out["pixel_values"] = torch.cat(valid_pixel_values, dim=0)
            
        valid_grid_thw = [s.image_grid_thw for s in samples if s.image_grid_thw is not None]
        if valid_grid_thw:
            batch_out["image_grid_thw"] = torch.cat(valid_grid_thw, dim=0)
            
        return batch_out
