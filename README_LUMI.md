# Run this repo on LUMI server

## 1. Prepare environment

```bash
# Copy the repo to LUMI
git clone git@github.com:shanshanwangcsc/synth-data-bench-training.git

# Load the required modules
module purge
module use /appl/local/laifs/modules                  
module load lumi-aif-singularity-bindings

# Create python virtual environment and install additional packages on top of the LUMI container 
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
singularity shell $SIF
Singularity> python -m venv vlm-env --system-site-packages
Singularity> source vlm-env/bin/activate

# Install extra packages 
pip install megatron-energon
```

## 2. Generate Synthetic datasets

```bash
cd synth-data-bench-training

# Generate vqa dataset
python src/generate.py configs/vqa.toml

# Generate interleaved dataset
python src/generate.py configs/interleaved.toml

# Generate cap_pretrain dataset (i increased num_samples to 100k since we will use this dataset for the vlm-training repo)
python src/generate.py configs/cap_pretrain.toml
```
## 3. Prepare dataset for Energon

```bash
# Prepare vqa dataset
energon prepare data/vqa --non-interactive --split-ratio 1.0,0,0 --sample-type CrudeWebdataset
# Prepare interleaved dataset
energon prepare data/interleaved --non-interactive --split-ratio 1.0,0,0 --sample-type CrudeWebdataset
# Prepare cap_pretrain dataset
energon prepare data/cap_pretrain --non-interactive --split-ratio 1.0,0,0 --sample-type CrudeWebdataset
```
## 4. Visualization
```bash
# Visualize vqa dataset
python src/viz_synthetic.py \
    --dataset data/vqa \
    --encoder-module DataPackingEncoder \
    --output visualizations/vqa_tokens_lumi.png

# Visualize interleaved dataset
python src/viz_synthetic.py \
    --dataset data/interleaved \
    --encoder-module DataPackingEncoder \
    --output visualizations/interleaved_tokens_lumi.png

# Visualize cap_pretrain dataset
python src/viz_synthetic.py \
    --dataset data/cap_pretrain \
    --encoder-module DataPackingEncoder \
    --output visualizations/cap_pretrain_tokens_lumi.png
``` 

