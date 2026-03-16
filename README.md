# Few-shot Domain Adaptation for Stable Diffusion v1.5

This repository studies parameter-efficient few-shot adaptation of Stable Diffusion v1.5 with three methods:

- `lora`
- `lora_prior`
- `dora`

The repo includes:

- training: `train.py`
- inference: `inference.py`
- evaluation: `evaluate.py`
- plotting: `visualize.py`
- batch runs across datasets: `train_all_datasets.sh`

The main goal is to adapt a pretrained text-to-image model to a new visual domain from only 1 to 5 images, then compare fidelity, diversity, and prompt controllability across methods.

## What Is In This Repo

The project is organized around a fixed experiment pipeline:

1. prepare a few-shot dataset in `datasets/<experiment_name>/`
2. train adapters with `train.py`
3. generate samples with `inference.py` or through `evaluate.py`
4. compute evaluation CSVs with `evaluate.py`
5. make figures with `visualize.py`

Important folders:

- `datasets/`: few-shot training sets, full evaluation reference sets, prior prompts
- `output/`: training outputs and final adapters
- `results/`: generated samples, evaluation CSVs, and figures

## Methods

- `lora`: standard Low-Rank Adaptation on Stable Diffusion UNet attention layers
- `lora_prior`: LoRA plus prior-preservation prompts during training
- `dora`: DoRA, a weight-decomposed variant of LoRA

The code freezes the base model backbone and trains only adapter parameters.

## Repository Layout

```text
.
+-- train.py
+-- inference.py
+-- evaluate.py
+-- visualize.py
+-- train_all_datasets.sh
+-- requirements.txt
+-- datasets/
|   +-- 1_shots_Anime_Faces/
|   +-- 5_shots_Anime_Faces/
|   +-- 5_flower_birdofparadise/
|   +-- 5_stanford_car/
|   +-- Anime_Faces/
|   +-- flower_birdofparadise/
|   +-- stanford_car/
|   `-- prior_prompts.txt
+-- output/
`-- results/
```

## Environment Setup

Recommended: Python 3.10 or 3.11 with a CUDA-enabled PyTorch install.

Create an environment:

```bash
python -m venv .venv
```

Activate it:

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On bash:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Model Access

The default base model is:

```text
runwayml/stable-diffusion-v1-5
```

You may need Hugging Face access configured locally before running training or inference. If model download fails, log in with:

```bash
huggingface-cli login
```

## Dataset Format

Each few-shot training folder should contain paired image-caption files:

```text
datasets/5_shots_Anime_Faces/
+-- 1.png
+-- 1.txt
+-- 2.png
+-- 2.txt
`-- ...
```

Each `.txt` file should contain one caption describing the matching image.

Examples from the current repo:

- anime: `a close-up portrait of a pink-haired anime girl with large eyes, soft shading, clean line art`
- flower: `a photo of a bird of paradise flower`
- car: `a photo of a car`

For evaluation, the repo also expects full-domain reference folders:

- `datasets/Anime_Faces`
- `datasets/flower_birdofparadise`
- `datasets/stanford_car`

These are used by `evaluate.py` when computing CLIP similarity and FID.

## Prior Prompts for `lora_prior`

`lora_prior` uses a plain text file with one generic prompt per line:

- default path used by batch training: `datasets/prior_prompts.txt`
- default path in `train.py`: `dataset/prior_prompts.txt`

The batch script already resolves this mismatch and prefers `datasets/prior_prompts.txt` if it exists.

Current examples:

- `a portrait photo of a person`
- `a landscape photo at sunset`
- `a close-up photo of a cat`
- `a photo of a city street`
- `a product photo on white background`

If no prompt file is found, `train.py` falls back to an internal default prompt list.

## Quick Start

Train a single LoRA adapter on the default dataset:

```bash
python train.py --method lora
```

Train LoRA with prior preservation:

```bash
python train.py --method lora_prior --data_dir datasets/5_shots_Anime_Faces
```

Train DoRA:

```bash
python train.py --method dora --data_dir datasets/5_stanford_car
```

Run zero-shot inference:

```bash
python inference.py --method zero_shot --prompt "a photo of a car"
```

Run inference with a trained LoRA adapter:

```bash
python inference.py --method lora --adapter_path output/5_stanford_car/lora/final --prompt "a photo of a car"
```

## Training

### Single Run

Core arguments for `train.py`:

- `--method`: `lora`, `lora_prior`, or `dora`
- `--data_dir`: few-shot training folder
- `--output_dir`: where checkpoints and final adapter are written
- `--train_steps`: default `1200`
- `--save_every`: checkpoint frequency
- `--learning_rate`: default `1e-4`
- `--lora_rank`: default `8`
- `--mixed_precision`: `no`, `fp16`, `bf16`

Example:

```bash
python train.py ^
  --method lora ^
  --data_dir datasets/5_shots_Anime_Faces ^
  --output_dir output/5_shots_Anime_Faces/lora ^
  --train_steps 1200 ^
  --save_every 300
```

On bash use `\` instead of `^` for line continuation.

Training outputs:

- `output/<experiment>/<method>/checkpoint-*`
- `output/<experiment>/<method>/final/`
- `output/<experiment>/<method>/quickcheck/`
- `output/<experiment>/<method>/logs/`

### Batch Training

`train_all_datasets.sh` runs all datasets and all three methods:

```bash
bash train_all_datasets.sh
```

It currently runs:

- `datasets/1_shots_Anime_Faces`
- `datasets/5_shots_Anime_Faces`
- `datasets/5_flower_birdofparadise`
- `datasets/5_stanford_car`

Pass extra training args through to every run:

```bash
bash train_all_datasets.sh --train_steps 200 --save_every 100
```

If you are on Windows, run this script through Git Bash or WSL. It is a bash script, not a PowerShell script.

Important behavior:

- the script deletes the target output directory before each run
- outputs are written to `output/<dataset_name>/<method>/`

## Inference

`inference.py` supports:

- `zero_shot`
- `lora`
- `lora_prior`
- `dora`

Examples:

Zero-shot:

```bash
python inference.py --method zero_shot --prompt "anime girl face portrait"
```

LoRA:

```bash
python inference.py --method lora --adapter_path output/5_shots_Anime_Faces/lora/final --prompt "anime girl face portrait"
```

LoRA + Prior:

```bash
python inference.py --method lora_prior --adapter_path output/5_shots_Anime_Faces/lora_prior/final --prompt "anime girl face portrait"
```

DoRA:

```bash
python inference.py --method dora --adapter_path output/5_stanford_car/dora/final --prompt "a photo of a car"
```

Useful flags:

- `--negative_prompt`
- `--num_images`
- `--seed`
- `--height`
- `--width`
- `--guidance_scale`
- `--num_inference_steps`
- `--device cuda|cpu`
- `--dtype fp16|bf16|fp32`

Inference outputs are saved to:

```text
results/<method>/samples/
```

If `--adapter_path` points to a directory containing `checkpoint-*`, use `--include_final` to also sample the final adapter.

## Evaluation

`evaluate.py` is the end-to-end evaluation script used for the project report. It:

- discovers experiments under `output/`
- loads the matching few-shot and reference datasets from `datasets/`
- generates samples for each method
- computes prompt controllability, LPIPS diversity, CLIP image similarity, and FID
- saves merged CSV summaries

Run:

```bash
python evaluate.py
```

Main outputs:

- `results/all_evaluation_results.csv`
- `results/all_evaluation_summary.csv`
- `results/plot_ready_summary.csv`
- `results/<experiment_name>/...`

## Visualization

After evaluation, create summary figures:

```bash
python visualize.py
```

Figures are written to:

```text
results/figures/
```

## Typical End-to-End Workflow

1. Install dependencies.
2. Make sure the few-shot datasets and full reference datasets are present under `datasets/`.
3. Run training:

```bash
bash train_all_datasets.sh
```

4. Run evaluation:

```bash
python evaluate.py
```

5. Generate plots:

```bash
python visualize.py
```

At that point, your final metrics and figures will be under `results/`.

## Common Issues

### `peft` does not support DoRA

If `train.py` raises an error about `use_dora=True`, upgrade `peft`:

```bash
pip install -U peft
```

### CUDA requested but not available

Inference will fall back to CPU if CUDA is unavailable. Training on CPU is technically possible but usually impractical for Stable Diffusion.

### Prior prompt file not found

`train.py` falls back to internal prompts if the prompt file is missing. For reproducible `lora_prior` runs, use `datasets/prior_prompts.txt` explicitly.

### Output directory is overwritten

`train_all_datasets.sh` deletes each target output directory before training. Do not point it at an output folder you need to keep.

## Notes on Reports

The repository also contains:

- `mid_term_report.tex`
- `Final_Report.tex`

These are project writeups and are not required to run the code.
