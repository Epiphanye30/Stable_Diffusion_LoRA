# Stable Diffusion LoRA / LoRA+Prior / DoRA

This repo now has a unified training entrypoint and a unified inference entrypoint built by minimally extending the existing vanilla LoRA code.

## Updated entrypoints

- `train.py`: unified training for `lora`, `lora_prior`, `dora`
- `inference.py`: unified inference for `zero_shot`, `lora`, `lora_prior`, `dora`

The original baseline scripts are kept unchanged:
- `train_lora.py`
- `infer_base.py`
- `infer_lora.py`

## Dataset

Training data default path:
- `datasets/5_shots_Anime_Faces/`

Expected format is unchanged from baseline (paired image + caption text file).

## Prior prompts (LoRA+Prior)

Default prompt file:
- `dataset/prior_prompts.txt`

One prompt per line. If the file is missing or empty, `train.py` falls back to an internal generic prompt list.

## Training

Run vanilla LoRA:

```bash
python train.py --method lora
```

Run LoRA + prior regularization:

```bash
python train.py --method lora_prior
```

Run DoRA:

```bash
python train.py --method dora
```

Notes:
- `--output_dir` is optional. If omitted, outputs are written to `output/<method>/`.
- Checkpoints are saved as `checkpoint-*` and final adapter is saved to `final/`.
- After training, a quick smoke generation (2 prompts x 2 seeds) is written to `output/<method>/quickcheck/`.

Batch training for all four datasets and all three methods:

```bash
bash train_all_datasets.sh
```

You can pass extra training arguments through to every run. Example:

```bash
bash train_all_datasets.sh --train_steps 200 --save_every 100
```

Batch outputs are stored under:
- `output/<dataset_name>/<method>/`

The script deletes the existing target output directory before each run, so the new output replaces the old output for that dataset/method.
If you want to force a specific prior prompt file for all `lora_prior` runs, you can set `PRIOR_PROMPTS_PATH=/your/path.txt` before the command.

## Inference

Zero-shot inference:

```bash
python inference.py --method zero_shot --prompt "your prompt"
```

LoRA inference (single adapter folder or parent checkpoint folder):

```bash
python inference.py --method lora --adapter_path output/lora/final --prompt "your prompt"
```

LoRA+Prior inference:

```bash
python inference.py --method lora_prior --adapter_path output/lora_prior/final --prompt "your prompt"
```

DoRA inference:

```bash
python inference.py --method dora --adapter_path output/dora/final --prompt "your prompt"
```

Notes:
- `--adapter_path` is required for `lora_prior` and `dora`.
- Inference results are saved under `results/<method>/samples/` by default.
- If `--adapter_path` points to a parent folder with `checkpoint-*`, you can sample each checkpoint (and optionally `final/` via `--include_final`).

## DoRA compatibility

If your installed `peft` does not support `LoraConfig(..., use_dora=True)`, `train.py` raises a clear error asking to upgrade `peft`.
