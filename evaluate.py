import os
import re
import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights

import scipy.linalg
from transformers import CLIPProcessor, CLIPModel
import lpips


# =========================================================
# Global settings
# =========================================================
EXPERIMENT_ROOT = "output"
DATASET_ROOT = "datasets"
RESULTS_ROOT = "results"

BASE_MODEL_NAME = "runwayml/stable-diffusion-v1-5"

NUM_SAMPLES_PER_PROMPT = 5
SEEDS = [100, 101, 102, 103, 104]

METHODS = ["base", "dora", "lora", "lora_prior"]


# =========================================================
# Domain-specific prompts and real-data directory mapping
# =========================================================
ANIME_PROMPTS = [
    "anime girl face portrait",
    "anime girl face portrait, smiling",
    "anime girl face portrait, serious expression",
    "anime girl face portrait, surprised expression",
    "anime girl face portrait, neutral expression",

    "anime girl face portrait, looking at viewer",
    "anime girl face portrait, looking to the side",
    "anime girl face portrait, side profile",

    "anime girl face portrait with long hair",
    "anime girl face portrait with short hair",

    "anime girl face portrait with glasses",
    "anime girl face portrait, soft lighting",
]
FLOWER_PROMPTS = [
    "a photo of a bird of paradise flower",
    "a close-up photo of a bird of paradise flower",
    "a bird of paradise flower in a garden",
    "a bird of paradise flower with green leaves",

    "a single bird of paradise flower on plain background",
    "a bird of paradise flower in sunlight",

    "a bird of paradise flower from the side",
    "a front view of a bird of paradise flower",

    "a detailed botanical photo of a bird of paradise flower",
    "a macro photo of a bird of paradise flower",

    "a bird of paradise flower with tropical background",
    "a natural photo of a bird of paradise flower",
]
CAR_PROMPTS = [
    "a photo of a car",
    "a photo of a sedan car",
    "a photo of a sports car",

    "a car parked on a street",
    "a car driving on a road",

    "a front view photo of a car",
    "a side view photo of a car",
    "a rear view photo of a car",

    "a car under daylight",
    "a car under cloudy sky",

    "a close-up photo of a car",
    "a clean product-style photo of a car",
]
DOMAIN_CONFIGS = {
    "Anime_Faces": {
        "real_dir": os.path.join(DATASET_ROOT, "Anime_Faces"),
        "prompts": ANIME_PROMPTS,
        "negative_prompt": "lowres, blurry, bad anatomy, extra fingers, text, watermark, realistic",
    },

    "flower_birdofparadise": {
        "real_dir": os.path.join(DATASET_ROOT, "flower_birdofparadise"),
        "prompts": FLOWER_PROMPTS,
        "negative_prompt": "lowres, blurry, bad petals, distorted flower, text, watermark",
    },

    "stanford_car": {
        "real_dir": os.path.join(DATASET_ROOT, "stanford_car"),
        "prompts": CAR_PROMPTS,
        "negative_prompt": "lowres, blurry, bad wheels, distorted car, text, watermark",
    },
}


# =========================================================
# Experiment config
# =========================================================
@dataclass
class ExperimentConfig:
    experiment_name: str
    shots: Optional[int]
    domain_key: str
    fewshot_dir: str
    real_dir: str
    prompts: List[str]
    negative_prompt: str
    checkpoint_dirs: Dict[str, Optional[str]]


def parse_experiment_name(exp_name: str):
    """
    Examples:
      1_shots_Anime_Faces
      5_shots_Anime_Faces
      5_flower_birdofparadise
      5_stanford_car
    """
    m = re.match(r"^(\d+)(?:_shots)?_(.+)$", exp_name)
    if not m:
        return None, None
    shots = int(m.group(1))
    domain_key = m.group(2)
    return shots, domain_key


def discover_experiments(experiment_root: str) -> List[ExperimentConfig]:
    experiments = []

    if not os.path.exists(experiment_root):
        print(f"[Error] Experiment root does not exist: {experiment_root}")
        return experiments

    for exp_name in sorted(os.listdir(experiment_root)):
        exp_path = os.path.join(experiment_root, exp_name)
        if not os.path.isdir(exp_path):
            continue

        shots, domain_key = parse_experiment_name(exp_name)
        if domain_key is None:
            print(f"[Skip] Cannot parse experiment name: {exp_name}")
            continue

        if domain_key not in DOMAIN_CONFIGS:
            print(f"[Skip] Unknown domain '{domain_key}' in experiment '{exp_name}'")
            continue

        domain_cfg = DOMAIN_CONFIGS[domain_key]

        # few-shot dir should match experiment name exactly
        fewshot_dir = os.path.join(DATASET_ROOT, exp_name)
        real_dir = domain_cfg["real_dir"]

        checkpoint_dirs = {
            "base": None,
            "dora": os.path.join(exp_path, "dora"),
            "lora": os.path.join(exp_path, "lora"),
            "lora_prior": os.path.join(exp_path, "lora_prior"),
        }

        experiments.append(
            ExperimentConfig(
                experiment_name=exp_name,
                shots=shots,
                domain_key=domain_key,
                fewshot_dir=fewshot_dir,
                real_dir=real_dir,
                prompts=domain_cfg["prompts"],
                negative_prompt=domain_cfg["negative_prompt"],
                checkpoint_dirs=checkpoint_dirs,
            )
        )

    return experiments


# =========================================================
# CLIP
# =========================================================
def load_clip(device="cuda"):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


def image_embedding(model, processor, image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(model.device)

    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output
        emb = model.visual_projection(pooled)

    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb


def text_embedding(model, processor, text: str):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        text_outputs = model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = text_outputs.pooler_output
        emb = model.text_projection(pooled)

    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb


# =========================================================
# LPIPS
# =========================================================
def compute_lpips_loss(img1, img2, loss_fn):
    with torch.no_grad():
        val = loss_fn(img1, img2)
    return val.item()


# =========================================================
# Inception / FID
# =========================================================
class InceptionV3FID(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        self.model = inception_v3(weights=weights, transform_input=False).to(device)
        self.model.eval()

    def forward(self, x):
        x = self.model.Conv2d_1a_3x3(x)
        x = self.model.Conv2d_2a_3x3(x)
        x = self.model.Conv2d_2b_3x3(x)
        x = self.model.maxpool1(x)

        x = self.model.Conv2d_3b_1x1(x)
        x = self.model.Conv2d_4a_3x3(x)
        x = self.model.maxpool2(x)

        x = self.model.Mixed_5b(x)
        x = self.model.Mixed_5c(x)
        x = self.model.Mixed_5d(x)
        x = self.model.Mixed_6a(x)
        x = self.model.Mixed_6b(x)
        x = self.model.Mixed_6c(x)
        x = self.model.Mixed_6d(x)
        x = self.model.Mixed_6e(x)
        x = self.model.Mixed_7a(x)
        x = self.model.Mixed_7b(x)
        x = self.model.Mixed_7c(x)

        x = self.model.avgpool(x)
        x = self.model.dropout(x)
        x = torch.flatten(x, 1)
        return x


def load_inception(device="cuda"):
    model = InceptionV3FID(device=device)
    model.eval()
    return model


def get_inception_features(images, model, device="cuda", batch_size=16):
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    processed = []
    for img in images:
        if isinstance(img, Image.Image):
            processed.append(preprocess(img))
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

    if len(processed) == 0:
        return np.empty((0, 2048), dtype=np.float32)

    feats = []
    with torch.no_grad():
        for i in range(0, len(processed), batch_size):
            batch = torch.stack(processed[i:i + batch_size], dim=0).to(device)
            f = model(batch)
            feats.append(f.cpu().numpy())

    return np.concatenate(feats, axis=0)


def calculate_fid(real_features, gen_features, eps=1e-6):
    if len(real_features) < 2 or len(gen_features) < 2:
        return np.nan

    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(gen_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(gen_features, rowvar=False)

    sigma_real = sigma_real + np.eye(sigma_real.shape[0]) * eps
    sigma_gen = sigma_gen + np.eye(sigma_gen.shape[0]) * eps

    diff = mu_real - mu_gen
    covmean, _ = scipy.linalg.sqrtm(sigma_real @ sigma_gen, disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma_real + offset) @ (sigma_gen + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2.0 * covmean)
    return float(fid)


# =========================================================
# IO helpers
# =========================================================
def load_images_from_dir(directory, max_images=1000):
    images = []
    if not os.path.exists(directory):
        print(f"[Warning] Directory does not exist: {directory}")
        return images

    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            path = os.path.join(directory, fname)
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"[Warning] Failed to load image {path}: {e}")
            if len(images) >= max_images:
                break
    return images


# =========================================================
# Diffusion pipeline
# =========================================================
def load_pipeline(checkpoint_dir, device="cuda"):
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    from peft import PeftModel

    dtype = torch.float16 if device == "cuda" else torch.float32

    if checkpoint_dir is None:
        pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(
            BASE_MODEL_NAME,
            subfolder="unet",
            torch_dtype=dtype,
        )
        unet = PeftModel.from_pretrained(unet, checkpoint_dir)

        pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_NAME,
            unet=unet,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# =========================================================
# Summary
# =========================================================
def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(["experiment", "shots", "domain", "method"], as_index=False)
        .agg({
            "prompt_controllability": "mean",
            "generative_diversity": "mean",
            "domain_fidelity_clip": "mean",
            "domain_fidelity_fid": "first",
        })
        .sort_values(["experiment", "method"])
        .reset_index(drop=True)
    )
    return summary


# =========================================================
# Evaluate one experiment
# =========================================================
def evaluate_experiment(exp_cfg, clip_model, clip_processor, lpips_loss_fn, inception_model, device):
    print(f"\n{'=' * 100}")
    print(f"Evaluating experiment: {exp_cfg.experiment_name}")
    print(f"shots={exp_cfg.shots}, domain={exp_cfg.domain_key}")
    print(f"fewshot_dir={exp_cfg.fewshot_dir}")
    print(f"real_dir={exp_cfg.real_dir}")
    print(f"{'=' * 100}")

    prompts = exp_cfg.prompts
    negative_prompt = exp_cfg.negative_prompt

    if device == "cuda":
        autocast_dtype = torch.float16
        autocast_enabled = True
    else:
        autocast_dtype = torch.float32
        autocast_enabled = False

    # real images for FID
    real_images = load_images_from_dir(exp_cfg.real_dir)
    if len(real_images) == 0:
        print("[Warning] No real images loaded. FID will be NaN.")
        real_features = np.empty((0, 2048), dtype=np.float32)
    else:
        print(f"Loaded {len(real_images)} real images for FID.")
        real_features = get_inception_features(real_images, inception_model, device)

    # few-shot images for CLIP prototype
    fewshot_images = load_images_from_dir(exp_cfg.fewshot_dir)
    train_embs = []
    for im in fewshot_images:
        try:
            train_embs.append(image_embedding(clip_model, clip_processor, im))
        except Exception as e:
            print(f"[Warning] CLIP embedding failed on few-shot image: {e}")

    if train_embs:
        prototype = torch.stack(train_embs, dim=0).mean(dim=0)
        prototype = prototype / prototype.norm(p=2, dim=-1, keepdim=True)
        print(f"Built CLIP prototype from {len(train_embs)} few-shot images.")
    else:
        prototype = None
        print("[Warning] No few-shot prototype built; domain_fidelity_clip will be NaN.")

    results = []

    for method, checkpoint_root in exp_cfg.checkpoint_dirs.items():
        if method != "base":
            ckpt_dir = os.path.join(checkpoint_root, "final")
            adapter_config = os.path.join(ckpt_dir, "adapter_config.json")
            if not os.path.exists(adapter_config):
                print(f"[Skip] adapter_config.json not found: {adapter_config}")
                continue
        else:
            ckpt_dir = None

        print(f"\nEvaluating method: {method}")
        pipe = load_pipeline(ckpt_dir, device)
        label = "base_model" if method == "base" else f"{method}_final"

        all_gen_images = []

        for prompt_idx, prompt in enumerate(prompts, start=1):
            print(f"[{method}] Prompt {prompt_idx}/{len(prompts)}: {prompt}")

            gen_images = []
            gen_embs = []

            txt_emb = text_embedding(clip_model, clip_processor, prompt)

            for seed in SEEDS[:NUM_SAMPLES_PER_PROMPT]:
                generator = torch.Generator(device=device).manual_seed(seed)

                with torch.no_grad():
                    with torch.autocast(
                        device_type=device,
                        dtype=autocast_dtype,
                        enabled=autocast_enabled
                    ):
                        img = pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=20,
                            guidance_scale=7.5,
                            generator=generator,
                        ).images[0]

                        # save generated sample
                        sample_dir = os.path.join(
                            RESULTS_ROOT,
                            exp_cfg.experiment_name,
                            "samples",
                            label
                        )

                        os.makedirs(sample_dir, exist_ok=True)

                        safe_prompt = prompt.replace(" ", "_").replace(",", "").replace("/", "_")

                        sample_path = os.path.join(
                            sample_dir,
                            f"p{prompt_idx:02d}_s{seed}_{safe_prompt[:60]}.png"
                        )

                        img.save(sample_path)

                gen_images.append(img)
                all_gen_images.append(img)
                gen_embs.append(image_embedding(clip_model, clip_processor, img))

            # prompt controllability
            sims = [(txt_emb @ emb.t()).item() for emb in gen_embs]
            prompt_score = float(np.mean(sims)) if sims else np.nan

            # diversity (LPIPS)
            lpips_vals = []
            for img1, img2 in itertools.combinations(gen_images, 2):
                t1 = torch.from_numpy(np.array(img1)).permute(2, 0, 1).unsqueeze(0).float()
                t2 = torch.from_numpy(np.array(img2)).permute(2, 0, 1).unsqueeze(0).float()
                t1 = t1 / 127.5 - 1.0
                t2 = t2 / 127.5 - 1.0
                lpips_vals.append(compute_lpips_loss(t1.to(device), t2.to(device), lpips_loss_fn))

            diversity_score = float(np.mean(lpips_vals)) if lpips_vals else np.nan

            # domain fidelity via CLIP prototype
            if prototype is not None:
                dom_sims = [(prototype @ emb.t()).item() for emb in gen_embs]
                domain_score = float(np.mean(dom_sims)) if dom_sims else np.nan
            else:
                domain_score = np.nan

            results.append({
                "experiment": exp_cfg.experiment_name,
                "shots": exp_cfg.shots,
                "domain": exp_cfg.domain_key,
                "method": label,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_samples_per_prompt": NUM_SAMPLES_PER_PROMPT,
                "num_prompts": len(prompts),
                "prompt_controllability": prompt_score,
                "generative_diversity": diversity_score,
                "domain_fidelity_clip": domain_score,
            })

        # FID for this method within this experiment
        if len(all_gen_images) == 0 or len(real_features) < 2:
            fid_score = np.nan
        else:
            gen_features = get_inception_features(all_gen_images, inception_model, device)
            fid_score = calculate_fid(real_features, gen_features)

        for res in results:
            if res["method"] == label:
                res["domain_fidelity_fid"] = fid_score

        print(f"[{method}] FID: {fid_score:.4f}" if not np.isnan(fid_score) else f"[{method}] FID: NaN")

        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(results)


# =========================================================
# Main
# =========================================================
def main():
    assert len(SEEDS) >= NUM_SAMPLES_PER_PROMPT, "Not enough seeds for NUM_SAMPLES_PER_PROMPT."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    experiments = discover_experiments(EXPERIMENT_ROOT)
    if len(experiments) == 0:
        print("[Error] No valid experiments found.")
        return

    print("\nDiscovered experiments:")
    for exp in experiments:
        print(
            f"  - {exp.experiment_name}: "
            f"shots={exp.shots}, domain={exp.domain_key}, "
            f"fewshot_dir={exp.fewshot_dir}, real_dir={exp.real_dir}"
        )

    clip_model, clip_processor = load_clip(device)
    lpips_loss_fn = lpips.LPIPS(net="alex").to(device).eval()
    inception_model = load_inception(device)

    os.makedirs(RESULTS_ROOT, exist_ok=True)

    all_detail_dfs = []
    all_summary_dfs = []

    for exp_cfg in experiments:
        df = evaluate_experiment(
            exp_cfg=exp_cfg,
            clip_model=clip_model,
            clip_processor=clip_processor,
            lpips_loss_fn=lpips_loss_fn,
            inception_model=inception_model,
            device=device,
        )

        exp_result_dir = os.path.join(RESULTS_ROOT, exp_cfg.experiment_name)
        os.makedirs(exp_result_dir, exist_ok=True)

        detail_path = os.path.join(exp_result_dir, "evaluation_results.csv")
        summary_path = os.path.join(exp_result_dir, "evaluation_summary.csv")

        df.to_csv(detail_path, index=False)

        summary_df = build_summary_table(df)
        summary_df.to_csv(summary_path, index=False)

        print(f"Saved detailed results to: {detail_path}")
        print(f"Saved summary results to: {summary_path}")

        all_detail_dfs.append(df)
        all_summary_dfs.append(summary_df)

    # merged outputs for downstream plotting
    if all_detail_dfs:
        merged_detail = pd.concat(all_detail_dfs, ignore_index=True)
        merged_detail_path = os.path.join(RESULTS_ROOT, "all_evaluation_results.csv")
        merged_detail.to_csv(merged_detail_path, index=False)
        print(f"\nSaved merged detailed results to: {merged_detail_path}")

    if all_summary_dfs:
        merged_summary = pd.concat(all_summary_dfs, ignore_index=True)
        merged_summary_path = os.path.join(RESULTS_ROOT, "all_evaluation_summary.csv")
        merged_summary.to_csv(merged_summary_path, index=False)
        print(f"Saved merged summary results to: {merged_summary_path}")

        # extra pivot-friendly aggregated table
        aggregate = (
            merged_summary.groupby(["shots", "domain", "method"], as_index=False)
            .agg({
                "prompt_controllability": "mean",
                "generative_diversity": "mean",
                "domain_fidelity_clip": "mean",
                "domain_fidelity_fid": "mean",
            })
            .sort_values(["domain", "shots", "method"])
            .reset_index(drop=True)
        )
        aggregate_path = os.path.join(RESULTS_ROOT, "plot_ready_summary.csv")
        aggregate.to_csv(aggregate_path, index=False)
        print(f"Saved plot-ready summary to: {aggregate_path}")

        print("\n=== Plot-ready summary ===")
        print(aggregate)


if __name__ == "__main__":
    main()