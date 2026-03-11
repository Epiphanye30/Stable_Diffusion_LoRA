import argparse
import copy
import inspect
import os
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer


FALLBACK_PRIOR_PROMPTS = [
    "a portrait photo of a person",
    "a landscape photo at sunset",
    "a close-up photo of a cat",
    "a photo of a city street",
    "a product photo on white background",
]


class FewShotImageTextDataset(Dataset):
    def __init__(self, data_dir: str, resolution: int = 512, center_crop: bool = True):
        self.data_dir = data_dir
        self.resolution = resolution
        self.center_crop = center_crop

        items = []
        for fn in sorted(os.listdir(data_dir)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                stem = os.path.splitext(fn)[0]
                txt = os.path.join(data_dir, stem + ".txt")
                img = os.path.join(data_dir, fn)
                if os.path.exists(txt):
                    items.append((img, txt))
        if len(items) == 0:
            raise ValueError(
                f"No (image, caption) pairs found in {data_dir}. "
                "Expected 00.png + 00.txt style pairs."
            )
        self.items = items

    def __len__(self):
        return len(self.items)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")

        w, h = img.size
        scale = self.resolution / min(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = img.resize((new_w, new_h), resample=Image.BICUBIC)

        left = (new_w - self.resolution) // 2
        top = (new_h - self.resolution) // 2
        img = img.crop((left, top, left + self.resolution, top + self.resolution))

        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img * 2.0 - 1.0
        return img

    def _load_caption(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        return txt

    def __getitem__(self, idx):
        img_path, txt_path = self.items[idx]
        image = self._load_image(img_path)
        caption = self._load_caption(txt_path)
        return {"pixel_values": image, "caption": caption, "img_path": img_path}


@dataclass
class Batch:
    pixel_values: torch.Tensor
    captions: List[str]
    img_paths: List[str]


def collate_fn(examples) -> Batch:
    pixel_values = torch.stack([e["pixel_values"] for e in examples], dim=0)
    captions = [e["caption"] for e in examples]
    img_paths = [e["img_path"] for e in examples]
    return Batch(pixel_values=pixel_values, captions=captions, img_paths=img_paths)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--method", type=str, default="lora", choices=["lora", "lora_prior", "dora"])

    p.add_argument("--data_dir", type=str, default="datasets/5_shots_Anime_Faces")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--save_every", type=int, default=300)

    p.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )

    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--center_crop", action="store_true")
    p.add_argument("--train_steps", type=int, default=1200)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["constant", "cosine", "linear", "constant_with_warmup"],
    )
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.0)

    p.add_argument("--lambda_prior", type=float, default=0.5)
    p.add_argument("--prior_every", type=int, default=2)
    p.add_argument("--prior_prompts_path", type=str, default="dataset/prior_prompts.txt")

    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    return p.parse_args()


def load_prior_prompts(path: str) -> List[str]:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        if prompts:
            return prompts
    return FALLBACK_PRIOR_PROMPTS


def resolve_output_dir(args) -> str:
    if args.output_dir:
        return args.output_dir
    return os.path.join("output", args.method)


def build_lora_config(args) -> LoraConfig:
    kwargs = dict(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    if args.method == "dora":
        if "use_dora" not in inspect.signature(LoraConfig).parameters:
            raise RuntimeError(
                "Current peft version does not support DoRA (`use_dora=True`). "
                "Please upgrade peft to a version that includes DoRA support."
            )
        kwargs["use_dora"] = True
    return LoraConfig(**kwargs)


def adapter_param_name(name: str) -> bool:
    if "lora_" in name:
        return True
    if "magnitude_vector" in name:
        return True
    if "modules_to_save" in name:
        return True
    return False


def assert_frozen_backbone(unet: torch.nn.Module, text_encoder: torch.nn.Module, vae: torch.nn.Module):
    if any(p.requires_grad for p in text_encoder.parameters()):
        raise RuntimeError("text_encoder has trainable parameters; expected frozen.")
    if any(p.requires_grad for p in vae.parameters()):
        raise RuntimeError("vae has trainable parameters; expected frozen.")

    bad = [name for name, p in unet.named_parameters() if p.requires_grad and not adapter_param_name(name)]
    if bad:
        preview = ", ".join(bad[:5])
        raise RuntimeError(f"Found non-adapter trainable parameters in UNet: {preview}")


def resolve_infer_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def save_training_args(path: str, args):
    with open(path, "w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")


def select_quickcheck_prompts(data_dir: str) -> List[str]:
    dataset_name = os.path.basename(os.path.normpath(data_dir)).lower()

    if "anime" in dataset_name:
        return [
            "anime style face illustration, detailed eyes",
            "close-up anime portrait, clean line art, soft shading",
        ]
    if "flower" in dataset_name or "birdofparadise" in dataset_name:
        return [
            "a close-up photo of a bird of paradise flower, natural lighting",
            "botanical photograph of a bird of paradise flower, detailed petals, clean background",
        ]
    if "car" in dataset_name:
        return [
            "a photo of a sports car, studio lighting, front three-quarter view",
            "a photo of a car parked on the street, detailed body lines, realistic lighting",
        ]

    return [
        "a high-quality photo of the subject, natural lighting",
        "a detailed image of the subject, clean composition",
    ]


def run_quickcheck(
    args,
    adapter_dir: str,
    quickcheck_dir: str,
    accelerator: Accelerator,
):
    prompts = select_quickcheck_prompts(args.data_dir)
    seeds = [args.seed, args.seed + 1]

    os.makedirs(quickcheck_dir, exist_ok=True)

    dtype = resolve_infer_dtype(args.mixed_precision)
    if accelerator.device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    base_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=dtype,
    )
    adapted_unet = PeftModel.from_pretrained(base_unet, adapter_dir)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=adapted_unet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(accelerator.device)

    for prompt_idx, prompt in enumerate(prompts):
        for seed in seeds:
            generator = torch.Generator(device=accelerator.device.type).manual_seed(seed)
            image = pipe(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                generator=generator,
            ).images[0]
            out_name = f"quick_p{prompt_idx}_s{seed}.png"
            out_path = os.path.join(quickcheck_dir, out_name)
            image.save(out_path)
            print(f"[QuickCheck] {out_path}")

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    args.output_dir = resolve_output_dir(args)
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, "logs"),
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(f"{args.method}_sdv15_fewshot")

    set_seed(args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    use_disable_adapter_teacher = (
        args.method == "lora_prior" and hasattr(PeftModel, "disable_adapter")
    )
    teacher_unet = None
    if args.method == "lora_prior" and not use_disable_adapter_teacher:
        teacher_unet = copy.deepcopy(unet)
        teacher_unet.requires_grad_(False)
        teacher_unet.eval()

    lora_config = build_lora_config(args)
    unet = get_peft_model(unet, lora_config)
    unet.train()

    assert_frozen_backbone(unet, text_encoder, vae)

    if accelerator.is_main_process:
        try:
            unet.print_trainable_parameters()
        except Exception:
            trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
            total = sum(p.numel() for p in unet.parameters())
            print(f"[{args.method}] Trainable params: {trainable} / {total}")
        print(f"[Info] method={args.method} output_dir={args.output_dir}")

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError(
            "No trainable parameters found. LoRA target_modules likely did not match UNet modules."
        )

    dataset = FewShotImageTextDataset(
        data_dir=args.data_dir,
        resolution=args.resolution,
        center_crop=args.center_crop,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    if accelerator.is_main_process:
        sample = dataset[0]
        pv = sample["pixel_values"]
        cap = sample["caption"]
        print(f"[Data] num_pairs={len(dataset)}")
        print(f"[Data] pixel_values shape={tuple(pv.shape)} range=({pv.min().item():.3f},{pv.max().item():.3f})")
        print(f"[Data] caption[0]={repr(cap[:120])}")
        if pv.shape[-1] != args.resolution or pv.shape[-2] != args.resolution:
            raise ValueError("Loaded image resolution mismatch after preprocessing.")
        if (args.resolution % 8) != 0:
            raise ValueError("resolution must be divisible by 8 for SD VAE/UNet.")
        if len(cap.strip()) == 0:
            raise ValueError("Empty caption detected.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps,
    )

    unet, optimizer, dataloader, lr_scheduler, text_encoder, vae = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler, text_encoder, vae
    )
    unwrapped_unet = accelerator.unwrap_model(unet)
    if teacher_unet is not None:
        teacher_dtype = next(unwrapped_unet.parameters()).dtype
        teacher_unet = teacher_unet.to(device=accelerator.device, dtype=teacher_dtype)
        teacher_unet.eval()

    text_encoder.eval()
    vae.eval()

    prior_prompts = None
    if args.method == "lora_prior":
        prior_prompts = load_prior_prompts(args.prior_prompts_path)
        if accelerator.is_main_process:
            src = args.prior_prompts_path if os.path.exists(args.prior_prompts_path) else "fallback list"
            print(f"[Prior] loaded {len(prior_prompts)} prompts from {src}")

    global_step = 0
    progress = tqdm(range(args.train_steps), disable=not accelerator.is_local_main_process)
    data_iter = iter(dataloader)

    while global_step < args.train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        with accelerator.accumulate(unet):
            pixel_values = batch.pixel_values.to(accelerator.device, non_blocking=True)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                inputs = tokenizer(
                    batch.captions,
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                input_ids = inputs.input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids)[0]

            target_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            target_loss = F.mse_loss(target_pred.float(), noise.float(), reduction="mean")
            prior_loss = torch.zeros((), device=target_loss.device, dtype=target_loss.dtype)
            loss = target_loss

            should_run_prior = (
                args.method == "lora_prior"
                and args.prior_every > 0
                and ((global_step + 1) % args.prior_every == 0)
            )
            if should_run_prior:
                prior_batch_captions = random.choices(prior_prompts, k=bsz)
                with torch.no_grad():
                    prior_inputs = tokenizer(
                        prior_batch_captions,
                        padding="max_length",
                        truncation=True,
                        max_length=tokenizer.model_max_length,
                        return_tensors="pt",
                    )
                    prior_input_ids = prior_inputs.input_ids.to(accelerator.device)
                    prior_hidden_states = text_encoder(prior_input_ids)[0]

                prior_latents = torch.randn_like(latents)
                prior_noise = torch.randn_like(prior_latents)
                prior_timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                    dtype=torch.long,
                )
                prior_noisy_latents = noise_scheduler.add_noise(
                    prior_latents, prior_noise, prior_timesteps
                )

                with torch.no_grad():
                    if use_disable_adapter_teacher:
                        with unwrapped_unet.disable_adapter():
                            teacher_pred = unet(
                                prior_noisy_latents, prior_timesteps, prior_hidden_states
                            ).sample
                    else:
                        teacher_pred = teacher_unet(
                            prior_noisy_latents, prior_timesteps, prior_hidden_states
                        ).sample

                student_pred = unet(prior_noisy_latents, prior_timesteps, prior_hidden_states).sample
                prior_loss = F.mse_loss(student_pred.float(), teacher_pred.float(), reduction="mean")
                loss = target_loss + args.lambda_prior * prior_loss

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if accelerator.is_main_process:
            logs = {
                "train_loss": loss.detach().item(),
                "target_loss": target_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            if args.method == "lora_prior":
                logs["prior_loss"] = prior_loss.detach().item()
            accelerator.log(logs, step=global_step)

        if accelerator.is_main_process and (global_step + 1) % args.save_every == 0:
            save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step + 1}")
            os.makedirs(save_dir, exist_ok=True)
            unet_to_save = accelerator.unwrap_model(unet)
            unet_to_save.save_pretrained(save_dir, safe_serialization=True)
            save_training_args(os.path.join(save_dir, "train_args.txt"), args)

        progress.update(1)
        global_step += 1

    final_dir = None
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        unet_to_save = accelerator.unwrap_model(unet)
        unet_to_save.save_pretrained(final_dir, safe_serialization=True)
        save_training_args(os.path.join(final_dir, "train_args.txt"), args)

        quickcheck_dir = os.path.join(args.output_dir, "quickcheck")
        try:
            run_quickcheck(args, final_dir, quickcheck_dir, accelerator)
        except Exception as exc:
            print(f"[Warn] quickcheck generation failed: {exc}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
