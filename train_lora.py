import os
import math
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np

from PIL import Image
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler

from peft import LoraConfig, get_peft_model


# -----------------------------
# Dataset
# -----------------------------
class FewShotImageTextDataset(Dataset):
    def __init__(self, data_dir: str, resolution: int = 512, center_crop: bool = True):
        self.data_dir = data_dir
        self.resolution = resolution
        self.center_crop = center_crop

        # collect pairs (img_path, txt_path)
        items = []
        for fn in sorted(os.listdir(data_dir)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                stem = os.path.splitext(fn)[0]
                txt = os.path.join(data_dir, stem + ".txt")
                img = os.path.join(data_dir, fn)
                if os.path.exists(txt):
                    items.append((img, txt))
        if len(items) == 0:
            raise ValueError(f"No (image, caption) pairs found in {data_dir}. "
                             f"Expected 00.png + 00.txt style pairs.")
        self.items = items

    def __len__(self):
        return len(self.items)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")

        # Resize
        w, h = img.size
        scale = self.resolution / min(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = img.resize((new_w, new_h), resample=Image.BICUBIC)

        # Crop
        left = (new_w - self.resolution) // 2
        top = (new_h - self.resolution) // 2
        img = img.crop((left, top, left + self.resolution, top + self.resolution))

        img = np.array(img).astype(np.float32) / 255.0   # (H, W, 3)
        img = torch.from_numpy(img).permute(2, 0, 1)     # (3, H, W)
        img = img * 2.0 - 1.0                            # [-1, 1]
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


# -----------------------------
# Training
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()

    # data / output
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="lora_out")
    p.add_argument("--save_every", type=int, default=300)

    # model
    p.add_argument("--pretrained_model_name_or_path", type=str,
                   default="runwayml/stable-diffusion-v1-5")

    # training
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--center_crop", action="store_true")
    p.add_argument("--train_steps", type=int, default=1200)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lr_scheduler", type=str, default="constant",
                   choices=["constant", "cosine", "linear", "constant_with_warmup"])
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    # LoRA
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.0)

    # precision / device
    p.add_argument("--mixed_precision", type=str, default="fp16",
                   choices=["no", "fp16", "bf16"])

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, "logs"),
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("lora_sdv15_fewshot")

    set_seed(args.seed)

    # Load tokenizer & text encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )

    # Load VAE & UNet
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

    # Freeze everything first
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Inject LoRA ONLY into attention projections in UNet
    # Targets are common projection module names in diffusers UNet attention:
    # to_q, to_k, to_v, to_out.0
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    unet.train()

    # Sanity: confirm LoRA actually inserted trainable params
    if accelerator.is_main_process:
        try:
            unet.print_trainable_parameters()
        except Exception:
            # fallback if print_trainable_parameters is unavailable in your peft version
            trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
            total = sum(p.numel() for p in unet.parameters())
            print(f"Trainable params: {trainable} / {total}")

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found. LoRA target_modules likely didn't match UNet module names.")

    # Build dataset & loader
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

    # Optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # LR Scheduler
    # total updates = train_steps (we control by manual loop)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps,
    )

    # Prepare for accelerate
    unet, optimizer, dataloader, lr_scheduler, text_encoder, vae = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler, text_encoder, vae
    )

    # For fp16: move inference-only modules to eval
    text_encoder.eval()
    vae.eval()

    # Training loop
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

            # Encode images to latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Sample noise & timesteps
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,),
                device=latents.device, dtype=torch.long
            )

            # Add noise
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Tokenize captions
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

            # UNet predicts noise
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # MSE loss on noise prediction
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # logging
        if accelerator.is_main_process:
            accelerator.log(
                {"train_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]},
                step=global_step
            )

        # save checkpoint
        if accelerator.is_main_process and (global_step + 1) % args.save_every == 0:
            save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step+1}")
            os.makedirs(save_dir, exist_ok=True)
            # Save LoRA adapter weights only
            unet_to_save = accelerator.unwrap_model(unet)
            unet_to_save.save_pretrained(save_dir, safe_serialization=True)
            # also store args
            with open(os.path.join(save_dir, "train_args.txt"), "w", encoding="utf-8") as f:
                for k, v in sorted(vars(args).items()):
                    f.write(f"{k}: {v}\n")

        progress.update(1)
        global_step += 1

    # final save
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        unet_to_save = accelerator.unwrap_model(unet)
        unet_to_save.save_pretrained(final_dir, safe_serialization=True)
        with open(os.path.join(final_dir, "train_args.txt"), "w", encoding="utf-8") as f:
            for k, v in sorted(vars(args).items()):
                f.write(f"{k}: {v}\n")

    accelerator.end_training()


if __name__ == "__main__":
    main()