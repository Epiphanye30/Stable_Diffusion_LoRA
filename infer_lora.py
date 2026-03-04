import os
import re
import argparse
from typing import List

import torch
from peft import PeftModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel


def parse_args():
    parser = argparse.ArgumentParser(description="Infer images from SD1.5 LoRA checkpoints")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="LoRA checkpoint dir or parent dir containing checkpoint-* and/or final")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output_dir", type=str, default="infer_out")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_images_per_ckpt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--include_final", action="store_true", help="When ckpt_dir is a parent folder, also include final/")
    return parser.parse_args()


def resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    return torch.float32


def find_adapter_dirs(ckpt_dir: str, include_final: bool) -> List[str]:
    ckpt_dir = os.path.abspath(ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"ckpt_dir not found: {ckpt_dir}")

    marker_files = {"adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"}
    if any(os.path.exists(os.path.join(ckpt_dir, m)) for m in marker_files):
        return [ckpt_dir]

    names = [n for n in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, n))]

    def step_of(name: str):
        m = re.match(r"^checkpoint-(\d+)$", name)
        return int(m.group(1)) if m else None

    pairs = [(name, step_of(name)) for name in names]
    ckpts = [os.path.join(ckpt_dir, n) for n, s in sorted(pairs, key=lambda x: (x[1] is None, x[1])) if s is not None]

    if include_final and "final" in names:
        ckpts.append(os.path.join(ckpt_dir, "final"))

    if not ckpts:
        raise ValueError(
            f"No checkpoint-* subdirs found under {ckpt_dir}. "
            "You can also pass a direct LoRA adapter folder containing adapter_config.json."
        )
    return ckpts


def build_pipe(base_model: str, adapter_dir: str, device: str, torch_dtype: torch.dtype) -> StableDiffusionPipeline:
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", torch_dtype=torch_dtype)
    unet = PeftModel.from_pretrained(unet, adapter_dir)

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        unet=unet,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    if device == "cuda" and torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
    return pipe


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    if args.dtype in {"fp16", "bf16"} and args.device == "cpu":
        print("[Warn] fp16/bf16 on CPU is not recommended. Switching dtype to fp32.")
        args.dtype = "fp32"

    torch_dtype = resolve_dtype(args.dtype)
    adapter_dirs = find_adapter_dirs(args.ckpt_dir, args.include_final)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[Info] Found {len(adapter_dirs)} adapter(s).")

    for index, adapter_dir in enumerate(adapter_dirs):
        tag = os.path.basename(adapter_dir)
        print(f"\n[Info] ({index + 1}/{len(adapter_dirs)}) Loading {tag}")

        pipe = build_pipe(args.base_model, adapter_dir, args.device, torch_dtype)

        for image_index in range(args.num_images_per_ckpt):
            seed = args.seed + index * 10000 + image_index
            generator = torch.Generator(device=args.device).manual_seed(seed)

            result = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt if args.negative_prompt else None,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
            image = result.images[0]
            out_name = f"{tag}_s{seed}_{image_index:02d}.png"
            out_path = os.path.join(args.output_dir, out_name)
            image.save(out_path)
            print(f"[Save] {out_path}")

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[Done] Inference finished.")


if __name__ == "__main__":
    main()
