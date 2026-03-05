import argparse
import os
import re
from typing import List, Optional

import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Unified inference for SD1.5 zero-shot/LoRA/LoRA+Prior/DoRA")
    parser.add_argument(
        "--method",
        type=str,
        default="zero_shot",
        choices=["zero_shot", "lora", "lora_prior", "dora"],
        help="Inference method",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output_dir", type=str, default="results")

    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Adapter folder path or parent folder with checkpoint-* subfolders",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Backward-compatible alias of --adapter_path",
    )
    parser.add_argument(
        "--include_final",
        action="store_true",
        help="When adapter_path points to parent folder, also include final/",
    )

    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--num_images_per_ckpt", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    return parser.parse_args()


def resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    return torch.float32


def find_adapter_dirs(adapter_path: str, include_final: bool) -> List[str]:
    adapter_path = os.path.abspath(adapter_path)
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"adapter_path not found: {adapter_path}")

    marker_files = {"adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"}
    if any(os.path.exists(os.path.join(adapter_path, m)) for m in marker_files):
        return [adapter_path]

    names = [n for n in os.listdir(adapter_path) if os.path.isdir(os.path.join(adapter_path, n))]

    def step_of(name: str):
        m = re.match(r"^checkpoint-(\d+)$", name)
        return int(m.group(1)) if m else None

    pairs = [(name, step_of(name)) for name in names]
    ckpts = [
        os.path.join(adapter_path, n)
        for n, s in sorted(pairs, key=lambda x: (x[1] is None, x[1]))
        if s is not None
    ]

    if include_final and "final" in names:
        ckpts.append(os.path.join(adapter_path, "final"))

    if not ckpts:
        raise ValueError(
            f"No checkpoint-* subdirs found under {adapter_path}. "
            "You can also pass a direct adapter folder containing adapter_config.json."
        )
    return ckpts


def resolve_adapter_dirs(args) -> List[Optional[str]]:
    if args.method == "zero_shot":
        return [None]

    adapter_root = args.adapter_path or args.ckpt_dir
    if args.method in {"lora_prior", "dora"} and not adapter_root:
        raise ValueError(f"--adapter_path is required for method={args.method}")
    if args.method == "lora" and not adapter_root:
        raise ValueError("--adapter_path (or --ckpt_dir) is required for method=lora")

    return find_adapter_dirs(adapter_root, args.include_final)


def build_pipe(
    method: str,
    base_model: str,
    adapter_dir: Optional[str],
    device: str,
    torch_dtype: torch.dtype,
) -> StableDiffusionPipeline:
    if method == "zero_shot":
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(
            base_model, subfolder="unet", torch_dtype=torch_dtype
        )
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


def allocate_output_paths(samples_dir: str, tag: str, seed: int, image_index: int):
    idx = image_index
    while True:
        stem = f"{tag}_s{seed}_{idx:02d}"
        image_path = os.path.join(samples_dir, f"{stem}.png")
        prompt_path = os.path.join(samples_dir, f"{stem}.txt")
        if not os.path.exists(image_path) and not os.path.exists(prompt_path):
            return stem, image_path, prompt_path
        idx += 1


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    if args.dtype in {"fp16", "bf16"} and args.device == "cpu":
        print("[Warn] fp16/bf16 on CPU is not recommended. Switching dtype to fp32.")
        args.dtype = "fp32"

    torch_dtype = resolve_dtype(args.dtype)
    adapter_dirs = resolve_adapter_dirs(args)
    num_images = args.num_images_per_ckpt if args.num_images_per_ckpt is not None else args.num_images

    samples_dir = os.path.join(args.output_dir, args.method, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    print(f"[Info] method={args.method}")
    if args.method != "zero_shot":
        print(f"[Info] Found {len(adapter_dirs)} adapter(s).")

    for index, adapter_dir in enumerate(adapter_dirs):
        if adapter_dir is None:
            tag = "base"
        else:
            tag = os.path.basename(adapter_dir.rstrip("/"))
            print(f"\n[Info] ({index + 1}/{len(adapter_dirs)}) Loading {tag}")

        pipe = build_pipe(args.method, args.base_model, adapter_dir, args.device, torch_dtype)

        for image_index in range(num_images):
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
            stem, out_path, prompt_path = allocate_output_paths(
                samples_dir=samples_dir,
                tag=tag,
                seed=seed,
                image_index=image_index,
            )
            image.save(out_path)
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(f"prompt: {args.prompt}\n")
                f.write(f"negative_prompt: {args.negative_prompt}\n")
                f.write(f"method: {args.method}\n")
                f.write(f"seed: {seed}\n")
            print(f"[Save] {out_path}")
            print(f"[Save] {prompt_path}")

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[Done] Inference finished.")


if __name__ == "__main__":
    main()
