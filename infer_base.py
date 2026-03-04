import os
import argparse

import torch
from diffusers import StableDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Infer images with base Stable Diffusion model (no LoRA)")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output_dir", type=str, default="infer_base_out")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_images", type=int, default=1)
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


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    if args.dtype in {"fp16", "bf16"} and args.device == "cpu":
        print("[Warn] fp16/bf16 on CPU is not recommended. Switching dtype to fp32.")
        args.dtype = "fp32"

    torch_dtype = resolve_dtype(args.dtype)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    if args.device == "cuda":
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    for image_index in range(args.num_images):
        seed = args.seed + image_index
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
        out_name = f"base_s{seed}_{image_index:02d}.png"
        out_path = os.path.join(args.output_dir, out_name)
        image.save(out_path)
        print(f"[Save] {out_path}")

    print("[Done] Base model inference finished.")


if __name__ == "__main__":
    main()
