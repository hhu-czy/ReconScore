import argparse
import time
import os
import gc
import torch
from diffusers import ZImagePipeline
from dreamsim import dreamsim
from scripts.image_generator import generate_image
from scripts.similarity_evaluator import evaluate_similarity

def parse_args():
    parser = argparse.ArgumentParser(description="Single Image Generation and Similarity Evaluation (Main Control Script)")
    
    parser.add_argument("--caption", type=str, required=True, help="Text description of the image")
    parser.add_argument("--ref_img_path", type=str, required=True, help="Path to the reference image")
    parser.add_argument("--output_path", type=str, default="./output_single_image.png", help="Path to save the generated image")
    parser.add_argument("--model_path", type=str, default="./models/Z-Image", help="Path to the diffusion model weights")
    parser.add_argument("--max_side", type=int, default=1024, help="Maximum side length of the generated image")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps for the diffusi on model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the models on")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if not os.path.exists(args.ref_img_path):
        raise FileNotFoundError(f"Reference image not found at {args.ref_img_path}")

    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"\n[*] [Phase 1] Loading ZImagePipeline from {args.model_path} onto {device}...")
    torch_dtype = torch.bfloat16 if "cuda" in device else torch.float32
    pipe = ZImagePipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype, use_safetensors=True)
    pipe.to(device)
    
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass

    start_time = time.time()
    
    # Pass the pipeline and the new steps argument to the encapsulated function
    generate_image(
        pipe=pipe,
        caption=args.caption,
        ref_img_path=args.ref_img_path,
        output_path=args.output_path,
        max_side=args.max_side,
        seed=args.seed,
        num_inference_steps=args.steps
    )
    
    gen_end_time = time.time()
    
    print("\n[*] [Cleanup] Freeing up VRAM before evaluation...")
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[*] [Phase 2] Loading DreamSim model onto {device}...")
    eval_start_time = time.time()
    
    # Initialize DreamSim in the main script
    model_dreamsim, preprocess = dreamsim(pretrained=True, device=device)
    model_dreamsim.eval()
    
    # Pass the model to the encapsulated function
    score = evaluate_similarity(
        ref_img_path=args.ref_img_path,
        gen_img_path=args.output_path,
        eval_model=model_dreamsim,
        preprocess_fn=preprocess,
        eval_device=device
    )
    
    eval_end_time = time.time()
    
    # Optionally clean up DreamSim VRAM after evaluation is complete
    del model_dreamsim
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print(f"- Final ReconScore (Similarity): {100*score:.4f}")
    print("=" * 50)