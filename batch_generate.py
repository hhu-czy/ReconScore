import os
import json
import argparse
import torch
from tqdm import tqdm
from diffusers import ZImagePipeline
from scripts.image_generator import generate_image

def parse_args():
    parser = argparse.ArgumentParser(description="Batch generation script utilizing the modular encapsulated function")
    parser.add_argument("--json_dir", type=str, required=True, help="Directory containing input JSON files")
    parser.add_argument("--ref_dir", type=str, required=True, help="Directory containing reference original images")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save generated images")
    parser.add_argument("--model_path", type=str, required=True, default="./models/Z-Image", help="Path to the Z-Image model weights")
    parser.add_argument("--max_side", type=int, default=1024, help="Maximum side length of the generated image")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device (e.g., cuda:0)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    # ==========================================
    # 1. Scan files and build a resume-capable task queue
    # ==========================================
    json_files = [f for f in os.listdir(args.json_dir) if f.endswith('.json')]
    tasks = []
    
    for j_file in json_files:
        json_path = os.path.join(args.json_dir, j_file)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        orig_filename = data.get("filename")
        caption = data.get("caption")
        
        if not orig_filename or not caption:
            continue
            
        parts = orig_filename.split('.')
        gen_filename = f"{parts[0]}_n.{parts[1]}"
        out_img_path = os.path.join(args.out_dir, gen_filename)
        ref_img_path = os.path.join(args.ref_dir, orig_filename)
        
        # Resume capability: Skip if image already exists
        if not os.path.exists(out_img_path):
            tasks.append({
                "caption": caption,
                "ref_img_path": ref_img_path,
                "out_img_path": out_img_path
            })

    if not tasks:
        print("[*] All images have been generated. No pending tasks.")
        exit(0)

    # ==========================================
    # 2. Load model globally (Only load once)
    # ==========================================
    print(f"[*] Preparing to generate {len(tasks)} images. Loading model...")
    torch_dtype = torch.bfloat16 if "cuda" in device else torch.float32
    pipe = ZImagePipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype, use_safetensors=True)
    pipe.to(device)
    try: pipe.vae.enable_tiling()
    except: pass

    # ==========================================
    # 3. Loop through tasks and call the encapsulated function
    # ==========================================
    for task in tqdm(tasks, desc="Generating Images"):
        try:
            # Directly call the function from image_generator.py
            generate_image(
                pipe=pipe,
                caption=task["caption"],
                ref_img_path=task["ref_img_path"],
                output_path=task["out_img_path"],
                max_side=args.max_side,
                seed=args.seed,
                num_inference_steps=args.steps
            )
        except Exception as e:
            print(f"\n[!] Error generating {task['out_img_path']}: {e}")
            
    print("\n[*] All generation tasks completed successfully!")