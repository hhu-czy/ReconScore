import torch
import os
import time
import argparse
import gc

from transformers import AutoModelForImageTextToText, AutoProcessor
from diffusers import ZImagePipeline
from dreamsim import dreamsim
from scripts.image_generator import generate_image
from scripts.similarity_evaluator import evaluate_similarity

from qwen_vl_utils import process_vision_info


SYSTEM_PROMPT = (
    "You are a professional expert in the field of remote sensing, specializing in captioning remote sensing images. "
    "You will receive a remote sensing image. Your goal is to generate a informative and accurate description of a remote sensing image."
    "Guidelines:"
    "1. Extract key objects and details as much as possible."
    "2. You should describe the attributes of objects in detail, including object quantity, color, material, shape, size, and spatial position and relative position."
    "3. Don't generate non-sense content, inaccuracies, and irrelevant information. Highlight essential visual elements and don't describe feeling or atmosphere."
    "4. You should first describe the overall scene of the image, followed by describing specific object."
    "5. The output should be coherent, concise and logical."
)

def Image_captioning(img_path, model, processor, device):
    """Generates a single caption using Qwen3-VL"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": SYSTEM_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Generate diverse descriptions (do_sample=True)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Single Image Best Caption Selection Pipeline")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the original remote sensing image")
    parser.add_argument("--output_dir", type=str, default="./temp_candidates", help="Directory to temporarily store generated candidate images")
    parser.add_argument("--qwen_model_path", type=str, default="./models/Qwen3-VL-8B-Instruct", help="Path to Qwen3-VL model")
    parser.add_argument("--zimage_model_path", type=str, default="./models/Z-Image", help="Path to Z-Image model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device")
    parser.add_argument("--num_candidates", type=int, default=4, help="Number of candidate captions to generate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"Input image not found: {args.img_path}")

    # Ensure output directory exists for temporarily storing Z-Image generated images
    os.makedirs(args.output_dir, exist_ok=True)
    base_filename = os.path.basename(args.img_path)
    file_no_ext = os.path.splitext(base_filename)[0]

    candidates = []

    # ==========================================
    # Phase 1: Load Qwen to generate candidate captions
    # ==========================================
    print(f"\n[Phase 1] Loading Qwen3-VL onto {device}...")
    qwen_model = AutoModelForImageTextToText.from_pretrained(
        args.qwen_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(args.qwen_model_path, device_map=device, trust_remote_code=True)

    print(f"[*] Generating {args.num_candidates} candidate captions for {base_filename}...")
    for i in range(args.num_candidates):
        try:
            caption = Image_captioning(args.img_path, qwen_model, processor, device)
            candidates.append({"id": i + 1, "caption": caption, "score": 0.0, "gen_img_path": ""})
            print(f"    - Caption {i + 1} generated successfully.")
        except Exception as e:
            print(f"[!] Failed to generate Caption {i + 1}: {e}")

    # Unload Qwen and clear VRAM
    print("[*] Unloading Qwen3-VL to free up VRAM...")
    del qwen_model, processor
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================
    # Phase 2: Load Z-Image to generate candidate images
    # ==========================================
    print(f"\n[Phase 2] Loading ZImagePipeline onto {device}...")
    torch_dtype = torch.bfloat16 if "cuda" in device else torch.float32
    pipe = ZImagePipeline.from_pretrained(args.zimage_model_path, torch_dtype=torch_dtype, use_safetensors=True)
    pipe.to(device)
    try: pipe.vae.enable_tiling()
    except: pass

    print(f"[*] Generating {len(candidates)} candidate images...")
    for idx, candidate in enumerate(candidates):
        gen_img_path = os.path.join(args.output_dir, f"{file_no_ext}_candidate_{candidate['id']}.png")
        candidate["gen_img_path"] = gen_img_path
        
        try:
            generate_image(
                pipe=pipe,
                caption=candidate["caption"],
                ref_img_path=args.img_path,
                output_path=gen_img_path,
                max_side=1024,
                seed=42 + idx, 
                num_inference_steps=28
            )
        except Exception as e:
            print(f"[!] Failed to generate image for candidate {candidate['id']}: {e}")

    # Unload Z-Image and clear VRAM
    print("[*] Unloading Z-Image to free up VRAM...")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================
    # Phase 3: Load DreamSim to calculate similarity scores
    # ==========================================
    print(f"\n[Phase 3] Loading DreamSim onto {device}...")
    model_dreamsim, preprocess_fn = dreamsim(pretrained=True, device=device)
    model_dreamsim.eval()

    print(f"[*] Calculating ReconScore for candidates...")
    best_candidate = None
    highest_score = -1.0

    for candidate in candidates:
        if not os.path.exists(candidate["gen_img_path"]):
            continue
            
        try:
            score = evaluate_similarity(
                ref_img_path=args.img_path,
                gen_img_path=candidate["gen_img_path"],
                eval_model=model_dreamsim,
                preprocess_fn=preprocess_fn,
                eval_device=device
            )
            candidate["score"] = score
            print(f"    - Candidate {candidate['id']} Score: {score:.4f}")
            
            if score > highest_score:
                highest_score = score
                best_candidate = candidate
        except Exception as e:
            print(f"[!] Failed to evaluate candidate {candidate['id']}: {e}")

    # Unload DreamSim and clear VRAM
    print("[*] Unloading DreamSim to free up VRAM...")
    del model_dreamsim
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================
    # Phase 4: Print the best result (Removed JSON saving)
    # ==========================================
    print("\n" + "="*80)
    print(" FINAL EVALUATION RESULTS")
    print("="*80)
    
    if best_candidate:
        print(best_candidate["caption"])
    else:
        print("[!] The pipeline failed to select a best candidate (errors occurred during generation or evaluation).")