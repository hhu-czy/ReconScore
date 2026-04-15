import torch
from PIL import Image
import os

def get_closest_multiple_of_16(val):
    """Ensure the dimension is a multiple of 16."""
    return int(round(val / 16) * 16)

def calculate_target_size(raw_w, raw_h, max_side=1024):
    """Smart size calculation: Maintain aspect ratio."""
    long_side = max(raw_w, raw_h)
    scale = max_side / long_side if long_side > max_side else 1.0
    target_w = raw_w * scale
    target_h = raw_h * scale
    return get_closest_multiple_of_16(target_w), get_closest_multiple_of_16(target_h)

def generate_image(pipe, caption, ref_img_path, output_path, max_side=1024, seed=42, num_inference_steps=28):
    """
    Receives a loaded model pipeline and generates a single image.
    Now includes num_inference_steps as a controllable parameter.
    """
    # 1. Dynamically calculate dimensions
    gen_width, gen_height = max_side, max_side
    if ref_img_path and os.path.exists(ref_img_path):
        try:
            with Image.open(ref_img_path) as ref_img:
                raw_w, raw_h = ref_img.size
                gen_width, gen_height = calculate_target_size(raw_w, raw_h, max_side)
                print(f"[*] Target size calculated from reference: {gen_width}x{gen_height}")
        except Exception as e:
            print(f"[!] Failed to read {ref_img_path}: {e}. Using default {max_side}x{max_side}.")

    # 2. Assemble the Prompt
    prompt = "I want a remote sensing image with a realistic satellite perspective view. " + caption + " Remember, I want a vertical remote sensing satellite perspective from top to bottom."
    
    # 3. Inference and generation
    print(f"[*] Generating image for: '{caption}' with {num_inference_steps} steps.")
    result = pipe(
        prompt=prompt,
        height=gen_height,
        width=gen_width,
        num_inference_steps=num_inference_steps, # <--- Updated here
        guidance_scale=0.0, 
        generator=torch.Generator("cpu").manual_seed(seed),
    )
    
    # 4. Save the image
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    result.images[0].save(output_path)
    
    return output_path