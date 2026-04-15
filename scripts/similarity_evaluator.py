import torch
from PIL import Image

def evaluate_similarity(ref_img_path, gen_img_path, eval_model, preprocess_fn, eval_device):
    """
    Receives the loaded DreamSim model and preprocessing function to calculate the ReconScore similarity.
    """
    print(f"[*] Calculating similarity between reference and generated image...")
    
    # Preprocess and convert to tensor
    img1 = preprocess_fn(Image.open(ref_img_path).convert("RGB")).to(eval_device)
    img2 = preprocess_fn(Image.open(gen_img_path).convert("RGB")).to(eval_device)
    
    # Calculate distance
    with torch.no_grad():
        distance = eval_model(img1, img2)
        
    score = 1 - (distance.item() / 2)
    return score