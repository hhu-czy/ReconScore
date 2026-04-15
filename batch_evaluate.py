import os
import json
import argparse
import torch
from tqdm import tqdm
from dreamsim import dreamsim
from scripts.similarity_evaluator import evaluate_similarity

def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluation script utilizing the modular encapsulated function")
    parser.add_argument("--json_dir", type=str, required=True, help="Directory containing input JSON files")
    parser.add_argument("--ref_dir", type=str, required=True, help="Directory containing reference original images")
    parser.add_argument("--gen_dir", type=str, required=True, help="Directory containing generated images")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device (e.g., cuda:0)")
    parser.add_argument("--save_score", action="store_true", help="Whether to write the ReconScore back to the JSON files")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing ReconScores in the JSON files")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # ==========================================
    # 1. Scan files and build a resume-capable task queue
    # ==========================================
    json_files = [f for f in os.listdir(args.json_dir) if f.endswith('.json')]
    eval_tasks = []
    
    print("[*] Scanning files for evaluation...")
    for j_file in json_files:
        json_path = os.path.join(args.json_dir, j_file)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Skip if score exists and overwrite flag is not set
        if not args.overwrite and "ReconScore" in data:
            continue
            
        orig_filename = data.get("filename")
        if not orig_filename: continue
        
        parts = orig_filename.split('.')
        gen_filename = f"{parts[0]}_n.{parts[1]}"
        
        ref_path = os.path.join(args.ref_dir, orig_filename)
        gen_path = os.path.join(args.gen_dir, gen_filename)
        
        # Ensure both reference and generated images exist
        if os.path.exists(ref_path) and os.path.exists(gen_path):
            eval_tasks.append({
                "json_path": json_path,
                "ref_path": ref_path,
                "gen_path": gen_path
            })

    total_tasks = len(eval_tasks)
    if total_tasks == 0:
        print("[*] No tasks to evaluate (scores might all be generated or images are missing).")
        exit(0)

    # ==========================================
    # 2. Load evaluation model globally
    # ==========================================
    print(f"[*] Loading DreamSim model (Total tasks: {total_tasks})...")
    model_dreamsim, preprocess = dreamsim(pretrained=True, device=device)
    model_dreamsim.eval()

    # ==========================================
    # 3. Loop through tasks and evaluate single images
    # ==========================================
    total_score = 0.0
    
    # Process image pairs one by one using the single-image evaluator
    for task in tqdm(eval_tasks, desc="Evaluating Images"):
        
        # Directly call the function from similarity_evaluator.py
        score = evaluate_similarity(
            ref_img_path=task["ref_path"],
            gen_img_path=task["gen_path"],
            eval_model=model_dreamsim,
            preprocess_fn=preprocess,
            eval_device=device
        )
        
        total_score += score
        
        # Write the result back to the corresponding JSON file
        if args.save_score:
            with open(task["json_path"], 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data["ReconScore"] = score
            
            with open(task["json_path"], 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

    avg_score = total_score / total_tasks
    print("\n" + "="*50)
    print(f"[*] Evaluation completed!")
    print(f"[*] Total images processed: {total_tasks}")
    print(f"[*] Average ReconScore: {100*avg_score:.4f}")
    if args.save_score:
        print(f"[*] Individual scores have been saved to the respective JSON files.")
    print("="*50)