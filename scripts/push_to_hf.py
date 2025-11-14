"""
Script to push the pretrained model to Hugging Face Hub.
Run this once to upload your model checkpoint.

Usage:
    python scripts/push_to_hf.py --repo-id YOUR_USERNAME/signdetr-pretrained
"""

import argparse
import os
from huggingface_hub import HfApi, create_repo
import torch

def push_model_to_hf(checkpoint_path: str, repo_id: str, token: str = None):
    """
    Push model checkpoint to Hugging Face Hub.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        repo_id: Hugging Face repo ID (e.g., "username/model-name")
        token: HF token (optional, will use saved token if not provided)
    """
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Uploading {checkpoint_path} to {repo_id}...")
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True, token=token, private=False)
        print(f"Repository created/verified: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Could not create repo: {e}")
        print("Attempting upload anyway...")
    
    # Upload the file
    api = HfApi()
    api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo=os.path.basename(checkpoint_path),
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    
    print(f"Model uploaded successfully!")
    print(f"View at: https://huggingface.co/{repo_id}")
    print(f"\nUpdate your code to use:")
    print(f"   model.load_pretrained_from_hf('{repo_id}', '{os.path.basename(checkpoint_path)}')")

def main():
    parser = argparse.ArgumentParser(description="Push model to Hugging Face Hub")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="pretrained/4426_model.pt",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--repo-id", 
        type=str, 
        required=True,
        help="Hugging Face repo ID (e.g., 'username/signdetr-pretrained')"
    )
    parser.add_argument(
        "--token", 
        type=str, 
        default=None,
        help="HF token (optional if already logged in via 'huggingface-cli login')"
    )
    
    args = parser.parse_args()
    
    push_model_to_hf(args.checkpoint, args.repo_id, args.token)

if __name__ == "__main__":
    main()
