#!/usr/bin/env python3
import subprocess
import sys
import os

def download_pretrained():
    # Create ckpt/pretrained directory if it doesn't exist
    os.makedirs("ckpt/pretrained", exist_ok=True)
    
    # Model URL
    url = "https://drive.google.com/file/d/1ajONZOgiG9pEYsQ-eBmgkVbMDuHgPCaY/view?usp=drive_link"
    output_path = "ckpt/pretrained/TPS-ResNet-BiLSTM-Attn.pth"
    
    try:
        subprocess.run(["gdown", "-O", output_path, "--fuzzy", url], check=True)
        print(f"Successfully downloaded model to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: gdown not found. Please install it using 'pip install gdown'", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    download_pretrained() 