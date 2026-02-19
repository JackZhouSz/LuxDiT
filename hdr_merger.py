import argparse
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import tqdm
import glob

import numpy as np
import torch
import imageio.v3 as imagio

from src.models.hdr_model import HDR_MLP, HDR_CNN

def seam_blending(img, blend_width=100):
    """
    Apply a smooth blend at the seam created by rolling the image.
    
    Args:
        img: The HDR image after rolling
        blend_width: Width of the blending region on each side of the seam
    
    Returns:
        Blended HDR image with smooth transition at the seam
    """
    height, width = img.shape[:2]
    result = img.copy()
    
    # Create blending weight arrays
    left_weights = np.linspace(1, 0.5, blend_width)
    right_weights = 1  - left_weights
    
    # Extract regions around the seam
    left_region = img[:, -blend_width:].copy()
    right_region = img[:, :blend_width].copy()
    
    # Create blended regions
    for i in range(blend_width):
        # Blend the left edge (end of image)
        result[:, width-blend_width+i] = (
            left_region[:, i] * left_weights[i] + 
            right_region[:, blend_width-1-i] * right_weights[i]
        )
        
        # Blend the right edge (beginning of image)
        result[:, blend_width-1-i] = (
            right_region[:, blend_width-1-i] * left_weights[i] + 
            left_region[:, i] * right_weights[i]
        )
    
    return result


def main(args):
    device = args.device
    HDR_NN = HDR_MLP if args.model_type == "mlp" else HDR_CNN
    hdr_model = HDR_NN.from_pretrained(args.model_path)

    hdr_model.to(device)
    hdr_model.eval()

    input_dir = args.input_dir
    img_list = sorted(glob.glob(os.path.join(input_dir, "*_ldr.png")))
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for ldr_path in tqdm.tqdm(img_list):
            log_path = ldr_path.replace("_ldr.png", "_log.png")
            img_name = os.path.basename(ldr_path)
            hdr_path = os.path.join(output_dir, img_name.replace("_ldr.png", ".exr"))

            # if os.path.exists(hdr_path):
            #     print(f"Skipping {hdr_path}, already exists.")
            #     continue

            ldr_img = imagio.imread(ldr_path) / 255.0 * 2.0 - 1
            log_img = imagio.imread(log_path) / 255.0 * 2.0 - 1

            ldr_img = torch.from_numpy(ldr_img).float().to(device)
            log_img = torch.from_numpy(log_img).float().to(device)

            hdr_img = hdr_model(ldr_img, log_img).cpu().detach().numpy()

            if not args.disable_env_roll:
                width = hdr_img.shape[1]
                if not args.disable_seam_blending:
                    hdr_img = seam_blending(hdr_img, blend_width=int(width * 0.02))

                hdr_img = np.roll(hdr_img, width // 2, axis=1)
                
            if not args.disable_env_flip:
                hdr_img = np.flip(hdr_img, axis=1)

            imagio.imwrite(hdr_path, hdr_img, plugin='opencv')
        print("HDR merging completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDR Merger")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing LDR images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save merged HDR images")
    parser.add_argument("--model_type", type=str, choices=["mlp", "cnn"], default="mlp", help="Model type to use for HDR merging")
    parser.add_argument("--model_path", type=str, help="Path to the pre-trained model", default="checkpoints/hdr_merge_mlp")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    parser.add_argument("--disable_env_roll", action="store_true", help="Disable environment roll for HDR merging")
    parser.add_argument("--disable_env_flip", action="store_true", help="Disable environment flip for HDR merging")
    parser.add_argument("--disable_seam_blending", action="store_true", help="Disable seam blending to the HDR image")
    args = parser.parse_args()
    main(args)