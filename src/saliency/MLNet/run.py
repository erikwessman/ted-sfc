"""This file is modified from:
https://github.com/Cogito2012/DRIVE/blob/master/main_saliency.py
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.io import write_video
import torchvision.transforms as transforms

from saliency.MLNet.model import MLNet
from saliency.MLNet.loader import MLNetLoader
from saliency.MLNet.data_transform import ProcessImages, padding_inv
import helper


MODEL_PATH = "models/saliency/mlnet_25.pth"
INPUT_SHAPE = [480, 640]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Saliency implementation")
    parser.add_argument("data_path", help="")
    parser.add_argument("output_path", help="")
    parser.add_argument("config_path", help="")
    parser.add_argument(
        "--cpu", help="Use CPU instead of GPU.", action=argparse.BooleanOptionalAction
    )
    return parser.parse_args()


def main(
    data_path: str,
    output_path: str,
    config_path: str,
    use_cpu: bool = False,
):
    # Load config
    config = helper.load_yml(config_path)
    grid_config = config["grid_config"]

    # Set up CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    if use_cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise Exception(
            "No CUDA device found. This code requires a CUDA-enabled GPU and OpenCV with CUDA support."
        )

    os.makedirs(output_path, exist_ok=True)

    # Set up data loader
    transform_image = transforms.Compose([ProcessImages(INPUT_SHAPE)])
    params_norm = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    test_data = MLNetLoader(
        data_path, transforms=transform_image, params_norm=params_norm
    )
    testdata_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    # Load model
    model = MLNet(INPUT_SHAPE).to(device)  # ~700MiB
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        pbar = tqdm(testdata_loader, desc="Creating heatmap videos")
        for video, video_info in pbar:
            video_id = str(video_info[0][0])
            num_frames, height, width = video_info[1:]
            num_frames = num_frames.item()
            height = height.item()
            width = width.item()

            pbar.set_description(f"Processing folder {video_id}")

            output_video_dir = os.path.join(output_path, video_id)
            os.makedirs(output_video_dir, exist_ok=True)

            output_file = os.path.join(output_video_dir, f"{video_id}_heatmap.avi")

            if os.path.exists(output_file):
                continue

            pred_video = []
            for fid in tqdm(range(num_frames), desc="Processing frames", leave=False):
                frame_data = video[:, fid].to(device, dtype=torch.float)

                # Forward
                out = model(frame_data)
                out = out.cpu().numpy() if out.is_cuda else out.detach().numpy()
                out = np.squeeze(out)

                # Decode results
                pred_saliency = padding_inv(out, height, width)
                pred_saliency = np.tile(
                    np.expand_dims(np.uint8(pred_saliency), axis=-1), (1, 1, 3)
                )
                pred_video.append(pred_saliency)

            pred_video = np.array(pred_video, dtype=np.uint8)  # (T, H, W, C)
            write_video(output_file, torch.from_numpy(pred_video), grid_config["fps"])


if __name__ == "__main__":
    args = parse_arguments()
    main(args.data_path, args.output_path, args.config_path, args.cpu)
