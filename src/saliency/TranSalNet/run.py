"""This file is modified from:
https://github.com/LJOVO/TranSalNet/blob/master/testing.ipynb
"""

import os
import torch
import numpy as np
import argparse
from torchvision import transforms
from torchvision.io import write_video
from saliency.TranSalNet.data_process import preprocess_img, postprocess_img
from saliency.TranSalNet.model import TranSalNet
from tqdm import tqdm

import helper


MODEL_PATH = "models/saliency/TranSalNet_Res.pth"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Saliency implementation")
    parser.add_argument("data_path", help="")
    parser.add_argument("output_path", help="")
    parser.add_argument("config_path", help="")
    parser.add_argument(
        "--cpu", help="Use CPU instead of GPU.", action=argparse.BooleanOptionalAction
    )
    return parser.parse_args()


def main(data_path: str, output_path: str, config_path: str, use_cpu: bool = False):
    # Load config
    config = helper.load_yml(config_path)
    grid_config = config["grid_config"]

    if use_cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise Exception(
            "No CUDA device found. This code requires a CUDA-enabled GPU and OpenCV with CUDA support."
        )

    model = TranSalNet()
    model.load_state_dict(torch.load(MODEL_PATH))

    model = model.to(device)
    model.eval()

    for video_dir, video_id, _ in helper.traverse_videos(data_path):
        video_path = os.path.join(video_dir, f"{video_id}.avi")
        list_frames, original_dims = helper.video_to_frames(video_path)

        output_video_dir = os.path.join(output_path, video_id)
        os.makedirs(output_video_dir, exist_ok=True)
        output_file = os.path.join(output_video_dir, f"{video_id}_heatmap.avi")

        if os.path.exists(output_file):
            continue

        video_sal_maps = []

        for img in tqdm(list_frames, desc="Processing frames", leave=False):
            img = preprocess_img(img)  # Pad and resize image to 384x288
            img = np.array(img) / 255.0
            img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
            img = torch.from_numpy(img)
            img = img.type(torch.cuda.FloatTensor).to(device)
            pred_saliency = model(img)
            toPIL = transforms.ToPILImage()
            pic = toPIL(pred_saliency.squeeze())

            # Restore the image to its original size as the result
            pred_saliency = postprocess_img(pic, original_dims)
            video_sal_maps.append(pred_saliency)

        video_tensor = torch.stack(
            [
                torch.from_numpy(frame).unsqueeze(-1).repeat(1, 1, 3)
                for frame in video_sal_maps
            ]
        )
        write_video(output_file, video_tensor.numpy(), fps=grid_config["fps"])


if __name__ == "__main__":
    args = parse_arguments()
    main(args.data_path, args.output_path, args.config_path, args.cpu)
