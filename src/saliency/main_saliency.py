""" This file is modified from:
https://github.com/Cogito2012/DRIVE/blob/master/main_saliency.py
"""
import os
import argparse
import torch
import yaml
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.io import write_video
import torchvision.transforms as transforms

from saliency.mlnet import MLNet
from saliency.tasednet import TASED_v2
from saliency.ted_loader import TEDLoader
from saliency.data_transform import ProcessImages, padding_inv


MODEL = "TASEDNet"
MLNET_MODEL_PATH = "models/saliency/mlnet_25.pth"
TASEDNET_MODEL_PATH = "models/saliency/tasednet_iter_1000.pt"
INPUT_SHAPE = [480, 640]


def load_config(file_path) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Saliency implementation")
    parser.add_argument("data_path", help="")
    parser.add_argument("output_path", help="")
    parser.add_argument("config_path", help="")
    parser.add_argument("--gpu_id", type=int, default=0, metavar="N", help="")
    return parser.parse_args()


def main(data_path: str, output_path: str, config_path: str, gpu_id: int = 0):
    # Load config
    config = load_config(config_path)
    grid_config = config["grid_config"]

    # Set up CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    os.makedirs(output_path, exist_ok=True)

    # Set up data loader
    transform_image = transforms.Compose([ProcessImages(INPUT_SHAPE)])
    params_norm = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    test_data = TEDLoader(data_path, transforms=transform_image, params_norm=params_norm)
    testdata_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    # Load model
    if MODEL == "MLNet":
        ckpt = torch.load(MLNET_MODEL_PATH, map_location=device)
        model = MLNet(INPUT_SHAPE).to(device)
        model.load_state_dict(ckpt["model"])
    elif MODEL == "TASEDNet":
        ckpt = torch.load(MLNET_MODEL_PATH, map_location=device)
        model = TASED_v2(INPUT_SHAPE).to(device)
        model_dict = model.state_dict()

        for name, param in ckpt.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print(' size? ' + name, param.size(), model_dict[name].size())
            else:
                print(' name? ' + name)

        model.load_state_dict(model_dict)

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

            result_dir = os.path.join(output_path, video_id)
            os.makedirs(result_dir, exist_ok=True)

            result_videofile = os.path.join(result_dir, f"{video_id}_heatmap.avi")

            if os.path.exists(result_videofile):
                continue

            pred_video = []
            for fid in range(num_frames):
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
            write_video(result_videofile, torch.from_numpy(pred_video), grid_config["fps"])

            pbar.set_description("Processing folders")


if __name__ == "__main__":
    args = parse_arguments()
    main(args.data_path, args.output_path, args.config_path, args.gpu_id)
