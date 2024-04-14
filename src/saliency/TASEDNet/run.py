""" This file is modified from:
https://github.com/MichiganCOG/TASED-Net/blob/master/run_example.py
"""
import os
import numpy as np
import cv2
import torch
import argparse
from torchvision.io import write_video
from scipy.ndimage import gaussian_filter

from saliency.TASEDNet.model import TASED_v2
import helper


MODEL_PATH = 'models/saliency/tasednet_iter_1000.pt'
LEN_TEMPORAL = 32


def parse_arguments():
    parser = argparse.ArgumentParser(description="Saliency implementation")
    parser.add_argument("data_path", help="")
    parser.add_argument("output_path", help="")
    parser.add_argument("config_path", help="")
    return parser.parse_args()


def main(data_path: str, output_path: str, config_path: str):
    # Load config
    config = helper.load_yml(config_path)
    grid_config = config["grid_config"]

    # Load model
    model = TASED_v2()
    weight_dict = torch.load(MODEL_PATH)
    model_dict = model.state_dict()
    for name, param in weight_dict.items():
        if 'module' in name:
            name = '.'.join(name.split('.')[1:])
        if name in model_dict:
            if param.size() == model_dict[name].size():
                model_dict[name].copy_(param)
            else:
                raise ValueError("Mismatch in model dimensions")
        else:
            raise ValueError("Missing model weights")

    model = model.cuda()
    torch.backends.cudnn.benchmark = False
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

        if len(list_frames) >= 2 * LEN_TEMPORAL - 1:
            snippet = []
            for i, img in enumerate(list_frames):
                img = cv2.resize(img, (384, 224))
                img = img[..., ::-1]  # BGR to RGB
                snippet.append(img)

                if i >= LEN_TEMPORAL - 1:
                    clip = transform(snippet)
                    sal_map = process(model, clip, original_dims)
                    video_sal_maps.append(sal_map)
                    if i < 2 * LEN_TEMPORAL - 2:
                        flipped_clip = torch.flip(clip, [2])
                        sal_map = process(model, flipped_clip, original_dims)
                        video_sal_maps.append(sal_map)
                    del snippet[0]

            video_tensor = torch.stack([torch.from_numpy(frame).unsqueeze(-1).repeat(1, 1, 3) for frame in video_sal_maps])
            write_video(output_file, video_tensor.numpy(), fps=grid_config["fps"])
        else:
            raise ValueError("More frames needed")


def transform(snippet):
    """Normalize and stack snippets for model input."""
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.).sub_(255).div(255)
    return snippet.view(1, -1, 3, snippet.size(1), snippet.size(2)).permute(0, 2, 1, 3, 4)


def process(model, clip, original_dims):
    """Process clip through model, resize and return the saliency map."""
    with torch.no_grad():
        smap = model(clip.cuda()).cpu().data[0]
    smap = gaussian_filter(smap.squeeze(), sigma=7)
    normalized_smap = (smap / smap.max() * 255).astype(np.uint8)
    resized_smap = cv2.resize(normalized_smap, (original_dims[1], original_dims[0]))  # Width, Height
    return resized_smap


if __name__ == '__main__':
    args = parse_arguments()
    main(args.data_path, args.output_path, args.config_path)
