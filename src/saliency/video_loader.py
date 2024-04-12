from torchvision import transforms as T
import os
import cv2
from torch.utils.data import Dataset
import torch


class VideoSequenceLoader(Dataset):
    def __init__(self, root_path, sequence_length=5, transforms=None, params_norm=None):
        self.root_path = root_path
        self.sequence_length = sequence_length
        self.transforms = transforms
        self.params_norm = params_norm
        self.data_list = self.get_data_list()

    def get_data_list(self):
        video_data_list = []
        for video_id in sorted(os.listdir(self.root_path)):
            video_dir = os.path.join(self.root_path, video_id)
            frame_files = sorted(os.listdir(video_dir))
            if len(frame_files) >= self.sequence_length:
                video_data_list.append((video_id, frame_files))
        return video_data_list

    def __getitem__(self, index):
        video_id, frame_files = self.data_list[index]
        video_frames = []
        for i in range(self.sequence_length):
            frame_path = os.path.join(self.root_path, video_id, frame_files[i])
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transforms:
                frame = self.transforms(frame)
            video_frames.append(frame)

        # Stack frames along a new dimension
        video_tensor = torch.stack(video_frames)
        if self.params_norm:
            mean = self.params_norm['mean']
            std = self.params_norm['std']
            video_tensor = (video_tensor - mean) / std  # Normalize if required

        return video_tensor, video_id

    def __len__(self):
        return len(self.data_list)


# Usage example
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((480, 640)),
    T.ToTensor()
])
video_loader = VideoSequenceLoader(
    "/path/to/videos", sequence_length=5, transforms=transform)
