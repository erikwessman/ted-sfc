""" This file is modified from:
https://github.com/Cogito2012/DRIVE/blob/master/src/DADALoader.py
"""
import os
import numpy as np
import cv2
from torch.utils.data import Dataset


class Dataloader(Dataset):
    def __init__(self, root_path, transforms=None, params_norm=None):
        self.root_path = root_path
        self.transforms = transforms
        self.params_norm = params_norm

        self.data_list = self.get_data_list()

    def get_data_list(self):
        data_list = []

        for video_id in sorted(os.listdir(self.root_path)):
            data_list.append(video_id)

        return data_list

    def gather_info(self, index, video_data):
        video_id = self.data_list[index]
        nr_frames = video_data.shape[0]
        height = video_data.shape[1]
        width = video_data.shape[2]
        return [video_id, nr_frames, height, width]

    def get_video_data(self, index):
        video_id = self.data_list[index]
        video_path = os.path.join(self.root_path, video_id, f"{video_id}.avi")
        assert os.path.exists(video_path), "Path does not exist: %s" % (video_path)

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()

        video_data = []

        while ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_data.append(frame)
            ret, frame = cap.read()

        video_data = np.array(video_data, dtype=np.float32)
        return video_data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        video = self.get_video_data(index)
        video_info = self.gather_info(index, video)

        if self.transforms is not None:
            video = self.transforms(video)
            if self.params_norm is not None:
                for i in range(video.shape[1]):
                    video[:, i] = (
                        video[:, i] - self.params_norm["mean"][i]
                    ) / self.params_norm["std"][i]

        return video, video_info
