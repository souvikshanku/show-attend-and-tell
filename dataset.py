import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image


class MultiMnist(Dataset):
    def __init__(self, annotations_file, img_dir, device):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.device = device

        self.num_classes = 12
        self.start_token = 10
        self.end_token = 11
        self.seq_len = 4

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)

        label = self.img_labels.iloc[idx, 1]
        label = self.one_hot_encode(label)

        image = image.to(self.device)
        label = label.to(self.device)

        return image, label

    def one_hot_encode(self, label):
        tokens = [self.start_token]

        # 42 -> [<start>, 4, 2]
        tokens += list(map(int, str(label)))
        # [<start>, 4, 2] -> [<start>, 4, 2, <end>]
        tokens += [self.end_token] * (self.seq_len - len(tokens))

        y = F.one_hot(torch.tensor(tokens), self.num_classes)
        return y
