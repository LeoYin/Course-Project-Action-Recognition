import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class Dataset(Dataset):
    def __init__(self, dataset_path,input_shape, sequence_length, training):
        self.training = training
        self.label_index = {'Phoning':0, 'ApplyingMakeUpOnLips':1, 'BrushingTeeth':2, 'CleaningFloor':3,'CleaningWindows':4, 'Drinking': 5, 'FoldingTextile':6, 'Ironing':7, 'PlayingHarmonica':8, 'TakingPhotosOrVideos':9}
        self.sequences = self._extract_sequence_paths(dataset_path)
        self.sequence_length = sequence_length
        self.label_names = ['Phoning','ApplyingMakeUpOnLips','BrushingTeeth','CleaningFloor','CleaningWindows','Drinking','FoldingTextile','Ironing','PlayingHarmonica','TakingPhotosOrVideos']
        self.num_classes = len(self.label_names)
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def _extract_sequence_paths(self, dataset_path):
        """ Extracts paths to sequences given the specified train / test split """
        sequence_paths = []
        lines=glob.glob(os.path.join(dataset_path, "*","*"))
        for line in lines:
                seq_name = line.split(".mp4")[0]
                sequence_paths += [seq_name]

        return sequence_paths

    def _activity_from_path(self, path):
        """ Extracts activity name from filepath """
        return path.split("/")[-2]

    def _frame_number(self, image_path):
        """ Extracts frame number from filepath """
        return int(image_path.split("/")[-1].split(".jpg")[0])

    def _pad_to_length(self, sequence):
        """ Pads the sequence to required sequence length """
        left_pad = sequence[0]
        if self.sequence_length is not None:
            while len(sequence) < self.sequence_length:
                sequence.insert(0, left_pad)
        return sequence

    def __getitem__(self, index):
        sequence_path = self.sequences[index % len(self)]
        # Sort frame sequence based on frame number
        image_paths = sorted(glob.glob(f"{sequence_path}/*.jpg"), key=lambda path: self._frame_number(path))
        # Pad frames sequences shorter than `self.sequence_length` to length
        image_paths = self._pad_to_length(image_paths)
        if self.training:
            # Randomly choose sample interval and start frame
            sample_interval = np.random.randint(1, len(image_paths) // self.sequence_length + 1)
            start_i = np.random.randint(0, len(image_paths) - sample_interval * self.sequence_length + 1)
            flip = np.random.random() < 0.5
        else:
            # Start at first frame and sample uniformly over sequence
            start_i = 0
            sample_interval = 1 if self.sequence_length is None else len(image_paths) // self.sequence_length
            flip = False
        # Extract frames as tensors
        image_sequence = []
        for i in range(start_i, len(image_paths), sample_interval):
            if self.sequence_length is None or len(image_sequence) < self.sequence_length:
                image_tensor = self.transform(Image.open(image_paths[i]))
                if flip:
                    image_tensor = torch.flip(image_tensor, (-1,))
                image_sequence.append(image_tensor)
        image_sequence = torch.stack(image_sequence)
        target = self._activity_from_path(sequence_path)
        return image_sequence, self.label_index[target]

    def __len__(self):
        return len(self.sequences)
