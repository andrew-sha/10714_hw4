import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        super().__init__(transforms=transforms)

        self.base_folder = base_folder
        self.train = train
        self.p = p

        # Determine which batch files to load
        if train:
            batch_files = [
                "data_batch_1",
                "data_batch_2",
                "data_batch_3",
                "data_batch_4",
                "data_batch_5",
            ]
        else:
            batch_files = ["test_batch"]

        xs = []
        ys = []
        for bf in batch_files:
            path = os.path.join(base_folder, bf)
            with open(path, "rb") as f:
                try:
                    entry = pickle.load(f, encoding="latin1")
                except TypeError:
                    entry = pickle.load(f)

            # Handle str or bytes keys
            data_key = "data" if "data" in entry else b"data"
            label_key = "labels" if "labels" in entry else ("fine_labels" if "fine_labels" in entry else b"labels")

            data = np.array(entry[data_key], dtype=np.uint8)
            labels = np.array(entry[label_key], dtype=np.int64)

            xs.append(data)
            ys.append(labels)

        X = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)

        # Reshape from (N, 3072) to (N, 32, 32, 3) then scale to [0,1]
        N = X.shape[0]
        X = X.reshape(N, 3, 32, 32).transpose(0, 2, 3, 1)  # N,H,W,C
        X = X.astype(np.float32) / 255.0

        self.X = X
        self.y = y

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        img = self.X[index]
        lbl = int(self.y[index])

        # Apply transforms expecting H x W x C
        img = self.apply_transforms(img)

        # Return in CHW order
        img = img.transpose(2, 0, 1).astype(np.float32, copy=False)
        return img, lbl

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        return self.X.shape[0]
