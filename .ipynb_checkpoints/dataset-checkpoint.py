import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pickle
from itertools import combinations
from util import BioZernikeMoment

class ProteinPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame = None, max_pairs=None, cache_path=None):
        self.features = []
        self.labels = []

        if cache_path is not None:
            # Load precomputed distances and labels from cache
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            self.features = cache["features"]
            self.labels = cache["labels"]

        elif df is not None:
            # Compute distances and labels on-the-fly
            self.pairs = list(combinations(range(len(df)), 2))
            if max_pairs:
                self.pairs = self.pairs[:max_pairs]

            for i, j in self.pairs:
                row_i = df.iloc[i]
                row_j = df.iloc[j]

                label = int(row_i[0] == row_j[0])

                geom_i = row_i[1:18].values
                geom_j = row_j[1:18].values

                zern_i = row_i[18:].values
                zern_j = row_j[18:].values

                bio_i = BioZernikeMoment(geom_i, zern_i)
                bio_j = BioZernikeMoment(geom_j, zern_j)

                dist = bio_i.distance_vector(bio_j)

                self.features.append(torch.tensor(dist, dtype=torch.float32))
                self.labels.append(torch.tensor(label, dtype=torch.float32))

        else:
            raise ValueError("Provide either 'df' or 'cache_path' to ProteinPairDataset.")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
