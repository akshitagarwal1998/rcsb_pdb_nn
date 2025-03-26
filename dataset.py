import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from itertools import combinations
from util import BioZernikeMoment

class ProteinPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_pairs=None):
        self.pairs = list(combinations(range(len(df)), 2))
        if max_pairs:
            self.pairs = self.pairs[:max_pairs]

        self.features = []
        self.labels = []

        for i, j in self.pairs:
            row_i, row_j = df.iloc[i], df.iloc[j]
            label = int(row_i[0] == row_j[0])

            geom_i, geom_j = row_i[1:18].values, row_j[1:18].values
            zern_i, zern_j = row_i[18:].values, row_j[18:].values

            bio_i = BioZernikeMoment(geom_i, zern_i)
            bio_j = BioZernikeMoment(geom_j, zern_j)
            dist = bio_i.distance_vector(bio_j)

            self.features.append(torch.tensor(dist, dtype=torch.float32))
            self.labels.append(torch.tensor(label, dtype=torch.float32))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
