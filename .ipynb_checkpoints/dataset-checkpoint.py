import torch
import itertools
from torch.utils.data import Dataset
import pickle
from util import BioZernikeMoment
from itertools import combinations

class ProteinPairDataset(Dataset):
    def __init__(self, df=None, cache_path=None, features=None, labels=None):
        """
        Use one of the following:
        - `df`: Compute on-the-fly (slow for large datasets)
        - `cache_path`: Load cached (recommended)
        - `(features, labels)`: Preloaded from merged cache parts
        """
        self.features = []
        self.labels = []

        if cache_path:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            self.features = cache["features"]
            self.labels = cache["labels"]

        elif features is not None and labels is not None:
            self.features = features
            self.labels = labels

        elif df is not None:
            from itertools import combinations
            from util import BioZernikeMoment 

            self.pairs = list(combinations(range(len(df)), 2))
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

                dist = bio_i.distance_vector(bio_j)  # Full vector 

                self.features.append(torch.tensor(dist, dtype=torch.float32))
                self.labels.append(torch.tensor(label, dtype=torch.float32))

        else:
            raise ValueError("Must provide either df, cache_path, or (features + labels)")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class StreamingProteinPairDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.num_proteins = len(df)

        # Build index mapping for combinations 
        self.index_pairs = list(itertools.combinations(range(self.num_proteins), 2))

    def __len__(self):
        return len(self.index_pairs)

    def __getitem__(self, idx):
        i, j = self.index_pairs[idx]

        row_i = self.df.iloc[i]
        row_j = self.df.iloc[j]

        label = int(row_i[0] == row_j[0])
        geom_i = row_i[1:18].values
        geom_j = row_j[1:18].values
        zern_i = row_i[18:].values
        zern_j = row_j[18:].values

        bio_i = BioZernikeMoment(geom_i, zern_i)
        bio_j = BioZernikeMoment(geom_j, zern_j)

        distance_vector = bio_i.distance_vector(bio_j)

        return torch.tensor(distance_vector, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

