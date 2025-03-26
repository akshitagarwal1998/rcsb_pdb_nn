import torch
from torch.utils.data import Dataset
import pickle

class ProteinPairDataset(Dataset):
    def __init__(self, df=None, cache_path=None, features=None, labels=None):
        """
        You must provide one of:
        - df (for dynamic pair generation),
        - cache_path (precomputed pickle file),
        - or (features, labels) pair from merged parts

        dataset = ProteinPairDataset(cache_path="cache/cath_pairs.pkl")

        features, labels = load_cached_parts("cache/cath_buffered")
        dataset = ProteinPairDataset(features=features, labels=labels)

        dataset = ProteinPairDataset(df=cath_df.head(100))    
        
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
            from util import BioZernikeMoment  # Imported here to avoid circular issues

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

                dist = bio_i.distance_vector(bio_j)

                self.features.append(torch.tensor(dist, dtype=torch.float32))
                self.labels.append(torch.tensor(label, dtype=torch.float32))

        else:
            raise ValueError("Must provide either df, cache_path, or (features + labels)")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
