import torch
import itertools
from torch.utils.data import Dataset
import pickle
from util import BioZernikeMoment
from itertools import combinations
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from weight_strategy import inverse_class_weighting
import multiprocessing
import time

class ProteinPairDataset(Dataset):
    def __init__(self, df=None, cache_path=None, features=None, labels=None):
        """
        Use one of the following:
        - `df`: Compute on-the-fly (slow for large datasets)
        - `cache_path`: Load cached (recommended)
        - `(features, labels)`: Preloaded from merged cache parts
        """
        start = time.time()
        self.features = []
        self.labels = []

        if cache_path:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            self.features = cache["features"]
            self.labels = cache["labels"]
            print(f"[INFO] Loading cached dataset from: {cache_path}")
            print(f"[INFO] Loaded {len(self.features)} samples from cache.")

        elif features is not None and labels is not None:
            self.features = features
            self.labels = labels
            print(f"[INFO] Using preloaded feature/label tensors with {len(features)} pairs.")

        elif df is not None:
            print(f"[INFO] Generating nC2 pairs from raw DataFrame of size {len(df)}...")
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
            print(f"[INFO] Finished pairwise generation: total {len(self.features)} pairs.")

        else:
            raise ValueError("Must provide either df, cache_path, or (features + labels)")

        print(f"[INFO] Dataset init completed in {time.time() - start:.2f} seconds")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class StreamingProteinPairDataset(Dataset):
    def __init__(self, df):
        print(f"[INFO] Initializing Streaming Dataset from DataFrame of size {len(df)}")
        start = time.time()
        self.df = df.reset_index(drop=True)
        self.num_proteins = len(df)

        # Build index mapping for combinations 
        self.index_pairs = list(itertools.combinations(range(self.num_proteins), 2))
        print(f"[INFO] Generated index pairs: {len(self.index_pairs)} total")
        print(f"[INFO] Streaming init done in {time.time() - start:.2f} seconds")

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
        # print(f"[INFO] distance_vector shape... streaming={bio_i.bio_vector_length()}")

        return torch.tensor(distance_vector, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def create_dataloaders(protein_df=None, features=None, labels=None, batch_size=64, val_split=0.2, streaming=False):
    print(f"[INFO] Creating dataloaders... streaming={streaming}")
    num_threads = multiprocessing.cpu_count()

    if features is not None and labels is not None:
        dataset = ProteinPairDataset(features=features, labels=labels)
        dataset_size = len(dataset)
        split = int(val_split * dataset_size)
        indices = list(range(dataset_size))
        train_indices, val_indices = indices[split:], indices[:split]
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
    elif protein_df is not None:
        train_df, val_df = train_test_split(protein_df, test_size=val_split, random_state=42, shuffle=True)
        if streaming:
            train_dataset = StreamingProteinPairDataset(train_df)
            val_dataset = StreamingProteinPairDataset(val_df)
        else:
            train_dataset = ProteinPairDataset(df=train_df)
            val_dataset = ProteinPairDataset(df=val_df)
    else:
        raise ValueError("Must provide either (features+labels) or protein_df")

    train_weights = inverse_class_weighting(train_dataset)
    sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_threads,
        prefetch_factor=2,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_threads,
        prefetch_factor=2,
        persistent_workers=True
    )

    return train_loader, val_loader
