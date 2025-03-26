import pickle
import pandas as pd
from dataset import ProteinPairDataset

def cache_pairwise_data(df: pd.DataFrame, cache_path: str, max_pairs=None):
    """
    Precompute protein pair features and labels and save them to a pickle file.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing BioZernike descriptors.
        cache_path (str): Path to save the output .pkl file.
        max_pairs (int, optional): Optional limit on number of pairs to compute.
    """
    print(f"Generating pairwise data from {len(df)} proteins...")

    dataset = ProteinPairDataset(df=df, max_pairs=max_pairs)

    cache = {
        "features": dataset.features,
        "labels": dataset.labels
    }

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    print(f"Saved {len(dataset)} precomputed protein pairs to: {cache_path}")
