import torch
import pickle
import pandas as pd
from itertools import combinations
from multiprocessing import Pool, cpu_count
import os
import sys

from util import BioZernikeMoment

# Shared global DataFrame for multiprocessing
_global_df = None

def init_worker(df):
    global _global_df
    _global_df = df

def compute_pair_static(pair):
    i, j = pair
    row_i = _global_df.iloc[i]
    row_j = _global_df.iloc[j]

    label = int(row_i[0] == row_j[0])
    geom_i = row_i[1:18].values
    geom_j = row_j[1:18].values
    zern_i = row_i[18:].values
    zern_j = row_j[18:].values

    bio_i = BioZernikeMoment(geom_i, zern_i)
    bio_j = BioZernikeMoment(geom_j, zern_j)
    dist_vec = bio_i.distance_vector(bio_j)

    return (
        torch.tensor(dist_vec, dtype=torch.float32),
        torch.tensor(label, dtype=torch.float32)
    )

def estimate_buffer_size(features, labels):
    return sys.getsizeof(features) + sys.getsizeof(labels)

def flush_buffer(buffer_id, buffer_features, buffer_labels, cache_dir):
    path = os.path.join(cache_dir, f"part_{buffer_id}.pkl")
    with open(path, "wb") as f:
        pickle.dump({
            "features": buffer_features,
            "labels": buffer_labels
        }, f)
    print(f"Flushed buffer {buffer_id} with {len(buffer_features)} items → {path}")

def cache_pairwise_data(df: pd.DataFrame, cache_dir: str, max_pairs=None, buffer_limit_mb=512):
    """
    Process all protein pairs in parallel and flush in 512MB chunks using double buffering.
    """
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Generating pairwise data from {len(df)} proteins using {cpu_count()} cores...")

    all_pairs = list(combinations(range(len(df)), 2))
    if max_pairs:
        all_pairs = all_pairs[:max_pairs]

    buffer_limit_bytes = buffer_limit_mb * 1024 * 1024
    buffer_features = []
    buffer_labels = []
    buffer_id = 0

    with Pool(processes=cpu_count(), initializer=init_worker, initargs=(df,)) as pool:
        for idx, result in enumerate(pool.imap(compute_pair_static, all_pairs, chunksize=64)):
            feat, label = result
            buffer_features.append(feat)
            buffer_labels.append(label)

            if idx % 1000 == 0 or estimate_buffer_size(buffer_features, buffer_labels) > buffer_limit_bytes:
                flush_buffer(buffer_id, buffer_features, buffer_labels, cache_dir)
                buffer_features, buffer_labels = [], []
                buffer_id += 1

    # Final flush
    if buffer_features:
        flush_buffer(buffer_id, buffer_features, buffer_labels, cache_dir)

    print(f" Done. Total buffers flushed: {buffer_id + 1}")
