import argparse
import os
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

import config
from storage import DataLog


MultiAssetData = Dict[str, Tuple[Dict[int, np.ndarray], List[float], List[int]]]


def _coerce_asset_vector_to_dtype(vector: Any, expected_dim: int, dtype: np.dtype) -> np.ndarray:
    try:
        arr = np.asarray(vector, dtype=dtype).reshape(-1)
        if arr.ndim != 1 or arr.shape[0] != expected_dim:
            return np.zeros((expected_dim,), dtype=dtype)
        return arr
    except Exception:
        return np.zeros((expected_dim,), dtype=dtype)


def coerce_plaintext_cache_to_dtype(datalog: DataLog, dtype: np.dtype) -> int:
    """
    Convert any legacy plaintext_cache entries (lists or wrong dtypes) into
    arrays of the requested dtype. Also handle older format where the per-uid value
    is a list-of-lists (one per asset) instead of a dict keyed by asset name.

    Returns the number of vectors converted.
    """
    converted = 0
    assets = list(config.ASSETS)
    dims = {a: int(config.ASSET_EMBEDDING_DIMS[a]) for a in assets}

    for step in datalog.plaintext_cache:
                                                                     
        for uid, entry in list(step.items()):
                                                                                      
            if isinstance(entry, list) and len(entry) == len(assets) and all(
                isinstance(x, (list, np.ndarray)) for x in entry
            ):
                entry_dict: Dict[str, np.ndarray] = {}
                for i, asset in enumerate(assets):
                    entry_dict[asset] = _coerce_asset_vector_to_dtype(entry[i], dims[asset], dtype)
                    converted += 1
                step[uid] = entry_dict
                continue

                                                          
            if isinstance(entry, dict):
                new_entry: Dict[str, np.ndarray] = {}
                for asset in assets:
                    vec = entry.get(asset)
                    new_entry[asset] = _coerce_asset_vector_to_dtype(vec, dims[asset], dtype)
                    converted += 1
                step[uid] = new_entry
            else:
                                                                    
                step[uid] = {a: np.zeros((dims[a],), dtype=dtype) for a in assets}
                converted += len(assets)

    return converted

                              
def coerce_plaintext_cache_to_np16(datalog: DataLog) -> int:
    return coerce_plaintext_cache_to_dtype(datalog, np.float16)


def _filter_unchanged_prices(prices: List[float], max_unchanged: int) -> np.ndarray:
    if not prices:
        return np.array([])
    price_array = np.array(prices, dtype=np.float64)
    is_stagnant = np.zeros_like(price_array, dtype=bool)
    unchanged_count = 0
    last_price = None
    for i, price in enumerate(price_array):
        if last_price is None or price != last_price:
            unchanged_count = 0
        else:
            unchanged_count += 1
        if unchanged_count > max_unchanged:
            is_stagnant[i] = True
        last_price = price
    price_array[is_stagnant] = np.nan
    return price_array


def build_training_data(
    datalog: DataLog,
    max_block_number: Optional[int] = None,
    skip_initial_timesteps: int = 1000,
    max_uids: Optional[int] = None,
    dtype: np.dtype = np.float16,
) -> Optional[MultiAssetData]:
    if not datalog.plaintext_cache:
        return None

    if max_block_number is not None:
        end_idx = next((i for i, b in enumerate(datalog.blocks) if b >= max_block_number), len(datalog.blocks))
        if end_idx == 0:
            return None
        blocks = datalog.blocks[:end_idx]
        asset_prices = datalog.asset_prices[:end_idx] if datalog.asset_prices else []
        plaintext_cache = datalog.plaintext_cache[:end_idx]
    else:
        blocks = datalog.blocks
        asset_prices = datalog.asset_prices if datalog.asset_prices else []
        plaintext_cache = datalog.plaintext_cache

    if len(blocks) <= skip_initial_timesteps:
        return None

    blocks = blocks[skip_initial_timesteps:]
    asset_prices = asset_prices[skip_initial_timesteps:] if asset_prices else []
    plaintext_cache = plaintext_cache[skip_initial_timesteps:]

    if len(blocks) < config.LAG * 2 + 1:
        return None
    if not asset_prices:
        return None

    T_total = len(blocks)

                                                
                                                                                  
    uid_counts: Dict[int, int] = {}
    for step in plaintext_cache:
        for uid in step.keys():
            uid_counts[uid] = uid_counts.get(uid, 0) + 1
    all_uids: List[int] = sorted(uid_counts.keys())
    if not all_uids:
        return None

                                                                  
    selected_uids: List[int]
    if max_uids is not None and len(all_uids) > int(max_uids):
                                                                    
        top = sorted(all_uids, key=lambda u: (-uid_counts[u], u))[: int(max_uids)]
        selected_uids = sorted(top)
    else:
        selected_uids = all_uids

    block_to_idx = {b: i for i, b in enumerate(blocks)}
    TARGET_BLOCK_DIFF = 300

    result: MultiAssetData = {}
    assets = list(config.ASSETS)
    dims = {a: int(config.ASSET_EMBEDDING_DIMS[a]) for a in assets}

    for asset in assets:
                                             
        price_series: List[float] = []
        for t in range(T_total):
            if t < len(asset_prices) and asset_prices[t] and asset in asset_prices[t]:
                price_series.append(float(asset_prices[t][asset]))
            else:
                price_series.append(np.nan)

        price_series = _filter_unchanged_prices(price_series, int(config.MAX_UNCHANGED_TIMESTEPS))

        asset_returns: List[float] = []
        valid_embedding_indices: List[int] = []

        for t_idx, p_initial in enumerate(price_series):
            target_block = blocks[t_idx] + TARGET_BLOCK_DIFF
            j = block_to_idx.get(target_block)
            if j is None:
                continue
            p_final = price_series[j]
            if not np.isnan(p_initial) and not np.isnan(p_final) and p_initial > 0:
                asset_returns.append(float((p_final - p_initial) / p_initial))
                valid_embedding_indices.append(t_idx)

        if not asset_returns:
            continue

        T_valid = len(valid_embedding_indices)
        emb_dim = dims[asset]

                                                                                         
        num_uids = len(selected_uids)
        buffer = np.zeros((num_uids, T_valid, emb_dim), dtype=dtype)
        uid_to_row = {uid: i for i, uid in enumerate(selected_uids)}
        history_dict: Dict[int, np.ndarray] = {}
        for uid, row in uid_to_row.items():
            history_dict[uid] = buffer[row]

                                                           
        for row_idx, t_idx in enumerate(valid_embedding_indices):
            if t_idx >= len(plaintext_cache):
                continue
            step = plaintext_cache[t_idx]
                                                             
                                                                        
            for uid, entry in list(step.items()):
                if isinstance(entry, list):
                                                 
                    coerced = _coerce_asset_vector_to_dtype(entry[assets.index(asset)] if assets.index(asset) < len(entry) else [], emb_dim, dtype)
                elif isinstance(entry, dict):
                    coerced = _coerce_asset_vector_to_dtype(entry.get(asset), emb_dim, dtype)
                else:
                    coerced = np.zeros((emb_dim,), dtype=dtype)
                if uid in history_dict:
                    history_dict[uid][row_idx, :] = coerced

        sample_blocks = [int(blocks[t_idx]) for t_idx in valid_embedding_indices]
        result[asset] = (history_dict, asset_returns, sample_blocks)

    if not result:
        return None
    return result


def load_or_download_datalog(archive_path: Optional[str]) -> DataLog:
    local_path = archive_path or os.path.join(config.STORAGE_DIR, "mantis_datalog.pkl.gz")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    return DataLog.load(local_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=None, help="Path to local datalog .pkl.gz (defaults to STORAGE_DIR/mantis_datalog.pkl.gz)")
    ap.add_argument("--max_block", type=int, default=None, help="Optional maximum block number to include")
    ap.add_argument("--skip_initial", type=int, default=1000, help="Timesteps to skip from the beginning")
    args = ap.parse_args()

    datalog = load_or_download_datalog(args.archive)

    num_converted = coerce_plaintext_cache_to_np16(datalog)
    print(f"Coerced {num_converted} asset vectors to np.float16 (includes already-correct entries).")

    data = build_training_data(datalog, max_block_number=args.max_block, skip_initial_timesteps=args.skip_initial)
    if not data:
        print("No training data available.")
        return

    total_assets = len(data)
    total_uids = len({uid for asset in data.values() for uid in asset[0].keys()})
    total_samples = next((len(v[1]) for v in data.values()), 0)
    print(f"Built training data: assets={total_assets}, uids={total_uids}, samples={total_samples}")

    for asset, (hist, returns, blocks) in data.items():
        emb_dim = int(config.ASSET_EMBEDDING_DIMS[asset])
        any_uid = next(iter(hist)) if hist else None
        shape = hist[any_uid].shape if any_uid is not None else (0, emb_dim)
        print(f"- {asset}: T={len(returns)}, D={emb_dim}, per-uid array shape={shape}")


if __name__ == "__main__":
    main()


