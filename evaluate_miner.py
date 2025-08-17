import argparse
import gzip
import json
import logging
import os
import pickle
from typing import Dict, Any

import numpy as np

import config
from storage import DataLog
import model


def get_embeddings(block_number: int) -> Dict[str, np.ndarray]:
    """
    User-provided hook to generate embeddings for a miner at a specific block.

    Parameters
    - block_number: the chain block number corresponding to a timestep we want
      to generate embeddings for.

    Returns
    - A mapping of asset -> ndarray[emb_dim]. For now this stub just returns zeros
      for every configured asset.
    """
    out: Dict[str, np.ndarray] = {}
    for asset in config.ASSETS:
        emb_dim = int(config.ASSET_EMBEDDING_DIMS[asset])
        out[asset] = np.zeros((emb_dim,), dtype=np.float16)
    return out


def _load_datalog(archive_path: str | None, prefer_local: bool) -> DataLog:
    os.makedirs(config.STORAGE_DIR, exist_ok=True)
    path = archive_path or os.path.join(config.STORAGE_DIR, "mantis_datalog.pkl.gz")
    logger = logging.getLogger("loader")
    if prefer_local and os.path.exists(path):
        logger.info("Loading local datalog from %s", path)
        try:
            with gzip.open(path, "rb") as f:
                log = pickle.load(f)
            try:
                log.raw_payloads = {}
                logger.info("Pruned raw_payloads after local load (prefer_local).")
            except Exception:
                pass
            return log
        except Exception as e:
            logger.warning("Local load failed (%s). Falling back to download.", e)
    return DataLog.load(path)


def _inject_miner_embeddings(
    multi_asset_data: Dict[str, tuple[dict[int, list], list[float], list[int]]],
    target_uid: int,
    last_days: float,
) -> Dict[str, tuple[dict[int, list], list[float], list[int]]]:
    """
    For each asset, ensure the target_uid exists and inject embeddings only in the
    last `last_days` days of the timeline (earlier entries remain zeros).
    """
    seconds_per_block = 12
    last_blocks = int(round((last_days * 24 * 60 * 60) / seconds_per_block))

    out: Dict[str, tuple[dict[int, list], list[float], list[int]]] = {}

    for asset, (history_dict, returns, blocks) in multi_asset_data.items():
        # Determine mask for last_days
        if blocks and last_blocks > 0:
            cutoff_block = blocks[-1] - last_blocks
            mask = (np.asarray(blocks, dtype=np.int64) >= cutoff_block)
        else:
            mask = np.zeros(len(blocks), dtype=bool)

        emb_dim = int(config.ASSET_EMBEDDING_DIMS[asset])
        T = len(returns)

        # Prepare a zero array for the entire timeline for the target UID
        target_arr = np.zeros((T, emb_dim), dtype=np.float16)

        # Populate only masked timesteps by calling the user hook per block
        for t_idx in range(T):
            if not mask[t_idx]:
                continue
            block_num = int(blocks[t_idx]) if t_idx < len(blocks) else None
            try:
                emap = get_embeddings(block_num) if block_num is not None else {}
            except Exception:
                emap = {}
            vec = emap.get(asset)
            if vec is None:
                continue
            try:
                row = np.asarray(vec, dtype=np.float16).reshape(-1)
            except Exception:
                continue
            if row.ndim == 1 and row.shape[0] == emb_dim:
                target_arr[t_idx, :] = row
            else:
                logging.getLogger("evaluate_miner").warning(
                    "Ignoring embedding for asset %s at block %s due to shape %s (expected %d)",
                    asset, str(block_num), getattr(row, "shape", None), emb_dim,
                )

        new_hist = dict(history_dict)
        new_hist[target_uid] = target_arr
        out[asset] = (new_hist, returns, blocks)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=str, default=None, help="Path to local datalog .pkl.gz (defaults to STORAGE_DIR/mantis_datalog.pkl.gz)")
    ap.add_argument("--prefer_local", action="store_true", help="If set, load existing local archive if it exists; otherwise download")
    ap.add_argument("--uid", type=int, default=0, help="Target UID to evaluate (default: 0)")
    ap.add_argument("--last_days", type=float, default=15.0, help="Simulate embeddings only in the last N days (earlier entries zeros)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("evaluate_miner")

    # Load (or download) datalog and build training data
    log = _load_datalog(args.archive, args.prefer_local)
    logger.info("Preparing training data from datalog...")
    multi_asset_data = log.get_training_data()
    if not multi_asset_data:
        logger.error("No training data available.")
        return

    # Ensure FP16 numpy arrays to save memory downstream
    logger.info("Converting training data to float16 structures...")
    multi_asset_data = model.convert_multi_asset_data_to_np16(multi_asset_data)

    # Inject the miner's embeddings for the last N days
    logger.info("Injecting miner embeddings for UID %d over last %.1f days...", args.uid, args.last_days)
    injected = _inject_miner_embeddings(multi_asset_data, target_uid=args.uid, last_days=args.last_days)

    # Compute weights
    logger.info("Computing salience-based weights...")
    weights = model.salience(injected)
    if not weights:
        logger.error("Empty weights computed; the evaluation window may be too short or data is insufficient.")
        return

    # Report target UID share
    target_weight = float(weights.get(args.uid, 0.0))
    logger.info("UID %d weight: %.6f (%.2f%%)", args.uid, target_weight, target_weight * 100.0)
    print(json.dumps({"uid": args.uid, "weight": target_weight, "percent": target_weight * 100.0}, indent=2))


if __name__ == "__main__":
    main() 
