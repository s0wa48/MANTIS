# MIT License
#
# Copyright (c) 2024 MANTIS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import config
import os, time, csv
from datetime import datetime

import torch
import logging
import random
import numpy as np
import xgboost as xgb
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)
logger.info("Salience computations will run on %s", DEVICE)

try:
    _NUM_CPU = max(1, os.cpu_count() or 1)
    torch.set_num_threads(_NUM_CPU)
    torch.set_num_interop_threads(_NUM_CPU)
    logger.info("Torch thread pools set to %d", _NUM_CPU)
except Exception as e:
    logger.warning("Could not set torch thread counts: %s", e)

def set_global_seed(seed: int) -> None:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        logger.info("Deterministic PyTorch algorithms enabled.")
    except (RuntimeError, AttributeError) as e:
        logger.warning(f"Could not enable deterministic algorithms: {e}")


def ten_day_saliences(
    multi_asset_data: dict[str, tuple[dict[int, list[list[float]]], list[float]]],
) -> dict[int, float]:
    window = 60 * 24 * 10
    trimmed = {}
    for asset, (hist, ret) in multi_asset_data.items():
        slice_start = max(0, len(ret) - window)
        trimmed_hist = {uid: h[slice_start:] for uid, h in hist.items()}
        trimmed[asset] = (trimmed_hist, ret[slice_start:])
    return salience(trimmed)


def salience(
    multi_asset_data: dict[str, tuple[dict[int, list[list[float]]], list[float]]],
) -> dict[int, float]:

    set_global_seed(config.SEED)
    if not multi_asset_data:
        return {}

    all_uids = sorted({k for asset in multi_asset_data.values() for k in asset[0].keys()})
    if not all_uids:
        return {}

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 0,
    }
    if DEVICE == "cuda":
        xgb_params.update(device="cuda", tree_method="gpu_hist", predictor="gpu_predictor")
    else:
        xgb_params["nthread"] = os.cpu_count()

    xgb_rounds = 50
    CHUNK_SIZE = 2000
    LAG = config.LAG

    asset_uid_importance: dict[str, dict[int, float]] = {}

    for asset_name, (history_dict, asset_returns) in multi_asset_data.items():
        logger.info(f"[ALT] Processing asset: {asset_name}")
        if not history_dict or not asset_returns:
            continue

        emb_dim = config.ASSET_EMBEDDING_DIMS[asset_name]
        uids = sorted(history_dict.keys())
        uid_to_idx = {u: i for i, u in enumerate(uids)}
        num_uids = len(uids)
        T = len(asset_returns)

        X = np.zeros((T, num_uids * emb_dim), dtype=np.float32)
        for uid, hist in history_dict.items():
            if uid in uid_to_idx and len(hist) == T:
                idx = uid_to_idx[uid]
                X[:, idx * emb_dim : (idx + 1) * emb_dim] = np.array(hist, dtype=np.float32)
        y_bin = (np.array(asset_returns, dtype=np.float32) > 0).astype(float)

        valid_rows = ~np.isnan(y_bin)
        X = X[valid_rows]
        y_bin = y_bin[valid_rows]
        T = X.shape[0]

        total_uid_imp = np.zeros(num_uids)
        num_windows = 0
        total_val_acc = 0.0
        acc_windows = 0

        train_end = CHUNK_SIZE
        bst = None
        pbar_total = 0
        tmp = CHUNK_SIZE
        while (tmp + CHUNK_SIZE) <= T:
            pbar_total += 1
            tmp += CHUNK_SIZE
        pbar = tqdm(total=pbar_total, desc=f"ALT Walk-fwd {asset_name}")

        while True:
            val_start = train_end + LAG
            val_end = train_end + CHUNK_SIZE
            if val_end > T:
                break

            X_train, y_train = X[:train_end], y_bin[:train_end]
            X_val, y_val = X[val_start:val_end], y_bin[val_start:val_end]

            if len(X_train) < 100 or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                train_end = val_end
                pbar.update(1)
                continue

            dtrain = xgb.DMatrix(X_train, label=y_train)
            bst = xgb.train(xgb_params, dtrain, num_boost_round=xgb_rounds, xgb_model=bst, verbose_eval=False)

            dval = xgb.DMatrix(X_val, label=y_val)
            base_preds = (bst.predict(dval) > 0.5).astype(int)
            base_acc = np.mean(base_preds == y_val)
            total_val_acc += base_acc
            acc_windows += 1

            window_imp = np.zeros(num_uids)
            for i, uid in enumerate(uids):
                cols = slice(i * emb_dim, (i + 1) * emb_dim)
                X_perm = X_val.copy()
                X_perm[:, cols] = np.random.permutation(X_perm[:, cols])
                perm_preds = (bst.predict(xgb.DMatrix(X_perm, label=y_val)) > 0.5).astype(int)
                window_imp[i] = max(base_acc - np.mean(perm_preds == y_val), 0)

            total_uid_imp += window_imp
            num_windows += 1
            train_end = val_end
            pbar.update(1)
        pbar.close()

        if acc_windows:
            logger.info(f"[ALT] Avg val-acc for {asset_name}: {total_val_acc/acc_windows:.4f}")
        if num_windows:
            asset_uid_importance[asset_name] = dict(zip(uids, total_uid_imp / num_windows))

    if not asset_uid_importance:
        return {}

    final_imp = {uid: 0.0 for uid in all_uids}
    for imp in asset_uid_importance.values():
        for uid, score in imp.items():
            final_imp[uid] += score
    total = sum(final_imp.values())
    return {uid: s/total for uid, s in final_imp.items()} if total > 1e-9 else {}




