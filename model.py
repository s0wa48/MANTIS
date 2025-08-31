from __future__ import annotations

import logging
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

import config


LAST_DEBUG: dict = {}


logger = logging.getLogger(__name__)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Salience computations will run on %s", DEVICE)


try:
    _NUM_CPU = max(1, os.cpu_count() or 1)
    torch.set_num_threads(_NUM_CPU)
    torch.set_num_interop_threads(_NUM_CPU)
    logger.info("Torch thread pools set to %d", _NUM_CPU)
except Exception as e:
    logger.warning("Could not set torch thread counts: %s", e)


def set_global_seed(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        logger.info("Deterministic PyTorch algorithms enabled.")
    except (RuntimeError, AttributeError) as e:
        logger.warning(f"Could not enable deterministic algorithms: {e}")


def _xgb_params() -> Dict[str, object]:
    params: Dict[str, object] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 2,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 0,
        "tree_method": "hist",
    }
    if DEVICE == "cuda":
        params.update({"device": "cuda", "tree_method": "gpu_hist", "predictor": "gpu_predictor"})
    else:
        params["nthread"] = os.cpu_count()
    return params


def _reshape_X_to_hotkey_dim(X: np.ndarray, H: int, D: int) -> np.ndarray:
    """Reshape flattened features (N, H*D) into (N, H, D) without copying if possible."""
    if X.ndim != 2 or X.shape[1] != H * D:
        raise ValueError(f"Unexpected X shape {X.shape}, expected (*, {H*D}) for H={H}, D={D}")
    return X.reshape(X.shape[0], H, D)


def salience(
    training_data: Dict[str, Tuple[Tuple[np.ndarray, Dict[str, int]], np.ndarray]]
) -> Dict[str, float]:
    """
    Compute salience scores per hotkey, aggregated across assets, using
    walk-forward logistic selections + XGBoost with permutation importance.

    Input format (per asset): (hist, y)
      - hist = (X, hk2idx)
        * X: np.ndarray, shape (T, H*D), dtype float16 (or convertible). T is timesteps/samples
        * hk2idx: dict mapping hotkey string -> index [0..H-1]
      - y: np.ndarray, shape (T,), numeric returns (float32). Direction label is y > 0.

    Returns: dict mapping hotkey string -> normalized salience weight (sums to 1 if any > 0).
    """
    global LAST_DEBUG
    set_global_seed(config.SEED)

    if not training_data:
        return {}

    xgb_params = _xgb_params()
    xgb_rounds = 250
    CHUNK_SIZE = 2000
    LAG = int(config.LAG)
    EMBARGO_IDX = LAG
    TOP_K = 20

    WINDOWS_HALF_LIFE = 10
    recency_gamma = float(0.5 ** (1.0 / max(1, WINDOWS_HALF_LIFE)))

    asset_hotkey_importance: Dict[str, Dict[str, float]] = {}
    asset_window_weights: Dict[str, List[float]] = {}
    asset_last_selected: Dict[str, List[Tuple[str, float]]] = {}
    asset_avg_val_acc_map: Dict[str, float | None] = {}
    asset_avg_val_auc_unscaled_map: Dict[str, float | None] = {}
    asset_weighted_val_auc_map: Dict[str, float | None] = {}
    asset_salience_share: Dict[str, float] = {}

    for asset_name, (hist, asset_returns) in training_data.items():
        logger.info(f"[SAL] Processing asset: {asset_name}")
        if not isinstance(hist, tuple) or len(hist) != 2:
            logger.warning(f"[{asset_name}] Invalid hist format; expected (X, hk2idx)")
            continue
        X_flat, hk2idx = hist
        if X_flat is None or asset_returns is None:
            continue

        dim = config.ASSET_EMBEDDING_DIMS.get(asset_name)
        if dim is None:
            logger.warning(f"No embedding dim for asset {asset_name}; skipping.")
            continue
        if not isinstance(hk2idx, dict) or not hk2idx:
            logger.info(f"[{asset_name}] No hotkeys present; skipping.")
            continue

        try:
            X_flat = np.asarray(X_flat, dtype=np.float32)
        except Exception:
            logger.warning(f"[{asset_name}] Could not coerce X to float32; skipping.")
            continue
        y = np.asarray(asset_returns, dtype=np.float32)
        if X_flat.shape[0] != y.shape[0]:
            logger.warning(f"[{asset_name}] X and y length mismatch: {X_flat.shape[0]} vs {y.shape[0]}; skipping.")
            continue

        T = int(X_flat.shape[0])
        if T < 500:
            logger.info(f"[{asset_name}] Not enough samples (T={T}); skipping.")
            continue

        H = int(X_flat.shape[1] // dim)
        if H <= 0 or H * dim != X_flat.shape[1]:
            logger.warning(f"[{asset_name}] Inconsistent shape {X_flat.shape} for dim={dim}")
            continue

        y_bin = (y > 0).astype(np.float32)
        if len(np.unique(y_bin)) < 2:
            logger.info(f"[{asset_name}] y has <2 classes; skipping.")
            continue

        X = _reshape_X_to_hotkey_dim(X_flat, H, dim)

        first_nz_idx = np.full(H, T, dtype=np.int32)
        for j in range(H):
            row_j = X[:, j, :]
            nz = (row_j != 0).any(axis=1)
            nz_idx = np.flatnonzero(nz)
            if nz_idx.size > 0:
                first_nz_idx[j] = int(nz_idx[0])

        indices: List[Tuple[int, int, int]] = []
        start = 0
        while True:
            val_start_idx = start + LAG
            if val_start_idx >= T:
                break
            end_idx = min(start + CHUNK_SIZE, T)
            if end_idx <= start:
                break
            indices.append((start, val_start_idx, end_idx))
            start = end_idx

        if not indices:
            continue

        pbar = tqdm(total=len(indices), desc=f"SAL Walk-fwd {asset_name}")

        total_hk_imp = np.zeros(H, dtype=np.float32)
        total_val_auc = 0.0
        total_val_acc = 0.0
        auc_windows = 0
        acc_windows = 0
        total_weight = 0.0
        weighted_auc_sum = 0.0
        weighted_w_sum = 0.0
        window_index = 0
        window_weights_used: List[float] = []
        last_sel_auc: List[Tuple[str, float]] = []

        idx2hk = [None] * H
        try:
            for hk, idx in hk2idx.items():
                if 0 <= idx < H:
                    idx2hk[idx] = hk
        except Exception:
            idx2hk = [str(i) for i in range(H)]

        for (train_start, val_start, val_end) in indices:
            train_end = val_start
            y_train_all = y_bin[:train_end]
            y_val = y_bin[val_start:val_end]

            if (
                train_end < 200
                or len(np.unique(y_train_all)) < 2
                or len(np.unique(y_val)) < 2
            ):
                pbar.update(1)
                continue

            sel_eval_end = train_end
            sel_eval_start = max(0, sel_eval_end - CHUNK_SIZE)
            sel_fit_end = max(0, sel_eval_start - EMBARGO_IDX)
            if sel_fit_end < 50:
                pbar.update(1)
                continue

            sel_auc = np.zeros(H, dtype=np.float32)
            for j in range(H):
                if first_nz_idx[j] >= sel_fit_end:
                    sel_auc[j] = 0.5
                    continue
                Xi_fit = X[:sel_fit_end, j, :].astype(np.float32, copy=False)
                yi_fit = y_bin[:sel_fit_end]
                if len(np.unique(yi_fit)) < 2:
                    sel_auc[j] = 0.5
                    continue
                try:
                    clf = LogisticRegression(
                        penalty="l2",
                        C=0.5,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=200,
                    )
                    clf.fit(Xi_fit, yi_fit)
                    Xi_eval = X[sel_eval_start:sel_eval_end, j, :].astype(np.float32, copy=False)
                    yi_eval = y_bin[sel_eval_start:sel_eval_end]
                    if len(np.unique(yi_eval)) < 2 or Xi_eval.shape[0] == 0:
                        sel_auc[j] = 0.5
                    else:
                        scores = clf.decision_function(Xi_eval)
                        sel_auc[j] = float(roc_auc_score(yi_eval, scores))
                except Exception:
                    sel_auc[j] = 0.5

            top_k = min(TOP_K, H)
            selected_idx = np.argsort(-sel_auc)[:top_k]
            selected_idx.sort()
            if selected_idx.size == 0:
                pbar.update(1)
                continue

            fit_end_pred = max(0, val_start - EMBARGO_IDX)
            if fit_end_pred <= 0:
                pbar.update(1)
                continue

            X_train_sel = np.zeros((fit_end_pred, selected_idx.size), dtype=np.float32)
            X_val_sel = np.zeros((val_end - val_start, selected_idx.size), dtype=np.float32)

            oos_segments: List[Tuple[int, int, int]] = []
            start_oos = 0
            while True:
                val_start_oos = start_oos + LAG
                if val_start_oos >= fit_end_pred:
                    break
                end_idx_oos = min(start_oos + CHUNK_SIZE, fit_end_pred)
                if end_idx_oos <= val_start_oos:
                    break
                oos_segments.append((start_oos, val_start_oos, end_idx_oos))
                start_oos = end_idx_oos

            for col_idx, j in enumerate(selected_idx):
                if first_nz_idx[j] >= fit_end_pred or fit_end_pred < 50:
                    continue
                Xi_all = X[:, j, :].astype(np.float32, copy=False)

                for (oos_train_start, oos_val_start, oos_val_end) in oos_segments:
                    tr_fit_end_oos = max(0, oos_val_start - LAG)
                    if tr_fit_end_oos < 50:
                        continue
                    Xi_fit_oos = Xi_all[:tr_fit_end_oos]
                    yi_fit_oos = y_bin[:tr_fit_end_oos]
                    if len(np.unique(yi_fit_oos)) < 2:
                        continue
                    try:
                        clf_oos = LogisticRegression(
                            penalty="l2",
                            C=0.5,
                            class_weight="balanced",
                            solver="lbfgs",
                            max_iter=200,
                        )
                        clf_oos.fit(Xi_fit_oos, yi_fit_oos)
                        X_train_sel[oos_val_start:oos_val_end, col_idx] = clf_oos.decision_function(
                            Xi_all[oos_val_start:oos_val_end]
                        )
                    except Exception:
                        continue

                try:
                    Xi_fit = Xi_all[:fit_end_pred]
                    yi_fit = y_bin[:fit_end_pred]
                    if len(np.unique(yi_fit)) < 2:
                        continue
                    clf_val = LogisticRegression(
                        penalty="l2",
                        C=0.5,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=200,
                    )
                    clf_val.fit(Xi_fit, yi_fit)
                    X_val_sel[:, col_idx] = clf_val.decision_function(Xi_all[val_start:val_end])
                except Exception:
                    continue

            y_train_head = y_bin[:fit_end_pred]

            try:
                dtrain = xgb.DMatrix(X_train_sel, label=y_train_head)
                bst = xgb.train(xgb_params, dtrain, num_boost_round=xgb_rounds, verbose_eval=False)

                dval = xgb.DMatrix(X_val_sel, label=y_val)
                base_probs = bst.predict(dval)
                base_auc = float(roc_auc_score(y_val, base_probs))
                total_val_auc += base_auc
                auc_windows += 1
                base_preds = (base_probs > 0.5).astype(np.int8)
                base_acc = float(np.mean((base_preds == y_val).astype(np.float32)))
                total_val_acc += base_acc
                acc_windows += 1
            except Exception as e:
                logger.warning(f"[{asset_name}] XGBoost training/eval failed: {e}")
                pbar.update(1)
                continue
            finally:
                try:
                    del dtrain
                    del dval
                except Exception:
                    pass

            window_imp = np.zeros(H, dtype=np.float32)
            for local_col, j in enumerate(selected_idx):
                col = X_val_sel[:, local_col].copy()
                perm_idx = np.random.permutation(col.shape[0])
                X_val_sel[:, local_col] = col[perm_idx]
                try:
                    dval_perm = xgb.DMatrix(X_val_sel, label=y_val)
                    perm_probs = bst.predict(dval_perm)
                    perm_auc = float(roc_auc_score(y_val, perm_probs))
                    delta = base_auc - perm_auc
                    window_imp[j] = delta if delta > 0.0 else 0.0
                except Exception:
                    window_imp[j] = 0.0
                finally:
                    X_val_sel[:, local_col] = col
                    try:
                        del dval_perm
                    except Exception:
                        pass

            scale = max((base_auc - 0.5) / 0.5, 0.0)
            if scale <= 0:
                window_imp[:] = 0.0
            else:
                window_imp *= scale

            w = recency_gamma ** (max(0, len(indices) - 1 - window_index))
            window_weights_used.append(float(w))
            total_hk_imp += (w * window_imp).astype(np.float32)
            total_weight += w
            weighted_auc_sum += (w * base_auc)
            weighted_w_sum += w
            window_index += 1

            last_sel_auc = []
            for j in selected_idx:
                hk = idx2hk[j] if j < len(idx2hk) and idx2hk[j] is not None else str(j)
                last_sel_auc.append((hk, float(sel_auc[j])))

            pbar.update(1)

        pbar.close()

        if auc_windows:
            logger.info(f"[SAL] Avg val-auc for {asset_name}: {total_val_auc/auc_windows:.4f}")
        if weighted_w_sum > 0:
            logger.info(f"[SAL] Weighted val-auc for {asset_name}: {weighted_auc_sum/weighted_w_sum:.4f}")

        if total_weight > 0:
            norm_imp = (total_hk_imp / total_weight).tolist()
            imp_map: Dict[str, float] = {}
            for j, score in enumerate(norm_imp):
                hk = idx2hk[j] if j < len(idx2hk) and idx2hk[j] is not None else str(j)
                imp_map[hk] = float(score)
            asset_hotkey_importance[asset_name] = imp_map
            asset_last_selected[asset_name] = last_sel_auc if last_sel_auc else []
            asset_window_weights[asset_name] = window_weights_used
            avg_val_acc = float(total_val_acc / acc_windows) if acc_windows else None
            asset_avg_val_acc_map[asset_name] = avg_val_acc
            avg_val_auc_unscaled = float(total_val_auc / auc_windows) if auc_windows else None
            weighted_val_auc = float(weighted_auc_sum / weighted_w_sum) if weighted_w_sum > 0 else None
            asset_avg_val_auc_unscaled_map[asset_name] = avg_val_auc_unscaled
            asset_weighted_val_auc_map[asset_name] = weighted_val_auc

    if not asset_hotkey_importance:
        return {}

    try:
        asset_contrib = {a: float(sum(max(0.0, v) for v in imp.values())) for a, imp in asset_hotkey_importance.items()}
        total_contrib = float(sum(asset_contrib.values()))
        if total_contrib > 0.0:
            asset_salience_share = {a: (c / total_contrib) for a, c in asset_contrib.items()}
            for a in sorted(asset_salience_share.keys()):
                logger.info(f"[SAL] Asset salience share {a}: {asset_salience_share[a]*100:.2f}%")
    except Exception as e:
        logger.warning(f"Failed to compute per-asset salience shares: {e}")

    all_hotkeys = set(hk for imp in asset_hotkey_importance.values() for hk in imp.keys())
    final_imp: Dict[str, float] = {hk: 0.0 for hk in all_hotkeys}
    for imp in asset_hotkey_importance.values():
        for hk, score in imp.items():
            final_imp[hk] += max(0.0, float(score))

    total = float(sum(final_imp.values()))
    result = {hk: (score / total) for hk, score in final_imp.items()} if total > 1e-9 else {}

    LAST_DEBUG = {
        "window_weights": asset_window_weights,
        "selected_auc": asset_last_selected,
        "asset_avg_val_acc": asset_avg_val_acc_map,
        "asset_avg_val_auc": asset_avg_val_auc_unscaled_map,
        "asset_weighted_val_auc": asset_weighted_val_auc_map,
        "asset_salience_share": asset_salience_share,
    }
    return result

