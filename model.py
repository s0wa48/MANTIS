
import config
import os
import logging
import random
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import torch
from sklearn.model_selection import TimeSeriesSplit

LAST_DEBUG = {}

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


def salience(
    multi_asset_data: dict[str, tuple[dict[int, list[list[float]]], list[float], list[int]]],
) -> dict[int, float]:

    global LAST_DEBUG
    set_global_seed(config.SEED)
    if not multi_asset_data:
        return {}

    all_uids = sorted({k for asset in multi_asset_data.values() for k in asset[0].keys()})
    if not all_uids:
        return {}

   
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 2,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 0,
    }
    if DEVICE == "cuda":
        xgb_params.update(device="cuda", tree_method="gpu_hist", predictor="gpu_predictor")
    else:
        xgb_params["nthread"] = os.cpu_count()

    xgb_rounds = 250
    CHUNK_SIZE = 2000
    LAG = config.LAG
    EMBARGO_IDX = LAG 
    TOP_K = 20 

    asset_uid_importance: dict[str, dict[int, float]] = {}
    asset_window_weights: dict[str, list[float]] = {}
    asset_last_selected: dict[str, list[tuple[int, float]]] = {}
    asset_avg_val_acc_map: dict[str, float | None] = {}
    asset_avg_val_auc_unscaled_map: dict[str, float | None] = {}
    asset_weighted_val_auc_map: dict[str, float | None] = {}
    asset_salience_share: dict[str, float] = {}

    for asset_name, (history_dict, asset_returns, sample_blocks) in multi_asset_data.items():
        logger.info(f"[SAL] Processing asset: {asset_name}")
        if (not history_dict) or (len(asset_returns) == 0):
            continue

        emb_dim = config.ASSET_EMBEDDING_DIMS[asset_name]
        uids = sorted(history_dict.keys())
        uid_to_idx = {u: i for i, u in enumerate(uids)}
        num_uids = len(uids)
        T = len(asset_returns)

                                                                               
        y_bin = (np.asarray(asset_returns, dtype=np.float32) > 0).astype(np.float32)
        block_arr = np.asarray(sample_blocks, dtype=np.int64)
        if T < 500 or len(np.unique(y_bin)) < 2:
            continue

        total_uid_imp = np.zeros(num_uids, dtype=np.float32)
        total_val_auc = 0.0
        total_val_acc = 0.0
        auc_windows = 0
        acc_windows = 0
        total_weight = 0.0
        weighted_auc_sum = 0.0
        weighted_w_sum = 0.0

       
        target_block_span = CHUNK_SIZE * 60 // 12
        indices = []
        start = 0
        while True:
            val_start_idx = start + LAG
            if val_start_idx >= T:
                break
            target_block = block_arr[start] + target_block_span
            end_idx = int(np.searchsorted(block_arr, target_block, side="left"))
            end_idx = min(end_idx, start + CHUNK_SIZE, T)
            if end_idx <= start:
                break
            indices.append((start, val_start_idx, end_idx))
            start = end_idx

        pbar_total = len(indices)
        if pbar_total == 0:
            continue
        pbar = tqdm(total=pbar_total, desc=f"SAL Walk-fwd {asset_name}")

       
       
        seconds_per_block = 12
        blocks_in_10d = (10 * 24 * 60 * 60) // seconds_per_block
        windows_for_10d = max(1, int(np.ceil(blocks_in_10d / float(target_block_span))))
        recency_gamma = float(0.5 ** (1.0 / windows_for_10d))
        window_index = 0
        window_weights_used: list[float] = []
        last_sel_auc: list[tuple[int, float]] = []

       
        first_nz_idx = np.full(num_uids, T, dtype=np.int32)
        for i, uid in enumerate(uids):
            hist = history_dict.get(uid)
            if hist is None or len(hist) != T:
                continue
            nz = (hist != 0).any(axis=1)
            nz_idx = np.flatnonzero(nz)
            if nz_idx.size > 0:
                first_nz_idx[i] = int(nz_idx[0])

        for (train_start, val_start, val_end) in indices:
            train_end = val_start
            y_train = y_bin[:train_end]
            y_val = y_bin[val_start:val_end]

           
            if (train_end < 200
                or len(np.unique(y_train)) < 2
                or len(np.unique(y_val)) < 2):
                pbar.update(1)
                continue

           
            tr_fit_end = max(0, train_end - LAG)
            if tr_fit_end < 50 or train_end - tr_fit_end < 20:
                pbar.update(1)
                continue

           
            sel_eval_end = train_end
            sel_eval_start = max(0, sel_eval_end - CHUNK_SIZE)
            sel_fit_end = max(0, sel_eval_start - EMBARGO_IDX)
            sel_auc = np.zeros(num_uids, dtype=np.float32)
            for i, uid in enumerate(uids):
                if first_nz_idx[i] >= sel_fit_end or sel_fit_end < 50:
                    sel_auc[i] = 0.5
                    continue
                hist = history_dict.get(uid)
                if hist is None or len(hist) != T:
                    sel_auc[i] = 0.5
                    continue
                                                                         
                Xi_fit = hist[:sel_fit_end].astype(np.float32, copy=False)
                yi_fit = y_bin[:sel_fit_end]
                if len(np.unique(yi_fit)) < 2:
                    sel_auc[i] = 0.5
                    continue
                try:
                    clf = LogisticRegression(penalty='l2', C=0.5, class_weight='balanced', solver='lbfgs', max_iter=200)
                    clf.fit(Xi_fit, yi_fit)
                    Xi_eval = hist[sel_eval_start:sel_eval_end].astype(np.float32, copy=False)
                    yi_eval = y_bin[sel_eval_start:sel_eval_end]
                    if len(np.unique(yi_eval)) < 2 or Xi_eval.shape[0] == 0:
                        sel_auc[i] = 0.5
                    else:
                        sel_auc[i] = float(roc_auc_score(yi_eval, clf.decision_function(Xi_eval)))
                except Exception:
                    sel_auc[i] = 0.5

            top_k = min(TOP_K, num_uids)
            selected_idx = np.argsort(-sel_auc)[:top_k]
            selected_idx.sort()
            if selected_idx.size == 0:
                pbar.update(1)
                continue

           
            fit_end_pred = max(0, val_start - EMBARGO_IDX)
           
            X_train_sel = np.zeros((fit_end_pred, selected_idx.size), dtype=np.float32)
            X_val_sel = np.zeros((val_end - val_start, selected_idx.size), dtype=np.float32)

           
            oos_segments: list[tuple[int, int, int]] = []
            start_oos = 0
            block_arr_train = block_arr[:fit_end_pred]
            while True:
                val_start_oos = start_oos + LAG
                if val_start_oos >= fit_end_pred:
                    break
                target_block_oos = block_arr_train[start_oos] + target_block_span
                end_idx_oos = int(np.searchsorted(block_arr_train, target_block_oos, side="left"))
                end_idx_oos = min(end_idx_oos, start_oos + CHUNK_SIZE, fit_end_pred)
                if end_idx_oos <= val_start_oos:
                    break
                oos_segments.append((start_oos, val_start_oos, end_idx_oos))
                start_oos = end_idx_oos

            for j, i in enumerate(selected_idx):
                if first_nz_idx[i] >= fit_end_pred or fit_end_pred < 50:
                    continue
                uid = uids[int(i)]
                hist = history_dict.get(uid)
                if hist is None or len(hist) != T:
                    continue
                Xi_all = hist.astype(np.float32, copy=False)

                for (oos_train_start, oos_val_start, oos_val_end) in oos_segments:
                    tr_fit_end_oos = max(0, oos_val_start - LAG)
                    if tr_fit_end_oos < 50:
                        continue
                    Xi_fit_oos = Xi_all[:tr_fit_end_oos]
                    yi_fit_oos = y_bin[:tr_fit_end_oos]
                    if len(np.unique(yi_fit_oos)) < 2:
                        continue
                    try:
                        clf_oos = LogisticRegression(penalty='l2', C=0.5, class_weight='balanced', solver='lbfgs', max_iter=200)
                        clf_oos.fit(Xi_fit_oos, yi_fit_oos)
                        X_train_sel[oos_val_start:oos_val_end, j] = clf_oos.decision_function(Xi_all[oos_val_start:oos_val_end])
                    except Exception:
                        continue

                try:
                    Xi_fit = Xi_all[:fit_end_pred]
                    yi_fit = y_bin[:fit_end_pred]
                    if len(np.unique(yi_fit)) < 2:
                        continue
                    clf_val = LogisticRegression(penalty='l2', C=0.5, class_weight='balanced', solver='lbfgs', max_iter=200)
                    clf_val.fit(Xi_fit, yi_fit)
                    X_val_sel[:, j] = clf_val.decision_function(Xi_all[val_start:val_end])
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

           
            window_imp = np.zeros(num_uids, dtype=np.float32)
            for j, col_idx in enumerate(selected_idx):
                                                                                                
                col = X_val_sel[:, j].copy()
                perm_idx = np.random.permutation(col.shape[0])
                X_val_sel[:, j] = col[perm_idx]
                try:
                    dval_perm = xgb.DMatrix(X_val_sel, label=y_val)
                    perm_probs = bst.predict(dval_perm)
                    perm_auc = float(roc_auc_score(y_val, perm_probs))
                    delta = base_auc - perm_auc
                    window_imp[col_idx] = delta if delta > 0.0 else 0.0
                except Exception:
                    window_imp[col_idx] = 0.0
                finally:
                    X_val_sel[:, j] = col
                    try:
                        del dval_perm
                    except Exception:
                        pass

           
            scale = max((base_auc - 0.5) / 0.5, 0.0)
            if scale <= 0:
                window_imp[:] = 0.0
            else:
                window_imp *= scale

           
            w = recency_gamma ** (max(0, pbar_total - 1 - window_index))
            window_weights_used.append(float(w))
            total_uid_imp += (w * window_imp).astype(np.float32)
            total_weight += w
            weighted_auc_sum += (w * base_auc)
            weighted_w_sum += w
            window_index += 1

           
            last_sel_auc = [(int(uids[i]), float(sel_auc[i])) for i in selected_idx]

            pbar.update(1)

        pbar.close()

        if auc_windows:
            logger.info(f"[SAL] Avg val-auc for {asset_name}: {total_val_auc/auc_windows:.4f}")
        if weighted_w_sum > 0:
            logger.info(f"[SAL] Weighted val-auc for {asset_name}: {weighted_auc_sum/weighted_w_sum:.4f}")

        if total_weight > 0:
            asset_uid_importance[asset_name] = dict(zip(uids, (total_uid_imp / total_weight).tolist()))
            asset_last_selected[asset_name] = last_sel_auc if last_sel_auc else []
            asset_window_weights[asset_name] = window_weights_used
            avg_val_acc = float(total_val_acc / acc_windows) if acc_windows else None
            asset_avg_val_acc_map[asset_name] = avg_val_acc
           
            avg_val_auc_unscaled = float(total_val_auc / auc_windows) if auc_windows else None
            weighted_val_auc = float(weighted_auc_sum / weighted_w_sum) if weighted_w_sum > 0 else None
            asset_avg_val_auc_unscaled_map[asset_name] = avg_val_auc_unscaled
            asset_weighted_val_auc_map[asset_name] = weighted_val_auc

    if not asset_uid_importance:
        return {}

   
    try:
        asset_contrib = {a: float(sum(imp.values())) for a, imp in asset_uid_importance.items()}
        total_contrib = float(sum(asset_contrib.values()))
        if total_contrib > 0.0:
            asset_salience_share = {a: (c / total_contrib) for a, c in asset_contrib.items()}
            for a in sorted(asset_salience_share.keys()):
                logger.info(f"[SAL] Asset salience share {a}: {asset_salience_share[a]*100:.2f}%")
    except Exception as e:
        logger.warning(f"Failed to compute per-asset salience shares: {e}")

   
    final_imp = {uid: 0.0 for uid in all_uids}
    for imp in asset_uid_importance.values():
        for uid, score in imp.items():
            final_imp[uid] += score
    total = sum(final_imp.values())
    result = {uid: s/total for uid, s in final_imp.items()} if total > 1e-9 else {}

    global LAST_DEBUG
    LAST_DEBUG = {
        "window_weights": asset_window_weights,
        "selected_auc": asset_last_selected,
        "asset_avg_val_acc": asset_avg_val_acc_map,
        "asset_avg_val_auc": asset_avg_val_auc_unscaled_map,
        "asset_weighted_val_auc": asset_weighted_val_auc_map,
        "asset_salience_share": asset_salience_share,
    }
    return result

def convert_multi_asset_data_to_np16(
    multi_asset_data: dict[str, tuple[dict[int, list[list[float]]], list[float], list[int]]]
) -> dict[str, tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]]:
    converted: dict[str, tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]] = {}
    for asset, (history_dict, asset_returns, sample_blocks) in multi_asset_data.items():
        new_hist: dict[int, np.ndarray] = {}
        for uid, hist in history_dict.items():
            new_hist[uid] = np.asarray(hist, dtype=np.float16)
        new_returns = np.asarray(asset_returns, dtype=np.float32)
        new_blocks = np.asarray(sample_blocks, dtype=np.int64)
        converted[asset] = (new_hist, new_returns, new_blocks)
    return converted


