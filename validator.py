from __future__ import annotations

import argparse
import logging
import os
import threading
import time
import asyncio
import copy
import json
import numpy as np 

import bittensor as bt
import torch
import aiohttp
from dotenv import load_dotenv
import requests

import config
from cycle import get_miner_payloads
from model import salience as sal_fn
from ledger import DataLog

LOG_DIR = os.path.expanduser("~/new_system_mantis")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "main.log"), mode="a"),
    ],
)

weights_logger = logging.getLogger("weights")
weights_logger.setLevel(logging.DEBUG)
weights_logger.addHandler(
    logging.FileHandler(os.path.join(LOG_DIR, "weights.log"), mode="a")
)

for noisy in ("websockets", "aiohttp"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

load_dotenv()

os.makedirs(config.STORAGE_DIR, exist_ok=True)
DATALOG_PATH = os.path.join(config.STORAGE_DIR, "mantis_datalog.pkl")
SAVE_INTERVAL = 480


async def _fetch_price_source(session, url, parse_json=True):
    async with session.get(url, timeout=5) as resp:
        resp.raise_for_status()
        if parse_json:
            return await resp.json()
        else:
            return await resp.text()

async def _get_price_from_sources(session, source_list):
    for name, url, parser in source_list:
        try:
            parse_json = not url.endswith("e=csv")
            data = await _fetch_price_source(session, url, parse_json=parse_json)
            price = parser(data)
            if price is not None:
                return price
        except Exception:
            continue
    return None


async def get_asset_prices(session: aiohttp.ClientSession) -> dict[str, float] | None:
    base = {asset: 0.0 for asset in config.ASSETS}
    try:
        async with session.get(config.PRICE_DATA_URL) as resp:
            resp.raise_for_status()
            text = await resp.text()
            data = json.loads(text)
            fetched = data.get("prices", {}) or {}
            out = dict(base)
            for asset in config.ASSETS:
                v = fetched.get(asset)
                try:
                    fv = float(v)
                    if fv > 0 and fv < float('inf'):
                        out[asset] = fv
                except Exception:
                    pass
            logging.info(f"Fetched prices, filled zeros where missing: {out}")
            return out
    except Exception as e:
        logging.error(f"Failed to fetch prices from {config.PRICE_DATA_URL}: {e}")
        return base

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wallet.name", required=True)
    p.add_argument("--wallet.hotkey", required=True)
    p.add_argument("--network", default="finney")
    p.add_argument("--netuid", type=int, default=config.NETUID)
    p.add_argument(
        "--do_save",
        action="store_true",
        default=False,
        help="Whether to save the datalog periodically."
    )
    p.add_argument(
        "--save-every-seconds",
        type=int,
        default=SAVE_INTERVAL * 12,
        help="How often to save the datalog, in seconds (default: SAVE_INTERVAL blocks * 12s).",
    )
    args = p.parse_args()

    while True:
        try:
            sub = bt.subtensor(network=args.network)
            wallet = bt.wallet(name=getattr(args, "wallet.name"), hotkey=getattr(args, "wallet.hotkey"))
            mg = bt.metagraph(netuid=args.netuid, network=args.network, sync=True)
            break
        except Exception as e:
            logging.exception("Subtensor connect failed")
            time.sleep(30)
            continue

    if not os.path.exists(DATALOG_PATH):
        try:
            logging.info(f"Attempting to download initial datalog from {config.DATALOG_ARCHIVE_URL} → {DATALOG_PATH}")
            os.makedirs(os.path.dirname(DATALOG_PATH), exist_ok=True)
            r = requests.get(config.DATALOG_ARCHIVE_URL, timeout=600, stream=True)
            r.raise_for_status()
            tmp = DATALOG_PATH + ".tmp"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp, DATALOG_PATH)
            logging.info("Downloaded datalog to local storage.")
        except Exception:
            logging.info("Remote datalog unavailable; starting with a new, empty ledger.")
    logging.info(f"Loading datalog from {DATALOG_PATH}...")
    datalog = DataLog.load(DATALOG_PATH)
        
    stop_event = asyncio.Event()

    try:
        asyncio.run(run_main_loop(args, sub, wallet, mg, datalog, stop_event))
    except KeyboardInterrupt:
        logging.info("Exit signal received. Shutting down.")
    finally:
        stop_event.set()
        if args.do_save:
            try:
                logging.info("Final save on shutdown...")
                asyncio.run(datalog.save(DATALOG_PATH))
            except Exception:
                logging.exception("Final save failed.")
        logging.info("Shutdown complete.")


async def decrypt_loop(datalog: DataLog, stop_event: asyncio.Event):
    logging.info("Decryption loop started.")
    while not stop_event.is_set():
        try:
            await datalog.process_pending_payloads()
        except asyncio.CancelledError:
            break
        except Exception:
            logging.exception("An error occurred in the decryption loop.")
        await asyncio.sleep(5)
    logging.info("Decryption loop stopped.")


async def save_loop(datalog: DataLog, do_save: bool, save_every_seconds: int, stop_event: asyncio.Event):
    if not do_save:
        logging.info("DO_SAVE is False, skipping periodic saves.")
        return
    logging.info("Save loop started.")
    try:
        logging.info("Initiating initial datalog save...")
        await datalog.save(DATALOG_PATH)
    except Exception:
        logging.exception("Initial save failed.")
    while not stop_event.is_set():
        try:
            await asyncio.sleep(save_every_seconds)
            logging.info("Initiating periodic datalog save...")
            await datalog.save(DATALOG_PATH)
        except asyncio.CancelledError:
            break
        except Exception:
            logging.exception("An error occurred in the save loop.")
    logging.info("⏹️ Save loop stopped.")


subtensor_lock = threading.Lock()


async def get_current_block_with_retry(sub: bt.subtensor, lock: threading.Lock, timeout: int = 10) -> int:
    """
    Fetches the current block number from the subtensor with a retry mechanism.
    This function will attempt to fetch the block indefinitely until it succeeds.
    """
    retry_delay = 5
    while True:
        try:
            def get_block():
                with lock:
                    return sub.get_current_block()

            current_block = await asyncio.wait_for(
                asyncio.to_thread(get_block),
                timeout=float(timeout)
            )
            return current_block
        except asyncio.TimeoutError:
            logging.warning(
                f"Getting current block timed out after {timeout}s. "
                f"Retrying in {retry_delay}s..."
            )
        except Exception as e:
            logging.error(
                f"An unexpected error occurred while getting current block: {e}. "
                f"Retrying in {retry_delay}s..."
            )
        await asyncio.sleep(retry_delay)


async def run_main_loop(
    args: argparse.Namespace,
    sub: bt.subtensor,
    wallet: bt.wallet,
    mg: bt.metagraph,
    datalog: DataLog,
    stop_event: asyncio.Event,
):
    last_block = await get_current_block_with_retry(sub, subtensor_lock)
    weight_thread: threading.Thread | None = None

    tasks = [
        asyncio.create_task(decrypt_loop(datalog, stop_event)),
        asyncio.create_task(save_loop(datalog, args.do_save, args.save_every_seconds, stop_event)),
    ]

    async with aiohttp.ClientSession() as session:
        while not stop_event.is_set():
            try:
                current_block = await get_current_block_with_retry(sub, subtensor_lock)
                
                if current_block == last_block:
                    await asyncio.sleep(1)
                    continue
                last_block = current_block

                if current_block % config.SAMPLE_EVERY != 0:
                    continue
                logging.info(f"Sampled block {current_block}")

                if current_block % 100 == 0:
                    with subtensor_lock:
                        mg.sync(subtensor=sub)
                    logging.info("Metagraph synced.")
                    async with datalog._lock:
                        datalog.prune_hotkeys(mg.hotkeys)

                asset_prices = await get_asset_prices(session)
                if not asset_prices:
                    logging.error("Failed to fetch prices for required assets.")
                    continue
                
                payloads = await get_miner_payloads(netuid=args.netuid, mg=mg)
                logging.info(f"Retrieved {len(payloads)} payloads from miners. Hotkey sample: {list(payloads.keys())[:3]}")
                await datalog.append_step(current_block, asset_prices, payloads, mg)

                
                hotkey_to_uid = {hk: uid for uid, hk in zip(mg.uids.tolist(), mg.hotkeys)}
                uids_in_mg = mg.uids.tolist()
                for uid in uids_in_mg:
                    if uid not in datalog.uid_age_in_blocks:
                        datalog.uid_age_in_blocks[uid] = 0
                
                current_uids = set(uids_in_mg)
                for uid in list(datalog.uid_age_in_blocks.keys()):
                    if uid not in current_uids:
                        del datalog.uid_age_in_blocks[uid]


                if (
                    current_block % config.TASK_INTERVAL == 0
                    and (weight_thread is None or not weight_thread.is_alive())
                    and len(datalog.blocks) >= config.LAG * 2 + 1
                ):
                    datalog_clone = copy.deepcopy(datalog)

                    def worker(dlog, uid_ages, block_snapshot, metagraph, cli_args):
                        training_data = dlog.get_training_data_sync(
                            max_block_number=block_snapshot - config.TASK_INTERVAL
                        )
                        if not training_data:
                            weights_logger.warning("Not enough data for salience, but checking for young UIDs.")
                        
                        weights_logger.info(f"=== Starting weight calculation | block {block_snapshot} ===")
                        
                        weights_logger.info("Calculating salience...")
                        general_sal_hk = sal_fn(training_data) if training_data else {}
                        if not general_sal_hk:
                            weights_logger.info("Salience computation returned empty.")

                        hk2uid = {hk: uid for uid, hk in zip(metagraph.uids.tolist(), metagraph.hotkeys)}
                        general_sal = {hk2uid.get(hk, -1): s for hk, s in general_sal_hk.items() if hk in hk2uid}
                        sal = {uid: score for uid, score in general_sal.items() if uid != -1}

                        if not sal:
                            weights_logger.info("Salience is empty. Assigning weights only to young UIDs if any.")
                        
                        uids = metagraph.uids.tolist()
                        
                        SAMPLE_EVERY = int(config.SAMPLE_EVERY)
                        young_threshold = 36000
                        hk2idx = getattr(dlog, 'hk2idx', {}) if hasattr(dlog, 'hk2idx') else {}
                        idx2hk = {idx: hk for hk, idx in hk2idx.items()}

                        hotkey_first_block: dict[str, int] = {}
                        for asset, ds in dlog.datasets.items():
                            if not ds.challenges:
                                continue
                            ch = ds.challenges[0]
                            for sidx, emb in ch.emb_sparse.items():
                                if emb is None or emb.ndim != 2:
                                    continue
                                H = emb.shape[0]
                                for hk_idx in range(H):
                                    hk = idx2hk.get(hk_idx)
                                    if not hk or hk in hotkey_first_block:
                                        continue
                                    row = emb[hk_idx, :]
                                    if np.any(row != 0):
                                        hotkey_first_block[hk] = int(sidx) * SAMPLE_EVERY

                        young_uids = set()
                        for hk, first_block in hotkey_first_block.items():
                            age_blocks = block_snapshot - int(first_block)
                            if age_blocks < young_threshold:
                                uid = hk2uid.get(hk)
                                if uid is not None:
                                    young_uids.add(uid)
                        weights_logger.info(f"Found {len(young_uids)} young UIDs by hotkey-first-nonzero (<{young_threshold} blocks).")
                        

                        mature_uid_scores = {uid: score for uid, score in sal.items() if uid not in young_uids}
                        
                        total_mature_score = sum(mature_uid_scores.values())
                        
                        fixed_weight_per_young_uid = 0.0001
                        
                        active_young_uids = {uid for uid in young_uids if uid in uids}
                        
                        weight_for_mature_uids = 1.0 - (len(active_young_uids) * fixed_weight_per_young_uid)
                        weight_for_mature_uids = max(0, weight_for_mature_uids)

                        final_weights = {}

                        if total_mature_score > 0 and weight_for_mature_uids > 0:
                            for uid, score in mature_uid_scores.items():
                                if uid in uids:
                                    final_weights[uid] = (score / total_mature_score) * weight_for_mature_uids
                        
                        for uid in active_young_uids:
                            final_weights[uid] = fixed_weight_per_young_uid

                        if not final_weights:
                            weights_logger.warning("No weights to set after processing.")
                            return

                        total_weight = sum(final_weights.values())
                        if total_weight <= 0:
                            weights_logger.warning("Total calculated weight is zero or negative, skipping set.")
                            return
                        
                        normalized_weights = {uid: w / total_weight for uid, w in final_weights.items()}

                        w = torch.tensor([normalized_weights.get(uid, 0.0) for uid in uids], dtype=torch.float32)
                        
                        if w.sum() > 0:
                            final_w = w / w.sum()
                        else:
                            weights_logger.warning("Zero-sum weights tensor, skipping set.")
                            return
                        
                        weights_to_log = {uid: f"{weight:.8f}" for uid, weight in normalized_weights.items() if uid in uids and weight > 0}
                        weights_logger.info(f"Normalized weights for block {block_snapshot}: {json.dumps(weights_to_log)}")
                        weights_logger.info(f"Final tensor sum before setting weights: {final_w.sum().item()}")
                        
                        try:
                            thread_sub = bt.subtensor(network=cli_args.network)
                            thread_wallet = bt.wallet(
                                name=getattr(cli_args, "wallet.name"), 
                                hotkey=getattr(cli_args, "wallet.hotkey")
                            )
                            thread_sub.set_weights(
                                netuid=cli_args.netuid, wallet=thread_wallet,
                                uids=metagraph.uids, weights=final_w,
                                wait_for_inclusion=False,
                            )
                            weights_logger.info(f"Weights set at block {block_snapshot} (max={final_w.max():.4f})")
                        except Exception as e:
                            weights_logger.error(f"Failed to set weights: {e}", exc_info=True)

                    async with datalog._lock:
                        uid_ages_copy = copy.deepcopy(datalog.uid_age_in_blocks)

                    weight_thread = threading.Thread(
                        target=worker,
                        args=(datalog_clone, uid_ages_copy, current_block, copy.deepcopy(mg), copy.deepcopy(args)),
                        daemon=True,
                    )
                    weight_thread.start()

            except KeyboardInterrupt:
                stop_event.set()
            except Exception:
                logging.error("Error in main loop", exc_info=True)
                await asyncio.sleep(10)
    
    logging.info("Main loop finished. Cleaning up background tasks.")
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    main()
