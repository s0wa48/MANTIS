from __future__ import annotations

import asyncio, copy, json, logging, os, time, numpy as np
import pickle, gzip
from typing import Dict, Any, List
import zipfile
from collections import defaultdict
import bittensor as bt

from timelock import Timelock
import aiohttp

import config

logger = logging.getLogger(__name__)

from dataclasses import dataclass, field

SECONDS_PER_BLOCK: int = 12
SAMPLE_EVERY: int = config.SAMPLE_EVERY

@dataclass
class ChallengeData:
    hotkeys: int
    dim: int
    emb_sparse: Dict[int, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        for sidx, arr in self.emb_sparse.items():
            if not isinstance(arr, np.ndarray):
                self.emb_sparse[sidx] = np.array(arr, dtype=np.float16)
            elif arr.dtype != np.float16:
                self.emb_sparse[sidx] = arr.astype(np.float16)

    def get_embedding_for_sidx(self, sidx: int) -> np.ndarray | None:
        return self.emb_sparse.get(sidx)

    def set_embedding_for_sidx(self, sidx: int, hotkey_idx: int, embedding: np.ndarray):
        if sidx not in self.emb_sparse:
            self.emb_sparse[sidx] = np.zeros((self.hotkeys, self.dim), dtype=np.float16)
        existing_tensor = self.emb_sparse[sidx]
        if existing_tensor.shape[0] < self.hotkeys:
            padded_tensor = np.zeros((self.hotkeys, self.dim), dtype=np.float16)
            padded_tensor[:existing_tensor.shape[0], :] = existing_tensor
            self.emb_sparse[sidx] = padded_tensor
        if hotkey_idx < self.hotkeys and len(embedding) == self.dim:
            self.emb_sparse[sidx][hotkey_idx, :] = embedding

    def prune_hotkeys(self, old_indices_to_keep: List[int]):
        new_hotkey_count = len(old_indices_to_keep)
        if not self.emb_sparse:
            self.hotkeys = new_hotkey_count
            return
        pruned_emb_sparse: Dict[int, np.ndarray] = {}
        for sidx, embeddings in self.emb_sparse.items():
            pruned_embeddings = np.zeros((new_hotkey_count, self.dim), dtype=np.float16)
            for new_row, old_row in enumerate(old_indices_to_keep):
                if 0 <= old_row < embeddings.shape[0]:
                    pruned_embeddings[new_row, :] = embeddings[old_row, :]
            pruned_emb_sparse[sidx] = pruned_embeddings
        self.emb_sparse = pruned_emb_sparse
        self.hotkeys = new_hotkey_count

@dataclass
class ChallengeDataset:
    challenges: List[ChallengeData] = field(default_factory=list)

    @classmethod
    def generate_dummy(
        cls,
        *,
        days: int = 0,
        embed_dims: List[int] = None,
        hotkeys: int | List[int] = 0,
    ) -> "ChallengeDataset":
        if embed_dims is None:
            embed_dims = []
        if isinstance(hotkeys, int) and embed_dims:
            hotkeys = [hotkeys] * len(embed_dims)
        challenges: List[ChallengeData] = []
        for dim, hk in zip(embed_dims, hotkeys):
            challenges.append(ChallengeData(emb_sparse={}, hotkeys=hk, dim=dim))
        return cls(challenges)

    def to_npz_dict(self) -> Dict[str, np.ndarray]:
        pack: Dict[str, np.ndarray] = {}
        for idx, ch in enumerate(self.challenges):
            if not ch.emb_sparse:
                continue
            p = f"c{idx}_"
            sorted_items = sorted(ch.emb_sparse.items())
            sindices = np.array([item[0] for item in sorted_items], dtype=np.int64)
            embeddings = np.array([item[1] for item in sorted_items], dtype=np.float16)
            pack[p + "sindices"] = sindices
            pack[p + "embeddings"] = embeddings
            pack[p + "hotkeys"] = np.array(ch.hotkeys, dtype=np.int32)
            pack[p + "dim"] = np.array(ch.dim, dtype=np.int32)
        return pack

DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)


class MultiAssetLedger:
    """Hotkey-centric, sparse ledger backed by dictionary-based ChallengeDataset objects."""

    def __init__(self):
        self.blocks: List[int] = []
        self.asset_prices: List[Dict[str, float]] = []
        self.raw_payloads: Dict[int, Dict[str, bytes]] = {}
        
        self.datasets: Dict[str, ChallengeDataset] = {
            asset: ChallengeDataset.generate_dummy(
                embed_dims=[config.ASSET_EMBEDDING_DIMS[asset]], 
                hotkeys=[0]
            )
            for asset in config.ASSETS
        }

        self.live_hotkeys: List[str] = []
        self.hk2idx: Dict[str, int] = {}
        self.uid_age_in_blocks: Dict[int, int] = {}

        self.tlock = Timelock(DRAND_PUBLIC_KEY)
        self._lock = asyncio.Lock()

    def __getstate__(self):
        """Support pickling by removing non-picklable members."""
        state = self.__dict__.copy()
        if "_lock" in state:
            del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = asyncio.Lock()

    def _ensure_hotkey(self, hk: str):
        if hk in self.hk2idx:
            return
        idx = len(self.live_hotkeys)
        self.live_hotkeys.append(hk)
        self.hk2idx[hk] = idx
        for ds in self.datasets.values():
            ds.challenges[0].hotkeys = len(self.live_hotkeys)

    def prune_hotkeys(self, active_hotkeys: List[str]):
        """Remove hotkeys that are no longer in the metagraph from the sparse embeddings."""
        active_set = set(active_hotkeys)
        to_remove = {hk for hk in self.live_hotkeys if hk not in active_set}
        if not to_remove:
            return
        
        logger.info(f"Pruning {len(to_remove)} inactive hotkeys.")
        
        new_live_hotkeys = [hk for hk in self.live_hotkeys if hk not in to_remove]
        old_indices_to_keep = [self.hk2idx[hk] for hk in new_live_hotkeys]
        
        self.live_hotkeys = new_live_hotkeys
        self.hk2idx = {hk: i for i, hk in enumerate(self.live_hotkeys)}
        
        for ds in self.datasets.values():
            ds.challenges[0].prune_hotkeys(old_indices_to_keep)

    def _zero_vecs(self) -> Dict[str, List[float]]:
        return {
            asset: [0.0] * config.ASSET_EMBEDDING_DIMS[asset] for asset in config.ASSETS
        }

    async def append_step(self, block: int, prices: Dict[str, float], payloads: Dict[str, bytes], metagraph: bt.metagraph):
        """Add one network sample, using the metagraph as the ground truth for active hotkeys."""
        async with self._lock:
            self.blocks.append(block)
            self.asset_prices.append(prices)
            ts = len(self.blocks) - 1
            self.raw_payloads[ts] = {}
            
            active_hotkeys = metagraph.hotkeys
            logger.info(f"[Ledger] Appending step for block {block}. Received {len(payloads)} payloads for {len(active_hotkeys)} active hotkeys.")

            for hk in active_hotkeys:
                self._ensure_hotkey(hk)
                ct_dict = payloads.get(hk)
                ct_bytes = json.dumps(ct_dict).encode('utf-8') if ct_dict else b"{}"
                self.raw_payloads[ts][hk] = ct_bytes

            if self.uid_age_in_blocks:
                for uid in self.uid_age_in_blocks:
                    self.uid_age_in_blocks[uid] += 1

    async def _get_drand_signature(self, round_num: int) -> bytes | None:
        try:
            await asyncio.sleep(0.02)
            url = f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/rounds/{round_num}"
            async with aiohttp.ClientSession() as sess:
                async with sess.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return bytes.fromhex(data["signature"])
        except Exception:
            pass
        return None

    async def process_pending_payloads(self):
        async with self._lock:
            payloads_copy = copy.deepcopy(self.raw_payloads)
            current_block = self.blocks[-1] if self.blocks else 0

        if not payloads_copy:
            return

        rounds: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        mature_payload_keys = set()
        
        for ts, by_hk in payloads_copy.items():
            if current_block - self.blocks[ts] >= 300:
                for hk, raw in by_hk.items():
                    mature_payload_keys.add((ts, hk))
                    try:
                        payload_dict = json.loads(raw.decode("utf-8")) if raw else {}
                        rnd = payload_dict.get("round", 0)
                        ct_hex = payload_dict.get("ciphertext", "")
                        rounds[rnd].append({"ts": ts, "hk": hk, "ct": ct_hex})
                    except Exception:
                        rounds[0].append({"ts": ts, "hk": hk, "ct": ""})

        if not mature_payload_keys:
            return
        
        decrypted_cache = {}

        async def _work(rnd, items):
            sig = await self._get_drand_signature(rnd) if rnd > 0 else None
            for item in items:
                ts, hk, ct_hex = item["ts"], item["hk"], item["ct"]
                vecs = self._zero_vecs()
                if sig and ct_hex:
                    try:
                        pt = self.tlock.tld(bytes.fromhex(ct_hex), sig).decode("utf-8")
                        emb_str, hk_in = pt.rsplit(":::", 1)
                        if hk_in != hk:
                            raise ValueError("hotkey mismatch")
                        submission = json.loads(emb_str)
                        vecs = self._validate_submission(submission)
                    except Exception:
                        pass
                decrypted_cache.setdefault(ts, {})[hk] = vecs
        
        ROUND_BATCH = 16
        round_items = list(rounds.items())
        for i in range(0, len(round_items), ROUND_BATCH):
            batch = round_items[i:i+ROUND_BATCH]
            await asyncio.gather(*(_work(r, items) for r, items in batch))
            await asyncio.sleep(0.1)

        async with self._lock:
            for ts, by_hk in decrypted_cache.items():
                if ts >= len(self.blocks): continue
                block_val = self.blocks[ts]
                if block_val % config.SAMPLE_EVERY == 0:
                    sidx = block_val // config.SAMPLE_EVERY
                    for hk, vecs in by_hk.items():
                        hk_idx = self.hk2idx.get(hk)
                        if hk_idx is None: continue
                        for asset, vec in vecs.items():
                            if any(v != 0.0 for v in vec):
                                ch = self.datasets[asset].challenges[0]
                                ch.set_embedding_for_sidx(sidx, hk_idx, np.array(vec, dtype=np.float16))
            
            for ts, hk in mature_payload_keys:
                if ts in self.raw_payloads and hk in self.raw_payloads[ts]:
                    del self.raw_payloads[ts][hk]
                    if not self.raw_payloads[ts]:
                        del self.raw_payloads[ts]

    def _validate_submission(self, submission: Any) -> Dict[str, List[float]]:
        if not isinstance(submission, list) or len(submission) != len(config.ASSETS):
            return self._zero_vecs()
        out = {}
        for i, asset in enumerate(config.ASSETS):
            vec = submission[i]
            dim = config.ASSET_EMBEDDING_DIMS[asset]
            if (isinstance(vec, list) and len(vec) == dim and
                    all(isinstance(v, (int, float)) and -1.0 <= v <= 1.0 for v in vec)):
                out[asset] = vec
            else:
                out[asset] = [0.0] * dim
        return out
    
    def get_training_data_sync(self, max_block_number: int | None = None) -> dict:
        TARGET_BLOCK_DIFF = 300 
        logger.info(f"Generating training data up to block {max_block_number}...")
        
        slice_idx = len(self.blocks)
        if max_block_number is not None:
            slice_idx = next((i for i, b in enumerate(self.blocks) if b > max_block_number), len(self.blocks))
        
        if slice_idx == 0:
            logger.warning("No blocks available for training data generation.")
            return {}

        effective_blocks = self.blocks[:slice_idx]
        effective_prices = self.asset_prices[:slice_idx]
        block_to_idx = {b: i for i, b in enumerate(self.blocks)}
        
        result = {}
        for asset in config.ASSETS:
            dim = config.ASSET_EMBEDDING_DIMS[asset]
            ch = self.datasets[asset].challenges[0]
            num_hks = ch.hotkeys

            X_rows, y_rows = [], []

            price_series = [p.get(asset, np.nan) if p else np.nan for p in effective_prices]
            n = len(price_series)
            keep_mask = [True] * n

            def _finalize_run(start_idx: int, end_idx: int, last_val):
                run_len = end_idx - start_idx
                if run_len > 5 and np.isfinite(last_val):
                    for k in range(start_idx, end_idx):
                        keep_mask[k] = False

            last_price = np.nan
            run_start = 0
            for idx in range(n):
                price = price_series[idx]
                if idx == 0:
                    last_price = price
                    run_start = 0
                    continue
                if not (np.isfinite(price) and np.isfinite(last_price)) or price != last_price:
                    _finalize_run(run_start, idx, last_price)
                    run_start = idx
                last_price = price
            _finalize_run(run_start, n, last_price)

            kept_indices = [i for i, keep in enumerate(keep_mask) if keep]

            for i_orig in kept_indices:
                p0_block = effective_blocks[i_orig]
                p1_block = p0_block + TARGET_BLOCK_DIFF
                j = block_to_idx.get(p1_block)
                if j is None or j >= len(self.asset_prices):
                    continue
                if j >= slice_idx or (j < len(keep_mask) and not keep_mask[j]):
                    continue

                sidx = p0_block // config.SAMPLE_EVERY
                embedding_tensor = ch.get_embedding_for_sidx(sidx)
                if embedding_tensor is None:
                    continue

                p0 = self.asset_prices[i_orig].get(asset, np.nan)
                p1 = self.asset_prices[j].get(asset, np.nan)

                if not (np.isnan(p0) or np.isnan(p1) or p0 <= 0 or p1 == 0):
                    if embedding_tensor.shape[0] < num_hks:
                        padded_tensor = np.zeros((num_hks, dim), dtype=np.float16)
                        padded_tensor[:embedding_tensor.shape[0], :] = embedding_tensor
                        X_rows.append(padded_tensor.flatten())
                    else:
                        X_rows.append(embedding_tensor.flatten())
                    y_rows.append((p1 - p0) / p0)

            if not X_rows:
                logger.warning(f"No valid training samples for {asset} after checking for embeddings.")
                continue

            logger.info(f"Generated {len(y_rows)} valid training samples for {asset}.")
            
            X = np.array(X_rows, dtype=np.float16)
            y = np.array(y_rows, dtype=np.float32)

            hist = (X, self.hk2idx)
            result[asset] = (hist, y)

        logger.info(f"Training data generated for {len(result)} assets.")
        return result if result else {}

    async def get_training_data(self, max_block_number: int | None = None) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_training_data_sync, max_block_number)

    async def save(self, path: str):
        """Persist the ledger using pickle (.pkl or .pkl.gz)."""
        async with self._lock:
            ledger_ref = self

        await asyncio.to_thread(self._deepcopy_and_save, ledger_ref, path)

    @staticmethod
    def _deepcopy_and_save(ledger_ref: "MultiAssetLedger", path: str):
        ledger_copy = copy.deepcopy(ledger_ref)
        MultiAssetLedger._save_pickle(ledger_copy, path)

    @staticmethod
    def _save_pickle(ledger_obj: "MultiAssetLedger", path: str):
        dir_name = os.path.dirname(path) or "."
        os.makedirs(dir_name, exist_ok=True)

        temp_path = f"{path}.tmp.{os.getpid()}"
        try:
            if path.endswith(".gz"):
                with gzip.open(temp_path, "wb") as f:
                    pickle.dump(ledger_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(temp_path, "wb") as f:
                    pickle.dump(ledger_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

        try:
            st = os.stat(path)
            logger.info(
                "Ledger saved → %s (size=%d bytes, mtime=%d)",
                path,
                st.st_size,
                int(st.st_mtime),
            )
        except Exception:
            logger.info("Ledger saved → %s", path)

    @staticmethod
    def load(path: str) -> "MultiAssetLedger":
        """Load a ledger from pickle by default, falling back to legacy .npz."""
        if not os.path.exists(path):
            logger.info("No ledger file found, creating a new one.")
            return MultiAssetLedger()

        logger.info("Loading ledger from %s", path)

        _, ext = os.path.splitext(path)

        def _load_pickle(p: str) -> MultiAssetLedger:
            try:
                if p.endswith(".gz"):
                    with gzip.open(p, "rb") as f:
                        return pickle.load(f)
                else:
                    with open(p, "rb") as f:
                        return pickle.load(f)
            except Exception as e:
                raise e

        def _load_legacy_npz(p: str) -> MultiAssetLedger:
            try:
                data = np.load(p, allow_pickle=True)
            except (zipfile.BadZipFile, ValueError, OSError) as e:
                raise e

            ledger = MultiAssetLedger()
            ledger.blocks = data["blocks"].tolist()
            ledger.asset_prices = json.loads(data["asset_prices"].tobytes().decode("utf-8"))
            ledger.live_hotkeys = data["live_hotkeys"].tolist()
            ledger.hk2idx = {hk: i for i, hk in enumerate(ledger.live_hotkeys)}

            if "uid_age_in_blocks" in data:
                uid_ages_str = data["uid_age_in_blocks"].tobytes().decode("utf-8")
                ledger.uid_age_in_blocks = {int(k): v for k, v in json.loads(uid_ages_str).items()}

            for asset in config.ASSETS:
                prefix = f"{asset}_"
                asset_specific_data = {k[len(prefix):]: v for k, v in data.items() if k.startswith(prefix)}
                if asset_specific_data:
                    logger.warning("Legacy NPZ import is no longer supported for challenge data; starting empty dataset for %s", asset)
                    ledger.datasets[asset] = ChallengeDataset.generate_dummy(embed_dims=[config.ASSET_EMBEDDING_DIMS[asset]], hotkeys=[len(ledger.live_hotkeys)])
                if not ledger.datasets[asset].challenges:
                    ledger.datasets[asset].challenges.append(ChallengeData(
                        hotkeys=len(ledger.live_hotkeys),
                        dim=config.ASSET_EMBEDDING_DIMS[asset],
                        emb_sparse={},
                    ))
                else:
                    ledger.datasets[asset].challenges[0].hotkeys = len(ledger.live_hotkeys)

            logger.info("Ledger loaded (legacy .npz) ← %s", p)
            return ledger

        try:
            if ext == ".npz":
                return _load_legacy_npz(path)
            return _load_pickle(path)
        except Exception:
            try:
                return _load_legacy_npz(path)
            except Exception as e2:
                logger.error("Failed to load ledger from %s: %s", path, e2)
                try:
                    corrupt_path = f"{path}.corrupt.{int(time.time())}"
                    os.replace(path, corrupt_path)
                    logger.warning("Renamed unreadable ledger to %s", corrupt_path)
                except Exception:
                    logger.warning("Could not rename unreadable ledger file at %s", path)
                logger.info("Starting with a new, empty ledger.")
                return MultiAssetLedger()

DataLog = MultiAssetLedger
