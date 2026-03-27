#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from beam_tracking_config import BeamTrackingConfig
from beam_tracking_model_m import BeamTrackingLSTMWithAttention


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {v}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Data2SeqDatasetV2(Dataset):
    """Dataset yielding historical and future normalized beam-space vectors."""

    def __init__(
        self,
        config: BeamTrackingConfig,
        data2_dir: str,
        neighbors_path: str,
        files: List[str],
        horizon: int = 1,
        station_hold_prob: float = 0.0,
        hold_only_on_change: bool = True,
        use_dilated: bool = False,
        s: int = 1,
    ):
        super().__init__()
        self.config = config
        self.data2_dir = data2_dir
        self.files = files
        self.horizon = max(1, int(horizon))
        self.station_hold_prob = float(station_hold_prob)
        self.hold_only_on_change = bool(hold_only_on_change)
        self.use_dilated = bool(use_dilated)
        self.s = max(1, int(s))

        with open(neighbors_path, "r") as f:
            self.neighbors = json.load(f)

        self.traj_data = []
        self.traj_meta = []
        for fn in self.files:
            fp = os.path.join(self.data2_dir, fn)
            with np.load(fp) as npz:
                rsrp = npz["rsrp_dbm"]
            self.traj_data.append(rsrp)
            self.traj_meta.append({"traj_id": os.path.splitext(fn)[0], "T": rsrp.shape[0]})

        self.sample_map = []
        min_t = self.s * self.config.M if self.use_dilated else self.config.M
        for traj_i, meta in enumerate(self.traj_meta):
            t_len = meta["T"]
            for t in range(min_t, t_len - self.horizon + 1):
                self.sample_map.append((traj_i, t))

    def __len__(self):
        return len(self.sample_map)

    def _select7_indices(self, gamma_bt: np.ndarray) -> List[int]:
        b_serv, _ = np.unravel_index(np.argmax(gamma_bt), gamma_bt.shape)
        nb = self.neighbors.get(str(int(b_serv))) or []
        others = [int(x) for x in nb if int(x) != int(b_serv)]
        return [int(b_serv)] + others[:6]

    @staticmethod
    def _flat7(gamma_bt: np.ndarray, bs7: List[int]) -> np.ndarray:
        return np.concatenate([gamma_bt[b] for b in bs7], axis=0).astype(np.float32)

    def _topk_mask(self, x: np.ndarray) -> np.ndarray:
        k = self.config.K
        if k >= len(x):
            m = np.ones_like(x, dtype=np.float32)
        else:
            idx = np.argpartition(-x, k - 1)[:k]
            m = np.zeros_like(x, dtype=np.float32)
            m[idx] = 1.0
        return m

    def __getitem__(self, idx: int):
        traj_i, t = self.sample_map[idx]
        traj_id = self.traj_meta[traj_i]["traj_id"]
        gamma_all = self.traj_data[traj_i]

        m = self.config.M
        h = self.horizon
        gamma_hist = []
        station_hist = []
        gamma_future = []
        station_future = []
        prev_bs7 = None

        if self.use_dilated and t >= self.s * m:
            time_indices = [t - self.s * m + i * self.s for i in range(m)]
        else:
            time_indices = list(range(t - m, t))

        for u in time_indices:
            g_bt = gamma_all[u]
            bs7_new = self._select7_indices(g_bt)
            bs7 = bs7_new
            if prev_bs7 is not None and self.station_hold_prob > 0.0:
                changed = (bs7_new != prev_bs7)
                if (not self.hold_only_on_change) or changed:
                    if random.random() < self.station_hold_prob:
                        bs7 = prev_bs7
            flat = self._flat7(g_bt, bs7)
            gamma_hist.append(self.config.normalize_rsrp(flat).astype(np.float32))
            station_hist.append(bs7)
            prev_bs7 = bs7

        for u in range(t, t + h):
            g_bt = gamma_all[u]
            bs7_new = self._select7_indices(g_bt)
            bs7 = bs7_new
            if prev_bs7 is not None and self.station_hold_prob > 0.0:
                changed = (bs7_new != prev_bs7)
                if (not self.hold_only_on_change) or changed:
                    if random.random() < self.station_hold_prob:
                        bs7 = prev_bs7
            flat = self._flat7(g_bt, bs7)
            gamma_future.append(self.config.normalize_rsrp(flat).astype(np.float32))
            station_future.append(bs7)
            prev_bs7 = bs7

        gamma_hist = np.stack(gamma_hist, axis=0).astype(np.float32)
        gamma_future = np.stack(gamma_future, axis=0).astype(np.float32)
        gamma0 = gamma_future[0]
        s_label = gamma0 * self._topk_mask(gamma0)
        rsrp_label = np.array([np.max(gamma0)], dtype=np.float32)

        return {
            "gamma_hist": torch.from_numpy(gamma_hist),
            "gamma_future": torch.from_numpy(gamma_future),
            "s_label": torch.from_numpy(s_label),
            "rsrp_label": torch.from_numpy(rsrp_label),
            "station_indices_hist": torch.tensor(station_hist, dtype=torch.long),
            "station_indices_future": torch.tensor(station_future, dtype=torch.long),
            "traj_id": traj_id,
            "time_step": t,
        }


class BeamTrackingLoss(nn.Module):
    """Training loss combining beam regression, link prediction, and KL regularization."""

    def __init__(self, config: BeamTrackingConfig, use_mse=True, use_link=True, use_kl=True, tau: float = 0.8):
        super().__init__()
        self.config = config
        self.lambda_weight = float(config.lambda_weight)
        self.use_mse = bool(use_mse)
        self.use_link = bool(use_link)
        self.use_kl = bool(use_kl)
        self.tau = float(tau)

    @staticmethod
    def _oracle_topk_mask(gamma_true: torch.Tensor, k: int) -> torch.Tensor:
        idx = torch.topk(gamma_true, k, dim=1).indices
        mask = torch.zeros_like(gamma_true)
        mask.scatter_(1, idx, 1.0)
        return mask

    def forward(self, pred_logits, rsrp_pred, gamma_true):
        loss_mse = torch.tensor(0.0, device=pred_logits.device)
        loss_link = torch.tensor(0.0, device=pred_logits.device)
        loss_kl = torch.tensor(0.0, device=pred_logits.device)

        if self.use_mse:
            mask = self._oracle_topk_mask(gamma_true, self.config.K)
            denom = mask.sum().clamp_min(1.0)
            loss_mse = (((pred_logits - gamma_true) ** 2) * mask).sum() / denom

        if self.use_link:
            rsrp_label = torch.max(gamma_true, dim=1, keepdim=True).values
            loss_link = F.mse_loss(rsrp_pred, rsrp_label)

        if self.use_kl:
            tau = self.tau
            target_prob = F.softmax(gamma_true / tau, dim=-1)
            log_pred = F.log_softmax(pred_logits / tau, dim=-1)
            loss_kl = F.kl_div(log_pred, target_prob, reduction="batchmean") * (tau * tau)

        total = loss_mse + self.lambda_weight * loss_link + loss_kl
        return total, {
            "total_loss": float(total.detach().cpu().item()),
            "mse": float(loss_mse.detach().cpu().item()),
            "link": float(loss_link.detach().cpu().item()),
            "kl": float(loss_kl.detach().cpu().item()),
        }


class BeamTrackingTrainer:
    """Trainer for teacher-forcing and rollout validation."""

    def __init__(
        self,
        config: BeamTrackingConfig,
        model: nn.Module,
        device: torch.device,
        use_mse: bool,
        use_link: bool,
        use_kl: bool,
        tau: float,
        weight_decay: float,
        disable_tqdm: bool,
    ):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.disable_tqdm = disable_tqdm

        self.criterion = BeamTrackingLoss(
            config,
            use_mse=use_mse,
            use_link=use_link,
            use_kl=use_kl,
            tau=tau,
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=10)

        self.train_history = {
            "train_loss": [],
            "train_mse": [],
            "train_link": [],
            "train_kl": [],
            "val_loss": [],
            "val_mse": [],
            "val_link": [],
            "val_kl": [],
            "val_hit_rate": [],
            "val_roll_hit_rate": [],
            "val_roll_regret": [],
        }

    @staticmethod
    def _topk_mask_from_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
        idx = torch.topk(logits, k, dim=1).indices
        mask = torch.zeros_like(logits)
        mask.scatter_(1, idx, 1.0)
        return mask

    @staticmethod
    def _oracle_topk_mask(gamma_true: torch.Tensor, k: int) -> torch.Tensor:
        idx = torch.topk(gamma_true, k, dim=1).indices
        mask = torch.zeros_like(gamma_true)
        mask.scatter_(1, idx, 1.0)
        return mask

    def _build_tf_input(self, gamma_hist, k, warmup_steps, rand_mask_prob, seq_rand_prob):
        b, m, d = gamma_hist.shape
        dev = gamma_hist.device
        meas = torch.zeros_like(gamma_hist)
        use_seq_rand = (torch.rand(b, device=dev) < seq_rand_prob)

        def random_k_mask(num):
            idx = torch.topk(torch.rand(num, d, device=dev), k, dim=1).indices
            mm = torch.zeros((num, d), device=dev, dtype=gamma_hist.dtype)
            mm.scatter_(1, idx, 1.0)
            return mm

        for i in range(m):
            g = gamma_hist[:, i, :]
            if i < warmup_steps:
                meas[:, i, :] = g
                continue
            mask = self._oracle_topk_mask(g, k)

            if use_seq_rand.any():
                seq_idx = use_seq_rand.nonzero(as_tuple=False).squeeze(1)
                mask[seq_idx] = random_k_mask(seq_idx.numel())

            if rand_mask_prob > 0:
                use_rand = (torch.rand(b, device=dev) < rand_mask_prob)
                if use_rand.any():
                    rand_idx = use_rand.nonzero(as_tuple=False).squeeze(1)
                    mask[rand_idx] = random_k_mask(rand_idx.numel())

            meas[:, i, :] = g * mask

        return meas

    def _forward_model(self, input_seq, station_idx):
        return self.model(input_seq, station_idx)

    def train_epoch(self, train_loader, epoch, warmup_steps, rand_mask_prob, train_rollout_horizon, ss_warmup_epochs, ss_ramp_epochs, seq_rand_prob):
        self.model.train()

        if epoch < ss_warmup_epochs:
            p_model = 0.0
        else:
            p_model = min(1.0, (epoch - ss_warmup_epochs) / max(1, ss_ramp_epochs))

        total = {"loss": 0.0, "mse": 0.0, "link": 0.0, "kl": 0.0, "n": 0}
        pbar = train_loader if self.disable_tqdm else tqdm(train_loader, desc=f"train p_model={p_model:.2f}")

        for batch in pbar:
            gamma_hist = batch["gamma_hist"].to(self.device)
            gamma_future = batch["gamma_future"].to(self.device)
            station_hist = batch["station_indices_hist"].to(self.device)
            station_future = batch["station_indices_future"].to(self.device)

            b = gamma_hist.shape[0]
            h_avail = gamma_future.shape[1]
            h = min(int(train_rollout_horizon), h_avail)

            meas_hist = self._build_tf_input(gamma_hist, self.config.K, warmup_steps, rand_mask_prob, seq_rand_prob)
            station_window = station_hist.clone()

            self.optimizer.zero_grad()
            loss_sum = torch.tensor(0.0, device=self.device)
            mse_sum = 0.0
            link_sum = 0.0
            kl_sum = 0.0

            for step in range(h):
                pred_logits, rsrp_pred = self._forward_model(meas_hist, station_window)
                gamma_true = gamma_future[:, step, :]

                loss, dct = self.criterion(pred_logits, rsrp_pred, gamma_true)
                loss_sum = loss_sum + loss
                mse_sum += dct["mse"]
                link_sum += dct["link"]
                kl_sum += dct["kl"]

                use_model = (torch.rand(b, device=self.device) < p_model)
                mask_oracle = self._oracle_topk_mask(gamma_true, self.config.K)
                mask_model = self._topk_mask_from_logits(pred_logits, self.config.K)
                mask_next = torch.where(use_model.unsqueeze(1), mask_model, mask_oracle)

                measured = gamma_true * mask_next
                meas_hist = torch.cat([meas_hist[:, 1:, :], measured.unsqueeze(1)], dim=1)

                st = station_future[:, step, :].unsqueeze(1)
                station_window = torch.cat([station_window[:, 1:, :], st], dim=1)

            loss_sum = loss_sum / max(1, h)
            loss_sum.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total["loss"] += float(loss_sum.detach().cpu().item())
            total["mse"] += mse_sum / max(1, h)
            total["link"] += link_sum / max(1, h)
            total["kl"] += kl_sum / max(1, h)
            total["n"] += 1

            if not self.disable_tqdm:
                pbar.set_postfix({
                    "loss": f"{total['loss']/max(1,total['n']):.4f}",
                    "mse": f"{total['mse']/max(1,total['n']):.4f}",
                    "kl": f"{total['kl']/max(1,total['n']):.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                })

        n = max(1, total["n"])
        return {
            "train_loss": total["loss"] / n,
            "train_mse": total["mse"] / n,
            "train_link": total["link"] / n,
            "train_kl": total["kl"] / n,
        }

    @torch.no_grad()
    def validate_epoch(self, val_loader, warmup_steps, rollout_horizon):
        self.model.eval()
        total = {"loss": 0.0, "mse": 0.0, "link": 0.0, "kl": 0.0, "n": 0}
        hit, count = 0, 0
        roll_hit, roll_count, roll_regret_sum = 0, 0, 0.0

        pbar = val_loader if self.disable_tqdm else tqdm(val_loader, desc="validate")
        for batch in pbar:
            gamma_hist = batch["gamma_hist"].to(self.device)
            gamma_future = batch["gamma_future"].to(self.device)
            station_hist = batch["station_indices_hist"].to(self.device)
            station_future = batch["station_indices_future"].to(self.device)

            h = min(int(rollout_horizon), gamma_future.shape[1])

            inp = self._build_tf_input(gamma_hist, self.config.K, warmup_steps, 0.0, 0.0)
            pred_logits, rsrp_pred = self._forward_model(inp, station_hist)
            gamma_true0 = gamma_future[:, 0, :]

            _, dct = self.criterion(pred_logits, rsrp_pred, gamma_true0)
            total["loss"] += dct["total_loss"]
            total["mse"] += dct["mse"]
            total["link"] += dct["link"]
            total["kl"] += dct["kl"]
            total["n"] += 1

            true_top1 = torch.argmax(gamma_true0, dim=1)
            pred_topk = torch.topk(pred_logits, self.config.K, dim=1).indices
            for i in range(pred_topk.shape[0]):
                if true_top1[i] in pred_topk[i]:
                    hit += 1
            count += pred_topk.shape[0]

            meas_hist = self._build_tf_input(gamma_hist, self.config.K, warmup_steps, 0.0, 0.0)
            station_window = station_hist.clone()

            for step in range(h):
                pred_h, _ = self._forward_model(meas_hist, station_window)
                mask_pred = self._topk_mask_from_logits(pred_h, self.config.K)
                g_true = gamma_future[:, step, :]

                t1 = torch.argmax(g_true, dim=1)
                ptk = torch.topk(pred_h, self.config.K, dim=1).indices
                for i in range(ptk.shape[0]):
                    if t1[i] in ptk[i]:
                        roll_hit += 1
                roll_count += ptk.shape[0]

                measured = g_true * mask_pred
                true_max = torch.max(g_true, dim=1).values
                meas_max = torch.max(measured, dim=1).values
                regret = (true_max - meas_max).clamp_min(0.0)
                roll_regret_sum += float(regret.mean().cpu().item()) * ptk.shape[0]

                meas_hist = torch.cat([meas_hist[:, 1:, :], measured.unsqueeze(1)], dim=1)
                st = station_future[:, step, :].unsqueeze(1)
                station_window = torch.cat([station_window[:, 1:, :], st], dim=1)

        n = max(1, total["n"])
        return {
            "val_loss": total["loss"] / n,
            "val_mse": total["mse"] / n,
            "val_link": total["link"] / n,
            "val_kl": total["kl"] / n,
            "val_hit_rate": hit / max(1, count),
            "val_roll_hit_rate": roll_hit / max(1, roll_count),
            "val_roll_regret": roll_regret_sum / max(1, roll_count),
        }

    def save_model(self, save_dir, filename):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "history": self.train_history,
        }, path)

    def train(self, train_loader, val_loader, num_epochs, save_dir, warmup_steps, rand_mask_prob, rollout_horizon, train_rollout_horizon, ss_warmup_epochs, ss_ramp_epochs, seq_rand_prob):
        best_roll_hit = -1.0
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            tr = self.train_epoch(
                train_loader,
                epoch,
                warmup_steps,
                rand_mask_prob,
                train_rollout_horizon,
                ss_warmup_epochs,
                ss_ramp_epochs,
                seq_rand_prob,
            )
            va = self.validate_epoch(val_loader, warmup_steps, rollout_horizon)
            self.scheduler.step(va["val_loss"])

            self.train_history["train_loss"].append(tr["train_loss"])
            self.train_history["train_mse"].append(tr["train_mse"])
            self.train_history["train_link"].append(tr["train_link"])
            self.train_history["train_kl"].append(tr["train_kl"])
            self.train_history["val_loss"].append(va["val_loss"])
            self.train_history["val_mse"].append(va["val_mse"])
            self.train_history["val_link"].append(va["val_link"])
            self.train_history["val_kl"].append(va["val_kl"])
            self.train_history["val_hit_rate"].append(va["val_hit_rate"])
            self.train_history["val_roll_hit_rate"].append(va["val_roll_hit_rate"])
            self.train_history["val_roll_regret"].append(va["val_roll_regret"])

            print(
                f"epoch={epoch+1}/{num_epochs} "
                f"train_loss={tr['train_loss']:.4f} val_loss={va['val_loss']:.4f} "
                f"hit@K={va['val_hit_rate']:.4f} roll_hit@K={va['val_roll_hit_rate']:.4f} "
                f"roll_regret={va['val_roll_regret']:.4f} lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )

            if (epoch + 1) % 50 == 0:
                self.save_model(save_dir, f"model_epoch_{epoch+1}.pth")

            if va["val_roll_hit_rate"] > best_roll_hit:
                best_roll_hit = va["val_roll_hit_rate"]
                self.save_model(save_dir, "best_roll_hit_model.pth")

            if va["val_loss"] < best_val_loss:
                best_val_loss = va["val_loss"]
                self.save_model(save_dir, "best_val_loss_model.pth")

        with open(os.path.join(save_dir, "training_history.json"), "w") as f:
            json.dump(self.train_history, f, indent=2)
        self.save_model(save_dir, "final_model.pth")


def split_dataset_files(data_dir, val_ratio=0.2, random_seed=42):
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])
    train_files, val_files = train_test_split(all_files, test_size=val_ratio, random_state=random_seed)
    return train_files, val_files


def main():
    parser = argparse.ArgumentParser(description="Beam tracking teacher-forcing training")

    parser.add_argument("--train_data_dir", type=str, default="/home/lyus1/wwp/eval/py/train_spd15_LOS")
    parser.add_argument("--neighbors_path", type=str, default=os.path.join(os.path.dirname(__file__), "..", "neighbors.json"))
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--disable_tqdm", type=str2bool, default=None)

    parser.add_argument("--M", type=int, default=4)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lambda_weight", type=float, default=1.0)
    parser.add_argument("--lstm_hidden_size", type=int, default=32)

    parser.add_argument("--warmup_steps", type=int, default=4)
    parser.add_argument("--rand_mask_prob", type=float, default=0.0)
    parser.add_argument("--seq_rand_prob", type=float, default=0.2)
    parser.add_argument("--rollout_horizon", type=int, default=4)
    parser.add_argument("--train_rollout_horizon", type=int, default=4)
    parser.add_argument("--ss_warmup_epochs", type=int, default=2)
    parser.add_argument("--ss_ramp_epochs", type=int, default=50)

    parser.add_argument("--use_dilated", type=str2bool, default=False)
    parser.add_argument("--s", type=int, default=1)
    parser.add_argument("--station_hold_prob", type=float, default=0.2)
    parser.add_argument("--hold_only_on_change", type=str2bool, default=True)

    parser.add_argument("--use_mse", type=str2bool, default=True)
    parser.add_argument("--use_link", type=str2bool, default=True)
    parser.add_argument("--use_kl", type=str2bool, default=True)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--use_attention", type=str2bool, default=True)
    parser.add_argument("--use_station_embedding", type=str2bool, default=True)

    args = parser.parse_args()
    if args.disable_tqdm is None:
        import sys
        args.disable_tqdm = not sys.stdout.isatty()

    set_seed(args.seed)

    config = BeamTrackingConfig()
    config.M = int(args.M)
    config.K = int(args.K)
    config.learning_rate = float(args.learning_rate)
    config.lambda_weight = float(args.lambda_weight)
    config.lstm_hidden_size = int(args.lstm_hidden_size)
    config.time_window_ms = config.M * config.ssb_period_ms

    selected_gnb = 7
    config.D = selected_gnb * config.codebook_size * config.num_rx_beams

    train_files, val_files = split_dataset_files(args.train_data_dir, val_ratio=args.val_ratio, random_seed=args.seed)

    train_ds = Data2SeqDatasetV2(
        config,
        args.train_data_dir,
        args.neighbors_path,
        train_files,
        horizon=max(1, args.train_rollout_horizon),
        station_hold_prob=args.station_hold_prob,
        hold_only_on_change=args.hold_only_on_change,
        use_dilated=args.use_dilated,
        s=args.s,
    )
    val_ds = Data2SeqDatasetV2(
        config,
        args.train_data_dir,
        args.neighbors_path,
        val_files,
        horizon=max(1, args.rollout_horizon),
        station_hold_prob=0.0,
        hold_only_on_change=True,
        use_dilated=args.use_dilated,
        s=args.s,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BeamTrackingLSTMWithAttention(
        config,
        use_attention=args.use_attention,
        use_station_embedding=args.use_station_embedding,
    ).to(device)

    trainer = BeamTrackingTrainer(
        config=config,
        model=model,
        device=device,
        use_mse=args.use_mse,
        use_link=args.use_link,
        use_kl=args.use_kl,
        tau=args.tau,
        weight_decay=args.weight_decay,
        disable_tqdm=args.disable_tqdm,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"PID: {os.getpid()}")
    print(f"device={device}, params={sum(p.numel() for p in model.parameters())}")

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        warmup_steps=args.warmup_steps,
        rand_mask_prob=args.rand_mask_prob,
        rollout_horizon=args.rollout_horizon,
        train_rollout_horizon=args.train_rollout_horizon,
        ss_warmup_epochs=args.ss_warmup_epochs,
        ss_ramp_epochs=args.ss_ramp_epochs,
        seq_rand_prob=args.seq_rand_prob,
    )

    print("training finished")


if __name__ == "__main__":
    main()
