#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import numpy as np


class BeamTrackingConfig:
    """Configuration for beam tracking training."""

    def __init__(self):
        cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {}
        else:
            cfg = {}

        self.M = int(cfg.get("M", 4))
        self.K = int(cfg.get("K", 64))
        self.num_stations = int(cfg.get("num_stations", 20))

        self.ssb_period_ms = int(cfg.get("ssb_period", 20))
        self.time_window_ms = self.M * self.ssb_period_ms

        self.rsrp_min_dbm = float(cfg.get("rsrp_min_dbm", -120.0))
        self.rsrp_max_dbm = float(cfg.get("rsrp_max_dbm", -60.0))

        self.lstm_hidden_size = int(cfg.get("lstm_hidden_size", 32))

        self.batch_size = int(cfg.get("batch_size", 32))
        self.learning_rate = float(cfg.get("learning_rate", 1e-4))
        self.num_epochs = int(cfg.get("num_epochs", 100))
        self.lambda_weight = float(cfg.get("lambda_weight", 1.0))

        self.num_gnb = int(cfg.get("num_gnb", 7))
        self.bs_antenna_rows = int(cfg.get("bs_antenna_rows", 8))
        self.bs_antenna_cols = int(cfg.get("bs_antenna_cols", 8))
        self.codebook_size = int(cfg.get("codebook_size", self.bs_antenna_rows * self.bs_antenna_cols))
        self.num_tx_beams = self.codebook_size
        self.num_rx_beams = int(cfg.get("ue_antenna_rows", 1)) * int(cfg.get("ue_antenna_cols", 1))
        self.num_rx_beams = max(1, self.num_rx_beams)

        self.D = self.num_gnb * self.num_tx_beams * self.num_rx_beams

    def normalize_rsrp(self, rsrp_dbm):
        rsrp_clipped = np.clip(rsrp_dbm, self.rsrp_min_dbm, self.rsrp_max_dbm)
        return (rsrp_clipped - self.rsrp_min_dbm) / (self.rsrp_max_dbm - self.rsrp_min_dbm + 1e-12)

    def denormalize_rsrp(self, normalized_rsrp):
        return normalized_rsrp * (self.rsrp_max_dbm - self.rsrp_min_dbm) + self.rsrp_min_dbm
