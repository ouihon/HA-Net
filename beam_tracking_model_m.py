#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from beam_tracking_config import BeamTrackingConfig


class BeamTrackingLSTMWithAttention(nn.Module):
    """LSTM beam tracker with temporal attention and optional station embeddings."""

    def __init__(self, config: BeamTrackingConfig, use_attention: bool = True, use_station_embedding: bool = True):
        super().__init__()
        self.config = config
        self.M = config.M
        self.D = config.D
        self.K = config.K
        self.num_stations = config.num_stations
        self.use_attention = bool(use_attention)
        self.use_station_embedding = bool(use_station_embedding)

        self.lstm1 = nn.LSTM(
            input_size=self.D,
            hidden_size=config.lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.lstm2 = nn.LSTM(
            input_size=config.lstm_hidden_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

        if self.use_station_embedding:
            self.station_embedding = nn.Embedding(self.num_stations, config.lstm_hidden_size)

        # Multi-head temporal attention over the encoded history.
        self.num_heads = 4
        self.head_dim = config.lstm_hidden_size // self.num_heads
        assert self.head_dim * self.num_heads == config.lstm_hidden_size, "hidden size must be divisible by num_heads"

        self.w_q = nn.Linear(config.lstm_hidden_size, config.lstm_hidden_size, bias=False)
        self.w_k = nn.Linear(config.lstm_hidden_size, config.lstm_hidden_size, bias=False)
        self.w_v = nn.Linear(config.lstm_hidden_size, config.lstm_hidden_size, bias=False)
        self.multihead_attention_out = nn.Linear(config.lstm_hidden_size, config.lstm_hidden_size)

        head_input_dim = config.lstm_hidden_size * 2 if self.use_station_embedding else config.lstm_hidden_size

        self.p_link_head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

        self.p_mcq_head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.D),
        )

    def multi_head_attention(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """Apply temporal multi-head attention to sequence features."""
        bsz, seq_len, hidden_size = lstm_out.size()

        q = self.w_q(lstm_out)
        k = self.w_k(lstm_out)
        v = self.w_v(lstm_out)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        weighted_v = torch.matmul(weights, v)

        weighted_v = weighted_v.transpose(1, 2).contiguous().view(bsz, seq_len, hidden_size)
        weighted_out = self.multihead_attention_out(weighted_v)
        return torch.mean(weighted_out, dim=1)

    def forward(self, x: torch.Tensor, station_idx: torch.Tensor):
        """Forward pass returning selection logits and link score."""
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)

        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)

        if self.use_attention:
            features = self.multi_head_attention(lstm2_out)
        else:
            features = torch.mean(lstm2_out, dim=1)

        if self.use_station_embedding:
            emb = self.station_embedding(station_idx[:, -1, 0])
            features = torch.cat([features, emb], dim=1)

        s_pred = self.p_mcq_head(features)
        link_pred = self.p_link_head(features)
        return s_pred, link_pred
