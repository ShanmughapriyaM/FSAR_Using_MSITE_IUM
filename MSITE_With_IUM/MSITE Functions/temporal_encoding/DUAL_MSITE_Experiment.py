import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
def segmented_multiscale_encoding(X, scales=(1, 2, 4)):
    """
    X : numpy array [T, D]
    returns: numpy array [sum(scales)*D]
    """
    T, D = X.shape
    encodings = []

    for s in scales:
        seg_len = T // s
        for i in range(s):
            start = i * seg_len
            end = T if i == s - 1 else (i + 1) * seg_len
            seg_feat = X[start:end]
            encodings.append(seg_feat.mean(axis=0))

    return np.concatenate(encodings, axis=0)
class LearnableTemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, out_dim=128):
        super().__init__()

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, out_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        x : Tensor [B, T, D]
        return: Tensor [B, out_dim]
        """
        x = x.transpose(1, 2)          # [B, D, T]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)   # [B, out_dim]
        return x
class DualMultiScaleTemporalEncoder:
    def __init__(self, feature_dim, scales=(1, 2, 4), device="cpu"):
        self.scales = scales
        self.device = device

        self.learnable_encoder = LearnableTemporalEncoder(
            input_dim=feature_dim,
            hidden_dim=128,
            out_dim=128
        ).to(device)

    def encode(self, X, use_learnable=True):
        """
        X : numpy array [T, D]
        returns: numpy array [Z]
        """

        # Branch-1: Segmented (training-free)
        z_segmented = segmented_multiscale_encoding(X, self.scales)

        if not use_learnable:
            return z_segmented

        # Branch-2: Learnable
        with torch.no_grad():  # training-free inference
            xt = torch.from_numpy(X).float().unsqueeze(0).to(self.device)
            z_learned = self.learnable_encoder(xt).cpu().numpy().squeeze(0)

        # Dual fusion
        return np.concatenate([z_segmented, z_learned], axis=0)
