"""
MSEM + PointRend-style PR-Head modules for Ultralytics YOLO segmentation.

Designed for research use with YOLO26s-seg / YOLOv8-style Segment heads.
The modules are intentionally self-contained so they can be registered into
Ultralytics without editing site-packages directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    """SE-style channel attention."""
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        mid = max(c // r, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, c, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x)).view(x.shape[0], x.shape[1], 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CrossScaleFusion(nn.Module):
    """Top-down P5->P4 and P4->P3 semantic fusion."""
    def __init__(self, hi_c: int, lo_c: int, out_c: int):
        super().__init__()
        self.phi = ConvBnAct(hi_c, out_c, 1)
        self.fuse = ConvBnAct(out_c + lo_c, out_c, 1)

    def forward(self, hi: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
        hi = F.interpolate(self.phi(hi), size=lo.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([hi, lo], dim=1))


class MSEM(nn.Module):
    """
    Multi-Scale Semantic Enhancement Module.

    Input and output are the three segmentation-head feature maps [P3, P4, P5].
    It is intended to be placed immediately before the Segment head.
    """
    def __init__(self, ch=(128, 256, 512)):
        super().__init__()
        c3, c4, c5 = [int(v) for v in ch]
        self.ca = nn.ModuleList([ChannelAttention(c) for c in (c3, c4, c5)])
        self.sa = nn.ModuleList([SpatialAttention(), SpatialAttention(), SpatialAttention()])
        self.f54 = CrossScaleFusion(c5, c4, c4)
        self.f43 = CrossScaleFusion(c4, c3, c3)
        self.proj = nn.ModuleList([ConvBnAct(c3, c3, 3), ConvBnAct(c4, c4, 3), ConvBnAct(c5, c5, 3)])

    def forward(self, x):
        p3, p4, p5 = x
        p3, p4, p5 = self.ca[0](p3), self.ca[1](p4), self.ca[2](p5)
        p3, p4, p5 = self.sa[0](p3), self.sa[1](p4), self.sa[2](p5)
        p4 = self.f54(p5, p4)
        p3 = self.f43(p4, p3)
        return [self.proj[0](p3) + p3, self.proj[1](p4) + p4, self.proj[2](p5) + p5]


class PRHead(nn.Module):
    """
    PointRend-inspired prototype refinement head.

    This module refines the prototype mask tensor produced by the Segment head
    using high-resolution P3 features. It is differentiable and can be trained
    through the normal segmentation mask loss because the refined prototype is
    passed back to Ultralytics loss/inference in place of the coarse prototype.

    Args:
        proto_c: prototype channels, usually 32 in YOLO segmentation.
        fine_c: high-resolution P3 feature channels.
        k_points: uncertain points per iteration.
        n_iters: refinement iterations.
        hidden: MLP width.
    """
    def __init__(self, proto_c: int = 32, fine_c: int = 128, k_points: int = 196, n_iters: int = 3, hidden: int = 128):
        super().__init__()
        self.proto_c = int(proto_c)
        self.fine_c = int(fine_c)
        self.k_points = int(k_points)
        self.n_iters = int(n_iters)
        self.mlp = nn.Sequential(
            nn.Conv1d(self.proto_c + self.fine_c, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, self.proto_c, 1),
        )

    @staticmethod
    def _idx_to_grid(idx: torch.Tensor, h: int, w: int) -> torch.Tensor:
        gy = (idx // w).float() / max(h - 1, 1) * 2 - 1
        gx = (idx % w).float() / max(w - 1, 1) * 2 - 1
        return torch.stack([gx, gy], dim=-1).unsqueeze(2)  # B,K,1,2

    @staticmethod
    def _uncertainty(proto: torch.Tensor) -> torch.Tensor:
        # B,C,H,W -> B,H*W. Higher value = closer to decision boundary.
        prob = torch.sigmoid(proto.float())
        return 1.0 - (prob - 0.5).abs().mean(dim=1).view(proto.shape[0], -1) * 2.0

    def forward(self, proto: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        b, c, h, w = proto.shape
        if c != self.proto_c:
            raise RuntimeError(f"PRHead expected proto_c={self.proto_c}, got {c}")

        out = proto
        for _ in range(self.n_iters):
            k = min(self.k_points, h * w)
            _, idx = self._uncertainty(out).topk(k, dim=1)

            out_flat = out.view(b, c, -1)
            coarse_pts = out_flat.gather(2, idx.unsqueeze(1).expand(b, c, k))

            grid = self._idx_to_grid(idx, h, w)
            fine_pts = F.grid_sample(p3.float(), grid.float(), mode="bilinear", align_corners=False, padding_mode="border").squeeze(-1).to(p3.dtype)

            delta = self.mlp(torch.cat([coarse_pts, fine_pts], dim=1))
            out_flat = out_flat.clone()
            out_flat.scatter_(2, idx.unsqueeze(1).expand(b, c, k), coarse_pts + delta)
            out = out_flat.view(b, c, h, w)
        return out


class BoundaryLoss(nn.Module):
    """Lightweight total-variation boundary regularizer for refined prototypes."""
    def __init__(self, weight: float = 0.02):
        super().__init__()
        self.weight = float(weight)

    def forward(self, refined_proto: torch.Tensor, coarse_proto: torch.Tensor | None = None) -> torch.Tensor:
        dy = (refined_proto[:, :, 1:, :] - refined_proto[:, :, :-1, :]).abs().mean()
        dx = (refined_proto[:, :, :, 1:] - refined_proto[:, :, :, :-1]).abs().mean()
        loss = dx + dy
        if coarse_proto is not None:
            # keep refinement stable; avoids the PR branch destroying prototypes early
            loss = loss + 0.25 * F.smooth_l1_loss(refined_proto, coarse_proto.detach())
        return self.weight * loss
