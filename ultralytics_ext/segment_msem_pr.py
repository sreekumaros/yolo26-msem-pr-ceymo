"""
Custom Segment head with integrated MSEM and PRHead.

This subclasses the installed Ultralytics Segment head. It enhances the incoming
[P3,P4,P5] features with MSEM before the regular Segment operations, then refines
the prototype tensor with PRHead. Because the refined prototype replaces the
coarse prototype in the returned prediction tuple, the standard Ultralytics
segmentation loss and inference path use PRHead outputs.
"""
from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn

try:
    from ultralytics.nn.modules.head import Segment26
except Exception as e:  # pragma: no cover
    raise ImportError("Could not import ultralytics.nn.modules.head.Segment. Install ultralytics first.") from e

from .msem_pr_modules import MSEM, PRHead



class MSEMPRSegment26(Segment26):
    """
    Drop-in replacement for Ultralytics Segment.

    YAML args expected by parse_model:
        MSEMPRSegment, [nc, nm, npr, pr_k, pr_iters]

    Ultralytics parse_model appends the feature-channel list `ch` automatically
    for head modules when this class is registered as a Segment-like module.
    If your installed parse_model does not do this automatically, use the
    provided patch script or manual tasks.py edit from docs/MANUAL_PATCH.md.
    """
    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, pr_k: int = 196, pr_iters: int = 3, reg_max: int = 16, end2end: bool = False, ch=()):
        super().__init__(nc=nc, nm=nm, npr=npr, reg_max=reg_max, end2end=end2end, ch=ch)
        if not ch or len(ch) < 3:
            raise ValueError(f"MSEMPRSegment requires three input channels [P3,P4,P5], got ch={ch}")
        self.msem = MSEM(ch=tuple(ch[:3]))
        self.prhead = PRHead(proto_c=nm, fine_c=int(ch[0]), k_points=pr_k, n_iters=pr_iters)
        self.pr_k = pr_k
        self.pr_iters = pr_iters
        self.pr_aux: dict[str, torch.Tensor] = {}

    @staticmethod
    def _replace_proto_in_output(y: Any, refined_proto: torch.Tensor):
        """
        Ultralytics Segment outputs differ slightly by version.
        Typical training return: (x, mc, p) where p is prototype tensor.
        Typical inference return: (pred, (x, mc, p)) or similar.
        This helper recursively replaces the last 4D prototype tensor with refined_proto.
        """
        if isinstance(y, torch.Tensor):
            return refined_proto if (y.ndim == 4 and y.shape[1] == refined_proto.shape[1]) else y
        if isinstance(y, tuple):
            items = list(y)
            replaced = False
            # Prefer exact-shape replacement from the end.
            for i in range(len(items) - 1, -1, -1):
                obj = items[i]
                if isinstance(obj, torch.Tensor) and obj.shape == refined_proto.shape:
                    items[i] = refined_proto
                    replaced = True
                    break
            if not replaced:
                items = [MSEMPRSegment26._replace_proto_in_output(v, refined_proto) for v in items]
            return tuple(items)
        if isinstance(y, list):
            return [MSEMPRSegment26._replace_proto_in_output(v, refined_proto) for v in y]
        return y

    @staticmethod
    def _find_proto(y: Any, nm: int):
        if isinstance(y, torch.Tensor) and y.ndim == 4 and y.shape[1] == nm:
            return y
        if isinstance(y, (tuple, list)):
            for v in reversed(y):
                found = MSEMPRSegment26._find_proto(v, nm)
                if found is not None:
                    return found
        return None

    def _refine_train_preds(self, preds: dict, p3: torch.Tensor) -> dict:
        """
        Training-mode path. Segment26 returns a dict here ({"one2many": {...}, "one2one": {...}}
        when end2end, else a flat dict) whose "proto" entry is a tensor or (proto, semantic).
        _find_proto() does not search dicts, so without this path PR-Head was skipped in
        training and only ran, untrained, at inference.
        """
        branch = preds["one2many"] if "one2many" in preds else preds
        proto, sem = branch["proto"] if isinstance(branch["proto"], tuple) else (branch["proto"], None)
        refined = self.prhead(proto, p3)
        self.pr_aux = {"coarse_proto": proto.detach(), "refined_proto": refined.detach()}  # detached: graph tensors break deepcopy (EMA)
        branch["proto"] = (refined, sem) if sem is not None else refined
        if "one2one" in preds:  # Segment26 feeds the one-to-one branch a detached prototype
            preds["one2one"]["proto"] = (refined.detach(), sem.detach()) if sem is not None else refined.detach()
        return preds

    def forward(self, x):
        # x is [P3,P4,P5] from YOLO neck.
        # getattr defaults keep checkpoints pickled before the ablation flags existed loadable.
        if getattr(self, "use_msem", True):
            x = self.msem(x)
        p3 = x[0]
        y = super().forward(x)
        if isinstance(y, dict) and getattr(self, "use_pr", True):
            return self._refine_train_preds(y, p3)
        proto = self._find_proto(y, self.nm) if getattr(self, "use_pr", True) else None
        if proto is None:
            self.pr_aux = {}
            return y
        refined = self.prhead(proto, p3)
        self.pr_aux = {"coarse_proto": proto.detach(), "refined_proto": refined.detach()}  # detached: graph tensors break deepcopy (EMA)
        return self._replace_proto_in_output(y, refined)


class MSEMOnlySegment26(MSEMPRSegment26):
    """Ablation variant: MSEM enabled, PR-Head removed."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.prhead
        self.use_pr = False


class PROnlySegment26(MSEMPRSegment26):
    """Ablation variant: PR-Head enabled, MSEM removed."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.msem
        self.use_msem = False


class PlainSegment26(MSEMPRSegment26):
    """Ablation control: neither MSEM nor PR-Head, but the same unscaled prototype width (npr)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.msem, self.prhead
        self.use_msem = False
        self.use_pr = False
