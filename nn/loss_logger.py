import os

from ultralytics.utils import LOGGER, RANK


_ENV_LTYPE       = 'YOLO_IOU_LOSS'
_ENV_MONOTONOUS  = 'YOLO_IOU_MONOTONOUS'
_ENV_INNER_RATIO = 'YOLO_IOU_INNER_RATIO'

_VALID_LTYPES = {
    'IoU', 'WIoU', 'GIoU', 'DIoU', 'CIoU', 'EIoU',
    'SIoU', 'MPDIoU', 'InnerIoU', 'WiseInnerMPDIoU',
}

_MONOTONOUS_MAP = {
    'true':  True,
    'false': False,
    'none':  None,
}

def _monotonous_label(m) -> str:
    """Human-readable label for the monotonous value."""
    return {
        True:  'True  (v2 — monotonic FM)',
        False: 'False (v3 — non-monotonic FM)',
        None:  'None  (v1 — no focusing)',
    }[m]

def read_loss_config() -> dict:
    """
    Read and validate the active loss config from environment variables.
 
    Returns a dict with keys: ltype, monotonous, inner_ratio.
    Raises ValueError for invalid values so the error surfaces at
    training start rather than mid-run.
    """
    # ── ltype ─────────────────────────────────────────────────────────
    ltype = os.environ.get(_ENV_LTYPE, 'WiseInnerMPDIoU').strip()
    if ltype not in _VALID_LTYPES:
        raise ValueError(
            f"YOLO_IOU_LOSS='{ltype}' is not valid. "
            f"Choose from: {sorted(_VALID_LTYPES)}"
        )
 
    # ── monotonous ────────────────────────────────────────────────────
    mono_raw = os.environ.get(_ENV_MONOTONOUS, 'false').strip().lower()
    if mono_raw not in _MONOTONOUS_MAP:
        raise ValueError(
            f"YOLO_IOU_MONOTONOUS='{mono_raw}' is not valid. "
            f"Use: 'true', 'false', or 'none'."
        )
    monotonous = _MONOTONOUS_MAP[mono_raw]
 
    # ── inner_ratio ───────────────────────────────────────────────────
    ratio_raw = os.environ.get(_ENV_INNER_RATIO, '0.7').strip()
    try:
        inner_ratio = float(ratio_raw)
    except ValueError:
        raise ValueError(
            f"YOLO_IOU_INNER_RATIO='{ratio_raw}' is not a valid float."
        )
    if not (0 < inner_ratio <= 1.0):
        raise ValueError(
            f"YOLO_IOU_INNER_RATIO={inner_ratio} must be in (0, 1]."
        )
 
    return dict(ltype=ltype, monotonous=monotonous, inner_ratio=inner_ratio)

def log_loss_config(trainer) -> None:
    """
    Ultralytics callback — fires once at on_pretrain_routine_end.
 
    Logs the active loss configuration through Ultralytics' LOGGER so it
    appears in the same output stream as the rest of training info, and
    is automatically suppressed on non-primary DDP workers (RANK != 0).
 
    Also validates the env vars early so a misconfiguration surfaces
    before the first training step rather than crashing mid-epoch.
 
    Register with:
        model.add_callback('on_pretrain_routine_end', log_loss_config)
    """
    if RANK not in {-1, 0}:
        return   # silent on DDP workers
 
    try:
        cfg = read_loss_config()
    except ValueError as e:
        LOGGER.error(f"❌  Invalid loss config: {e}")
        raise
 
    # Grab iou_mean from the live BboxLoss module if accessible
    iou_mean_str = 'n/a'
    try:
        # trainer.model is de-paralleled; the detect head is the last layer
        bbox_loss = trainer.model.model[-1].loss.bbox_loss
        if hasattr(bbox_loss, 'iouloss'):
            iou_mean = bbox_loss.iouloss.iou_mean.item()
            iou_mean_str = f'{iou_mean:.4f}'
    except Exception:
        pass   # model may not be fully wired yet; not critical
 
    sep = '─' * 44
    LOGGER.info(
        f'\nLoss config {sep}\n'
        f'  type        : {cfg["ltype"]}\n'
        f'  monotonous  : {_monotonous_label(cfg["monotonous"])}\n'
        f'  inner_ratio : {cfg["inner_ratio"]}'
        + (f'\n  iou_mean    : {iou_mean_str} (EMA buffer)' if iou_mean_str != 'n/a' else '')
        + f'\n{sep}\n'
    )

