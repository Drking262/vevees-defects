"""Patch an installed `ultralytics` package to support the YOLOV8-CDC custom
architecture (ADown2, C2f_DWRSeg, MultiDilatelocalAttention, from
yolov8_CDC.yaml), and fix a torch>=2.6 `weights_only` incompatibility with
older ultralytics releases.

Automates the original repo's manual setup (drop 3 custom nn.Module files
into `ultralytics/nn/Addmodules`, register them in `tasks.py`) against a
fresh `ultralytics==8.1.0` install (oldest version with `dist2rbox`, which
DWRSeg.py needs). Does NOT replace loss.py/metrics.py as the original README
also suggests -- those target an older internal API and would break
OBBMetrics/ClassifyMetrics in 8.1.0. Training uses stock CIoU loss instead
of the paper's custom WIoU/Focaleriou loss; the architecture itself is
intact.

Idempotent: safe to re-run.

Usage:
    pip install "ultralytics==8.1.0"
    python setup_cdc_ultralytics.py
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Locate the installed package WITHOUT importing it: importing ultralytics
# here would pull nn/tasks.py into sys.modules before we've patched it, and
# the patched-on-disk version would never take effect in this process.
_spec = importlib.util.find_spec("ultralytics")
if _spec is None or not _spec.submodule_search_locations:
    raise SystemExit("ultralytics is not installed -- run `pip install ultralytics==8.1.0` first")
UL_ROOT = Path(next(iter(_spec.submodule_search_locations)))
ADDMODULES_DIR = UL_ROOT / "nn" / "Addmodules"

SOURCE_FILES = {
    "c_adown.py": "C-ADown.py",
    "dwrseg.py": "DWRSeg.py",
    "msda.py": "MSDA.py",
}

INIT_PY = '''from .msda import MultiDilatelocalAttention
from .dwrseg import C2f_DWRSeg, Bottleneck_DWRSeg, DWRSeg_Conv
from .c_adown import ADown2

__all__ = [
    "ADown2",
    "MultiDilatelocalAttention",
    "C2f_DWRSeg",
    "Bottleneck_DWRSeg",
    "DWRSeg_Conv",
]
'''


def install_addmodules() -> None:
    ADDMODULES_DIR.mkdir(parents=True, exist_ok=True)
    for dest_name, src_name in SOURCE_FILES.items():
        shutil.copy(REPO_ROOT / src_name, ADDMODULES_DIR / dest_name)
    (ADDMODULES_DIR / "__init__.py").write_text(INIT_PY)
    print(f"installed Addmodules -> {ADDMODULES_DIR}")


def patch_tasks_py() -> None:
    path = UL_ROOT / "nn" / "tasks.py"
    text = path.read_text()

    if "from ultralytics.nn.Addmodules import ADown2, C2f_DWRSeg" not in text:
        text = text.replace(
            "from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load",
            "from ultralytics.nn.Addmodules import ADown2, C2f_DWRSeg\n"
            "from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load",
            1,
        )

    if re.search(r"\n\s+C3x,\n\s+RepC3,\n\s+ADown2,\n\s+C2f_DWRSeg,\n", text):
        n1 = 1  # already patched
    else:
        text, n1 = re.subn(
            r"(\n\s+C3x,\n\s+RepC3,\n)(\s+\):\n\s+c1, c2 = ch\[f\], args\[0\])",
            r"\1            ADown2,\n            C2f_DWRSeg,\n\2",
            text,
            count=1,
        )

    if "RepC3, C2f_DWRSeg):" in text:
        n2 = 1  # already patched
    else:
        text, n2 = re.subn(
            r"if m in \(BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3\):",
            "if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3, C2f_DWRSeg):",
            text,
            count=1,
        )

    if n1 == 0 or n2 == 0:
        raise SystemExit(
            "tasks.py structure didn't match the expected ultralytics==8.1.0 layout "
            "(parse_model's module tuple) -- inspect it manually before patching. "
            f"(matched channel-arg tuple: {bool(n1)}, matched repeat-insert tuple: {bool(n2)})"
        )

    path.write_text(text)
    print(f"patched {path}")


def patch_weights_only() -> None:
    """torch>=2.6 defaults torch.load(weights_only=True), which breaks loading
    plain ultralytics DetectionModel checkpoints on ultralytics<8.3ish. Force
    weights_only=False on the checkpoint loads used during our own training
    (safe: these are checkpoints this same job produced or a trusted .pt)."""
    torch_utils_path = UL_ROOT / "utils" / "torch_utils.py"
    text = torch_utils_path.read_text()
    old = 'x = torch.load(f, map_location=torch.device("cpu"))'
    new = 'x = torch.load(f, map_location=torch.device("cpu"), weights_only=False)'
    if old in text:
        torch_utils_path.write_text(text.replace(old, new, 1))
        print(f"patched {torch_utils_path}")

    tasks_path = UL_ROOT / "nn" / "tasks.py"
    text = tasks_path.read_text()
    old = 'return torch.load(file, map_location="cpu"), file  # load'
    new = 'return torch.load(file, map_location="cpu", weights_only=False), file  # load'
    if old in text:
        tasks_path.write_text(text.replace(old, new))
        print(f"patched {tasks_path} (torch_safe_load)")


def patch_numpy_trapz() -> None:
    """numpy>=2.0 removed np.trapz (renamed np.trapezoid); ultralytics==8.1.0's
    ap_per_class/compute_ap still calls np.trapz, which crashes validation
    (AttributeError) on any environment with a modern numpy. Safe no-op if
    already patched or if the installed numpy still has trapz."""
    metrics_path = UL_ROOT / "utils" / "metrics.py"
    text = metrics_path.read_text()
    old = "ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate"
    new = (
        "trapz_fn = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz  # numpy>=2.0 renamed trapz\n"
        "        ap = trapz_fn(np.interp(x, mrec, mpre), x)  # integrate"
    )
    if old in text:
        metrics_path.write_text(text.replace(old, new, 1))
        print(f"patched {metrics_path} (np.trapz -> np.trapezoid fallback)")


def main() -> None:
    print(f"patching ultralytics at {UL_ROOT}")
    install_addmodules()
    patch_tasks_py()
    patch_weights_only()
    patch_numpy_trapz()

    # Smoke test in a FRESH subprocess -- this process must never `import
    # ultralytics` itself, or it'd cache the pre-patch module in sys.modules.
    # Runs a real 1-epoch train+val on ultralytics' tiny built-in coco8
    # dataset (auto-downloads, 8 images) so it also exercises the validation
    # path (this is what caught the np.trapz incompatibility).
    smoke_test = (
        "from ultralytics import YOLO\n"
        f"m = YOLO(r'{REPO_ROOT / 'yolov8_CDC.yaml'}')\n"
        "m.train(data='coco8.yaml', epochs=1, imgsz=320, batch=2, workers=0, verbose=False)\n"
        "print('smoke test OK: train+val completed')\n"
    )
    result = subprocess.run([sys.executable, "-c", smoke_test], cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise SystemExit("smoke test failed -- see traceback above")


if __name__ == "__main__":
    main()
