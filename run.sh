#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh  —  CHIPPY AI Tutor launcher
#
# Fixes the cv2/PyQt5 Qt platform plugin conflict on Raspberry Pi by setting
# QT_QPA_PLATFORM_PLUGIN_PATH BEFORE Python starts, so cv2's bundled Qt
# plugins are never seen by PyQt5.
#
# Usage:
#   chmod +x run.sh
#   ./run.sh
#
# Optional env overrides (set before calling this script or in your .env):
#   DISPLAY_BACKEND=physical  →  HDMI on :0
#   DISPLAY_BACKEND=vnc       →  TigerVNC on :1
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")"          # always run from the project root

# ── 1. Find the venv Python ──────────────────────────────────────────────────
PYTHON="${VIRTUAL_ENV:-$(pwd)/venv}/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "       Activate your venv first:  source venv/bin/activate"
    exit 1
fi

# ── 2. Find the PyQt5 Qt5 plugins directory inside the venv ─────────────────
PYQT5_PLUGINS=$("$PYTHON" - << 'PYEOF'
import importlib.util, os, sys

# Walk possible locations in order of likelihood
candidates = []
for base in sys.path:
    for sub in ("PyQt5/Qt5/plugins", "PyQt5/Qt/plugins"):
        candidates.append(os.path.join(base, sub))

for c in candidates:
    if os.path.isdir(c):
        print(c)
        sys.exit(0)

# Final fallback: ask PyQt5 directly
try:
    from PyQt5.QtCore import QLibraryInfo
    p = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    if os.path.isdir(p):
        print(p)
        sys.exit(0)
except Exception:
    pass

sys.exit(1)
PYEOF
)

if [ -z "$PYQT5_PLUGINS" ]; then
    echo "ERROR: Could not locate PyQt5 Qt5 plugins directory."
    echo "       Is PyQt5 installed in your venv?  pip install PyQt5"
    exit 1
fi

echo "[run.sh] PyQt5 plugins: $PYQT5_PLUGINS"

# ── 3. Set Qt env vars at the SHELL level (before Python imports anything) ───
export QT_QPA_PLATFORM_PLUGIN_PATH="$PYQT5_PLUGINS"

# Use xcb when an X11 display is available, eglfs for direct framebuffer
if [ -n "$DISPLAY" ] || [ -n "$WAYLAND_DISPLAY" ]; then
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
else
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-eglfs}"
fi

# Suppress Qt debug noise in logs
export QT_LOGGING_RULES="*.debug=false;qt.qpa.*=false"

# Suppress OpenCV's own Qt plugin registration (key fix)
export OPENCV_IO_ENABLE_OPENEXR=0

echo "[run.sh] QT_QPA_PLATFORM=$QT_QPA_PLATFORM"
echo "[run.sh] QT_QPA_PLATFORM_PLUGIN_PATH=$QT_QPA_PLATFORM_PLUGIN_PATH"

# ── 4. Launch ────────────────────────────────────────────────────────────────
exec "$PYTHON" src/main.py "$@"