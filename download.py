"""Modal download entry for ffmpeg.

Run:
  modal run download.py::download
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal



app = modal.App("ffmpeg-download")


@app.local_entrypoint()
def download() -> None:
    print("No download step required for ffmpeg.")
