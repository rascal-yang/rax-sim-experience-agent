import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, x):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False))
