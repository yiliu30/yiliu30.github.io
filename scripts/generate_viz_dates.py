#!/usr/bin/env python3
"""Generate Hugo viz metadata from Git commit timestamps.

GitHub Actions checks out the full history, so each page's most recent Git
commit determines its position in the /viz/ index. No third-party packages.
"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIZ_DIR = ROOT / "static" / "viz"
OUTPUT = ROOT / "data" / "viz_dates.json"


def last_updated(path: Path) -> int:
    relative = path.relative_to(ROOT).as_posix()
    result = subprocess.run(
        ["git", "log", "-1", "--format=%ct", "--", relative],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    # Untracked pages can still appear in local previews before first commit.
    return int(result.stdout.strip()) if result.stdout.strip() else int(path.stat().st_mtime)


def main() -> None:
    if not VIZ_DIR.is_dir():
        raise SystemExit(f"Visualization directory not found: {VIZ_DIR}")
    dates = {}
    for path in sorted(VIZ_DIR.glob("*.html")):
        if path.name == "admin.html":
            continue
        timestamp = last_updated(path)
        dates[path.name] = {
            "timestamp": timestamp,
            "date": datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat(),
        }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(dates, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Generated dates for {len(dates)} visualization pages: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
