#!/usr/bin/env python3
"""Archive a CI run's raw output into docs/runs/ -- see docs/runs/README.md.

Takes the JSON file the Claude Code harness saves when
``mcp__github__get_job_logs`` (called with ``return_content: true``) exceeds
the inline token limit: ``{"job_id": ..., "logs_content": "<timestamped log
text, GitHub's escaping>", ...}``. Strips the per-line ``<timestamp>Z ``
prefix GitHub Actions adds to every log line, and -- if the log contains a
``print(json.dumps(result, indent=2))`` block (as
``scripts/train_neural_grounding.py``'s battery step does) -- extracts and
validates that block as its own ``battery.json``.

This exists so a run's raw output can be committed byte-verifiable rather
than hand-copied into a chat message or a doc, where a paraphrase can drop a
field or round a number with no way for the reader to check it against the
source (see docs/runs/README.md).

Usage
-----
    python3 scripts/archive_ci_run.py --job-log <path> --out docs/runs/run-14
    python3 scripts/archive_ci_run.py --job-log <path> --out docs/runs/regime-probe-4
"""

from __future__ import annotations

import argparse
import json
import os
import re


_TS = re.compile(r"(?=\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z )")
_TS_LINE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z (.*)$", re.S)


def load_clean_lines(job_log_path: str) -> list[str]:
    with open(job_log_path, encoding="utf-8", errors="replace") as f:
        wrapper = json.load(f)
    content = wrapper["logs_content"]
    lines = []
    for part in _TS.split(content):
        part = part.rstrip("\n")
        m = _TS_LINE.match(part)
        if m:
            lines.append(m.group(1))
    return lines


def extract_json_block(lines: list[str]) -> tuple[dict, str] | None:
    """Find and validate a top-level print(json.dumps(...)) block, if any."""
    starts = [i for i, l in enumerate(lines) if l.strip() == "{"]
    for start in starts:
        depth, buf = 0, []
        for i in range(start, len(lines)):
            buf.append(lines[i])
            depth += lines[i].count("{") - lines[i].count("}")
            if depth == 0:
                text = "\n".join(buf)
                try:
                    return json.loads(text), text
                except json.JSONDecodeError:
                    break  # not a clean top-level block; try the next '{'
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job-log", required=True,
                     help="path to a saved mcp__github__get_job_logs result JSON")
    ap.add_argument("--out", required=True,
                     help="output dir, e.g. docs/runs/run-14")
    args = ap.parse_args()

    lines = load_clean_lines(args.job_log)
    os.makedirs(args.out, exist_ok=True)

    full_log_path = os.path.join(args.out, "full_log.txt")
    with open(full_log_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {full_log_path} ({len(lines)} lines)")

    found = extract_json_block(lines)
    if found:
        data, _ = found
        battery_path = os.path.join(args.out, "battery.json")
        with open(battery_path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        print(f"wrote {battery_path} (top-level keys: {list(data.keys())})")
    else:
        print("no print(json.dumps(...)) block found -- "
              "full_log.txt only (expected for regime-decoding-probe runs)")


if __name__ == "__main__":
    main()
