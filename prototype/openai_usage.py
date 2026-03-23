#!/usr/bin/env python3
"""
OpenAI usage summary by model for a given project + API key name (or API key ID) + period.

What it prints:
- model
- number of requests
- estimated total cost (USD)
- estimated average cost per request (USD)
- token breakdown

Requirements:
    pip install requests

Examples:
python openai_usage.py \
    --admin-key "REDACTED_OPENAI_ADMIN_KEY" \
    --project-id proj_5WCNtDZa3wCoARByJJib2DD7 \
    --start 2026-03-19 \
    --end 2026-03-24 \
    --debug

Notes:
- This uses organization/project management + usage endpoints, which require an ADMIN API key.
- --api-key-name is the key name shown in the UI, e.g. "caching2".
- --api-key-id is the internal API key ID, e.g. "key_...".
- Average price is estimated from usage tokens × the pricing table below.
- Update MODEL_PRICING_PER_1M when OpenAI pricing changes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

import requests


# Project-specific defaults — set these env vars to avoid passing them every run:
#   export OPENAI_ADMIN_KEY="sk-admin-..."
#   export OPENAI_PROJECT_ID="proj_..."
#   export OPENAI_API_KEY_NAME="your-key-name"
_DEFAULT_ADMIN_KEY    = os.environ.get("OPENAI_ADMIN_KEY")
_DEFAULT_PROJECT_ID   = os.environ.get("OPENAI_PROJECT_ID")
_DEFAULT_API_KEY_NAME = os.environ.get("OPENAI_API_KEY_NAME")


BASE_URL = "https://api.openai.com/v1"

# USD per 1M tokens
# Fill in or update the models you actually use.
MODEL_PRICING_PER_1M: Dict[str, Dict[str, float]] = {
    "gpt-5.4": {
        "input": 2.50,
        "cached_input": 0.25,
        "output": 15.00,
    },
    "gpt-5.4-mini": {
        "input": 0.750,
        "cached_input": 0.075,
        "output": 4.500,
    },
    "gpt-5.4-nano": {
        "input": 0.20,
        "cached_input": 0.02,
        "output": 1.25,
    },
    "o3": {
        "input": 2.00,
        "cached_input": 0.50,
        "output": 8.00,
    },
    "o3-mini": {
        "input": 1.10,
        "cached_input": 0.55,
        "output": 4.40,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show model-wise request counts and estimated average cost from OpenAI usage."
    )
    parser.add_argument(
        "--admin-key",
        default=_DEFAULT_ADMIN_KEY,
        required=_DEFAULT_ADMIN_KEY is None,
        help="OpenAI ADMIN API key. Falls back to $OPENAI_ADMIN_KEY.",
    )
    parser.add_argument(
        "--project-id",
        default=_DEFAULT_PROJECT_ID,
        required=_DEFAULT_PROJECT_ID is None,
        help="Project ID, e.g. proj_... Falls back to $OPENAI_PROJECT_ID.",
    )
    parser.add_argument(
        "--api-key-id",
        default=None,
        help="Internal API key ID, e.g. key_...",
    )
    parser.add_argument(
        "--api-key-name",
        default=_DEFAULT_API_KEY_NAME,
        help="API key name shown in the UI. Falls back to $OPENAI_API_KEY_NAME.",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date inclusive in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date exclusive in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--bucket-width",
        choices=["1m", "1h", "1d"],
        default="1d",
        help="Usage API bucket width. Default: 1d",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Optional model filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout seconds.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of a text table.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw API URLs and responses for debugging.",
    )
    return parser.parse_args()


def date_to_unix_utc(date_str: str) -> int:
    try:
        d = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as e:
        raise SystemExit(f"Invalid date '{date_str}'. Use YYYY-MM-DD.") from e

    return int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp())


def safe_get(dct: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = dct.get(key, default)
    if value is None:
        return default
    return float(value)


def request_get(url: str, admin_key: str, timeout: int, params: Optional[Dict[str, Any]] = None, debug: bool = False) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {admin_key}",
        "Content-Type": "application/json",
    }
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    if debug:
        print(f"[debug] GET {resp.url}", file=sys.stderr)
        print(f"[debug] status: {resp.status_code}", file=sys.stderr)
        print(f"[debug] response: {resp.text[:2000]}", file=sys.stderr)
    if resp.status_code >= 400:
        raise RuntimeError(f"GET {url} failed ({resp.status_code}): {resp.text}")
    return resp.json()


def list_project_api_keys(admin_key: str, project_id: str, timeout: int) -> Dict[str, Any]:
    """
    List API keys in a project so we can resolve UI-visible key names to internal key IDs.
    """
    url = f"{BASE_URL}/organization/projects/{project_id}/api_keys"
    return request_get(url=url, admin_key=admin_key, timeout=timeout)


def resolve_api_key_id(
    admin_key: str,
    project_id: str,
    timeout: int,
    api_key_id: Optional[str] = None,
    api_key_name: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve either:
    - a directly provided api_key_id, or
    - a UI-visible api_key_name -> internal api_key_id
    """
    if api_key_id:
        return api_key_id

    if not api_key_name:
        return None

    payload = list_project_api_keys(admin_key=admin_key, project_id=project_id, timeout=timeout)
    data = payload.get("data", [])

    matches = [item for item in data if item.get("name") == api_key_name]

    if not matches:
        available = [item.get("name", "<unnamed>") for item in data]
        raise RuntimeError(
            f"Could not find API key named '{api_key_name}' in project '{project_id}'. "
            f"Available names: {available}"
        )

    if len(matches) > 1:
        ids = [item.get("id") for item in matches]
        raise RuntimeError(
            f"Multiple API keys named '{api_key_name}' found in project '{project_id}'. "
            f"Use --api-key-id explicitly. Matching IDs: {ids}"
        )

    return matches[0]["id"]


def fetch_usage_page(
    admin_key: str,
    start_time: int,
    end_time: int,
    bucket_width: str,
    project_id: str,
    api_key_id: Optional[str],
    models: Optional[list[str]],
    page: Optional[str],
    timeout: int,
    debug: bool = False,
) -> Dict[str, Any]:
    url = f"{BASE_URL}/organization/usage/completions"

    params: Dict[str, Any] = {
        "start_time": start_time,
        "end_time": end_time,
        "bucket_width": bucket_width,
        "group_by": ["model"],
        "project_ids": [project_id],
    }

    if api_key_id:
        params["api_key_ids"] = [api_key_id]
    if models:
        params["models"] = models
    if page:
        params["page"] = page

    return request_get(url=url, admin_key=admin_key, timeout=timeout, params=params, debug=debug)


def iter_all_usage_results(
    admin_key: str,
    start_time: int,
    end_time: int,
    bucket_width: str,
    project_id: str,
    api_key_id: Optional[str],
    models: Optional[list[str]],
    timeout: int,
    debug: bool = False,
) -> Iterable[Dict[str, Any]]:
    page: Optional[str] = None

    while True:
        payload = fetch_usage_page(
            admin_key=admin_key,
            start_time=start_time,
            end_time=end_time,
            bucket_width=bucket_width,
            project_id=project_id,
            api_key_id=api_key_id,
            models=models,
            page=page,
            timeout=timeout,
            debug=debug,
        )

        for bucket in payload.get("data", []):
            for result in bucket.get("results", []):
                yield result

        if not payload.get("has_more"):
            break

        page = payload.get("next_page")
        if not page:
            break


def normalize_model_name(model: str) -> str:
    """Strip trailing date suffix like -2025-01-31 from model names."""
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)


def estimate_cost_usd(model: str, usage: Dict[str, float]) -> Optional[float]:
    price = MODEL_PRICING_PER_1M.get(model) or MODEL_PRICING_PER_1M.get(normalize_model_name(model))
    if not price:
        return None

    input_tokens = usage.get("input_tokens", 0.0)
    cached_input_tokens = usage.get("input_cached_tokens", 0.0)
    uncached_input_tokens = max(0.0, input_tokens - cached_input_tokens)

    output_tokens = usage.get("output_tokens", 0.0)

    input_audio_tokens = usage.get("input_audio_tokens", 0.0)
    input_audio_cached_tokens = usage.get("input_audio_cached_tokens", 0.0)
    uncached_input_audio_tokens = max(0.0, input_audio_tokens - input_audio_cached_tokens)

    output_audio_tokens = usage.get("output_audio_tokens", 0.0)

    total = 0.0

    total += (uncached_input_tokens / 1_000_000.0) * price.get("input", 0.0)
    total += (cached_input_tokens / 1_000_000.0) * price.get("cached_input", price.get("input", 0.0))
    total += (output_tokens / 1_000_000.0) * price.get("output", 0.0)

    total += (uncached_input_audio_tokens / 1_000_000.0) * price.get("audio_input", 0.0)
    total += (input_audio_cached_tokens / 1_000_000.0) * price.get("audio_cached_input", price.get("audio_input", 0.0))
    total += (output_audio_tokens / 1_000_000.0) * price.get("audio_output", 0.0)

    return total


def aggregate_results(results: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    agg: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for r in results:
        model = r.get("model") or "UNKNOWN"

        agg[model]["requests"] += safe_get(r, "num_model_requests")
        agg[model]["input_tokens"] += safe_get(r, "input_tokens")
        agg[model]["input_cached_tokens"] += safe_get(r, "input_cached_tokens")
        agg[model]["output_tokens"] += safe_get(r, "output_tokens")
        agg[model]["input_audio_tokens"] += safe_get(r, "input_audio_tokens")
        agg[model]["input_audio_cached_tokens"] += safe_get(r, "input_audio_cached_tokens")
        agg[model]["output_audio_tokens"] += safe_get(r, "output_audio_tokens")

    for model, stats in agg.items():
        estimated_total = estimate_cost_usd(model, stats)
        stats["estimated_total_cost_usd"] = float("nan") if estimated_total is None else estimated_total

        req = stats["requests"]
        if estimated_total is None or req <= 0:
            stats["estimated_avg_cost_per_request_usd"] = float("nan")
        else:
            stats["estimated_avg_cost_per_request_usd"] = estimated_total / req

    return agg


def fmt_money(x: float) -> str:
    if math.isnan(x):
        return "N/A"
    return f"${x:,.6f}"


def fmt_intish(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x)):,}"
    return f"{x:,.2f}"


def print_table(agg: Dict[str, Dict[str, float]]) -> None:
    rows = []
    for model, s in sorted(agg.items(), key=lambda kv: (-kv[1]["requests"], kv[0])):
        rows.append([
            model,
            fmt_intish(s["requests"]),
            fmt_money(s["estimated_avg_cost_per_request_usd"]),
            fmt_money(s["estimated_total_cost_usd"]),
            fmt_intish(s["input_tokens"]),
            fmt_intish(s["input_cached_tokens"]),
            fmt_intish(s["output_tokens"]),
            fmt_intish(s["input_audio_tokens"]),
            fmt_intish(s["input_audio_cached_tokens"]),
            fmt_intish(s["output_audio_tokens"]),
        ])

    headers = [
        "model",
        "requests",
        "avg_cost/req",
        "total_cost",
        "input_tok",
        "cached_in",
        "output_tok",
        "audio_in",
        "audio_cached_in",
        "audio_out",
    ]

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render_row(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(render_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(render_row(row))


def main() -> None:
    args = parse_args()

    if not args.api_key_id and not args.api_key_name:
        print("[info] No API key filter — showing all keys in the project.", file=sys.stderr)

    start_time = date_to_unix_utc(args.start)
    end_time = date_to_unix_utc(args.end)

    if end_time <= start_time:
        raise SystemExit("--end must be later than --start.")

    resolved_api_key_id = resolve_api_key_id(
        admin_key=args.admin_key,
        project_id=args.project_id,
        timeout=args.timeout,
        api_key_id=args.api_key_id,
        api_key_name=args.api_key_name,
    )

    if not resolved_api_key_id and (args.api_key_id or args.api_key_name):
        raise SystemExit("Failed to resolve API key ID.")

    print(f"[info] project_id       : {args.project_id}")
    print(f"[info] resolved_key_id : {resolved_api_key_id}")
    if args.api_key_name:
        print(f"[info] api_key_name    : {args.api_key_name}")
    print(f"[info] period          : [{args.start}, {args.end}) UTC")
    print()

    results = list(
        iter_all_usage_results(
            admin_key=args.admin_key,
            start_time=start_time,
            end_time=end_time,
            bucket_width=args.bucket_width,
            project_id=args.project_id,
            api_key_id=resolved_api_key_id,
            models=args.model,
            timeout=args.timeout,
            debug=args.debug,
        )
    )

    agg = aggregate_results(results)

    if args.json:
        out = {
            "project_id": args.project_id,
            "api_key_id": resolved_api_key_id,
            "api_key_name": args.api_key_name,
            "start": args.start,
            "end": args.end,
            "models": {},
        }

        for model, s in agg.items():
            out["models"][model] = {
                "requests": int(round(s["requests"])),
                "estimated_total_cost_usd": None if math.isnan(s["estimated_total_cost_usd"]) else s["estimated_total_cost_usd"],
                "estimated_avg_cost_per_request_usd": None if math.isnan(s["estimated_avg_cost_per_request_usd"]) else s["estimated_avg_cost_per_request_usd"],
                "input_tokens": int(round(s["input_tokens"])),
                "input_cached_tokens": int(round(s["input_cached_tokens"])),
                "output_tokens": int(round(s["output_tokens"])),
                "input_audio_tokens": int(round(s["input_audio_tokens"])),
                "input_audio_cached_tokens": int(round(s["input_audio_cached_tokens"])),
                "output_audio_tokens": int(round(s["output_audio_tokens"])),
            }

        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        if not agg:
            print("No usage found for the given project / API key / period.")
            return

        print_table(agg)

        missing_models = [m for m in agg.keys() if normalize_model_name(m) not in MODEL_PRICING_PER_1M and m not in MODEL_PRICING_PER_1M]
        if missing_models:
            print("\n[warning] Pricing is not configured for these models, so cost fields are N/A:")
            for m in missing_models:
                print(f"  - {m}")
            print("Add them to MODEL_PRICING_PER_1M at the top of the script.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)