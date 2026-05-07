#!/usr/bin/env python3
"""Run a reproducible runtime generation smoke test through the Backoffice BFF."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an async generation request and store JSON/Markdown evidence."
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("AXIOMNODE_GATEWAY_URL", "https://axiomnode-gateway.amksandbox.cloud"),
        help="Public gateway or BFF base URL.",
    )
    parser.add_argument(
        "--service",
        default="microservice-quiz",
        choices=["microservice-quiz", "microservice-wordpass"],
        help="Backoffice service key to evaluate.",
    )
    parser.add_argument("--category-id", default="9", help="Generation category id.")
    parser.add_argument("--difficulty", type=int, default=50, help="Difficulty percentage.")
    parser.add_argument("--item-count", type=int, default=1, help="Requested item count.")
    parser.add_argument("--polls", type=int, default=6, help="Maximum polls after starting.")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between polls.")
    parser.add_argument(
        "--token",
        default=os.getenv("AXIOMNODE_EDGE_API_TOKEN"),
        help="Bearer token. Defaults to AXIOMNODE_EDGE_API_TOKEN.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("AXIOMNODE_EVAL_OUTPUT_DIR", "runtime-eval-output"),
        help="Directory where JSON and Markdown evidence will be written.",
    )
    return parser.parse_args()


def request_json(url: str, method: str, token: str | None, body: dict[str, Any] | None = None) -> tuple[int, Any]:
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, data=payload, headers=headers, method=method)
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else {}
    except HTTPError as error:
        raw = error.read().decode("utf-8")
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"message": raw}
        return error.code, payload
    except URLError as error:
        return 0, {"message": str(error.reason)}


def write_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = result["startedAt"].replace(":", "").replace("-", "")
    stem = f"runtime-generation-{result['service']}-{timestamp}"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"

    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


def extract_task_snapshot(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        nested_task = payload.get("task")
        if isinstance(nested_task, dict):
            return nested_task
        return payload
    return {}


def extract_task_id(payload: Any) -> str | None:
    snapshot = extract_task_snapshot(payload)
    task_id = snapshot.get("taskId")
    return task_id if isinstance(task_id, str) and task_id else None


def render_markdown(result: dict[str, Any]) -> str:
    final_snapshot = extract_task_snapshot(result.get("finalSnapshot"))
    rows = [
        "# Runtime generation evaluation",
        "",
        f"Date: {result['startedAt']}",
        f"Service: `{result['service']}`",
        f"Base URL: `{result['baseUrl']}`",
        "",
        "## Request",
        "",
        "```json",
        json.dumps(result["request"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Start response",
        "",
        f"HTTP status: `{result['startStatus']}`",
        "",
        "```json",
        json.dumps(result["startResponse"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Final snapshot",
        "",
        f"Status: `{final_snapshot.get('status', 'unknown')}`",
        f"Requested: `{final_snapshot.get('requested', 'unknown')}`",
        f"Processed: `{final_snapshot.get('processed', 'unknown')}`",
        f"Created: `{final_snapshot.get('created', 'unknown')}`",
        f"Failed: `{final_snapshot.get('failed', 'unknown')}`",
        f"Stalled: `{final_snapshot.get('stalled', 'unknown')}`",
        f"Idle seconds: `{final_snapshot.get('idleSeconds', 'unknown')}`",
        "",
        "```json",
        json.dumps(final_snapshot, indent=2, ensure_ascii=False),
        "```",
    ]
    return "\n".join(rows) + "\n"


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    endpoint_base = f"{base_url}/v1/backoffice/services/{args.service}/generation"
    request_body = {
        "categoryId": args.category_id,
        "difficultyPercentage": args.difficulty,
        "itemCount": args.item_count,
        "requestedBy": "backoffice",
    }
    started_at = datetime.now(timezone.utc).isoformat()

    start_status, start_response = request_json(
        f"{endpoint_base}/process",
        "POST",
        args.token,
        request_body,
    )
    task_id = extract_task_id(start_response)
    polls: list[dict[str, Any]] = []

    if task_id:
        for _ in range(args.polls):
            time.sleep(args.poll_interval)
            poll_status, poll_response = request_json(
                f"{endpoint_base}/process/{task_id}?includeItems=true",
                "GET",
                args.token,
            )
            polls.append({"statusCode": poll_status, "body": poll_response})
            if extract_task_snapshot(poll_response).get("status") in {"completed", "failed"}:
                break

    result = {
        "startedAt": started_at,
        "baseUrl": base_url,
        "service": args.service,
        "request": request_body,
        "startStatus": start_status,
        "startResponse": start_response,
        "polls": polls,
        "finalSnapshot": polls[-1]["body"] if polls else start_response,
    }
    write_outputs(Path(args.output_dir), result)
    return 0 if start_status in {200, 202} else 1


if __name__ == "__main__":
    sys.exit(main())