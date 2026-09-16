from __future__ import annotations

import argparse
import json
from urllib.request import urlopen

from recoflow.artifacts import load_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify one live personalized API request")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--user-artifact", default="artifacts/user_embeddings.npz")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    user_ids, _ = load_embeddings(args.user_artifact)
    if not user_ids:
        raise RuntimeError("user embedding artifact is empty")
    user_id = user_ids[0]
    url = f"{args.base_url}/v1/recommendations/{user_id}?limit={args.limit}"
    with urlopen(url, timeout=10) as response:
        payload = json.load(response)
    if payload.get("source") != "two_tower":
        raise RuntimeError(f"expected personalized retrieval, received: {payload}")
    print(json.dumps(payload, indent=2))
