#!/usr/bin/env python3
import argparse
import os
import random
import time
from typing import List, Dict

import numpy as np
from pymilvus import connections, Collection

# Zilliz Cloud configuration (via environment variables)
ZILLIZ_URI = os.getenv("ZILLIZ_URI", "http://localhost:19530")
ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY", "")  # Empty for localhost, required for Zilliz Cloud

DEFAULT_COLLECTION = "user_profiles"
EMBED_FIELD = "embeddings"
PK_FIELD = "userId"


def connect(uri: str = None, api_key: str = None):
    uri = uri or ZILLIZ_URI
    api_key = api_key or ZILLIZ_API_KEY
    
    if api_key:
        # Zilliz Cloud connection
        connections.connect(
            alias="default",
            uri=uri,
            token=api_key
        )
        print(f"Connected to Zilliz Cloud at {uri}")
    else:
        # Localhost connection (fallback)
        connections.connect(alias="default", host="localhost", port="19530")
        print("Connected to local Milvus")


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    data = sorted(data)
    k = (len(data) - 1) * p
    f = int(k)
    c = min(f + 1, len(data) - 1)
    if f == c:
        return data[int(k)]
    d0 = data[f] * (c - k)
    d1 = data[c] * (k - f)
    return d0 + d1


def load_collection(name: str) -> Collection:
    coll = Collection(name)
    coll.load()
    return coll


def fetch_ground_truth(coll: Collection, n: int) -> List[Dict]:
    total = coll.num_entities
    take = min(n, total)
    print(f"Fetching {take} ground-truth vectors (of {total} total)...")
    rows = coll.query(expr="", output_fields=[PK_FIELD, EMBED_FIELD], limit=take)
    return rows


def search_one(coll: Collection, vec, topk: int, ef: int) -> List[Dict]:
    res = coll.search(
        data=[vec],
        anns_field=EMBED_FIELD,
        param={"metric_type": "COSINE", "params": {"ef": ef}},
        limit=topk,
        output_fields=[PK_FIELD],
    )
    return res[0]


def recall_test(coll: Collection, ground: List[Dict], topk: int, ef: int, sample: int) -> float:
    sample = min(sample, len(ground))
    if sample == 0:
        return 0.0
    picked = random.sample(ground, sample)
    hits = 0
    for row in picked:
        gt_id = row[PK_FIELD]
        res = search_one(coll, row[EMBED_FIELD], topk, ef)
        found = any(hit.id == gt_id or hit.entity.get(PK_FIELD) == gt_id for hit in res)
        if found:
            hits += 1
    return hits / sample


def latency_test(coll: Collection, ground: List[Dict], topk: int, ef: int, runs: int):
    times = []
    for _ in range(runs):
        row = random.choice(ground)
        vec = row[EMBED_FIELD]
        t0 = time.perf_counter()
        _ = search_one(coll, vec, topk, ef)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return {
        "p50_ms": percentile(times, 0.50),
        "p95_ms": percentile(times, 0.95),
        "p99_ms": percentile(times, 0.99),
        "runs": runs,
    }


def batch_throughput_test(
    coll: Collection,
    ground: List[Dict],
    topk: int,
    ef: int,
    batch_sizes: List[int],
    runs_per_size: int,
):
    results = []
    for bs in batch_sizes:
        total_queries = 0
        total_time = 0.0
        for _ in range(runs_per_size):
            vecs = [random.choice(ground)[EMBED_FIELD] for _ in range(bs)]
            t0 = time.perf_counter()
            _ = coll.search(
                data=vecs,
                anns_field=EMBED_FIELD,
                param={"metric_type": "COSINE", "params": {"ef": ef}},
                limit=topk,
                output_fields=[PK_FIELD],
            )
            t1 = time.perf_counter()
            total_time += (t1 - t0)
            total_queries += bs
        qps = total_queries / total_time if total_time > 0 else 0
        results.append(
            {
                "batch_size": bs,
                "runs": runs_per_size,
                "qps": qps,
                "avg_ms_per_batch": (total_time / runs_per_size) * 1000.0,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Query & performance tests on Milvus collection")
    parser.add_argument("--collection", "-c", default=DEFAULT_COLLECTION, help="Collection name")
    parser.add_argument("--uri", default=None, help="Zilliz Cloud URI (or use ZILLIZ_URI env var)")
    parser.add_argument("--api-key", default=None, help="Zilliz Cloud API key (or use ZILLIZ_API_KEY env var)")
    parser.add_argument("--topk", type=int, default=10, help="top_k for search")
    parser.add_argument("--ef", type=int, default=64, help="HNSW search ef")
    parser.add_argument("--ground", type=int, default=1000, help="number of ground-truth vectors to sample")
    parser.add_argument("--recall-sample", type=int, default=200, help="number of self-queries for recall")
    parser.add_argument("--latency-runs", type=int, default=200, help="single-query latency runs")
    parser.add_argument("--batch-runs", type=int, default=50, help="runs per batch size")
    parser.add_argument("--batches", type=str, default="32,64,128", help="comma-separated batch sizes")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batches.split(",") if x.strip()]

    connect(args.uri, args.api_key)
    coll = load_collection(args.collection)

    ground = fetch_ground_truth(coll, args.ground)
    if not ground:
        print("No data in collection; aborting.")
        return

    recall = recall_test(coll, ground, args.topk, args.ef, args.recall_sample)
    latency = latency_test(coll, ground, args.topk, args.ef, args.latency_runs)
    throughput = batch_throughput_test(coll, ground, args.topk, args.ef, batch_sizes, args.batch_runs)

    print("\n=== Results ===")
    print(f"Recall@{args.topk} (self-query): {recall:.4f} over {min(args.recall_sample, len(ground))} queries")
    print(
        f"Single-query latency (ms): p50={latency['p50_ms']:.2f}, "
        f"p95={latency['p95_ms']:.2f}, p99={latency['p99_ms']:.2f} "
        f"(runs={latency['runs']})"
    )
    for r in throughput:
        print(
            f"Batch size {r['batch_size']}: "
            f"QPS={r['qps']:.2f}, avg_ms_per_batch={r['avg_ms_per_batch']:.2f} "
            f"(runs={r['runs']})"
        )


if __name__ == "__main__":
    main()

