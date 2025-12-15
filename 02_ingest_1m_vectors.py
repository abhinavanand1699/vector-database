#!/usr/bin/env python3
import argparse
import time
import random
from typing import List, Dict

import numpy as np
from pymilvus import connections, Collection

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "user_profiles"
EMBED_DIM = 768


def connect():
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("Connected to Milvus")


def generate_records(count: int, start_id: int = 0) -> List[Dict]:
    records = []
    for i in range(count):
        uid = start_id + i
        records.append(
            {
                "userId": uid,
                "height": random.uniform(150, 200),
                "latitude": 37.7 + random.uniform(-0.5, 0.5),
                "longitude": -122.4 + random.uniform(-0.5, 0.5),
                "dob": random.randint(1970, 2005) * 10_000 + random.randint(101, 1231),
                "createdAt": int(time.time() * 1000),
                "updatedAt": int(time.time() * 1000),
                "openToCatchFeelings": random.choice([True, False]),
                "datingPause": random.choice([True, False]),
                "singleImageExists": random.choice([True, False]),
                "gender": random.choice(["male", "female", "non-binary"]),
                "sexualOrientation": random.choice(["straight", "gay", "bisexual"]),
                "purpose": random.choice(["relationship", "casual", "friendship"]),
                "ethnicity": random.choice(["asian", "white", "black", "hispanic", "mixed", "other"]),
                "drinkStatus": random.choice(["never", "social", "regular"]),
                "smokeStatus": random.choice(["never", "social", "regular"]),
                "weedStatus": random.choice(["never", "social", "regular"]),
                "embeddings": np.random.rand(EMBED_DIM).astype("float32").tolist(),
            }
        )
    return records


def ingest(collection_name: str, total: int, batch_size: int):
    coll = Collection(collection_name)
    print(f"Using collection: {collection_name}")
    start_time = time.time()
    inserted = 0
    batch_num = 0

    while inserted < total:
        take = min(batch_size, total - inserted)
        batch = generate_records(take, start_id=inserted)
        batch_start = time.time()
        res = coll.insert(batch)
        batch_elapsed = time.time() - batch_start
        inserted += take
        batch_num += 1
        if batch_num % 10 == 0 or inserted == total:
            print(f"Batch {batch_num}: +{take} (total {inserted}/{total}), time {batch_elapsed:.2f}s")

    coll.flush()
    elapsed = time.time() - start_time
    print("\n=== Ingest Summary ===")
    print(f"Total vectors inserted: {inserted}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Throughput: {inserted/elapsed:.2f} vectors/sec")
    if res:
        print(f"Last insert IDs (count): {len(res.primary_keys)}")


def main():
    parser = argparse.ArgumentParser(description="Bulk ingest vectors into Milvus")
    parser.add_argument("--collection", "-c", type=str, default=COLLECTION_NAME, help="Collection name")
    parser.add_argument("--total", "-t", type=int, default=1_000_000, help="Total vectors to insert")
    parser.add_argument("--batch", "-b", type=int, default=10_000, help="Batch size per insert")
    args = parser.parse_args()

    connect()
    ingest(args.collection, args.total, args.batch)


if __name__ == "__main__":
    main()

