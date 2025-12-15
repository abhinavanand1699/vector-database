#!/usr/bin/env python3
import argparse
import os
import time
import random
from typing import List, Dict

import numpy as np
from dotenv import load_dotenv
from pymilvus import connections, Collection, db, utility

# Load environment variables from .env file
load_dotenv()

# Zilliz Cloud configuration (via environment variables or .env file)
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY")
DATABASE_NAME = os.getenv("DATABASE_NAME", "user_profile_db")  # Default: user_profile_db

COLLECTION_NAME = "user_profiles"
EMBED_DIM = 768


def connect():
    if not ZILLIZ_URI or not ZILLIZ_API_KEY:
        raise ValueError(
            "ZILLIZ_URI and ZILLIZ_API_KEY are required. "
            "Please set them in a .env file or as environment variables.\n"
            "Create a .env file with:\n"
            "  ZILLIZ_URI=https://in01-d180f5391055b77.aws-us-west-2.vectordb.zillizcloud.com:19541\n"
            "  ZILLIZ_API_KEY=your_api_key_here\n"
            "  DATABASE_NAME=user_profile_db"
        )
    
    # Zilliz Cloud connection
    connections.connect(
        alias="default",
        uri=ZILLIZ_URI,
        token=ZILLIZ_API_KEY
    )
    print(f"Connected to Zilliz Cloud at {ZILLIZ_URI}")
    
    # Switch to the correct database
    db.using_database(DATABASE_NAME)
    print(f"Using database: {DATABASE_NAME}")


def generate_records(count: int, start_id: int = 0) -> List[Dict]:
    """
    Generate mock user profile records matching the schema.
    All FLOAT fields are converted to float32 to match schema requirements.
    """
    records = []
    for i in range(count):
        uid = start_id + i
        records.append(
            {
                "userId": uid,
                # FLOAT fields: convert to float32 to match schema (FLOAT = 32-bit)
                "height": np.float32(random.uniform(150, 200)),
                "latitude": np.float32(37.7 + random.uniform(-0.5, 0.5)),
                "longitude": np.float32(-122.4 + random.uniform(-0.5, 0.5)),
                # INT64 fields: timestamps and dates
                "dob": random.randint(1970, 2005) * 10_000 + random.randint(101, 1231),
                "createdAt": int(time.time() * 1000),
                "updatedAt": int(time.time() * 1000),
                # BOOL fields
                "openToCatchFeelings": random.choice([True, False]),
                "datingPause": random.choice([True, False]),
                "singleImageExists": random.choice([True, False]),
                # VARCHAR fields: string values
                "gender": random.choice(["male", "female", "non-binary"]),
                "sexualOrientation": random.choice(["straight", "gay", "bisexual"]),
                "purpose": random.choice(["relationship", "casual", "friendship"]),
                "ethnicity": random.choice(["asian", "white", "black", "hispanic", "mixed", "other"]),
                "drinkStatus": random.choice(["never", "social", "regular"]),
                "smokeStatus": random.choice(["never", "social", "regular"]),
                "weedStatus": random.choice(["never", "social", "regular"]),
                # Vector field: 768-dimensional float32 vector
                "embeddings": np.random.rand(EMBED_DIM).astype(np.float32).tolist(),
            }
        )
    return records


def ingest(collection_name: str, total: int, batch_size: int):
    """
    Ingest records into the collection in batches.
    
    Args:
        collection_name: Name of the collection to insert into
        total: Total number of records to insert
        batch_size: Number of records per batch
    """
    # Check if collection exists
    if not utility.has_collection(collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist. "
                        f"Please run 01_create_user_collection.py first to create it.")
    
    coll = Collection(collection_name)
    print(f"Using collection: {collection_name}")
    
    # Check if collection has indexes (for informational purposes)
    # Note: Inserts work without indexes, but indexes are needed for efficient queries
    indexes = utility.list_indexes(collection_name)
    if not indexes:
        print("Warning: Collection has no indexes. Consider creating indexes before querying.")
    else:
        print(f"Collection has {len(indexes)} index(es): {', '.join(indexes[:5])}{'...' if len(indexes) > 5 else ''}")
    
    start_time = time.time()
    inserted = 0
    batch_num = 0

    print(f"\nStarting ingestion: {total:,} records in batches of {batch_size:,}")
    print("-" * 60)
    
    while inserted < total:
        take = min(batch_size, total - inserted)
        batch = generate_records(take, start_id=inserted)
        batch_start = time.time()
        
        try:
            res = coll.insert(batch)
            batch_elapsed = time.time() - batch_start
            inserted += take
            batch_num += 1
            
            # Print progress every 10 batches or on completion
            if batch_num % 10 == 0 or inserted == total:
                rate = take / batch_elapsed if batch_elapsed > 0 else 0
                print(f"Batch {batch_num}: +{take:,} records (total {inserted:,}/{total:,}), "
                      f"time {batch_elapsed:.2f}s, rate {rate:.0f} records/sec")
        except Exception as e:
            print(f"\nError inserting batch {batch_num}: {e}")
            raise

    # Flush to ensure all data is persisted
    print("\nFlushing collection to ensure all data is persisted...")
    coll.flush()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("=== Ingest Summary ===")
    print("=" * 60)
    print(f"Total records inserted: {inserted:,}")
    print(f"Total time: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
    print(f"Average throughput: {inserted/elapsed:.2f} records/sec")
    if res:
        print(f"Last insert IDs count: {len(res.primary_keys)}")
    
    # Get collection stats
    coll.load()  # Load collection to get accurate stats
    stats = coll.num_entities
    print(f"Collection now contains: {stats:,} total entities")


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

