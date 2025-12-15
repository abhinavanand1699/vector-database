#!/usr/bin/env python3
"""
API Operations Script for Zilliz Cloud Milvus

Provides three main operations:
1. print_collection_info() - Display database/collection details
2. upsert_entities() - Insert or update entities
3. hybrid_search() - Perform hybrid vector + scalar search
"""

import argparse
import json
import os
import time
from typing import List, Dict, Optional

import numpy as np
from dotenv import load_dotenv
from pymilvus import (
    connections,
    Collection,
    utility,
    db,
)

# Load environment variables from .env file
load_dotenv()

# Zilliz Cloud configuration (via environment variables or .env file)
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY")
DATABASE_NAME = os.getenv("DATABASE_NAME", "user_profile_db")  # Default: user_profile_db

COLLECTION_NAME = "user_profiles"
EMBED_DIM = 768
EMBED_FIELD = "embeddings"
PK_FIELD = "userId"

# FLOAT fields that need np.float32 conversion
FLOAT_FIELDS = ["height", "latitude", "longitude"]


def connect():
    """Connect to Zilliz Cloud."""
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


def print_collection_info(collection_name: str, db_name: str = None):
    """
    Display comprehensive collection information.
    
    Args:
        collection_name: Name of the collection
        db_name: Optional database name (uses current if not provided)
    """
    # Switch to database if provided
    if db_name:
        db.using_database(db_name)
        print(f"Using database: {db_name}")
    
    # Get current database
    current_db = db.list_database()
    active_db = db_name if db_name else (current_db[0] if current_db else "default")
    
    # Check if collection exists
    if not utility.has_collection(collection_name):
        print(f"Error: Collection '{collection_name}' does not exist.")
        return
    
    coll = Collection(collection_name)
    
    print("\n" + "=" * 80)
    print("COLLECTION INFORMATION")
    print("=" * 80)
    
    # Database and Collection Name
    print(f"\n📊 Database Name: {active_db}")
    print(f"📁 Collection Name: {collection_name}")
    
    # Number of Entities
    try:
        num_entities = coll.num_entities
        print(f"📈 Number of Entities: {num_entities:,}")
    except Exception as e:
        print(f"📈 Number of Entities: Unable to retrieve ({e})")
    
    # Collection Properties
    print(f"\n⚙️  Collection Properties:")
    print(f"   - Shards: {coll.shards_num if hasattr(coll, 'shards_num') else 'N/A'}")
    print(f"   - Consistency Level: {coll.consistency_level if hasattr(coll, 'consistency_level') else 'N/A'}")
    
    # Schema Information
    print(f"\n📋 Schema Fields ({len(coll.schema.fields)} fields):")
    print("-" * 80)
    for field in coll.schema.fields:
        field_type = field.dtype.name
        is_primary = " (PRIMARY KEY)" if field.is_primary else ""
        auto_id = " (AUTO_ID)" if field.auto_id else ""
        
        # Format field info
        info_parts = [f"   • {field.name}: {field_type}"]
        
        if field_type == "VARCHAR":
            info_parts.append(f"max_length={field.max_length}")
        elif field_type == "FLOAT_VECTOR":
            info_parts.append(f"dim={field.dim}")
        
        if hasattr(field, 'default_value') and field.default_value is not None:
            default_val = field.default_value
            if hasattr(default_val, 'string_data'):
                info_parts.append(f"default='{default_val.string_data}'")
            elif hasattr(default_val, 'float_data'):
                info_parts.append(f"default={default_val.float_data}")
            elif hasattr(default_val, 'int_data'):
                info_parts.append(f"default={default_val.int_data}")
            elif hasattr(default_val, 'bool_data'):
                info_parts.append(f"default={default_val.bool_data}")
            else:
                info_parts.append(f"default={default_val}")
        
        if hasattr(field, 'description') and field.description:
            info_parts.append(f"\n     Description: {field.description}")
        
        print(" ".join(info_parts) + is_primary + auto_id)
    
    # Index Information
    print(f"\n🔍 Indexes:")
    print("-" * 80)
    indexes = utility.list_indexes(collection_name)
    if indexes:
        # Get index details from collection.indexes
        index_map = {idx.index_name: idx for idx in coll.indexes}
        
        for idx_name in indexes:
            try:
                if idx_name in index_map:
                    idx = index_map[idx_name]
                    idx_type = idx.params.get('index_type', 'UNKNOWN')
                    metric_type = idx.params.get('metric_type', '')
                    params = idx.params.get('params', {})
                    field_name = idx.params.get('field_name', idx_name)
                    
                    print(f"   • {field_name} ({idx_name}):")
                    print(f"     - Type: {idx_type}")
                    if metric_type:
                        print(f"     - Metric: {metric_type}")
                    if params:
                        print(f"     - Parameters: {params}")
                else:
                    print(f"   • {idx_name}: Index exists but details unavailable")
            except Exception as e:
                print(f"   • {idx_name}: Unable to get details ({e})")
    else:
        print("   No indexes found")
    
    print("\n" + "=" * 80)


def prepare_entities(entities: List[Dict]) -> List[Dict]:
    """
    Prepare entities for upsert by converting FLOAT fields to float32.
    
    Args:
        entities: List of entity dictionaries
        
    Returns:
        Prepared entities with proper data types
    """
    prepared = []
    for entity in entities:
        prepared_entity = entity.copy()
        
        # Convert FLOAT fields to np.float32
        for field in FLOAT_FIELDS:
            if field in prepared_entity and isinstance(prepared_entity[field], (float, int)):
                prepared_entity[field] = np.float32(prepared_entity[field])
        
        # Ensure embeddings are float32
        if EMBED_FIELD in prepared_entity:
            if isinstance(prepared_entity[EMBED_FIELD], list):
                prepared_entity[EMBED_FIELD] = np.array(prepared_entity[EMBED_FIELD], dtype=np.float32).tolist()
            elif isinstance(prepared_entity[EMBED_FIELD], np.ndarray):
                prepared_entity[EMBED_FIELD] = prepared_entity[EMBED_FIELD].astype(np.float32).tolist()
        
        prepared.append(prepared_entity)
    
    return prepared


def upsert_entities(collection_name: str, entities: List[Dict], db_name: str = None, batch_size: int = 1000):
    """
    Insert or update entities in the collection.
    
    Args:
        collection_name: Name of collection
        entities: List of dictionaries matching schema fields
        db_name: Optional database name (uses current if not provided)
        batch_size: Number of entities to upsert per batch
        
    Returns:
        Dictionary with upsert results (ids, count, etc.)
    """
    # Switch to database if provided
    if db_name:
        db.using_database(db_name)
        print(f"Using database: {db_name}")
    
    # Check if collection exists
    if not utility.has_collection(collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist.")
    
    coll = Collection(collection_name)
    
    # Validate required fields
    if not entities:
        raise ValueError("No entities provided for upsert")
    
    required_fields = [PK_FIELD, EMBED_FIELD]
    for i, entity in enumerate(entities[:1]):  # Check first entity as sample
        missing = [f for f in required_fields if f not in entity]
        if missing:
            raise ValueError(f"Entity {i} missing required fields: {missing}")
    
    # Prepare entities (convert data types)
    print(f"Preparing {len(entities)} entities for upsert...")
    prepared_entities = prepare_entities(entities)
    
    # Upsert in batches
    total = len(prepared_entities)
    upserted = 0
    start_time = time.time()
    
    print(f"\nStarting upsert: {total:,} entities in batches of {batch_size:,}")
    print("-" * 80)
    
    results = []
    for i in range(0, total, batch_size):
        batch = prepared_entities[i:i + batch_size]
        batch_start = time.time()
        
        try:
            res = coll.upsert(batch)
            batch_elapsed = time.time() - batch_start
            upserted += len(batch)
            batch_num = (i // batch_size) + 1
            
            if batch_num % 10 == 0 or upserted == total:
                rate = len(batch) / batch_elapsed if batch_elapsed > 0 else 0
                print(f"Batch {batch_num}: +{len(batch):,} entities (total {upserted:,}/{total:,}), "
                      f"time {batch_elapsed:.2f}s, rate {rate:.0f} entities/sec")
            
            results.append(res)
        except Exception as e:
            print(f"\nError upserting batch {batch_num}: {e}")
            raise
    
    # Flush to ensure persistence
    print("\nFlushing collection to ensure all data is persisted...")
    coll.flush()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("=== Upsert Summary ===")
    print("=" * 80)
    print(f"Total entities upserted: {upserted:,}")
    print(f"Total time: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
    print(f"Average throughput: {upserted/elapsed:.2f} entities/sec")
    
    # Get updated entity count
    try:
        coll.load()
        new_count = coll.num_entities
        print(f"Collection now contains: {new_count:,} total entities")
    except Exception as e:
        print(f"Note: Unable to get updated entity count ({e})")
    
    return {
        "upserted_count": upserted,
        "total_time": elapsed,
        "throughput": upserted / elapsed if elapsed > 0 else 0,
        "results": results
    }


def hybrid_search(
    collection_name: str,
    query_vector: List[float],
    filter_expr: str = None,
    top_k: int = 10,
    ef: int = 100,
    output_fields: List[str] = None,
    db_name: str = None
) -> List[Dict]:
    """
    Perform hybrid search combining vector similarity and scalar filters.
    
    Args:
        collection_name: Collection to search
        query_vector: 768-dimensional query vector (will be converted to float32)
        filter_expr: Optional filter expression (e.g., "gender == 'male' AND height > 170")
        top_k: Number of results to return
        ef: HNSW search parameter (effort)
        output_fields: Fields to return (default: all fields)
        db_name: Optional database name
        
    Returns:
        List of search results with scores and distances
    """
    # Switch to database if provided
    if db_name:
        db.using_database(db_name)
        print(f"Using database: {db_name}")
    
    # Check if collection exists
    if not utility.has_collection(collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist.")
    
    coll = Collection(collection_name)
    
    # Check if collection has indexes
    indexes = utility.list_indexes(collection_name)
    if not indexes:
        print("Warning: Collection has no indexes. Loading collection...")
    
    # Ensure collection is loaded
    coll.load()
    
    # Convert query vector to float32
    query_vec = np.array(query_vector, dtype=np.float32)
    if len(query_vec) != EMBED_DIM:
        raise ValueError(f"Query vector dimension ({len(query_vec)}) does not match embedding dimension ({EMBED_DIM})")
    
    # Prepare search parameters
    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": ef}
    }
    
    # Perform search
    print(f"\nPerforming hybrid search...")
    print(f"  - Top K: {top_k}")
    print(f"  - EF: {ef}")
    if filter_expr:
        print(f"  - Filter: {filter_expr}")
    if output_fields:
        print(f"  - Output fields: {', '.join(output_fields)}")
    
    start_time = time.time()
    
    search_results = coll.search(
        data=[query_vec.tolist()],
        anns_field=EMBED_FIELD,
        param=search_params,
        limit=top_k,
        expr=filter_expr,
        output_fields=output_fields
    )
    
    elapsed = time.time() - start_time
    
    # Format results
    results = []
    if search_results and len(search_results) > 0:
        hits = search_results[0]
        for hit in hits:
            result = {
                "id": hit.id,
                "distance": hit.distance,
                "score": hit.score if hasattr(hit, 'score') else hit.distance
            }
            # Add output fields
            if hasattr(hit, 'entity'):
                result.update(hit.entity)
            results.append(result)
    
    print(f"\nSearch completed in {elapsed*1000:.2f}ms")
    print(f"Found {len(results)} results")
    print("\n" + "=" * 80)
    print("=== Search Results ===")
    print("=" * 80)
    
    for i, result in enumerate(results[:10], 1):  # Show first 10
        print(f"\nResult {i}:")
        print(f"  ID: {result.get('id', 'N/A')}")
        print(f"  Distance: {result.get('distance', 'N/A'):.6f}")
        if PK_FIELD in result:
            print(f"  {PK_FIELD}: {result[PK_FIELD]}")
        # Show a few other fields if available
        for field in ["gender", "height", "purpose"]:
            if field in result:
                print(f"  {field}: {result[field]}")
    
    if len(results) > 10:
        print(f"\n... and {len(results) - 10} more results")
    
    print("\n" + "=" * 80)
    
    return results


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Zilliz Cloud API Operations - Info, Upsert, and Hybrid Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print collection info
  python 03_api_operations.py --operation info --collection user_profiles
  
  # Upsert entities from JSON file
  python 03_api_operations.py --operation upsert --collection user_profiles --data entities.json
  
  # Hybrid search with filter
  python 03_api_operations.py --operation search --collection user_profiles \\
    --query-vector "[0.1,0.2,...]" --filter "gender == 'male' AND height > 170" --top-k 20
        """
    )
    
    parser.add_argument(
        "--operation", "-o",
        type=str,
        choices=["info", "upsert", "search"],
        required=True,
        help="Operation to perform: info, upsert, or search"
    )
    
    parser.add_argument(
        "--collection", "-c",
        type=str,
        default=COLLECTION_NAME,
        help=f"Collection name (default: {COLLECTION_NAME})"
    )
    
    parser.add_argument(
        "--database", "-d",
        type=str,
        default=None,
        help=f"Database name (default: {DATABASE_NAME})"
    )
    
    # Upsert-specific arguments
    parser.add_argument(
        "--data",
        type=str,
        help="JSON file path containing entities to upsert (for upsert operation)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for upsert operations (default: 1000)"
    )
    
    # Search-specific arguments
    parser.add_argument(
        "--query-vector",
        type=str,
        help="Query vector as JSON array string (e.g., '[0.1,0.2,...]') or path to JSON file"
    )
    
    parser.add_argument(
        "--filter",
        type=str,
        help="Filter expression (e.g., \"gender == 'male' AND height > 170\")"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)"
    )
    
    parser.add_argument(
        "--ef",
        type=int,
        default=100,
        help="HNSW search parameter ef (default: 100)"
    )
    
    parser.add_argument(
        "--output-fields",
        type=str,
        help="Comma-separated list of fields to return (default: all fields)"
    )
    
    args = parser.parse_args()
    
    # Connect to Zilliz Cloud
    connect()
    
    # Use database
    db_name = args.database or DATABASE_NAME
    db.using_database(db_name)
    
    # Execute operation
    if args.operation == "info":
        print_collection_info(args.collection, db_name)
    
    elif args.operation == "upsert":
        if not args.data:
            parser.error("--data is required for upsert operation")
        
        # Load entities from JSON file
        try:
            with open(args.data, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    entities = data
                elif isinstance(data, dict) and 'entities' in data:
                    entities = data['entities']
                else:
                    entities = [data]
        except FileNotFoundError:
            parser.error(f"File not found: {args.data}")
        except json.JSONDecodeError as e:
            parser.error(f"Invalid JSON file: {e}")
        
        upsert_entities(args.collection, entities, db_name, args.batch_size)
    
    elif args.operation == "search":
        if not args.query_vector:
            parser.error("--query-vector is required for search operation")
        
        # Parse query vector
        try:
            # Try as JSON string first
            if args.query_vector.startswith('[') or args.query_vector.startswith('{'):
                query_vector = json.loads(args.query_vector)
            else:
                # Try as file path
                with open(args.query_vector, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        query_vector = data
                    elif isinstance(data, dict) and 'vector' in data:
                        query_vector = data['vector']
                    else:
                        query_vector = data
        except (json.JSONDecodeError, FileNotFoundError) as e:
            parser.error(f"Invalid query vector: {e}")
        
        # Parse output fields
        output_fields = None
        if args.output_fields:
            output_fields = [f.strip() for f in args.output_fields.split(',')]
        
        hybrid_search(
            collection_name=args.collection,
            query_vector=query_vector,
            filter_expr=args.filter,
            top_k=args.top_k,
            ef=args.ef,
            output_fields=output_fields,
            db_name=db_name
        )


if __name__ == "__main__":
    main()

