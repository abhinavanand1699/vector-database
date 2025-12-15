#!/usr/bin/env python3
import os
import time
import numpy as np
from dotenv import load_dotenv
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
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
SHARDS = 2  # adjust for ingest throughput; 4–8 for higher write QPS


def connect():
    if not ZILLIZ_URI or not ZILLIZ_API_KEY:
        raise ValueError(
            "ZILLIZ_URI and ZILLIZ_API_KEY are required. "
            "Please set them in a .env file or as environment variables.\n"
            "Create a .env file with:\n"
            "  ZILLIZ_URI=https://in01-d180f5391055b77.aws-us-west-2.vectordb.zillizcloud.com:19541\n"
            "  ZILLIZ_API_KEY=your_api_key_here"
        )
    
    # Zilliz Cloud connection
    connections.connect(
        alias="default",
        uri=ZILLIZ_URI,
        token=ZILLIZ_API_KEY
    )
    print(f"Connected to Zilliz Cloud at {ZILLIZ_URI}")


def create_database(db_name: str):
    """
    Create database with properties.
    
    Database Properties Available:
    - database.replica.number (integer): Number of replicas (CLUSTER MODE ONLY)
    - database.resource_groups (string): Comma-separated resource group names
    - database.diskQuota.mb (integer): Max disk space in MB
    - database.max.collections (integer): Max number of collections allowed
    - database.force.deny.writing (boolean): Force deny write operations
    - database.force.deny.reading (boolean): Force deny read operations
    - timezone (string): IANA timezone identifier (e.g., "UTC", "America/Chicago")
    """
    # Database properties dictionary
    # Only include properties you want to set; others are optional
    db_properties = {
        # REQUIRED/COMMON PROPERTIES (uncomment and set as needed):
        "database.max.collections": "3",  # Limit max collections (good for resource management)
        "database.diskQuota.mb": "6000",  # 6 GB disk quota (for ~1M records)
        
        # REPLICAS - AVAILABLE IN ZILLIZ CLOUD DEDICATED CLUSTERS
        # Set number of replicas for high availability (default: 1, can increase via UI or here)
        # More replicas = better read performance and fault tolerance, but uses more resources
        "database.replica.number": "1",  # Set to 2 or 3 for better availability (check your cluster's replica capacity)
        
        # OPTIONAL PROPERTIES (commented out - uncomment if needed):
        
        # Resource groups - for isolating workloads (CLUSTER MODE)
        # "database.resource_groups": "rg1,rg2",  # Comma-separated resource group names
        
        # Force deny operations - for maintenance/emergency
        # "database.force.deny.writing": "false",  # Set to "true" to block all writes
        # "database.force.deny.reading": "false",  # Set to "true" to block all reads
        
        # Timezone - for TIMESTAMPTZ fields
        # "timezone": "UTC",  # Default timezone (e.g., "UTC", "America/Chicago", "Asia/Shanghai")
    }
    
    try:
        # Create database with properties
        db.create_database(db_name, properties=db_properties)
        print(f"Created database: {db_name}")
        print(f"  Properties: max_collections={db_properties.get('database.max.collections')}, "
              f"disk_quota={db_properties.get('database.diskQuota.mb')} MB, "
              f"replicas={db_properties.get('database.replica.number')}")
    except Exception as e:
        # Database might already exist
        if "already exist" in str(e).lower() or "already exists" in str(e).lower():
            print(f"Database '{db_name}' already exists - using existing database")
        else:
            raise e
    
    # Use the database (required before creating collections)
    db.using_database(db_name)
    print(f"Using database: {db_name}")


def build_schema() -> CollectionSchema:
    """
    Build collection schema with required and optional fields.
    
    Default values are set for optional fields. If a field is omitted during insert,
    Milvus will automatically use the default value.
    
    Required fields (must always be provided, no defaults):
    - userId (primary key) - cannot have default value
    - embeddings (vector field) - cannot have default value
    
    Optional fields (will use default values if omitted):
    - All scalar fields have default values set
    """
    fields = [
        # REQUIRED FIELDS (must always be provided, cannot have defaults)
        FieldSchema(
            name="userId", 
            dtype=DataType.INT64, 
            is_primary=True, 
            auto_id=False,
            description="Unique identifier for each user profile (primary key)"
        ),
        FieldSchema(
            name="embeddings", 
            dtype=DataType.FLOAT_VECTOR, 
            dim=EMBED_DIM,
            description="768-dimensional vector embedding representing user hobbies and deep passions for similarity search"
        ),
        
        # OPTIONAL FIELDS with default values (will be used if field is omitted during insert)
        # Note: FLOAT fields need np.float32() for default values (not Python's float which is double)
        FieldSchema(
            name="height", 
            dtype=DataType.FLOAT, 
            default_value=np.float32(0.0),
            description="User height in centimeters (0.0 if not provided)"
        ),
        FieldSchema(
            name="latitude", 
            dtype=DataType.FLOAT, 
            default_value=np.float32(0.0),
            description="Geographic latitude coordinate for location-based matching (0.0 if not provided)"
        ),
        FieldSchema(
            name="longitude", 
            dtype=DataType.FLOAT, 
            default_value=np.float32(0.0),
            description="Geographic longitude coordinate for location-based matching (0.0 if not provided)"
        ),
        FieldSchema(
            name="dob", 
            dtype=DataType.INT64, 
            default_value=0,
            description="Date of birth stored as epoch timestamp in milliseconds (0 if not provided)"
        ),
        FieldSchema(
            name="createdAt", 
            dtype=DataType.INT64, 
            default_value=0,
            description="Timestamp when user profile was created (epoch milliseconds, 0 if not provided)"
        ),
        FieldSchema(
            name="updatedAt", 
            dtype=DataType.INT64, 
            default_value=0,
            description="Timestamp when user profile was last updated (epoch milliseconds, 0 if not provided)"
        ),
        FieldSchema(
            name="openToCatchFeelings", 
            dtype=DataType.BOOL, 
            default_value=False,
            description="Whether user is open to developing romantic feelings (default: False)"
        ),
        FieldSchema(
            name="datingPause", 
            dtype=DataType.BOOL, 
            default_value=False,
            description="Whether user has paused their dating profile (default: False)"
        ),
        FieldSchema(
            name="singleImageExists", 
            dtype=DataType.BOOL, 
            default_value=False,
            description="Whether user has uploaded at least one profile image (default: False)"
        ),
        # VARCHAR fields: "NULL" string is the default (represents no value/None)
        FieldSchema(
            name="gender", 
            dtype=DataType.VARCHAR, 
            max_length=32, 
            default_value="NULL",
            description="User's gender identity (e.g., 'Male', 'Female', 'Non-binary', default: 'NULL')"
        ),
        FieldSchema(
            name="sexualOrientation", 
            dtype=DataType.VARCHAR, 
            max_length=32, 
            default_value="NULL",
            description="User's sexual orientation preference (e.g., 'Straight', 'Gay', 'Bisexual', default: 'NULL')"
        ),
        FieldSchema(
            name="purpose", 
            dtype=DataType.VARCHAR, 
            max_length=64, 
            default_value="NULL",
            description="User's dating purpose or intent (e.g., 'Long-term', 'Casual', 'Friendship', default: 'NULL')"
        ),
        FieldSchema(
            name="ethnicity", 
            dtype=DataType.VARCHAR, 
            max_length=64, 
            default_value="NULL",
            description="User's ethnicity or cultural background (default: 'NULL')"
        ),
        FieldSchema(
            name="drinkStatus", 
            dtype=DataType.VARCHAR, 
            max_length=32, 
            default_value="NULL",
            description="User's drinking preference (e.g., 'Never', 'Socially', 'Regularly', default: 'NULL')"
        ),
        FieldSchema(
            name="smokeStatus", 
            dtype=DataType.VARCHAR, 
            max_length=32, 
            default_value="NULL",
            description="User's smoking preference (e.g., 'Never', 'Occasionally', 'Regularly', default: 'NULL')"
        ),
        FieldSchema(
            name="weedStatus", 
            dtype=DataType.VARCHAR, 
            max_length=32, 
            default_value="NULL",
            description="User's cannabis/marijuana usage preference (e.g., 'Never', 'Occasionally', 'Regularly', default: 'NULL')"
        ),
    ]
    return CollectionSchema(
        fields=fields,
        description="User dating profiles with preference embeddings",
        enable_dynamic_field=False,  # Strict schema - only defined fields allowed
    )


def prepare_record_for_insert(record: dict) -> dict:
    """
    Helper function to handle None/null values before inserting.
    
    Since Milvus doesn't support default values in schema, you have two options:
    1. Omit the field entirely (works with enable_dynamic_field=True) - RECOMMENDED
    2. Set a default value here (e.g., "" for strings, 0 for numbers, False for bools)
    
    This function demonstrates option 2 - setting defaults for None values.
    """
    defaults = {
        "height": 0.0,
        "latitude": 0.0,
        "longitude": 0.0,
        "dob": 0,
        "createdAt": int(time.time() * 1000) if "createdAt" not in record else record["createdAt"],
        "updatedAt": int(time.time() * 1000) if "updatedAt" not in record else record["updatedAt"],
        "openToCatchFeelings": False,
        "datingPause": False,
        "singleImageExists": False,
        "gender": "",
        "sexualOrientation": "",
        "purpose": "",
        "ethnicity": "",
        "drinkStatus": "",
        "smokeStatus": "",
        "weedStatus": "",
    }
    
    # Apply defaults for None values
    prepared = record.copy()
    for key, default_value in defaults.items():
        if key not in prepared or prepared[key] is None:
            prepared[key] = default_value
    
    # Remove None values entirely (alternative approach - works with dynamic fields)
    # prepared = {k: v for k, v in record.items() if v is not None}
    
    return prepared


def drop_if_exists(name: str):
    existing = set(utility.list_collections())
    if name in existing:
        Collection(name).drop()
        print(f"Dropped existing collection: {name}")


def create_collection(name: str, schema: CollectionSchema) -> Collection:
    return Collection(name=name, schema=schema, shards_num=SHARDS)


def create_vector_index(collection: Collection):
    collection.create_index(
        field_name="embeddings",
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",  # change to IP/L2 if your model needs it
            "params": {"M": 32, "efConstruction": 200},
        },
    )


def create_bitmap_indexes(collection: Collection):
    bitmap_fields = [
        "datingPause",
        "singleImageExists",
        "openToCatchFeelings",
        "gender",
        "sexualOrientation",
        "drinkStatus",
        "smokeStatus",
        "weedStatus",
        "purpose",
    ]
    for f in bitmap_fields:
        collection.create_index(field_name=f, index_params={"index_type": "BITMAP"})


def create_sort_indexes(collection: Collection):
    sort_fields = ["dob", "height", "createdAt", "updatedAt"]
    for f in sort_fields:
        collection.create_index(
            field_name=f,
            index_params={"index_type": "STL_SORT"},  # use INVERTED instead if preferred
        )


def load(collection: Collection):
    collection.load()


def print_schema_and_indexes(collection: Collection):
    print("\n=== Collection Schema ===")
    print(collection.schema)
    print("\n=== Indexes ===")
    print("list_indexes:", utility.list_indexes(collection.name))
    for idx in collection.indexes:
        info = {"index_name": idx.index_name}
        # idx.params usually contains field_name, index_type, metric_type, params
        info.update(idx.params)
        print(info)
    print()


def main():
    connect()
    create_database(DATABASE_NAME)
    drop_if_exists(COLLECTION_NAME)
    schema = build_schema()
    coll = create_collection(COLLECTION_NAME, schema)
    print(f"Created collection: {COLLECTION_NAME}")

    create_vector_index(coll)
    create_bitmap_indexes(coll)
    create_sort_indexes(coll)
    print_schema_and_indexes(coll)
    load(coll)
    print("Indexes created and collection loaded.")


if __name__ == "__main__":
    main()

