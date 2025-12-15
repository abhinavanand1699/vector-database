#!/usr/bin/env python3
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "user_profiles"
EMBED_DIM = 768
SHARDS = 2  # adjust for ingest throughput; 4–8 for higher write QPS


def connect():
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("Connected to Milvus")


def build_schema() -> CollectionSchema:
    fields = [
        FieldSchema(name="userId", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="height", dtype=DataType.FLOAT),
        FieldSchema(name="latitude", dtype=DataType.FLOAT),
        FieldSchema(name="longitude", dtype=DataType.FLOAT),
        FieldSchema(name="dob", dtype=DataType.INT64),
        FieldSchema(name="createdAt", dtype=DataType.INT64),
        FieldSchema(name="updatedAt", dtype=DataType.INT64),
        FieldSchema(name="openToCatchFeelings", dtype=DataType.BOOL),
        FieldSchema(name="datingPause", dtype=DataType.BOOL),
        FieldSchema(name="singleImageExists", dtype=DataType.BOOL),
        FieldSchema(name="gender", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="sexualOrientation", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="purpose", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="ethnicity", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="drinkStatus", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="smokeStatus", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="weedStatus", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
    ]
    return CollectionSchema(
        fields=fields,
        description="User dating profiles with preference embeddings",
        enable_dynamic_field=False,
    )


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

