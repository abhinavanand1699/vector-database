# Zilliz Vector Database Setup

This folder contains scripts for setting up and managing a Milvus vector database for user profiles on Zilliz Cloud.

## Scripts Overview

### 1. `01_create_user_collection.py`
Creates the Milvus collection with schema, indexes, and loads it into memory.

**Usage:**
```bash
python 01_create_user_collection.py
```

**What it does:**
- Connects to Zilliz Cloud
- Creates database `user_profile_db` with properties (max collections, disk quota, replicas)
- Creates collection `user_profiles` with schema (userId, embeddings, scalar fields)
- Creates HNSW index on embeddings (768-dim, COSINE metric)
- Creates BITMAP indexes on categorical fields
- Creates STL_SORT indexes on numeric/time fields
- Loads collection for search

### 2. `02_ingest_1m_vectors.py`
Bulk ingestion script to insert vectors into the collection.

**Usage:**
```bash
python 02_ingest_1m_vectors.py --total 1000000 --batch 10000
```

**Options:**
- `--total` or `-t`: Total vectors to insert (default: 1,000,000)
- `--batch` or `-b`: Batch size per insert (default: 10,000)
- `--collection` or `-c`: Collection name (default: user_profiles)

### 3. `03_api_operations.py`
API operations script for collection management, upsert, and hybrid search.

**Operations:**

#### Print Collection Info
```bash
python 03_api_operations.py --operation info --collection user_profiles
```
Displays database name, collection name, schema, indexes, and entity count.

#### Upsert Entities
```bash
python 03_api_operations.py --operation upsert --collection user_profiles --data test_entity.json
```

**Options:**
- `--operation` or `-o`: Operation type (`info`, `upsert`, `search`)
- `--collection` or `-c`: Collection name (default: user_profiles)
- `--database` or `-d`: Database name (default: user_profile_db)
- `--data`: JSON file path containing entities to upsert
- `--batch-size`: Batch size for upsert operations (default: 1000)

#### Hybrid Search
```bash
python 03_api_operations.py --operation search \
  --collection user_profiles \
  --query-vector "[0.1,0.2,...]" \
  --filter "gender == 'male' AND height > 170" \
  --top-k 20 \
  --ef 100
```

**Options:**
- `--query-vector`: Query vector as JSON array string or path to JSON file
- `--filter`: Filter expression (e.g., `"gender == 'male' AND height > 170"`)
- `--top-k`: Number of results to return (default: 10)
- `--ef`: HNSW search parameter ef (default: 100)
- `--output-fields`: Comma-separated list of fields to return (default: all fields)

### 4. `04_query_and_performance_tests.py`
Performance testing script for query latency, throughput, and recall.

**Usage:**
```bash
python 04_query_and_performance_tests.py --collection user_profiles --topk 10 --ef 64
```

**Options:**
- `--collection` or `-c`: Collection name (default: user_profiles)
- `--topk`: Number of results per query (default: 10)
- `--ef`: HNSW search ef parameter (default: 64)
- `--ground`: Number of ground-truth vectors to fetch (default: 1000)
- `--recall-sample`: Number of self-queries for recall test (default: 200)
- `--latency-runs`: Number of single-query latency tests (default: 200)
- `--batch-runs`: Runs per batch size (default: 50)
- `--batches`: Comma-separated batch sizes (default: "32,64,128")

### 5. `generate_test_entity.py`
Helper script to generate test entities for upsert testing.

**Usage:**
```bash
python generate_test_entity.py
```
Creates `test_entity.json` with 2 sample entities (IDs: 9999 and 10000).

## Prerequisites

1. **Zilliz Cloud Account:**
   - Zilliz Cloud account and cluster
   - Endpoint URI and API key from your cluster dashboard
   - Set environment variables (see Configuration below)

2. **Python 3.9+**
3. **Dependencies:**
   ```bash
   pip install pymilvus numpy python-dotenv
   ```

## Configuration

### Zilliz Cloud Setup

1. **Get your endpoint URI:**
   - Go to https://cloud.zilliz.com
   - Navigate to your cluster
   - Click "Connection Details" or "Connect"
   - Copy the endpoint URI (format: `https://{cluster-id}.{region}.vectordb.zillizcloud.com:19541`)

2. **Get your API key:**
   - Go to https://cloud.zilliz.com
   - Navigate to "API Keys" section
   - Create a new API key or use an existing one
   - Copy the API key

3. **Create `.env` file:**
   ```bash
   # Create .env file in zilliz-vector-database folder
   ZILLIZ_URI=https://in01-d180f5391055b77.aws-us-west-2.vectordb.zillizcloud.com:19541
   ZILLIZ_API_KEY=your_api_key_here
   DATABASE_NAME=user_profile_db
   ```

   Or set environment variables:
   ```bash
   export ZILLIZ_URI="https://in01-d180f5391055b77.aws-us-west-2.vectordb.zillizcloud.com:19541"
   export ZILLIZ_API_KEY="your_api_key_here"
   export DATABASE_NAME="user_profile_db"
   ```

## Quick Start

### Step 1: Create Collection
```bash
python 01_create_user_collection.py
```

### Step 2: Ingest Data (Optional)
```bash
# Bulk ingestion
python 02_ingest_1m_vectors.py --total 1000000

# Or upsert individual entities
python generate_test_entity.py
python 03_api_operations.py --operation upsert --collection user_profiles --data test_entity.json
```

### Step 3: Verify Collection
```bash
python 03_api_operations.py --operation info --collection user_profiles
```

### Step 4: Test Search
```bash
python 03_api_operations.py --operation search \
  --collection user_profiles \
  --query-vector "[0.1,0.2,...]" \
  --filter "gender == 'male'" \
  --top-k 10
```

## Upsert Guide

### Method 1: Using Generated Test File

1. **Generate test entities:**
   ```bash
   python generate_test_entity.py
   ```
   This creates `test_entity.json` with 2 sample entities.

2. **Run upsert:**
   ```bash
   python 03_api_operations.py --operation upsert --collection user_profiles --data test_entity.json
   ```

3. **Verify:**
   ```bash
   python 03_api_operations.py --operation info --collection user_profiles
   ```

### Method 2: Create Custom JSON File

Create a JSON file (e.g., `my_entities.json`):
```json
[
  {
    "userId": 12345,
    "height": 175.5,
    "latitude": 37.7749,
    "longitude": -122.4194,
    "dob": 19900101,
    "createdAt": 1700000000000,
    "updatedAt": 1700000000000,
    "openToCatchFeelings": true,
    "datingPause": false,
    "singleImageExists": true,
    "gender": "male",
    "sexualOrientation": "straight",
    "purpose": "relationship",
    "ethnicity": "asian",
    "drinkStatus": "social",
    "smokeStatus": "never",
    "weedStatus": "never",
    "embeddings": [0.1, 0.2, 0.3, ...]  // Must be exactly 768 dimensions
  }
]
```

**Generate 768-dimensional embeddings:**
```python
import numpy as np
embeddings = np.random.rand(768).astype(np.float32).tolist()
```

**Run upsert:**
```bash
python 03_api_operations.py --operation upsert --collection user_profiles --data my_entities.json
```

### Method 3: Update Existing Entity

To update an existing entity, use the same `userId`:

```bash
# First upsert
python 03_api_operations.py --operation upsert --collection user_profiles --data test_entity.json

# Modify test_entity.json - change fields but keep same userId
# Then upsert again (will update existing entity)
python 03_api_operations.py --operation upsert --collection user_profiles --data test_entity.json
```

### Upsert Features

- **Automatic type conversion**: FLOAT fields are converted to float32 automatically
- **Batch processing**: Processes entities in batches (default: 1000 per batch)
- **Progress tracking**: Shows progress and throughput statistics
- **Validation**: Checks for required fields (`userId`, `embeddings`)
- **Auto-flush**: Flushes data after upsert to ensure persistence

### Required Fields

- **userId** (INT64): Primary key - required
- **embeddings** (FLOAT_VECTOR): 768-dimensional vector - required

### Optional Fields (use defaults if omitted)

- `height`, `latitude`, `longitude` (FLOAT) - default: 0.0
- `dob`, `createdAt`, `updatedAt` (INT64) - default: 0
- `openToCatchFeelings`, `datingPause`, `singleImageExists` (BOOL) - default: false
- `gender`, `sexualOrientation`, `purpose`, `ethnicity`, `drinkStatus`, `smokeStatus`, `weedStatus` (VARCHAR) - default: "NULL"

## Collection Schema

- **Primary Key**: `userId` (INT64)
- **Vector Field**: `embeddings` (FLOAT_VECTOR, dim=768)
- **Scalar Fields**: 
  - Numeric: `height` (FLOAT), `latitude` (FLOAT), `longitude` (FLOAT)
  - Timestamps: `dob` (INT64), `createdAt` (INT64), `updatedAt` (INT64)
  - Boolean: `openToCatchFeelings`, `datingPause`, `singleImageExists`
  - Categorical: `gender`, `sexualOrientation`, `purpose`, `ethnicity`, `drinkStatus`, `smokeStatus`, `weedStatus` (VARCHAR)

## Index Configuration

- **HNSW Index** on `embeddings`: M=32, efConstruction=200, metric=COSINE
- **BITMAP Indexes** on: datingPause, singleImageExists, openToCatchFeelings, gender, sexualOrientation, drinkStatus, smokeStatus, weedStatus, purpose
- **STL_SORT Indexes** on: dob, height, createdAt, updatedAt

## Troubleshooting

### Common Issues

1. **"Entity missing required fields"**
   - Ensure `userId` and `embeddings` are present in your JSON
   - Check that `embeddings` is exactly 768 dimensions

2. **"Collection does not exist"**
   - Run `01_create_user_collection.py` first to create the collection

3. **"ZILLIZ_URI and ZILLIZ_API_KEY are required"**
   - Create a `.env` file with your credentials (see Configuration section)

4. **Type errors**
   - FLOAT fields are automatically converted to float32
   - Ensure embeddings are a list of floats (not numpy array)

5. **"AmbiguousIndexName" error**
   - This is fixed in the scripts - ensure you're using the latest version

### Verify Operations

**Check collection info:**
```bash
python 03_api_operations.py --operation info --collection user_profiles
```

**Search for specific entity:**
```bash
python 03_api_operations.py --operation search \
  --collection user_profiles \
  --query-vector "[0.1,0.2,...]" \
  --filter "userId == 9999" \
  --top-k 1
```

## Additional Documentation

- `STORAGE_ARCHITECTURE.md` - Detailed explanation of disk quota and storage architecture
- `test_entity.json` - Sample test entities (generated by `generate_test_entity.py`)
