#!/usr/bin/env python3
"""
Helper script to generate test entities for upsert testing.
"""

import json
import numpy as np

# Generate test entities
entities = [
    {
        "userId": 9999,
        "height": 180.5,
        "latitude": 37.7749,
        "longitude": -122.4194,
        "dob": 19900101,
        "createdAt": 1700000000000,
        "updatedAt": 1700000000000,
        "openToCatchFeelings": True,
        "datingPause": False,
        "singleImageExists": True,
        "gender": "male",
        "sexualOrientation": "straight",
        "purpose": "relationship",
        "ethnicity": "asian",
        "drinkStatus": "social",
        "smokeStatus": "never",
        "weedStatus": "never",
        "embeddings": np.random.rand(768).astype(np.float32).tolist()
    },
    {
        "userId": 10000,
        "height": 165.0,
        "latitude": 37.7849,
        "longitude": -122.4094,
        "dob": 19950101,
        "createdAt": 1700001000000,
        "updatedAt": 1700001000000,
        "openToCatchFeelings": False,
        "datingPause": False,
        "singleImageExists": True,
        "gender": "female",
        "sexualOrientation": "bisexual",
        "purpose": "casual",
        "ethnicity": "white",
        "drinkStatus": "regular",
        "smokeStatus": "social",
        "weedStatus": "never",
        "embeddings": np.random.rand(768).astype(np.float32).tolist()
    }
]

# Save to JSON file
with open("test_entity.json", "w") as f:
    json.dump(entities, f, indent=2)

print(f"Generated test_entity.json with {len(entities)} entities")
for i, entity in enumerate(entities, 1):
    print(f"  Entity {i}: ID={entity['userId']}, gender={entity['gender']}, embedding_dim={len(entity['embeddings'])}")

