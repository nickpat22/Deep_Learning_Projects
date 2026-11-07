from datetime import datetime, timezone
from functools import lru_cache
import os

from pymongo import MongoClient, DESCENDING


@lru_cache(maxsize=1)
def get_mongo_client():
    """
    Reuse one MongoClient per warm function instance.
    MongoDB URI must be configured as an environment variable.
    """
    uri = os.getenv("MONGODB_URI")
    if not uri:
        return None

    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        appname="Reusable-AI-ML-Template",
    )
    client.admin.command("ping")
    return client


def get_collection():
    client = get_mongo_client()
    if client is None:
        return None

    database_name = os.getenv("MONGODB_DATABASE", "ai_ml_projects")
    collection_name = os.getenv("MONGODB_COLLECTION", "prediction_history")
    return client[database_name][collection_name]


def save_prediction(collection, result):
    collection.insert_one(
        {
            "project": os.getenv("PROJECT_NAME", "crack-detection"),
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "raw_model_output": result["raw_model_output"],
            "created_at": datetime.now(timezone.utc),
        }
    )


def get_recent_predictions(collection, limit=10):
    documents = collection.find(
        {},
        {
            "_id": 0,
            "project": 1,
            "prediction": 1,
            "confidence": 1,
            "created_at": 1,
        },
    ).sort("created_at", DESCENDING).limit(limit)

    items = []
    for document in documents:
        created_at = document.get("created_at")
        items.append(
            {
                "project": document.get("project"),
                "prediction": document.get("prediction"),
                "confidence": document.get("confidence"),
                "created_at": created_at.isoformat() if created_at else None,
            }
        )
    return items
