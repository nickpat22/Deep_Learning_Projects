from datetime import datetime, timezone
from functools import lru_cache
import os

import streamlit as st
from pymongo import MongoClient, DESCENDING


def _setting(name: str, default=None):
    """Read a setting from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name, None)
    except Exception:
        value = None
    if value not in (None, ""):
        return value
    return os.getenv(name, default)


def database_is_configured() -> bool:
    return bool(_setting("MONGODB_URI"))


@st.cache_resource
def get_mongo_client():
    uri = _setting("MONGODB_URI")
    if not uri:
        return None

    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        appname="Crack-Detection-Streamlit",
    )
    client.admin.command("ping")
    return client


def get_collection():
    client = get_mongo_client()
    if client is None:
        return None

    database_name = _setting("MONGODB_DATABASE", "ai_ml_projects")
    collection_name = _setting("MONGODB_COLLECTION", "prediction_history")
    return client[database_name][collection_name]


def save_prediction(result: dict) -> None:
    collection = get_collection()
    if collection is None:
        raise RuntimeError("MONGODB_URI is not configured.")

    collection.insert_one(
        {
            "project": _setting("PROJECT_NAME", "crack-detection"),
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "raw_model_output": result["raw_model_output"],
            "created_at": datetime.now(timezone.utc),
        }
    )


def get_recent_predictions(limit: int = 50) -> list[dict]:
    collection = get_collection()
    if collection is None:
        return []

    documents = collection.find(
        {},
        {
            "_id": 0,
            "project": 1,
            "prediction": 1,
            "confidence": 1,
            "raw_model_output": 1,
            "created_at": 1,
        },
    ).sort("created_at", DESCENDING).limit(limit)

    records = []
    for document in documents:
        created_at = document.get("created_at")
        records.append(
            {
                "project": document.get("project"),
                "prediction": document.get("prediction"),
                "confidence": document.get("confidence"),
                "raw_model_output": document.get("raw_model_output"),
                "created_at": created_at.isoformat() if created_at else None,
            }
        )
    return records
