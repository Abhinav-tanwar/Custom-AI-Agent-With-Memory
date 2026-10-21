"""Local Chroma storage; embeddings are supplied explicitly, never auto-downloaded."""

import hashlib
import re
from datetime import datetime, timezone

import chromadb


class MemoryStore:
    def __init__(self, path, embeddings):
        self.embeddings = embeddings
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            "personal_memories",
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def save(self, text):
        text = text.strip()
        if not text or len(text) > 2000:
            raise ValueError("Memory must contain between 1 and 2,000 characters.")
        # Reduce accidental credential storage; this is not a comprehensive DLP filter.
        if re.search(r"sk-[A-Za-z0-9_-]{12,}|-----BEGIN .*PRIVATE KEY-----", text):
            raise ValueError("Remove credentials before saving this memory.")
        identifier = hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()
        if self.collection.get(ids=[identifier])["ids"]:
            return identifier
        vector = self.embeddings.embed_documents([text])[0]
        self.collection.upsert(
            ids=[identifier],
            documents=[text],
            embeddings=[vector],
            metadatas=[{"created_at": datetime.now(timezone.utc).isoformat()}],
        )
        return identifier

    def list(self):
        rows = self.collection.get(include=["documents", "metadatas"])
        return sorted(
            [
                {"id": identifier, "text": text, "created_at": metadata["created_at"]}
                for identifier, text, metadata in zip(
                    rows["ids"], rows["documents"], rows["metadatas"], strict=True
                )
            ],
            key=lambda row: row["created_at"],
            reverse=True,
        )

    def search(self, query, limit=4):
        count = self.collection.count()
        if not count:
            return []
        result = self.collection.query(
            query_embeddings=[self.embeddings.embed_query(query)],
            n_results=min(limit, count),
            include=["documents", "distances"],
        )
        # Cosine distance cutoff prevents unrelated memories being forced into context.
        return [
            {"id": identifier, "text": text}
            for identifier, text, distance in zip(
                result["ids"][0], result["documents"][0], result["distances"][0], strict=True
            )
            if distance <= 0.65
        ]

    def delete(self, identifier):
        self.collection.delete(ids=[identifier])

    def clear(self):
        ids = self.collection.get()["ids"]
        if ids:
            self.collection.delete(ids=ids)
