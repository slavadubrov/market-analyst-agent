"""Document Memory: File-based storage for accumulated knowledge.

This module implements the third tier of the three-tier memory architecture:
1. Hot memory (PostgreSQL checkpoints) - short-term state persistence
2. Cold memory (Qdrant vectors) - long-term semantic memory
3. Document memory (Files) - structured knowledge accumulation

Document memory provides namespace organization for different types of content:
- research: Analysis reports and market research
- conventions: Established patterns and preferences
- learnings: Episodic knowledge from past runs
- user-profiles: User-specific configurations
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class DocumentMetadata:
    """Metadata for a stored document."""

    timestamp: str
    user_id: str
    namespace: str
    tags: list[str] | None = None
    execution_mode: str | None = None
    ticker: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class DocumentMemory:
    """File-based storage for accumulated knowledge with namespaces.

    Organizes documents into namespaces:
    - research/: Market analysis reports
    - conventions/: Established patterns (e.g., report style)
    - learnings/: Episodic knowledge from past executions
    - user-profiles/: User preferences and configurations

    Each document is stored as JSON with content and metadata.
    """

    def __init__(self, base_path: Path | str = Path("memory/documents")):
        """Initialize document memory.

        Args:
            base_path: Root directory for document storage
        """
        self.base_path = Path(base_path)
        self.namespaces = ["research", "conventions", "learnings", "user-profiles"]

        # Create namespace directories
        for namespace in self.namespaces:
            (self.base_path / namespace).mkdir(parents=True, exist_ok=True)

    def write_doc(
        self,
        namespace: str,
        key: str,
        content: str,
        metadata: DocumentMetadata | None = None,
    ) -> Path:
        """Write a document to the specified namespace.

        Args:
            namespace: Namespace category (research, conventions, etc.)
            key: Document identifier (will be sanitized for filesystem)
            content: Document content (markdown, text, etc.)
            metadata: Optional metadata for the document

        Returns:
            Path to the written file

        Raises:
            ValueError: If namespace is invalid
        """
        if namespace not in self.namespaces:
            raise ValueError(f"Invalid namespace '{namespace}'. Must be one of: {self.namespaces}")

        # Sanitize key for filesystem
        safe_key = self._sanitize_key(key)
        filepath = self.base_path / namespace / f"{safe_key}.json"

        # Prepare document structure
        doc = {
            "content": content,
            "metadata": metadata.to_dict() if metadata else {},
            "created_at": datetime.now().isoformat(),
        }

        # Write to file
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(json.dumps(doc, indent=2))

        return filepath

    def read_doc(self, namespace: str, key: str) -> dict[str, Any] | None:
        """Read a document from the specified namespace.

        Args:
            namespace: Namespace category
            key: Document identifier

        Returns:
            Dictionary with 'content' and 'metadata' keys, or None if not found
        """
        if namespace not in self.namespaces:
            return None

        safe_key = self._sanitize_key(key)
        filepath = self.base_path / namespace / f"{safe_key}.json"

        if not filepath.exists():
            return None

        try:
            data = json.loads(filepath.read_text())
            return {
                "content": data.get("content", ""),
                "metadata": data.get("metadata", {}),
                "created_at": data.get("created_at"),
            }
        except (json.JSONDecodeError, OSError):
            return None

    def list_docs(self, namespace: str, pattern: str = "*") -> list[dict[str, Any]]:
        """List documents in a namespace with optional pattern matching.

        Args:
            namespace: Namespace category
            pattern: Glob pattern for filtering (default: all files)

        Returns:
            List of document metadata dictionaries, sorted by creation time (newest first)
        """
        if namespace not in self.namespaces:
            return []

        namespace_dir = self.base_path / namespace
        if not namespace_dir.exists():
            return []

        # Find matching files
        files = sorted(
            namespace_dir.glob(f"{pattern}.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Load metadata for each file
        docs = []
        for filepath in files:
            try:
                data = json.loads(filepath.read_text())
                docs.append(
                    {
                        "key": filepath.stem,
                        "path": str(filepath),
                        "metadata": data.get("metadata", {}),
                        "created_at": data.get("created_at"),
                    }
                )
            except (json.JSONDecodeError, OSError):
                continue

        return docs

    def search_docs(self, namespace: str, query: str) -> list[dict[str, Any]]:
        """Search documents by content or metadata.

        Performs simple text-based search across document content and metadata.
        For production use, consider integrating with embedding-based search.

        Args:
            namespace: Namespace category to search
            query: Search query string

        Returns:
            List of matching documents with content and metadata
        """
        if namespace not in self.namespaces:
            return []

        namespace_dir = self.base_path / namespace
        if not namespace_dir.exists():
            return []

        query_lower = query.lower()
        results = []

        # Search all documents in namespace
        for filepath in namespace_dir.glob("*.json"):
            try:
                data = json.loads(filepath.read_text())
                content = data.get("content", "")
                metadata = data.get("metadata", {})

                # Simple text search in content and metadata
                searchable = f"{content} {json.dumps(metadata)}".lower()

                if query_lower in searchable:
                    results.append(
                        {
                            "key": filepath.stem,
                            "path": str(filepath),
                            "content": content,
                            "metadata": metadata,
                            "created_at": data.get("created_at"),
                        }
                    )
            except (json.JSONDecodeError, OSError):
                continue

        # Sort by modification time (newest first)
        results.sort(
            key=lambda x: Path(x["path"]).stat().st_mtime,
            reverse=True,
        )

        return results

    def delete_doc(self, namespace: str, key: str) -> bool:
        """Delete a document from the specified namespace.

        Args:
            namespace: Namespace category
            key: Document identifier

        Returns:
            True if document was deleted, False if not found
        """
        if namespace not in self.namespaces:
            return False

        safe_key = self._sanitize_key(key)
        filepath = self.base_path / namespace / f"{safe_key}.json"

        if filepath.exists():
            filepath.unlink()
            return True

        return False

    @staticmethod
    def _sanitize_key(key: str) -> str:
        """Sanitize a key for safe filesystem usage.

        Args:
            key: Raw key string

        Returns:
            Filesystem-safe key
        """
        # Replace unsafe characters with underscores
        safe = key.replace("/", "_").replace("\\", "_").replace(" ", "_")
        # Remove any other potentially problematic characters
        safe = "".join(c for c in safe if c.isalnum() or c in "._-")
        return safe


def get_document_memory() -> DocumentMemory:
    """Get a DocumentMemory instance with default configuration.

    Returns:
        DocumentMemory instance
    """
    return DocumentMemory()
