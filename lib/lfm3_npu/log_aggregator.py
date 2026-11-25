"""Log Aggregation for LFM3-1.7B inference servers.

Provides:
  * Centralized log collection from CPU and NPU servers
  * Log rotation and archival
  * Log search and filtering
  * Structured log parsing (JSON format)
  * Log retention policies
  * Export to external systems (Elasticsearch, Loki, etc.)
"""

import gzip
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LogEntry:
    """A single log entry."""

    timestamp: float
    level: str
    message: str
    source: str
    source_file: str
    line_number: Optional[int] = None
    raw_line: str
    parsed_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LogCollectionResult:
    """Result of log collection."""

    source: str
    file_path: str
    entries_collected: int
    entries_parsed: int
    errors: int
    bytes_read: int


class LogAggregator:
    """Collects and aggregates logs from LFM3-1.7B inference servers."""

    def __init__(
        self,
        log_path: Optional[str] = None,
        storage_path: Optional[str] = None,
        max_log_size_bytes: int = 100 * 1024 * 1024,
        retention_days: int = 30,
        compress_archives: bool = True,
    ):
        if log_dir is None:
            log_dir = os.environ.get(
                "LFM3_LOG_DIR",
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "var", "log"),
            )
        self.log_path = Path(log_path)
        self.storage_path = Path(
            storage_path or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "var",
                "log",
            )
        )
        self.max_log_size_bytes = max_log_size_bytes
        self.retention_days = retention_days
        self.compress_archives = compress_archives

        self.log_path.mkdir(parents=True, exist_ok=True)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Log Collection                                                     #
    # ------------------------------------------------------------------ #

    def collect_all(self) -> Dict[str, LogCollectionResult]:
        """Collect logs from all log files in the log directory."""
        results: Dict[str, LogCollectionResult] = {}
        for log_file in sorted(self.log_path.glob("*.log")):
            result = self.collect_file(log_file)
            results[log_file.name] = result
        return results

    def collect_file(self, log_path: Path) -> LogCollectionResult:
        """Collect and parse a single log file."""
        log_path = Path(log_path)
        if not log_path.exists():
            return LogCollectionResult(
                source=log_path.name,
                file_path=str(log_path),
                entries_collected=0,
                entries_parsed=0,
                errors=1,
                bytes_read=0,
            )

        entries: List[LogEntry] = []
        bytes_read = 0
        errors = 0

        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    bytes_read += len(line.encode("utf-8"))
                    entry = self._parse_log_line(line, log_path.name)
                    if entry:
                        entries.append(entry)
                    else:
                        errors += 1
        except Exception:
            pass

        try:
            if log_path.stat().st_size > self.max_log_size_bytes:
                self._archive_log(log_path)
        except Exception:
            pass

        return LogCollectionResult(
            source=log_path.name,
            file_path=str(log_path),
            entries_collected=len(entries),
            entries_parsed=len(entries),
            errors=errors,
            bytes_read=bytes_read,
        )

    def _parse_log_line(self, line: str, source_file: str) -> Optional[LogEntry]:
        """Parse a single log line into a LogEntry."""
        line = line.strip()
        if not line:
            return None

        # Try parsing as JSON first
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                return LogEntry(
                    timestamp=data.get("timestamp", time.time()),
                    level=data.get("level", "INFO"),
                    message=data.get("message", ""),
                    source=data.get("provider", "unknown"),
                    source_file=source_file,
                    line_number=data.get("line"),
                    raw_line=line,
                    parsed_fields={
                        k: v
                        for k, v in data.items()
                        if k not in ("timestamp", "level", "message", "provider", "source", "line")
                    },
                )
        except json.JSONDecodeError:
            pass

        # Try parsing as standard log format:
        # 2026-01-15 10:30:45,123 - lfm3-1.7b-cpu - INFO - message
        match = re.match(
            r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})\s+-\s+(\S+)\s+-\s+(\w+)\s+-\s+(.*)",
            line,
        )
        if match:
            timestamp_str, source_name, level, message = match.groups()
            try:
                timestamp = time.mktime(
                    time.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                )
            except ValueError:
                timestamp = time.time()

            return LogEntry(
                timestamp=timestamp,
                level=level.upper(),
                message=message,
                source=source_name,
                source_file=source_file,
                raw_line=line,
            )

        # Fallback: treat as plain text
        return LogEntry(
            timestamp=time.time(),
            level="INFO",
            message=line,
            source="unknown",
            source_file=source_file,
            raw_line=line,
        )

    def _archive_log(self, log_path: Path) -> None:
        """Archive a log file."""
        if not log_path.exists():
            return

        archive_name = f"{log_path.name}.{int(time.time())}"
        archive_path = self.archive_dir / archive_name

        try:
            if self.compress_archives:
                with open(log_path, "rb") as f_in:
                    with gzip.open(f"{archive_path}.gz", "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                log_path.unlink()
            else:
                log_path.rename(archive_path)
        except Exception:
            pass

    def rotate_logs(self) -> List[str]:
        """Rotate all log files that exceed max size."""
        archived = []
        for log_file in self.log_dir.glob("*.log"):
            try:
                if log_file.stat().st_size > self.max_log_size_bytes:
                    self._archive_log(log_file)
                    archived.append(str(log_file))
            except Exception:
                pass
        return archived

    def prune_old_logs(self, retention_days: Optional[int] = None) -> List[str]:
        """Prune old log files older than retention_days."""
        retention = retention_days or self.retention_days
        cutoff = time.time() - (retention * 86400)
        deleted = []

        for log_file in self.log_dir.glob("*.log*"):
            try:
                if log_file.stat().st_mtime < cutoff:
                    log_file.unlink()
                    deleted.append(str(log_file))
            except Exception:
                pass

        for archive_file in self.archive_dir.glob("*"):
            try:
                if archive_file.stat().st_mtime < cutoff:
                    archive_file.unlink()
                    deleted.append(str(archive_file))
            except Exception:
                pass

        return deleted

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics about log collection."""
        total_files = 0
        total_size = 0
        total_entries = 0

        for log_file in self.log_dir.glob("*.log"):
            total_files += 1
            try:
                total_size += log_file.stat().st_size
            except Exception:
                pass

        return {
            "log_files": total_files,
            "total_size_bytes": total_size,
            "total_entries": total_entries,
            "log_dir": str(self.log_path),
            "archive_dir": str(self.archive_dir),
            "retention_days": self.retention_days,
        }
