"""
Backup and restore for LFM3-1.7B inference server configuration and state.

Provides:
  * Backup of configuration files, TLS certificates, feature flags,
    quality monitor state, and audit logs
  * Restore from backup with integrity verification
  * Backup listing and metadata
  * Backup integrity verification (checksums)
  * Automatic pruning of old backups

Backup format: tar.gz archive with a manifest.json containing:
  * Timestamp, version, file list with SHA-256 checksums
  * Backup type (full or incremental)

All backups stored in /media/scott/data/lfm3-npu-tool/var/backups/
by default, configurable via LFM3_BACKUP_DIR.
"""

import hashlib
import json
import os
import shutil
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

DEFAULT_BACKUP_DIR = Path("/media/scott/data/lfm3-npu-tool/var/backups")
DEFAULT_BACKUP_RETENTION = 30  # days

# Files and directories to include in a full backup
BACKUP_SOURCES: List[str] = [
    "/media/scott/data/lfm3-npu-tool/etc/lfm3-npu-config.yaml",
    "/media/scott/data/lfm3-npu-tool/etc/lfm3-npu-cpu.service",
    "/media/scott/data/lfm3-npu-tool/etc/lfm3-npu-npu.service",
    "/media/scott/data/lfm3-npu-tool/etc/lfm3-npu-metrics.service",
    "/media/scott/data/lfm3-npu-tool/etc/lfm3-npu-quality-monitor.service",
    "/media/scott/data/lfm3-npu-tool/etc/lfm3-npu-watchdog.service",
    "/media/scott/data/lfm3-npu-tool/etc/lfm3-npu-doctor.service",
    "/media/scott/data/lfm3-npu-tool/etc/lfm3-npu-health.service",
    "/media/scott/data/lfm3-npu-tool/etc/lfm3-npu-benchmark.service",
    "/media/scott/data/lfm3-npu-tool/etc/lfm3-npu-startup-validate.service",
    "/media/scott/data/lfm3-npu-tool/etc/logrotate-lfm3-npu.conf",
    "/media/scott/data/lfm3-npu-tool/etc/prometheus-rules-lfm3-npu.yml",
    "/media/scott/data/lfm3-npu-tool/etc/prometheus-lfm3.yml",
    "/media/scott/data/lfm3-npu-tool/etc/alertmanager-lfm3.yml",
    "/media/scott/data/lfm3-npu-tool/etc/grafana-dashboard-lfm3-npu.json",
    "/media/scott/data/lfm3-npu-tool/etc/tls",  # TLS certificates
    "/media/scott/data/lfm3-npu-tool/var/feature_flags.json",  # Feature flags
    "/media/scott/data/lfm3-npu-tool/var/quality_monitor_state.json",  # QM state
    "/media/scott/data/lfm3-npu-tool/var/audit.log",  # Audit log
]


class BackupError(Exception):
    """Raised when backup or restore operations fail."""


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class BackupManager:
    """
    Manages backups and restores for LFM3-1.7B inference server.

    Backups are tar.gz archives with an embedded manifest.json containing
    checksums for integrity verification.
    """

    def __init__(
        self,
        backup_dir: Optional[str] = None,
        retention_days: int = DEFAULT_BACKUP_RETENTION,
    ) -> None:
        self.backup_dir = Path(backup_dir or DEFAULT_BACKUP_DIR)
        self.retention_days = retention_days
        self._lock = threading.RLock()
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups with metadata."""
        with self._lock:
            backups = []
            for backup_file in sorted(self.backup_dir.glob("*.tar.gz"), reverse=True):
                manifest_file = self.backup_dir / (
                    backup_file.stem.replace(".tar", "") + ".manifest.json"
                )
                # Handle both .tar.gz.manifest.json and .tar.manifest.json
                if not manifest_file.exists():
                    alt = self.backup_dir / (backup_file.name + ".manifest.json")
                    if alt.exists():
                        manifest_file = alt
                    else:
                        continue

                try:
                    with open(manifest_file, "r") as f:
                        manifest = json.load(f)
                    backups.append(
                        {
                            "filename": backup_file.name,
                            "path": str(backup_file),
                            "size_bytes": backup_file.stat().st_size,
                            "created_at": manifest.get("created_at", ""),
                            "version": manifest.get("version", "1.0.0"),
                            "file_count": manifest.get("file_count", 0),
                            "total_size": manifest.get("total_size", 0),
                        }
                    )
                except (json.JSONDecodeError, OSError):
                    continue
            return backups

    def create_backup(
        self,
        name: Optional[str] = None,
        sources: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> str:
        """
        Create a full backup.

        Returns the backup filename.
        """
        with self._lock:
            timestamp = datetime.now(timezone.utc)
            ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
            backup_name = name or f"lfm3-backup-{ts_str}"
            backup_file = self.backup_dir / f"{backup_name}.tar.gz"
            manifest_file = self.backup_dir / f"{backup_name}.manifest.json"

            sources = sources or BACKUP_SOURCES

            file_entries: List[Dict[str, Any]] = []
            total_size = 0

            with tarfile.open(backup_file, "w:gz") as tar:
                for source in sources:
                    src_path = Path(source)
                    if not src_path.exists():
                        continue
                    if src_path.is_file():
                        arcname = str(src_path)
                        tar.add(src_path, arcname=arcname)
                        file_entries.append(
                            {
                                "path": str(src_path),
                                "sha256": _sha256_file(src_path),
                                "size": src_path.stat().st_size,
                                "type": "file",
                            }
                        )
                        total_size += src_path.stat().st_size
                    elif src_path.is_dir():
                        for f in src_path.rglob("*"):
                            if f.is_file():
                                arcname = str(f)
                                tar.add(f, arcname=arcname)
                                file_entries.append(
                                    {
                                        "path": str(f),
                                        "sha256": _sha256_file(f),
                                        "size": f.stat().st_size,
                                        "type": "file",
                                    }
                                )
                                total_size += f.stat().st_size

            manifest = {
                "created_at": timestamp.isoformat(),
                "version": "1.0.0",
                "description": description or "",
                "backup_name": backup_name,
                "file_count": len(file_entries),
                "total_size": total_size,
                "files": file_entries,
            }

            with open(manifest_file, "w") as f:
                json.dump(manifest, f, indent=2)

            self._prune_old_backups()

            return backup_name

    def verify_backup(self, backup_name: str) -> Dict[str, Any]:
        """
        Verify the integrity of a backup.

        Returns a dict with 'valid' boolean and details.
        """
        with self._lock:
            backup_file = self.backup_dir / f"{backup_name}.tar.gz"
            manifest_file = self.backup_dir / f"{backup_name}.manifest.json"

            if not backup_file.exists():
                return {"valid": False, "error": "Backup file not found"}
            if not manifest_file.exists():
                return {"valid": False, "error": "Manifest file not found"}

            try:
                with open(manifest_file, "r") as f:
                    manifest = json.load(f)
            except json.JSONDecodeError as e:
                return {"valid": False, "error": f"Invalid manifest: {e}"}

            # Verify tar integrity
            try:
                with tarfile.open(backup_file, "r:gz") as tar:
                    tar_members = tar.getmembers()
            except tarfile.TarError as e:
                return {"valid": False, "error": f"Corrupt archive: {e}"}

            # Verify file checksums
            failed_files = []
            extracted_files: Dict[str, str] = {}  # path -> sha256 of extracted content

            with tarfile.open(backup_file, "r:gz") as tar:
                for entry in manifest.get("files", []):
                    file_path = entry["path"]
                    expected_hash = entry["sha256"]

                    # Find matching member in tar
                    member = None
                    for m in tar.getmembers():
                        if m.name == file_path or m.name == file_path.lstrip("/"):
                            member = m
                            break

                    if member is None:
                        failed_files.append(
                            {"path": file_path, "error": "Not in archive"}
                        )
                        continue

                    # Extract and hash
                    f = tar.extractfile(member)
                    if f is None:
                        failed_files.append(
                            {"path": file_path, "error": "Not a regular file"}
                        )
                        continue

                    content = f.read()
                    actual_hash = hashlib.sha256(content).hexdigest()
                    if actual_hash != expected_hash:
                        failed_files.append(
                            {
                                "path": file_path,
                                "error": "Checksum mismatch",
                                "expected": expected_hash,
                                "actual": actual_hash,
                            }
                        )
                    else:
                        extracted_files[file_path] = actual_hash

            valid = len(failed_files) == 0
            return {
                "valid": valid,
                "backup_name": backup_name,
                "file_count": len(manifest.get("files", [])),
                "verified_files": len(extracted_files),
                "failed_files": failed_files,
                "created_at": manifest.get("created_at", ""),
            }

    def restore_backup(
        self,
        backup_name: str,
        destination: Optional[str] = None,
        verify_first: bool = True,
    ) -> str:
        """
        Restore a backup to a destination directory.

        Returns the destination directory path.
        """
        with self._lock:
            backup_file = self.backup_dir / f"{backup_name}.tar.gz"

            if not backup_file.exists():
                raise BackupError(f"Backup file not found: {backup_file}")

            if verify_first:
                result = self.verify_backup(backup_name)
                if not result["valid"]:
                    raise BackupError(
                        f"Backup verification failed: {result.get('failed_files', [])}"
                    )

            dest_dir = Path(destination or "/")
            dest_dir.mkdir(parents=True, exist_ok=True)

            with tarfile.open(backup_file, "r:gz") as tar:
                tar.extractall(dest_dir, filter="data")

            return str(dest_dir)

    def delete_backup(self, backup_name: str) -> bool:
        """Delete a backup and its manifest."""
        with self._lock:
            backup_file = self.backup_dir / f"{backup_name}.tar.gz"
            manifest_file = self.backup_dir / f"{backup_name}.manifest.json"

            deleted = False
            if backup_file.exists():
                backup_file.unlink()
                deleted = True
            if manifest_file.exists():
                manifest_file.unlink()
                deleted = True
            return deleted

    def _prune_old_backups(self) -> None:
        """Remove backups older than retention_days."""
        if self.retention_days <= 0:
            return

        cutoff = time.time() - (self.retention_days * 86400)
        for backup_file in self.backup_dir.glob("*.tar.gz"):
            try:
                if backup_file.stat().st_mtime < cutoff:
                    name = backup_file.stem.replace(".tar", "")
                    self.delete_backup(name)
            except OSError:
                pass

    def get_backup_info(self, backup_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a specific backup."""
        for b in self.list_backups():
            if b["filename"] == f"{backup_name}.tar.gz":
                return b
        return None


# Global backup manager
_backup_manager: Optional[BackupManager] = None


def get_backup_manager() -> BackupManager:
    """Get the global backup manager (initializes on first use)."""
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupManager()
    return _backup_manager
