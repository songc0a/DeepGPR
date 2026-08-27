#!/usr/bin/env python3
"""Build the static website statistics snapshot from PyPI and GitHub APIs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


LOGGER = logging.getLogger("deepgpr.website_stats")
REPOSITORY_FALLBACK = "songc0a/DeepGPR"
PACKAGE_DEFAULT = "DeepGPR"
TIMEOUT_SECONDS = 20
USER_AGENT = "DeepGPR-website-stats/1.0 (+https://github.com/songc0a/DeepGPR)"
REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class StatsError(RuntimeError):
    """A recoverable statistics source or validation error."""


def utc_now() -> str:
    """Return an ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def default_snapshot(package: str, repository: str) -> dict[str, Any]:
    """Return a valid snapshot with unavailable values represented by null."""
    return {
        "schema_version": 2,
        "generated_at": None,
        "pypi": {
            "package": package,
            "latest_version": None,
            "downloads": {
                "last_day": None,
                "last_week": None,
                "last_month": None,
            },
        },
        "github": {
            "repository": repository,
            "stars": None,
            "forks": None,
            "open_issues": None,
            "default_branch": None,
            "updated_at": None,
        },
        "sources": {
            "pypi_recent": "unavailable",
            "pypi_metadata": "unavailable",
            "github_repository": "unavailable",
        },
    }


def require_optional_nonnegative_integer(value: Any, field: str) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise StatsError(f"{field} must be a non-negative integer or null")


def validate_snapshot(data: Any) -> None:
    """Validate the stable fields consumed by the website."""
    if not isinstance(data, dict):
        raise StatsError("snapshot root must be an object")
    if data.get("schema_version") != 2:
        raise StatsError("schema_version must be 2")
    if data.get("generated_at") is not None and not isinstance(data.get("generated_at"), str):
        raise StatsError("generated_at must be a string or null")

    pypi = data.get("pypi")
    github = data.get("github")
    sources = data.get("sources")
    if not isinstance(pypi, dict) or not isinstance(github, dict) or not isinstance(sources, dict):
        raise StatsError("pypi, github, and sources must be objects")
    if not isinstance(pypi.get("package"), str) or not pypi["package"]:
        raise StatsError("pypi.package must be a non-empty string")
    if pypi.get("latest_version") is not None and not isinstance(pypi.get("latest_version"), str):
        raise StatsError("pypi.latest_version must be a string or null")

    downloads = pypi.get("downloads")
    if not isinstance(downloads, dict):
        raise StatsError("pypi.downloads must be an object")
    for key in ("last_day", "last_week", "last_month"):
        require_optional_nonnegative_integer(downloads.get(key), f"pypi.downloads.{key}")

    repository = github.get("repository")
    if not isinstance(repository, str) or not REPOSITORY_PATTERN.fullmatch(repository):
        raise StatsError("github.repository must have owner/name form")
    for key in ("stars", "forks", "open_issues"):
        require_optional_nonnegative_integer(github.get(key), f"github.{key}")
    for key in ("default_branch", "updated_at"):
        if github.get(key) is not None and not isinstance(github.get(key), str):
            raise StatsError(f"github.{key} must be a string or null")

    expected_sources = {"pypi_recent", "pypi_metadata", "github_repository"}
    if not expected_sources.issubset(sources):
        raise StatsError("sources is missing one or more required status fields")
    if any(not isinstance(sources[key], str) for key in expected_sources):
        raise StatsError("source status values must be strings")


def get_json(url: str, token: str | None = None) -> Any:
    """Fetch and decode one JSON resource with bounded network time."""
    headers = {
        "Accept": "application/vnd.github+json, application/json",
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return json.loads(response.read().decode(charset))
    except HTTPError as exc:
        raise StatsError(f"HTTP {exc.code} from {urlparse(url).netloc}") from exc
    except URLError as exc:
        raise StatsError(f"network error from {urlparse(url).netloc}: {exc.reason}") from exc
    except (TimeoutError, json.JSONDecodeError) as exc:
        raise StatsError(f"invalid or timed-out response from {urlparse(url).netloc}") from exc


def parse_remote_repository(remote: str) -> str | None:
    """Extract owner/name from GitHub SSH or HTTPS remote syntax."""
    remote = remote.strip()
    if not remote:
        return None
    if remote.startswith("git@github.com:"):
        candidate = remote.split(":", 1)[1]
    else:
        parsed = urlparse(remote)
        if parsed.hostname not in {"github.com", "www.github.com"}:
            return None
        candidate = parsed.path.lstrip("/")
    if candidate.endswith(".git"):
        candidate = candidate[:-4]
    return candidate if REPOSITORY_PATTERN.fullmatch(candidate) else None


def repository_from_git() -> str | None:
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_remote_repository(result.stdout)


def detect_repository() -> str:
    """Resolve repository from Actions, git origin, then the documented fallback."""
    environment_repository = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if REPOSITORY_PATTERN.fullmatch(environment_repository):
        return environment_repository
    git_repository = repository_from_git()
    return git_repository or REPOSITORY_FALLBACK


def load_existing(path: Path, package: str, repository: str) -> dict[str, Any]:
    """Load a previous valid snapshot so temporary source failures retain known data."""
    if not path.is_file():
        return default_snapshot(package, repository)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("schema_version") == 1:
            migrated = default_snapshot(package, repository)
            migrated["generated_at"] = data.get("generated_at")
            legacy_pypi = data.get("pypi", {})
            legacy_github = data.get("github", {})
            legacy_sources = data.get("sources", {})
            migrated["pypi"] = {
                "package": legacy_pypi.get("package"),
                "latest_version": legacy_pypi.get("latest_version"),
                "downloads": legacy_pypi.get("downloads"),
            }
            migrated["github"] = {
                key: legacy_github.get(key)
                for key in (
                    "repository",
                    "stars",
                    "forks",
                    "open_issues",
                    "default_branch",
                    "updated_at",
                )
            }
            migrated["sources"] = {
                key: legacy_sources.get(key, "unavailable")
                for key in ("pypi_recent", "pypi_metadata", "github_repository")
            }
            validate_snapshot(migrated)
            LOGGER.info("Migrated statistics snapshot from schema 1 to schema 2")
            return migrated
        validate_snapshot(data)
        return data
    except (AttributeError, OSError, TypeError, json.JSONDecodeError, StatsError) as exc:
        LOGGER.warning("Ignoring invalid previous snapshot: %s", exc)
        return default_snapshot(package, repository)


def update_pypi_recent(snapshot: dict[str, Any], package: str) -> None:
    normalized_package = re.sub(r"[-_.]+", "-", package).lower()
    payload = get_json(f"https://pypistats.org/api/packages/{normalized_package}/recent")
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        raise StatsError("PyPI Stats response is missing data")
    values = {key: data.get(key) for key in ("last_day", "last_week", "last_month")}
    for key, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise StatsError(f"PyPI Stats returned an invalid {key}")
    snapshot["pypi"]["downloads"] = values


def update_pypi_metadata(snapshot: dict[str, Any], package: str) -> None:
    payload = get_json(f"https://pypi.org/pypi/{package}/json")
    info = payload.get("info") if isinstance(payload, dict) else None
    version = info.get("version") if isinstance(info, dict) else None
    if not isinstance(version, str) or not version:
        raise StatsError("PyPI metadata response is missing a version")
    snapshot["pypi"]["latest_version"] = version


def update_github_repository(snapshot: dict[str, Any], repository: str, token: str | None) -> None:
    payload = get_json(f"https://api.github.com/repos/{repository}", token)
    if not isinstance(payload, dict):
        raise StatsError("GitHub repository response is not an object")
    mapped = {
        "stars": payload.get("stargazers_count"),
        "forks": payload.get("forks_count"),
        "open_issues": payload.get("open_issues_count"),
    }
    for key, value in mapped.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise StatsError(f"GitHub repository response has invalid {key}")
    full_name = payload.get("full_name")
    if isinstance(full_name, str) and REPOSITORY_PATTERN.fullmatch(full_name):
        snapshot["github"]["repository"] = full_name
    snapshot["github"].update(mapped)
    for target, source in (("default_branch", "default_branch"), ("updated_at", "updated_at")):
        value = payload.get(source)
        snapshot["github"][target] = value if isinstance(value, str) and value else None


def run_source(name: str, snapshot: dict[str, Any], action: Any) -> None:
    try:
        action()
    except StatsError as exc:
        snapshot["sources"][name] = f"error: {exc}"
        LOGGER.warning("%s unavailable: %s", name, exc)
    else:
        snapshot["sources"][name] = "ok"
        LOGGER.info("%s updated", name)


def write_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
        handle.write(payload)
    temporary_path.replace(path)


def build_snapshot(path: Path, package: str) -> dict[str, Any]:
    repository = detect_repository()
    token = os.environ.get("GITHUB_TOKEN") or None
    snapshot = deepcopy(load_existing(path, package, repository))
    empty = default_snapshot(package, repository)
    if snapshot["pypi"]["package"].lower() != package.lower():
        snapshot["pypi"] = empty["pypi"]
    if snapshot["github"]["repository"].lower() != repository.lower():
        snapshot["github"] = empty["github"]
    snapshot["schema_version"] = 2
    snapshot["generated_at"] = utc_now()
    snapshot["pypi"]["package"] = package
    snapshot["github"]["repository"] = repository

    run_source("pypi_recent", snapshot, lambda: update_pypi_recent(snapshot, package))
    run_source("pypi_metadata", snapshot, lambda: update_pypi_metadata(snapshot, package))
    run_source(
        "github_repository", snapshot, lambda: update_github_repository(snapshot, repository, token)
    )
    validate_snapshot(snapshot)
    return snapshot


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default=PACKAGE_DEFAULT, help="PyPI package name")
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "website" / "data" / "stats.json",
        help="Snapshot path",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing snapshot without network access",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    output = args.output.resolve()
    if args.validate_only:
        try:
            validate_snapshot(json.loads(output.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, StatsError) as exc:
            LOGGER.error("Snapshot validation failed: %s", exc)
            return 1
        LOGGER.info("Snapshot is valid: %s", output)
        return 0

    snapshot = build_snapshot(output, args.package)
    write_snapshot(output, snapshot)
    LOGGER.info("Wrote valid statistics snapshot: %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
