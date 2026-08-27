#!/usr/bin/env python3
"""Build the static website statistics and persistent PyPI download history."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


LOGGER = logging.getLogger("deepgpr.website_stats")
PACKAGE_DEFAULT = "DeepGPR"
PROJECT_REPOSITORY_DEFAULT = "songc0a/DeepGPR"
SNAPSHOT_SCHEMA_VERSION = 3
HISTORY_SCHEMA_VERSION = 1
TRACKING_METHOD = "persistent_daily_history"
TIMEOUT_SECONDS = 30
USER_AGENT = "DeepGPR-website-stats/2.0 (+https://github.com/songc0a/DeepGPR)"
REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class StatsError(RuntimeError):
    """A statistics source, state, or validation error."""


def utc_now() -> str:
    """Return an ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalized_package_name(package: str) -> str:
    return re.sub(r"[-_.]+", "-", package).lower()


def require_optional_nonnegative_integer(value: Any, field: str) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise StatsError(f"{field} must be a non-negative integer or null")


def validate_date(value: Any, field: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str):
        raise StatsError(f"{field} must be an ISO date string")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise StatsError(f"{field} must be an ISO date string") from exc


def default_cumulative_public() -> dict[str, Any]:
    return {
        "total": None,
        "first_tracked_date": None,
        "last_tracked_date": None,
        "history_complete": False,
        "tracking_method": TRACKING_METHOD,
    }


def default_snapshot(package: str, repository: str) -> dict[str, Any]:
    """Return a valid public snapshot with unavailable values represented by null."""
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "generated_at": None,
        "pypi": {
            "package": package,
            "latest_version": None,
            "downloads": {
                "last_day": None,
                "last_week": None,
                "last_month": None,
                "total": None,
            },
            "cumulative": default_cumulative_public(),
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
            "pypi_history": "unavailable",
            "github_repository": "unavailable",
        },
    }


def validate_snapshot(data: Any) -> None:
    """Validate the stable public fields consumed by the website."""
    if not isinstance(data, dict):
        raise StatsError("snapshot root must be an object")
    if data.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise StatsError(f"schema_version must be {SNAPSHOT_SCHEMA_VERSION}")
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
    cumulative = pypi.get("cumulative")
    if not isinstance(downloads, dict) or not isinstance(cumulative, dict):
        raise StatsError("pypi.downloads and pypi.cumulative must be objects")
    for key in ("last_day", "last_week", "last_month", "total"):
        require_optional_nonnegative_integer(downloads.get(key), f"pypi.downloads.{key}")
    require_optional_nonnegative_integer(cumulative.get("total"), "pypi.cumulative.total")
    if downloads.get("total") != cumulative.get("total"):
        raise StatsError("public cumulative totals must match")
    for key in ("first_tracked_date", "last_tracked_date"):
        validate_date(cumulative.get(key), f"pypi.cumulative.{key}", optional=True)
    if not isinstance(cumulative.get("history_complete"), bool):
        raise StatsError("pypi.cumulative.history_complete must be a boolean")
    if cumulative.get("tracking_method") != TRACKING_METHOD:
        raise StatsError(f"pypi.cumulative.tracking_method must be {TRACKING_METHOD!r}")
    if cumulative.get("total") is None:
        if cumulative.get("first_tracked_date") is not None or cumulative.get("last_tracked_date") is not None:
            raise StatsError("unavailable cumulative totals cannot have tracked dates")
    elif cumulative.get("first_tracked_date") is None or cumulative.get("last_tracked_date") is None:
        raise StatsError("available cumulative totals require tracked dates")

    repository = github.get("repository")
    if not isinstance(repository, str) or not REPOSITORY_PATTERN.fullmatch(repository):
        raise StatsError("github.repository must have owner/name form")
    for key in ("stars", "forks", "open_issues"):
        require_optional_nonnegative_integer(github.get(key), f"github.{key}")
    for key in ("default_branch", "updated_at"):
        if github.get(key) is not None and not isinstance(github.get(key), str):
            raise StatsError(f"github.{key} must be a string or null")

    expected_sources = {
        "pypi_recent",
        "pypi_metadata",
        "pypi_history",
        "github_repository",
    }
    if not expected_sources.issubset(sources):
        raise StatsError("sources is missing one or more required status fields")
    if any(not isinstance(sources[key], str) for key in expected_sources):
        raise StatsError("source status values must be strings")


def validate_cumulative_history(data: Any, package: str | None = None) -> None:
    """Validate the internal persistent daily history."""
    if not isinstance(data, dict):
        raise StatsError("cumulative history root must be an object")
    if data.get("schema_version") != HISTORY_SCHEMA_VERSION:
        raise StatsError(f"history schema_version must be {HISTORY_SCHEMA_VERSION}")
    history_package = data.get("package")
    if not isinstance(history_package, str) or not history_package:
        raise StatsError("history package must be a non-empty string")
    if package is not None and normalized_package_name(history_package) != normalized_package_name(package):
        raise StatsError(f"history package {history_package!r} does not match {package!r}")
    if data.get("tracking_method") != TRACKING_METHOD:
        raise StatsError(f"history tracking_method must be {TRACKING_METHOD!r}")
    if data.get("mirrors_included") is not False:
        raise StatsError("history mirrors_included must be false")
    if not isinstance(data.get("history_complete"), bool):
        raise StatsError("history_complete must be a boolean")
    if not isinstance(data.get("updated_at"), str) or not data["updated_at"]:
        raise StatsError("history updated_at must be a non-empty string")
    validate_date(data.get("package_first_release_date"), "package_first_release_date", optional=True)

    daily = data.get("daily")
    if not isinstance(daily, dict):
        raise StatsError("history daily must be an object")
    for tracked_date, downloads in daily.items():
        validate_date(tracked_date, f"daily date {tracked_date!r}")
        require_optional_nonnegative_integer(downloads, f"daily[{tracked_date!r}]")
        if downloads is None:
            raise StatsError(f"daily[{tracked_date!r}] cannot be null")
    ordered_dates = sorted(daily)
    expected_first = ordered_dates[0] if ordered_dates else None
    expected_last = ordered_dates[-1] if ordered_dates else None
    if data.get("first_tracked_date") != expected_first:
        raise StatsError("first_tracked_date does not match daily history")
    if data.get("last_tracked_date") != expected_last:
        raise StatsError("last_tracked_date does not match daily history")
    total = data.get("total_downloads")
    require_optional_nonnegative_integer(total, "total_downloads")
    if total != sum(daily.values()):
        raise StatsError("total_downloads does not equal the daily history sum")


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


def fetch_pypi_recent(package: str) -> dict[str, int]:
    payload = get_json(
        f"https://pypistats.org/api/packages/{normalized_package_name(package)}/recent"
    )
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        raise StatsError("PyPI Stats recent response is missing data")
    values = {key: data.get(key) for key in ("last_day", "last_week", "last_month")}
    for key, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise StatsError(f"PyPI Stats returned an invalid {key}")
    return values


def first_release_date_from_metadata(payload: Any) -> str | None:
    releases = payload.get("releases") if isinstance(payload, dict) else None
    if not isinstance(releases, dict):
        return None
    upload_dates: list[str] = []
    for files in releases.values():
        if not isinstance(files, list):
            continue
        for file_metadata in files:
            if not isinstance(file_metadata, dict):
                continue
            uploaded = file_metadata.get("upload_time_iso_8601") or file_metadata.get("upload_time")
            if not isinstance(uploaded, str):
                continue
            try:
                parsed = datetime.fromisoformat(uploaded.replace("Z", "+00:00"))
            except ValueError:
                continue
            upload_dates.append(parsed.date().isoformat())
    return min(upload_dates) if upload_dates else None


def fetch_pypi_metadata(package: str) -> tuple[str, str | None]:
    payload = get_json(f"https://pypi.org/pypi/{package}/json")
    info = payload.get("info") if isinstance(payload, dict) else None
    version = info.get("version") if isinstance(info, dict) else None
    if not isinstance(version, str) or not version:
        raise StatsError("PyPI metadata response is missing a version")
    return version, first_release_date_from_metadata(payload)


def parse_pypi_daily_history(payload: Any) -> dict[str, int]:
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise StatsError("PyPI Stats overall response is missing its daily data")
    daily: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise StatsError("PyPI Stats overall response contains an invalid row")
        if row.get("category") != "without_mirrors":
            continue
        tracked_date = row.get("date")
        downloads = row.get("downloads")
        validate_date(tracked_date, "PyPI Stats daily date")
        if not isinstance(downloads, int) or isinstance(downloads, bool) or downloads < 0:
            raise StatsError(f"PyPI Stats returned invalid downloads for {tracked_date}")
        if tracked_date in daily:
            raise StatsError(f"PyPI Stats returned duplicate data for {tracked_date}")
        daily[tracked_date] = downloads
    if not daily:
        raise StatsError("PyPI Stats returned no without-mirror daily history")
    return complete_daily_range(daily)


def fetch_pypi_daily_history(package: str) -> dict[str, int]:
    query = urlencode({"mirrors": "false"})
    payload = get_json(
        f"https://pypistats.org/api/packages/{normalized_package_name(package)}/overall?{query}"
    )
    return parse_pypi_daily_history(payload)


def fetch_github_repository(repository: str, token: str | None) -> dict[str, Any]:
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
    if not isinstance(full_name, str) or not REPOSITORY_PATTERN.fullmatch(full_name):
        raise StatsError("GitHub repository response has an invalid full_name")
    return {
        "repository": full_name,
        **mapped,
        "default_branch": payload.get("default_branch") if isinstance(payload.get("default_branch"), str) else None,
        "updated_at": payload.get("updated_at") if isinstance(payload.get("updated_at"), str) else None,
    }


def load_cumulative_history(path: Path, package: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StatsError(f"cannot read cumulative history: {exc}") from exc
    validate_cumulative_history(data, package)
    return data


def complete_daily_range(daily: dict[str, int]) -> dict[str, int]:
    """Represent every calendar date between the first and last observation."""
    if not daily:
        return {}
    first = date.fromisoformat(min(daily))
    last = date.fromisoformat(max(daily))
    complete: dict[str, int] = {}
    tracked_date = first
    while tracked_date <= last:
        key = tracked_date.isoformat()
        complete[key] = daily.get(key, 0)
        tracked_date += timedelta(days=1)
    return complete


def merge_daily_history(existing: dict[str, int], current: dict[str, int]) -> dict[str, int]:
    """Merge current API data by date, preserving dates outside the API window."""
    merged = dict(existing)
    merged.update(current)
    return complete_daily_range(merged)


def calculate_cumulative_total(daily: dict[str, int]) -> int:
    return sum(daily.values())


def build_cumulative_history(
    package: str,
    existing: dict[str, Any] | None,
    current_daily: dict[str, int],
    package_first_release_date: str | None,
    updated_at: str,
) -> dict[str, Any]:
    """Create idempotent cumulative state from persistent and currently visible daily data."""
    existing_daily = existing["daily"] if existing is not None else {}
    merged = merge_daily_history(existing_daily, current_daily)
    if not merged:
        raise StatsError("cannot build cumulative history without daily data")
    first_tracked_date = min(merged)
    last_tracked_date = max(merged)
    known_release_date = package_first_release_date
    if known_release_date is None and existing is not None:
        known_release_date = existing.get("package_first_release_date")

    if existing is None:
        history_complete = bool(
            known_release_date is not None and first_tracked_date <= known_release_date
        )
    else:
        history_complete = existing["history_complete"]
        if known_release_date is None or first_tracked_date > known_release_date:
            history_complete = False

    history = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "package": package,
        "total_downloads": calculate_cumulative_total(merged),
        "first_tracked_date": first_tracked_date,
        "last_tracked_date": last_tracked_date,
        "package_first_release_date": known_release_date,
        "history_complete": history_complete,
        "tracking_method": TRACKING_METHOD,
        "mirrors_included": False,
        "daily": merged,
        "updated_at": updated_at,
    }
    validate_cumulative_history(history, package)
    return history


def apply_cumulative_to_snapshot(snapshot: dict[str, Any], history: dict[str, Any]) -> None:
    total = history["total_downloads"]
    snapshot["pypi"]["downloads"]["total"] = total
    snapshot["pypi"]["cumulative"] = {
        "total": total,
        "first_tracked_date": history["first_tracked_date"],
        "last_tracked_date": history["last_tracked_date"],
        "history_complete": history["history_complete"],
        "tracking_method": history["tracking_method"],
    }


def migrate_snapshot(data: Any, package: str, repository: str) -> dict[str, Any]:
    """Migrate valid-looking schema 1/2 snapshots without retaining removed release fields."""
    if not isinstance(data, dict) or data.get("schema_version") not in {1, 2}:
        raise StatsError("unsupported previous snapshot schema")
    pypi = data.get("pypi")
    github = data.get("github")
    sources = data.get("sources")
    if not isinstance(pypi, dict) or not isinstance(github, dict) or not isinstance(sources, dict):
        raise StatsError("previous snapshot is missing required objects")
    downloads = pypi.get("downloads")
    if not isinstance(downloads, dict):
        raise StatsError("previous snapshot downloads must be an object")
    migrated = default_snapshot(package, repository)
    migrated["generated_at"] = data.get("generated_at")
    migrated["pypi"]["package"] = pypi.get("package")
    migrated["pypi"]["latest_version"] = pypi.get("latest_version")
    for key in ("last_day", "last_week", "last_month"):
        migrated["pypi"]["downloads"][key] = downloads.get(key)
    for key in migrated["github"]:
        migrated["github"][key] = github.get(key)
    for key in migrated["sources"]:
        if key in sources:
            migrated["sources"][key] = sources[key]
    validate_snapshot(migrated)
    return migrated


def load_existing_snapshot(path: Path, package: str, repository: str) -> dict[str, Any]:
    if not path.is_file():
        return default_snapshot(package, repository)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("schema_version") in {1, 2}:
            return migrate_snapshot(data, package, repository)
        validate_snapshot(data)
        return data
    except (OSError, json.JSONDecodeError, StatsError) as exc:
        LOGGER.warning("Ignoring invalid previous public snapshot: %s", exc)
        return default_snapshot(package, repository)


def run_source(
    name: str,
    snapshot: dict[str, Any],
    action: Callable[[], Any],
) -> tuple[bool, Any]:
    try:
        result = action()
    except StatsError as exc:
        snapshot["sources"][name] = f"error: {exc}"
        LOGGER.warning("%s unavailable: %s", name, exc)
        return False, None
    snapshot["sources"][name] = "ok"
    LOGGER.info("%s updated", name)
    return True, result


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
        handle.write(payload)
    temporary_path.replace(path)


def build_snapshot(
    output_path: Path,
    history_path: Path,
    package: str,
    repository: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not REPOSITORY_PATTERN.fullmatch(repository):
        raise StatsError("repository must have owner/name form")
    token = os.environ.get("GITHUB_TOKEN") or None
    snapshot = deepcopy(load_existing_snapshot(output_path, package, repository))
    empty = default_snapshot(package, repository)
    if normalized_package_name(snapshot["pypi"]["package"]) != normalized_package_name(package):
        snapshot["pypi"] = empty["pypi"]
    if snapshot["github"]["repository"].lower() != repository.lower():
        snapshot["github"] = empty["github"]
    snapshot["schema_version"] = SNAPSHOT_SCHEMA_VERSION
    snapshot["generated_at"] = utc_now()
    snapshot["pypi"]["package"] = package
    snapshot["github"]["repository"] = repository

    recent_ok, recent = run_source("pypi_recent", snapshot, lambda: fetch_pypi_recent(package))
    if recent_ok:
        snapshot["pypi"]["downloads"].update(recent)

    metadata_ok, metadata = run_source("pypi_metadata", snapshot, lambda: fetch_pypi_metadata(package))
    package_first_release_date = None
    if metadata_ok:
        version, package_first_release_date = metadata
        snapshot["pypi"]["latest_version"] = version

    existing_history = load_cumulative_history(history_path, package)
    history_ok, current_daily = run_source(
        "pypi_history", snapshot, lambda: fetch_pypi_daily_history(package)
    )
    updated_history = None
    if history_ok:
        updated_history = build_cumulative_history(
            package,
            existing_history,
            current_daily,
            package_first_release_date,
            snapshot["generated_at"],
        )
        apply_cumulative_to_snapshot(snapshot, updated_history)
    elif existing_history is not None:
        apply_cumulative_to_snapshot(snapshot, existing_history)

    github_ok, github = run_source(
        "github_repository", snapshot, lambda: fetch_github_repository(repository, token)
    )
    if github_ok:
        snapshot["github"].update(github)

    validate_snapshot(snapshot)
    return snapshot, updated_history


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    package_default = os.environ.get("PYPI_PACKAGE", PACKAGE_DEFAULT)
    project_repository_default = os.environ.get(
        "PROJECT_REPOSITORY", PROJECT_REPOSITORY_DEFAULT
    )
    history_default = Path(
        os.environ.get(
            "PYPI_HISTORY_PATH",
            repository_root / ".stats-history" / "pypi_cumulative.json",
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default=package_default, help="PyPI package name")
    parser.add_argument(
        "--repository",
        default=project_repository_default,
        help="GitHub project in owner/name form",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=history_default,
        help="Persistent cumulative history path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "website" / "data" / "stats.json",
        help="Public snapshot path",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing JSON without network access",
    )
    parser.add_argument(
        "--require-history",
        action="store_true",
        help="Require and validate the persistent history during validation",
    )
    return parser.parse_args()


def log_summary(snapshot: dict[str, Any]) -> None:
    downloads = snapshot["pypi"]["downloads"]
    cumulative = snapshot["pypi"]["cumulative"]
    LOGGER.info("PyPI last day: %s", downloads["last_day"])
    LOGGER.info("PyPI last week: %s", downloads["last_week"])
    LOGGER.info("PyPI last month: %s", downloads["last_month"])
    LOGGER.info("PyPI cumulative total: %s", downloads["total"])
    LOGGER.info("First tracked date: %s", cumulative["first_tracked_date"])
    LOGGER.info("Last tracked date: %s", cumulative["last_tracked_date"])
    LOGGER.info("History complete: %s", cumulative["history_complete"])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    output = args.output.resolve()
    history_path = args.history.resolve()
    if args.validate_only:
        try:
            validate_snapshot(json.loads(output.read_text(encoding="utf-8")))
            if args.require_history:
                history = load_cumulative_history(history_path, args.package)
                if history is None:
                    raise StatsError(f"required cumulative history is missing: {history_path}")
        except (OSError, json.JSONDecodeError, StatsError) as exc:
            LOGGER.error("Statistics validation failed: %s", exc)
            return 1
        LOGGER.info("Statistics JSON is valid")
        return 0

    try:
        snapshot, updated_history = build_snapshot(
            output,
            history_path,
            args.package,
            args.repository,
        )
        if updated_history is not None:
            write_json_atomic(history_path, updated_history)
        write_json_atomic(output, snapshot)
    except StatsError as exc:
        LOGGER.error("Statistics update failed: %s", exc)
        return 1
    log_summary(snapshot)
    LOGGER.info("Wrote valid public statistics snapshot: %s", output)
    if updated_history is not None:
        LOGGER.info("Wrote persistent cumulative history: %s", history_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
