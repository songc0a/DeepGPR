import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import update_website_stats as stats


class PersistentPyPIDownloadHistoryTests(unittest.TestCase):
    def build(self, existing, current, updated_at="2026-08-27T00:00:00Z"):
        return stats.build_cumulative_history(
            "DeepGPR",
            existing,
            current,
            "2026-08-24",
            updated_at,
        )

    def test_first_run_sums_visible_daily_history(self):
        history = self.build(None, {"2026-08-24": 10, "2026-08-25": 20})

        self.assertEqual(history["total_downloads"], 30)
        self.assertEqual(history["first_tracked_date"], "2026-08-24")
        self.assertTrue(history["history_complete"])

    def test_second_run_adds_only_the_new_date(self):
        first = self.build(None, {"2026-08-24": 10, "2026-08-25": 20})
        second = self.build(
            first,
            {"2026-08-24": 10, "2026-08-25": 20, "2026-08-26": 15},
        )

        self.assertEqual(second["total_downloads"], 45)
        self.assertEqual(second["daily"]["2026-08-26"], 15)

    def test_repeated_run_is_idempotent(self):
        daily = {"2026-08-24": 10, "2026-08-25": 20, "2026-08-26": 15}
        first = self.build(None, daily)
        repeated = self.build(first, daily, "2026-08-27T01:00:00Z")

        self.assertEqual(repeated["total_downloads"], 45)
        self.assertEqual(repeated["daily"], first["daily"])

    def test_recent_date_correction_overwrites_instead_of_adding(self):
        first = self.build(
            None,
            {"2026-08-24": 10, "2026-08-25": 20, "2026-08-26": 15},
        )
        corrected = self.build(first, {"2026-08-26": 18})

        self.assertEqual(corrected["daily"]["2026-08-26"], 18)
        self.assertEqual(corrected["total_downloads"], 48)

    def test_dates_outside_current_api_window_are_preserved(self):
        existing = self.build(
            None,
            {"2026-04-15": 4, "2026-04-16": 6},
        )
        updated = self.build(existing, {"2026-08-26": 15})

        self.assertEqual(updated["daily"]["2026-04-15"], 4)
        self.assertEqual(updated["daily"]["2026-04-16"], 6)
        self.assertEqual(updated["daily"]["2026-08-26"], 15)
        self.assertEqual(updated["total_downloads"], 25)

    def test_daily_history_keeps_zero_download_dates(self):
        history = self.build(None, {"2026-08-24": 10, "2026-08-26": 15})

        self.assertEqual(history["daily"]["2026-08-25"], 0)
        self.assertEqual(history["total_downloads"], 25)

    def test_incomplete_window_is_not_marked_as_release_lifetime(self):
        history = stats.build_cumulative_history(
            "DeepGPR",
            None,
            {"2026-08-24": 10},
            "2026-01-01",
            "2026-08-27T00:00:00Z",
        )

        self.assertFalse(history["history_complete"])

    def test_api_failure_retains_persistent_total_in_public_snapshot(self):
        history = self.build(
            None,
            {"2026-08-24": 10, "2026-08-25": 20, "2026-08-26": 15},
        )
        unavailable = stats.StatsError("simulated provider outage")
        with TemporaryDirectory() as directory:
            history_path = Path(directory) / "pypi_cumulative.json"
            output_path = Path(directory) / "stats.json"
            stats.write_json_atomic(history_path, history)
            with (
                patch.object(stats, "fetch_pypi_recent", side_effect=unavailable),
                patch.object(stats, "fetch_pypi_metadata", side_effect=unavailable),
                patch.object(stats, "fetch_pypi_daily_history", side_effect=unavailable),
                patch.object(stats, "fetch_github_repository", side_effect=unavailable),
            ):
                snapshot, updated_history = stats.build_snapshot(
                    output_path,
                    history_path,
                    "DeepGPR",
                    "songc0a/DeepGPR",
                )

        self.assertIsNone(updated_history)
        self.assertEqual(snapshot["pypi"]["downloads"]["total"], 45)
        self.assertTrue(snapshot["sources"]["pypi_history"].startswith("error:"))

    def test_project_defaults_are_explicit(self):
        self.assertEqual(stats.PACKAGE_DEFAULT, "DeepGPR")
        self.assertEqual(stats.PROJECT_REPOSITORY_DEFAULT, "songc0a/DeepGPR")


if __name__ == "__main__":
    unittest.main()
