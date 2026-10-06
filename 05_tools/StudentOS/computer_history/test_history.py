from datetime import date, datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from history import collect, day_start, export, report


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.dest = self.root / "private" / "MacBook.json"
        self.now = datetime(2026, 10, 6, 8, tzinfo=timezone.utc)

    def summary(self, name, content="---\ntitle: Exercise\n---\nObserved evidence\n"):
        path = self.source / name
        path.write_text(content)
        return path

    def test_london_day_and_overlapping_summary(self):
        self.summary("2026-10-04T22-30-00-a-6h-memory-summary.md")
        self.summary("2026-10-04T22-40-00-b-10min-memory-summary.md")
        self.summary("2026-10-05T22-55-00-c-10min-memory-summary.md")
        data = collect(self.source, self.now)
        self.assertEqual(len(data["records"]), 2)
        self.assertEqual(len(report(data, date(2026, 10, 6), self.now)["records"]), 1)
        # DST changes the length of a London calendar day, not its boundaries.
        self.assertEqual((day_start(date(2026, 10, 26)) - day_start(date(2026, 10, 25))).total_seconds(), 25 * 3600)

    def test_source_deletion_propagates_without_touching_sources(self):
        p = self.summary("2026-10-05T12-00-00-a-10min-memory-summary.md")
        original = p.read_bytes()
        export(self.source, self.dest, self.now)
        self.assertEqual(p.read_bytes(), original)
        self.assertEqual(self.dest.stat().st_mode & 0o777, 0o600)
        p.unlink()
        export(self.source, self.dest, self.now)
        self.assertEqual(json.loads(self.dest.read_text())["records"], [])

    def test_unavailable_or_unreadable_source_preserves_snapshot(self):
        p = self.summary("2026-10-05T12-00-00-a-10min-memory-summary.md")
        export(self.source, self.dest, self.now)
        original = self.dest.read_bytes()
        with self.assertRaises(FileNotFoundError):
            export(self.root / "missing", self.dest, self.now)
        p.write_bytes(b"\xff")
        with self.assertRaises(UnicodeDecodeError):
            export(self.source, self.dest, self.now)
        self.assertEqual(self.dest.read_bytes(), original)

    def test_staleness_and_missing_records_do_not_imply_inactivity(self):
        data = collect(self.source, self.now)
        result = report(data, date(2026, 10, 5), datetime(2026, 10, 6, 10, tzinfo=timezone.utc))
        self.assertTrue(result["export_stale"])
        self.assertEqual(result["recorder_status"], "not_checked_by_exporter")
        self.assertEqual(result["records"], [])

    def test_no_symlink_read_or_unmanaged_overwrite(self):
        external = self.root / "external.md"
        external.write_text("unrelated")
        (self.source / "2026-10-05T12-00-00-a-10min-memory-summary.md").symlink_to(external)
        self.assertEqual(collect(self.source, self.now)["records"], [])
        self.dest.parent.mkdir()
        self.dest.write_text('{"owner":"someone-else"}')
        with self.assertRaises(ValueError):
            export(self.source, self.dest, self.now)
        self.assertEqual(self.dest.read_text(), '{"owner":"someone-else"}')


if __name__ == "__main__":
    unittest.main()
