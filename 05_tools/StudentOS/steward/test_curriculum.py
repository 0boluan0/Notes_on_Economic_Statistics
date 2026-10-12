import datetime as dt
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from zoneinfo import ZoneInfo

import curriculum as c


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vault = Path(self.tmp.name)
        self.current = dt.datetime(2026, 10, 8, 10, tzinfo=ZoneInfo("Europe/London"))
        self.write(c.TIMETABLE, {"registered_courses": ["ST445", "ST456"], "term_starts": {"AT": "2026-09-28", "WT": "2027-01-11"},
            "term_ends": {"AT": "2026-12-11", "WT": "2027-03-25"}, "series": [
                {"course": "ST445", "term": "AT", "title": "Data"}, {"course": "ST456", "term": "WT", "title": "Deep learning"}]})
        self.write(c.EVENTS, [{"course": "ST445", "start_iso": "2026-10-06T11:00:00+01:00", "end_iso": "2026-10-06T13:00:00+01:00"},
            {"course": "ST456", "date": "2027-01-12", "start_iso": "2027-01-12T16:00:00+00:00", "end_iso": "2027-01-12T18:00:00+00:00"}])
        self.write(c.POSITION, {"courses": [{"id": "ST445", "school": {"summary": "Week 2 materials released", "valid_until": "2026-10-11"}}]})
        self.state = {"tasks": [{"id": "t1", "title": "ST445｜Week 1 exercise", "path": "plan.md", "status": "completion_unknown", "ref": "plan.md#^one"}]}

    def write(self, path, value):
        target = self.vault / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value) if not isinstance(value, str) else value)

    def tearDown(self):
        self.tmp.cleanup()

    def view(self, reports=None):
        return c.overview(self.vault, self.state, reports or [], self.current)

    def test_school_schedule_and_materials_never_imply_learning(self):
        view = self.view()
        self.assertEqual(view["week"], 2)
        self.assertEqual(view["term_weeks"], 11)
        course = view["courses"][0]
        self.assertIsNotNone(course["schedule"]["last"])
        self.assertEqual(course["personal"]["summary"], "尚无已记录学习进度")
        self.assertEqual(view["courses"][1]["group"], "future")
        self.current += dt.timedelta(days=7)
        self.assertTrue(self.view()["courses"][0]["school"]["historical"])

    def test_latest_handoff_updates_only_its_course(self):
        self.write(c.POSITION, {"courses": [{"id": "ST445", "personal": {
            "summary": "Old position", "display": {"week": 1, "label": "Week 1", "units": ["Old unit"]}}}]})
        report = {"task_id": "t1", "created_at": "2026-10-08T09:00:00Z", "export_path": "private/report.md",
                  "payload": {"task_title": "ST445｜Week 1 exercise", "task_path": "plan.md", "summary": "Stopped at list sorting", "next_step": "Trace the return value"}}
        courses = self.view([report])["courses"]
        self.assertEqual(courses[0]["personal"]["summary"], "Stopped at list sorting")
        self.assertNotIn("display", courses[0]["personal"], "A new report must not inherit an old visual position or completion badge")
        self.assertEqual(courses[0]["next_task"]["id"], "t1")
        self.assertEqual(courses[1]["personal"]["summary"], "尚无已记录学习进度")
        self.assertEqual(self.state["tasks"][0]["status"], "completion_unknown")

    def test_changed_or_missing_evidence_is_not_silently_current(self):
        self.write("evidence.md", "Original observation")
        self.write(c.POSITION, {"courses": [{"id": "ST445", "personal": {"summary": "Last observation", "as_of": "2026-10-06", "sources": [
            {"path": "evidence.md", "sha256": hashlib.sha256(b"Original observation").hexdigest()}]}}]})
        self.assertFalse(self.view()["courses"][0]["personal"]["needs_review"])
        self.write("evidence.md", "Updated observation")
        self.assertTrue(self.view()["courses"][0]["personal"]["needs_review"])
        (self.vault / "evidence.md").unlink()
        self.assertEqual(self.view()["courses"][0]["personal"]["sources"][0]["status"], "missing")

    def test_finite_course_reads_original_checkboxes_and_keeps_gaps(self):
        self.write(c.POSITION, {"courses": [{"id": "Finite", "plan": "finite.md", "mode": "finite"}]})
        self.state["tasks"].extend({"id": str(i), "title": f"Lecture {i}", "path": "finite.md", "status": status} for i, status in enumerate(
            ["recorded_complete", "completion_unknown", "recorded_complete"], 1))
        course = self.view()["courses"][-1]
        self.assertIn("Lecture 1", course["personal"]["summary"])
        self.assertEqual(course["next_task"]["id"], "2")
        self.assertEqual(course["unit_progress"], {"completed": 2, "total": 3})
        self.state["tasks"] = []
        self.assertEqual(self.view()["courses"][-1]["next_step"], "原计划未读到可核验任务")

    def test_section_evidence_ignores_unrelated_edits_but_detects_relevant_changes(self):
        self.write("evidence.md", "# Record\n## Course\nActual stop\n## Other\nOld text\n")
        ref = {"path": "evidence.md#Course", "sha256": hashlib.sha256(c.source_bytes(self.vault, "evidence.md#Course")).hexdigest()}
        self.write("evidence.md", "# Record\n## Course\nActual stop\n## Other\nNew text\n")
        self.assertEqual(c.references(self.vault, [ref])[0]["status"], "unchanged")
        self.write("evidence.md", "# Record\n## Course\nNew stop\n## Other\nNew text\n")
        self.assertEqual(c.references(self.vault, [ref])[0]["status"], "changed")
        self.write("evidence.md", "# Record\n## Other\nNew text\n")
        self.assertEqual(c.references(self.vault, [ref])[0]["status"], "section_missing_or_ambiguous")
        self.assertEqual(c.references(self.vault, [{"path": "../outside.md#Course"}])[0]["status"], "missing")

    def test_unavailable_inputs_keep_registered_courses_visible(self):
        (self.vault / c.POSITION).write_text("broken JSON")
        view = self.view()
        self.assertEqual(len(view["courses"]), 2)
        self.assertEqual(view["issues"][0]["path"], c.POSITION)


if __name__ == "__main__":
    unittest.main()
