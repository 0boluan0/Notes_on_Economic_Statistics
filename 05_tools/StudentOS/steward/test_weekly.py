import copy
import datetime as dt
import unittest
from zoneinfo import ZoneInfo

from weekly import project, task_position


class WeeklyTests(unittest.TestCase):
    def setUp(self):
        self.current = dt.datetime(2026, 10, 9, 12, tzinfo=ZoneInfo("Europe/London"))
        self.start = dt.date(2026, 9, 28)
        self.end = dt.date(2026, 12, 11)

    def test_week_markers_and_lesson_dates_keep_original_placement(self):
        cases = [
            ({"title": "ST445｜Week 1 Lab", "raw": "Week 1 Lab ⏳ 2026-10-09"}, "1", "原任务 Week 1"),
            ({"title": "ST447｜2026-10-02 大课：笔记", "raw": "Note ⏳ 2026-10-09"}, "1", "课堂／任务日期"),
            ({"title": "Read", "raw": "Read ⏳ 2026-10-09"}, "2", "原计划日期"),
            ({"title": "Read", "raw": "Read"}, "unassigned", "原任务未标周次或日期"),
            ({"title": "Final", "raw": "Final 📅 2027-08-06"}, "outside", "截止日期"),
            ({"title": "2026-10-09｜ST445｜Week 2 prep", "raw": "Week 2 prep", "kind": "obligation"}, "2", "截止／必到日期"),
        ]
        for task, key, basis in cases:
            result = task_position(task, self.start, self.end)
            self.assertEqual(result[:2], (key, basis))
        self.assertIsNone(task_position({"title": "Week 2 exercise", "raw": "Week 2 exercise"}, self.start, self.end)[2], "A week label is not an exact task date")

    def test_completion_reports_plans_and_unknowns_are_not_conflated(self):
        def task(i, title, status):
            return {"id": i, "ref": "plan.md#^" + i, "title": title, "raw": title, "path": "plan.md", "status": status}
        tasks = [task("done", "ST445｜Week 1 Lab", "recorded_complete"),
                 task("partial", "ST445｜Week 2 prep", "completion_unknown"),
                 task("unknown", "ST445｜2026-10-06 review", "completion_unknown"),
                 task("future", "ST445｜2026-10-13 review", "completion_unknown"),
                 task("cancelled", "ST445｜Week 2 old plan", "cancelled")]
        before = copy.deepcopy(tasks)
        courses = [{"id": "ST445", "task_ids": [t["id"] for t in tasks], "personal": {"task_ref": "plan.md#^partial", "summary": "Some work exists"}}]
        due = {"raw": "- [ ] 2026-10-09 13:30 前｜ST445｜Prepare", "date": "2026-10-09", "path": "ddl.md", "line": 1,
               "ref": "ddl.md#^prep", "status": "completion_unknown"}
        plan = project(courses, tasks, [due, due], [], "2026-09-28", "2026-12-11", self.current)
        self.assertEqual(tasks, before)
        self.assertEqual(len(plan["weeks"]), 11)
        self.assertEqual(plan["counts"], {"total": 5, "done": 1, "recorded": 1, "unknown": 2, "planned": 1})
        self.assertEqual(len([t for t in plan["tasks"] if t.get("kind") == "obligation"]), 1)
        self.assertEqual(next(t for t in plan["tasks"] if t["id"] == "partial")["state"], "recorded")
        self.assertEqual(courses[0]["task_progress"], plan["counts"])


if __name__ == "__main__":
    unittest.main()
