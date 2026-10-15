/* Run with Obsidian eval. Checks the live view; restores filters, size and scroll. */
(() => {
  const view = app.workspace.getLeavesOfType("student-os-steward-view")[0]?.view;
  if (!view?.bridge.lastState?.curriculum.weekly) throw new Error("Open the steward home and let its records load first.");
  const el = view.contentEl, plan = view.bridge.lastState.curriculum.weekly;
  const saved = { section: view.section, course: view.selectedCourse, week: view.selectedWeek, filter: view.taskFilter, style: el.getAttribute("style"), scroll: el.scrollTop };
  const checks = { weeks: [], widths: [], failures: [] };
  try {
    view.section = "overview"; view.selectedCourse = null; view.taskFilter = "all";
    for (const key of [...plan.weeks.map(w => w.key), "unassigned", "outside"]) {
      view.selectedWeek = key; view.render();
      const degreeIds = view.bridge.lastState.curriculum.courses.filter(c => c.group === "degree").map(c => c.id);
      const expected = plan.tasks.filter(t => t.week === key && t.state !== "cancelled" && (!/^\d+$/.test(key) || degreeIds.includes(t.course_id))).length;
      const actual = el.querySelectorAll(".sos-week-task").length;
      checks.weeks.push({ key, expected, actual });
      if (actual !== expected) checks.failures.push(`Week ${key}: expected ${expected}, saw ${actual}`);
    }
    view.selectedWeek = "all"; view.taskFilter = "done"; view.render();
    checks.completed = { expected: plan.counts.done, actual: el.querySelectorAll(".sos-week-task.is-done").length };
    if (checks.completed.expected !== checks.completed.actual || el.querySelector(".sos-week-task:not(.is-done)")) checks.failures.push("Completed filter mismatch");
    view.selectedWeek = null; view.taskFilter = "all"; view.render();
    for (const width of [360, 540, 820, 1000, 1150]) {
      el.style.width = width + "px";
      el.style.maxWidth = "none";
      const overflow = el.scrollWidth - el.clientWidth;
      checks.widths.push({ width, overflow });
      if (overflow > 1) checks.failures.push(`Horizontal overflow at ${width}px: ${overflow}px`);
      const stickyTop = el.getBoundingClientRect().top + parseFloat(getComputedStyle(el).paddingTop) + 16;
      if (view.managerOpen && width >= 1000 && view.side.getBoundingClientRect().top > Math.max(view.board.getBoundingClientRect().top, stickyTop) + 1) {
        checks.failures.push(`The manager fell below the entire board at ${width}px`);
      }
    }
  } finally {
    view.section = saved.section; view.selectedCourse = saved.course; view.selectedWeek = saved.week; view.taskFilter = saved.filter;
    saved.style === null ? el.removeAttribute("style") : el.setAttribute("style", saved.style);
    view.render(); el.scrollTop = saved.scroll;
  }
  window.studentOsUiVerification = checks;
  return JSON.stringify(checks);
})()
