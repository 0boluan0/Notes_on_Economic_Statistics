/* Read-only live acceptance. Never saves progress, completes a task, or sends chat. */
(async () => {
  const view = app.workspace.getLeavesOfType("student-os-steward-view")[0]?.view;
  if (!view?.bridge.lastState) throw new Error("Open Student OS first.");
  if (view.taskDetail) throw new Error("Close the task editor before running this check, preserving any draft first.");
  const saved = view.getState(), style = view.contentEl.getAttribute("style");
  const checks = window.studentOsInlineVerification = { pending: true, failures: [], widths: [] };
  const ensure = (ok, message) => { if (!ok) checks.failures.push(message); };
  const waitFor = async predicate => {
    for (let n = 0; n < 120; n++) { if (predicate()) return; await new Promise(resolve => setTimeout(resolve, 50)); }
    throw new Error("Task context did not finish loading.");
  };
  try {
    await view.bridge.refresh();
    const data = view.bridge.lastState;
    view.section = "overview"; view.render();
    const tracks = [...view.board.querySelectorAll(".sos-learning-track")];
    checks.selfStudyCards = tracks.length;
    for (const card of tracks) {
      const course = data.curriculum.courses.find(c => c.id === card.dataset.courseDetails);
      const segments = [...card.querySelectorAll(":scope > summary .sos-track-segments > span")];
      if (course.replay_progress) {
        ensure(!segments.length, `${course.id} shows a full progress bar for an incomplete replay scope`);
        ensure(card.querySelector(":scope > summary .sos-study-map-hint").textContent.includes(`${course.unit_progress.completed} / ${course.unit_progress.total}`), "Replay card hides the historical course position");
      } else if (course.mode === "finite" && course.unit_progress?.total) {
        const count = course.unit_progress;
        ensure(segments.length === count.total, `${course.id} unit segments disagree with its plan`);
        ensure(segments.filter(s => s.classList.contains("is-done")).length === count.completed, `${course.id} segments overstate completed units`);
      } else ensure(!segments.length && !card.querySelector(".sos-track-count"), `${course.id} shows a finite percentage for open-ended practice`);
    }
    for (const width of [360, 540, 820, 1150]) {
      view.contentEl.style.width = width + "px"; view.contentEl.style.maxWidth = "none";
      ensure(tracks.every(card => card.scrollWidth <= card.clientWidth + 1), `A self-study card overflows at ${width}px`);
    }
    style === null ? view.contentEl.removeAttribute("style") : view.contentEl.setAttribute("style", style);
    const task = data.curriculum.weekly.tasks.find(t => t.checkpoints?.length > 1 && t.status === "completion_unknown");
    if (!task) throw new Error("A task with checkpoints is required for this check.");
    view.section = "overview"; view.selectTasks(task.course_id, task.week);
    const before = document.querySelectorAll(".modal-container").length;
    view.board.querySelector(`[data-key="task-${task.id}"] .sos-check-task`).click();
    await waitFor(() => !!view.taskDetail?.controller.resultEl && !view.taskDetail.controller.submit.disabled);
    const { host, controller } = view.taskDetail;
    ensure(document.querySelectorAll(".modal-container").length === before, "Opening a task created a popup");
    ensure(!!host.closest(".sos-week-task"), "Task did not expand within its weekly row");
    ensure(host.querySelectorAll(".sos-detail-checkpoints li").length === task.checkpoints.length, "Checkpoint route is incomplete");
    ensure(controller.completionPanel.hidden && controller.progressPanel.hidden, "Initial details expose a form before it is requested");
    ensure(!view.board.querySelector(".sos-course-sources,.sos-detail-original"), "Technical provenance leaked into learner navigation");
    host.querySelector(".sos-step-record").click();
    ensure(!controller.progressPanel.hidden && !!controller.progressForm.next.value, "Checkpoint action did not open the progress editor");
    controller.progressForm.summary.value = "UI verification draft only";
    const next = controller.progressForm.next.value;
    view.render();
    await view.bridge.refresh();
    ensure(host.isConnected && controller.progressForm.summary.value === "UI verification draft only" && controller.progressForm.next.value === next, "Refresh lost the editor or unsaved draft");
    controller.showMode("complete", false);
    controller.reportInput.value = "Unsaved completion draft";
    view.render();
    ensure(controller.reportInput.value === "Unsaved completion draft", "Refresh lost a completion draft");
    for (const width of [360, 540, 820, 1150]) {
      view.contentEl.style.width = width + "px"; view.contentEl.style.maxWidth = "none";
      const overflow = view.contentEl.scrollWidth - view.contentEl.clientWidth;
      checks.widths.push({ width, overflow }); ensure(overflow <= 1, `Inline task overflow at ${width}px`);
    }
    controller.close();
    ensure(!view.taskDetail && !host.isConnected, "Collapse did not remove the inline editor");
    const mit = view.bridge.lastState.curriculum.courses.find(c => c.id === "MIT 6.100L");
    if (mit?.replay_progress?.completed) {
      ensure(!mit.personal.summary.includes("尚未开始"), "An active replay still shows not started");
      const track = [...view.contentEl.querySelectorAll(".sos-learning-track")].find(n => n.dataset.courseDetails === mit.id);
      const wasOpen = track.open; track.querySelector("summary").click();
      ensure(track.open !== wasOpen, "Course does not expand/collapse inline");
      ensure(document.querySelectorAll(".modal-container").length === before, "Opening a course created a popup");
      ensure(track.querySelector(".sos-study-complete")?.textContent.includes("这次重看已完成") && track.textContent.includes("历史课程清单"), "Replay and historical course progress are not separated");
      track.open = true;
      [...track.querySelectorAll(".sos-study-unit-actions button")].find(button => button.textContent === "查看完成记录").click();
      await waitFor(() => !!view.taskDetail?.controller.resultEl);
      ensure(!!view.taskDetail.host.closest(".sos-study-inspector"), "Self-study task opened outside its course");
      view.render();
      ensure(!!view.taskDetail.host.closest(".sos-study-inspector"), "Refresh moved the self-study task out of its course");
      view.taskDetail.controller.close();
      checks.replay = mit.replay_progress; checks.historical = mit.unit_progress;
    }
    checks.canonicalRawUnchanged = data.tasks.every(t => view.bridge.lastState.tasks.some(current => current.id === t.id && current.raw === t.raw));
    checks.changedSources = data.tasks.filter(t => !view.bridge.lastState.tasks.some(current => current.id === t.id && current.raw === t.raw)).map(t => t.path);
    ensure(checks.canonicalRawUnchanged, "Canonical task state changed during a read-only check");
  } catch (error) { checks.failures.push(error.message); }
  finally {
    view.taskDetail?.controller.close();
    style === null ? view.contentEl.removeAttribute("style") : view.contentEl.setAttribute("style", style);
    await view.setState(saved); checks.pending = false;
  }
  return JSON.stringify(checks);
})()
