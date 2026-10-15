/* Read-only acceptance: route selection, task binding, draft retention and responsive layout. */
(async () => {
  const view = app.workspace.getLeavesOfType("student-os-steward-view")[0]?.view;
  if (!view?.bridge.lastState || view.taskDetail) throw new Error("Open Student OS and preserve/close any task editor first.");
  const saved = view.getState(), style = view.contentEl.getAttribute("style");
  const originalBegin = view.beginTask;
  const checks = window.studentOsStudyVerification = { pending: true, failures: [], widths: [] };
  const ensure = (ok, message) => { if (!ok) checks.failures.push(message); };
  const waitFor = async predicate => {
    for (let n = 0; n < 150; n++) { if (predicate()) return; await new Promise(resolve => setTimeout(resolve, 40)); }
    throw new Error("Study workspace did not settle.");
  };
  const card = id => [...view.board.querySelectorAll(".sos-learning-track")].find(row => row.dataset.courseDetails === id);
  const selected = id => card(id).querySelector(".sos-study-inspector").dataset.taskId;
  try {
    await view.bridge.refresh();
    const data = view.bridge.lastState, raw = JSON.stringify(data.tasks.map(t => [t.id, t.raw]));
    const popups = document.querySelectorAll(".modal-container").length;
    view.section = "overview"; view.studyUnits = {}; view.studyRoutes = {}; view.render();
    ensure(view.board.contains(view.classroom) && view.classroom.querySelector(".sos-launch-position") && view.classroom.querySelector(".sos-launch-next") && view.classroom.querySelector(".sos-launch-handoff"), "Overview omits the stopping point, next step or expandable completion condition");
    ensure(!view.board.querySelector(".sos-learning-nav"), "Extra workflow navigation has returned");
    ensure(!!(view.board.querySelector(".sos-self-study").compareDocumentPosition(view.board.querySelector(".sos-weekly")) & Node.DOCUMENT_POSITION_FOLLOWING), "Self-directed courses are buried after the weekly task list");
    const calculus = data.curriculum.courses.find(c => c.id === "MIT 18.01");
    card(calculus.id).open = true;
    const units = data.curriculum.weekly.tasks.filter(t => t.course_id === calculus.id && t.state !== "cancelled");
    ensure(selected(calculus.id) === calculus.next_task.id, "Calculus did not open at its actual next task");
    ensure(!card(calculus.id).querySelector(".sos-study-allunits").open, "Full historical route is expanded by default");
    ensure(card(calculus.id).querySelectorAll(".sos-route-peek-item").length <= 3, "Nearby route is overloaded");
    ensure([...card(calculus.id).querySelectorAll(".sos-route-peek-item")].every(node => node.querySelector("div > span").textContent), "Nearby route hides lecture topics behind numbers");
    const nodes = [...card(calculus.id).querySelectorAll(".sos-unit-node")];
    ensure(nodes.length === units.length, "Unit map omitted source tasks");
    ensure(nodes.every((node, i) => node.querySelector("strong").textContent === units[i].title.match(/Lecture\s+(\d+)/)[1].padStart(2, "0")), "Unit map changed original lecture numbers");
    nodes.at(-1).click();
    await waitFor(() => selected(calculus.id) === units.at(-1).id);
    ensure(!view.taskDetail, "Browsing a unit unexpectedly opened a completion form");
    card(calculus.id).querySelector('[aria-pressed="true"].sos-unit-node').dispatchEvent(new KeyboardEvent("keydown", { key: "Home", bubbles: true }));
    ensure(selected(calculus.id) === units[0].id, "Keyboard navigation did not select the first real lecture");
    card(calculus.id).querySelector('[aria-pressed="true"].sos-unit-node').dispatchEvent(new KeyboardEvent("keydown", { key: "End", bubbles: true }));
    ensure(selected(calculus.id) === units.at(-1).id, "Keyboard navigation did not select the last real lecture");
    card(calculus.id).querySelector(".sos-study-allunits").open = true;
    await waitFor(() => view.studyRoutes?.[calculus.id]);
    const restore = view.getState();
    view.studyUnits = {}; await view.setState(restore);
    ensure(selected(calculus.id) === units.at(-1).id && card(calculus.id).open, "Workspace state lost the selected unit or expanded course");
    ensure(card(calculus.id).querySelector(".sos-study-allunits").open, "Workspace state lost the expanded historical route");
    const replay = data.curriculum.courses.find(c => c.id === "MIT 6.100L");
    ensure(card(replay.id).querySelector(".sos-track-count strong").textContent === String(replay.replay_progress.completed), "Historical progress replaced the primary replay metric");
    ensure(card(replay.id).querySelector(".sos-study-complete")?.textContent.includes("重看已完成"), "Replay completion is not visible in its unit workspace");
    const historyTab = [...card(replay.id).querySelectorAll(".sos-study-track-tab")].find(b => b.textContent === "历史课程清单");
    historyTab.click();
    await waitFor(() => card(replay.id).querySelectorAll(".sos-unit-node").length === replay.unit_progress.total);
    ensure(card(replay.id).querySelectorAll(".sos-unit-node.is-done").length === replay.unit_progress.completed, "Historical and replay completion counts were merged");
    const python = data.curriculum.courses.find(c => c.id === "DSL Python");
    card(python.id).open = true;
    let prepared;
    view.beginTask = async task => { prepared = task; };
    card(python.id).querySelector(".sos-study-start").click();
    await waitFor(() => !!prepared);
    ensure(prepared.id === python.next_task.id, "Classroom action does not bind to the canonical selected task");
    view.beginTask = originalBegin;
    [...card(python.id).querySelectorAll(".sos-study-unit-actions button")].find(b => b.textContent === "记录进度").click();
    await waitFor(() => !!view.taskDetail?.controller.progressForm && !view.taskDetail.controller.submit.disabled);
    const { controller, host } = view.taskDetail;
    controller.progressForm.summary.value = "Unsaved study route check";
    controller.progressForm.next.value = "Resume this exact draft";
    const second = card(python.id).querySelectorAll(".sos-unit-node")[1];
    second.click();
    await waitFor(() => selected(python.id) === second.dataset.unitId);
    ensure(host.isConnected && !!host.closest(".sos-study-editing"), "Unit selection detached the active draft");
    await view.bridge.refresh();
    ensure(controller.progressForm.summary.value === "Unsaved study route check" && controller.progressForm.next.value === "Resume this exact draft", "Automatic refresh lost the active draft");
    ensure(!!host.closest(".sos-study-editing"), "Refresh moved the draft away from its course");
    ensure(!card(python.id).querySelector(".sos-study-inspector .sos-study-start"), "A different task can overwrite the active draft");
    controller.close();
    for (const width of [360, 540, 820, 1150]) {
      view.contentEl.style.width = width + "px"; view.contentEl.style.maxWidth = "none";
      view.board.querySelectorAll(".sos-learning-track").forEach(row => { row.open = true; });
      const overflow = view.contentEl.scrollWidth - view.contentEl.clientWidth;
      const inner = [...view.board.querySelectorAll(".sos-learning-track,.sos-study-inspector,.sos-unit-map")].filter(node => node.scrollWidth > node.clientWidth + 1).map(node => node.className);
      checks.widths.push({ width, overflow, inner });
      ensure(overflow <= 1 && !inner.length, `Study workspace overflows at ${width}px`);
    }
    ensure(document.querySelectorAll(".modal-container").length === popups, "Study navigation created a popup");
    ensure(JSON.stringify(view.bridge.lastState.tasks.map(t => [t.id, t.raw])) === raw, "Read-only browsing changed learning records");
    checks.courses = view.board.querySelectorAll(".sos-learning-track").length;
    checks.calculusLectures = nodes.length; checks.historical = replay.unit_progress; checks.replay = replay.replay_progress;
  } catch (error) { checks.failures.push(error.stack || error.message); }
  finally {
    view.beginTask = originalBegin;
    view.taskDetail?.controller.close();
    style === null ? view.contentEl.removeAttribute("style") : view.contentEl.setAttribute("style", style);
    await view.setState(saved); checks.pending = false;
  }
  return JSON.stringify(checks);
})()
