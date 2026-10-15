/* Run with Obsidian eval. Exercises shared task bindings and navigation; never completes real tasks or sends chat. */
(async () => {
  const leaf = app.workspace.getLeavesOfType("student-os-steward-view")[0], view = leaf?.view;
  if (!view?.bridge.lastState?.curriculum.weekly) throw new Error("Open the steward and load its records first.");
  const saved = view.getState(), originalInput = view.input, previousLeaves = new Set(app.workspace.getLeavesOfType("markdown"));
  const checks = window.studentOsWorkspaceVerification = { pending: true, navigation: [], widths: [], failures: [], drafts: [] };
  const ensure = (ok, message) => { if (!ok) checks.failures.push(message); };
  const waitFor = async predicate => {
    for (let n = 0; n < 100; n++) { if (predicate()) return; await new Promise(resolve => setTimeout(resolve, 50)); }
    throw new Error("The selected native page did not finish rendering.");
  };
  const frame = async () => {
    const win = require("electron").remote.getCurrentWindow(); win.show(); win.focus();
    await win.webContents.capturePage(undefined, { stayHidden: false, stayAwake: true });
  };
  const page = async section => {
    view.contentEl.querySelector(`[data-section="${section}"]`).click();
    await waitFor(() => view.section === section && (section === "overview" ? view.board.querySelector(".sos-weekly")
      : section === "deadlines" ? view.board.querySelector(".sos-deadline-board")
      : view.board.querySelector(".sos-management-note")?.dataset.source === (section === "today" ? view.bridge.lastState.today.path : `99_学习情况记录/${section}.md`)));
    await frame();
    ensure(view.input === originalInput, "Navigation replaced the chat composer");
    checks.drafts.push({ section, preserved: view.input.value === "联动验证草稿（不会发送）" });
    checks.navigation.push(section);
  };
  let openedLeaf;
  try {
    const weekly = view.bridge.lastState.curriculum.weekly;
    const obligation = weekly.tasks.find(t => t.kind === "obligation" && t.status === "completion_unknown" && t.days_until >= 0)
      || weekly.tasks.find(t => t.kind === "obligation" && t.status === "completion_unknown");
    if (!obligation) throw new Error("No canonical obligation available for the read-only navigation check.");
    view.input.value = "联动验证草稿（不会发送）";
    view.input.dispatchEvent(new Event("input", { bubbles: true }));
    view.discussTask(obligation);
    await page("workbench");
    const callout = [...view.board.querySelectorAll(".callout")].find(el => el.querySelector(".callout-title-inner")?.textContent.startsWith("ST447 ·"));
    if (callout?.classList.contains("is-collapsed")) callout.querySelector(".callout-title").click();
    await frame();
    await waitFor(() => callout?.querySelector(".plugin-tasks-list-item"));
    checks.workbenchTasks = callout.querySelectorAll(".plugin-tasks-list-item").length;
    ensure(checks.workbenchTasks > 0, "Workbench did not render canonical course tasks");
    const taskNodes = [...callout.querySelectorAll(".plugin-tasks-list-item")];
    checks.sourceBindings = taskNodes.map(el => el.querySelector(".task-description [data-code-space-source-path]")?.dataset.codeSpaceSourcePath).filter(Boolean);
    ensure(checks.sourceBindings.length > 0, "Workbench task sources are not bound");
    const editable = taskNodes.find(node => !["DONE", "CANCELLED"].includes(node.dataset.taskStatusType));
    editable.querySelector("input.task-list-item-checkbox").click();
    await waitFor(() => !!view.taskDetail);
    ensure(!editable.querySelector("input.task-list-item-checkbox").checked, "Workbench tick bypassed verification");
    checks.workbenchOpensInline = !!view.taskDetail.host.closest(".sos-home");
    view.taskDetail.controller.close();
    const leafCount = app.workspace.getLeavesOfType("markdown").length;
    view.board.querySelector('a[data-href="deadlines"]').click();
    await waitFor(() => view.section === "deadlines"); await frame();
    await waitFor(() => view.board.querySelector(".sos-deadline-board"));
    ensure(app.workspace.getLeavesOfType("markdown").length === leafCount, "Workbench to Deadlines created an extra note tab");
    checks.obligationRef = obligation.ref;
    const datedRow = view.board.querySelector(`[data-task-id="${obligation.id}"]`);
    const originalRaw = obligation.raw;
    datedRow.querySelector(".sos-deadline-check").click();
    await waitFor(() => view.taskDetail?.controller.task.ref === obligation.ref);
    ensure(obligation.raw === originalRaw && obligation.status === "completion_unknown", "Opening a deadline changed its canonical state");
    checks.deadlineOpensInline = !!view.taskDetail.host.closest(".sos-deadline-task");
    ensure(checks.deadlineOpensInline, "Deadline form did not open inside its row");
    const data = view.bridge.lastState;
    const actualIds = [...view.board.querySelectorAll(".sos-deadline-task")].map(row => row.dataset.taskId).sort();
    const expectedIds = data.obligations.filter(t => t.status !== "cancelled").map(t =>
      data.curriculum.weekly.tasks.find(unit => unit.path === t.path && unit.raw === t.raw)?.id || t.id).sort();
    ensure(JSON.stringify(actualIds) === JSON.stringify(expectedIds), "A canonical deadline was omitted or duplicated");
    checks.supplements = view.board.querySelectorAll(".sos-deadline-note").length;
    ensure(checks.supplements === 3, "The current optional and unresolved notices are incomplete");
    ensure(!view.board.querySelector(".sos-management-note,.sos-management-actions"), "Deadline page still embeds the administrative note");
    view.taskDetail.controller.close();
    for (const section of ["deadlines", "today", "overview"]) {
      await page(section);
      const style = view.contentEl.getAttribute("style");
      for (const width of [360, 540, 820, 1150]) {
        view.contentEl.style.width = `${width}px`; view.contentEl.style.maxWidth = "none";
        const overflow = view.contentEl.scrollWidth - view.contentEl.clientWidth;
        checks.widths.push({ section, width, overflow }); ensure(overflow <= 1, `${section} overflows at ${width}px`);
      }
      style === null ? view.contentEl.removeAttribute("style") : view.contentEl.setAttribute("style", style);
    }
    ensure(view.input.value === "联动验证草稿（不会发送）", "Navigation lost the unsent draft");
    ensure(view.chatContext?.ref === obligation.ref, "Navigation lost the exact task context");
    await view.openOriginal(obligation.ref);
    openedLeaf = app.workspace.getLeavesOfType("markdown").find(l => l.view.file?.path === obligation.path);
    ensure(!!openedLeaf?.view.containerEl.querySelector(".sos-return-steward"), "Original note has no return to the steward");
    openedLeaf?.view.containerEl.querySelector(".sos-return-steward")?.click();
    await waitFor(() => app.workspace.activeLeaf === leaf);
    ensure(app.workspace.getLeavesOfType("student-os-steward-view").length === 1, "Returning created a duplicate steward");
    ensure(view.chatContext?.ref === obligation.ref, "Returning lost the task context");
    checks.chatDraftPreserved = true;
    checks.zenComposerCount = document.querySelectorAll(".claudian-zen-composer").length;
  } catch (error) { checks.failures.push(error.message); }
  finally {
    view.taskDetail?.controller.close();
    await view.setState(saved); app.workspace.requestSaveLayout();
    await app.workspace.revealLeaf(leaf);
    if (openedLeaf && !previousLeaves.has(openedLeaf)) openedLeaf.detach();
    checks.pending = false;
  }
  window.studentOsWorkspaceVerification = checks;
  return JSON.stringify(checks);
})()
