/* Run with Obsidian eval. Exercises navigation and native task projections; never ticks real tasks or sends chat. */
(async () => {
  const leaf = app.workspace.getLeavesOfType("student-os-steward-view")[0], view = leaf?.view;
  if (!view?.bridge.lastState?.curriculum.weekly) throw new Error("Open the steward and load its records first.");
  const saved = view.getState(), originalInput = view.input, previousLeaves = new Set(app.workspace.getLeavesOfType("markdown"));
  const checks = { navigation: [], widths: [], failures: [] };
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
      : view.board.querySelector(".sos-management-note")?.dataset.source === (section === "today" ? view.bridge.lastState.today.path : `99_学习情况记录/${section}.md`)));
    await frame();
    ensure(view.input === originalInput, "Navigation replaced the chat composer");
    checks.navigation.push(section);
  };
  let openedLeaf;
  try {
    const weekly = view.bridge.lastState.curriculum.weekly;
    const obligation = weekly.tasks.find(t => t.kind === "obligation" && t.ref?.endsWith("#^st447-exercise1-submit-20261009"))
      || weekly.tasks.find(t => t.kind === "obligation" && t.state !== "done");
    if (!obligation) throw new Error("No canonical obligation available for the read-only navigation check.");
    view.input.value = "联动验证草稿（不会发送）";
    view.discussTask(obligation);
    await page("workbench");
    const callout = [...view.board.querySelectorAll(".callout")].find(el => el.querySelector(".callout-title-inner")?.textContent.startsWith("ST447 ·"));
    if (callout?.classList.contains("is-collapsed")) callout.querySelector(".callout-title").click();
    await frame();
    await waitFor(() => callout?.querySelector(".plugin-tasks-list-item"));
    checks.workbenchTasks = callout.querySelectorAll(".plugin-tasks-list-item").length;
    ensure(callout.textContent.includes("Exercise 1"), "Workbench did not render the original exercise task");
    const taskNodes = [...callout.querySelectorAll(".plugin-tasks-list-item")];
    checks.sourceBindings = taskNodes.map(el => el.querySelector(".task-description [data-code-space-source-path]")?.dataset.codeSpaceSourcePath).filter(Boolean);
    const leafCount = app.workspace.getLeavesOfType("markdown").length;
    view.board.querySelector('a[data-href="deadlines"]').click();
    await waitFor(() => view.section === "deadlines"); await frame();
    await waitFor(() => view.board.querySelector(".plugin-tasks-list-item"));
    ensure(app.workspace.getLeavesOfType("markdown").length === leafCount, "Workbench to Deadlines created an extra note tab");
    ensure(view.board.textContent.includes("提交 Exercise 1 PDF"), "The canonical submission is missing from Deadlines");
    ensure(view.radar.textContent.includes("提交 Exercise 1 PDF"), "The canonical submission is missing from the home radar");
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
    ensure(!!view.board.querySelector(".sos-due-today")?.textContent.includes("17:00"), "Today's 17:00 obligation is not prominent on the overview");
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
    await view.setState(saved); app.workspace.requestSaveLayout();
    await app.workspace.revealLeaf(leaf);
    if (openedLeaf && !previousLeaves.has(openedLeaf)) openedLeaf.detach();
  }
  window.studentOsWorkspaceVerification = checks;
  return JSON.stringify(checks);
})()
