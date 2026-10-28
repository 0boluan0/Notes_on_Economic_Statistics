/* Live Obsidian regression. Run via `obsidian eval` after deployment, then poll
   window.studentOsReviewVerification. All attempts/reports use an isolated copy
   under window.studentOsReviewVerificationWork (or the system temp directory).
   Only source buttons open original, read-only material in the real vault. */
(async () => {
  const checks = window.studentOsReviewVerification = { pending: true, failures: [], checks: [] };
  const fs = require("fs"), path = require("path"), os = require("os"), { spawn } = require("child_process");
  const sleep = ms => new Promise(resolve => require("timers").setTimeout(resolve, ms));
  const ensure = (ok, message) => { (ok ? checks.checks : checks.failures).push(message); };
  const until = async (predicate, message, timeout = 15000) => {
    const end = Date.now() + timeout;
    while (!predicate()) { if (Date.now() > end) throw new Error(message); await sleep(30); }
  };
  const run = (program, args, input = "") => new Promise((resolve, reject) => {
    const child = spawn(program, args, { stdio: ["pipe", "pipe", "pipe"] });
    let stdout = "", stderr = "";
    const timer = require("timers").setTimeout(() => { child.kill(); reject(new Error("Fixture process timed out")); }, 15000);
    child.stdout.on("data", data => { stdout += data; }); child.stderr.on("data", data => { stderr += data; });
    child.on("error", error => { require("timers").clearTimeout(timer); reject(error); });
    child.on("close", code => { require("timers").clearTimeout(timer); code === 0 ? resolve(stdout) : reject(new Error(stderr || stdout || `Exit ${code}`)); });
    child.stdin.end(input);
  });
  let view, originalPlugin, injected = false, fixture, transport;
  const ownedLeaves = new Set();
  try {
    const plugin = app.plugins.plugins["learning-progress-dashboard"];
    if (!plugin?.openReview) throw new Error("Deploy the review module before this verification.");
    const existing = app.workspace.getLeavesOfType("student-os-review-view")[0]?.view;
    if (existing?.data?.session || existing?.busy) throw new Error("A real review is open; preserve it and run this check from the review landing page.");
    await plugin.openReview();
    view = app.workspace.getLeavesOfType("student-os-review-view")[0]?.view;
    if (!view || view.data?.session) throw new Error("The review landing page is unavailable.");
    const visibleTiming = !!window.studentOsReviewVerifyVisibleTiming;
    if (visibleTiming) await until(() => !document.hidden && document.hasFocus(), "Bring this Obsidian window to the foreground before checking visible timing.", 40000);
    const vault = app.vault.adapter.getBasePath();
    const steward = path.join(vault, "05_tools/StudentOS/steward");
    const packsPath = path.join(steward, "review_packs");
    const packName = window.studentOsReviewVerificationPack || fs.readdirSync(packsPath).filter(file => file.endsWith(".json")).sort()[0];
    if (!packName || path.basename(packName) !== packName) throw new Error("Choose an existing review pack filename.");
    const pack = JSON.parse(fs.readFileSync(path.join(packsPath, packName), "utf8"));
    const work = window.studentOsReviewVerificationWork || os.tmpdir();
    fs.mkdirSync(work, { recursive: true });
    fixture = fs.mkdtempSync(path.join(work, "student-os-review-verification-"));
    const fixtureVault = path.join(fixture, "vault"), fixturePacks = path.join(fixture, "packs");
    fs.mkdirSync(fixtureVault); fs.mkdirSync(fixturePacks);
    fs.writeFileSync(path.join(fixtureVault, ".gitignore"), "99_学习情况记录/\n");
    fs.writeFileSync(path.join(fixtureVault, "canonical-sentinel.md"), "- [ ] Synthetic canonical task; must remain unchanged. #student-os/task\n");
    const originalHashes = new Map();
    const sha = bytes => require("crypto").createHash("sha256").update(bytes).digest("hex");
    for (const source of pack.source_refs) {
      const original = path.resolve(vault, source.path), copied = path.resolve(fixtureVault, source.path);
      if (!original.startsWith(vault + path.sep) || !copied.startsWith(fixtureVault + path.sep)) throw new Error("Source escaped the fixture vault.");
      originalHashes.set(original, sha(fs.readFileSync(original)));
      fs.mkdirSync(path.dirname(copied), { recursive: true }); fs.copyFileSync(original, copied);
    }
    fs.copyFileSync(path.join(packsPath, packName), path.join(fixturePacks, packName));
    await run("/usr/bin/git", ["init", "-q", fixtureVault]);
    const python = ["/opt/homebrew/bin/python3", "/usr/bin/python3"].find(fs.existsSync);
    if (!python) throw new Error("Python is unavailable.");
    const clockFile = path.join(fixture, "clock.json");
    const runner = [
      "import datetime as dt, json, pathlib, sys",
      "sys.path.insert(0, sys.argv[1])",
      "from steward import Store",
      "from service import Service",
      "from review import Review, ZONE",
      "root=pathlib.Path(sys.argv[2])",
      "clock_path=root/'clock.json'",
      "clock=lambda: dt.datetime.fromisoformat(json.loads(clock_path.read_text())) if clock_path.exists() else dt.datetime.now(ZONE)",
      "store=Store(root/'state', root/'vault', root/'codex')",
      "service=Service(store, clock=clock)",
      "engine=Review(store, clock=clock, packs=root/'packs', write_private=service.write_private)",
      "try: print(json.dumps(engine.handle(json.load(sys.stdin)), ensure_ascii=False))",
      "finally: store.close()",
    ].join("\n");
    const completed = {};
    transport = async input => {
      const result = JSON.parse(await run(python, ["-B", "-c", runner, steward, fixture], JSON.stringify(input)));
      completed[input.op] = (completed[input.op] || 0) + 1;
      return result;
    };
    originalPlugin = view.plugin;
    view.plugin = Object.assign(Object.create(originalPlugin), { steward: { refresh: async () => {} } });
    view.transport = transport; injected = true;
    await view.loadCatalog();
    ensure(view.catalog.lessons.length === 1 && !view.catalog.lessons[0].source_issues.length, "Copied real pack and source hashes validate");
    const click = async (op, within = view.contentEl) => {
      const button = within.querySelector(`[data-review-action="${op}"]`);
      if (!button || button.disabled) throw new Error(`Enabled ${op} button missing`);
      const command = op === "pause" ? "checkpoint" : op, before = completed[command] || 0;
      button.click(); await sleep(0);
      await until(() => (completed[command] || 0) > before && !view.busy && !view.flushing || !!view.error, `${op} did not finish`);
      if (view.error) throw new Error(`${op}: ${view.error}`);
    };
    const enter = (element, value) => {
      element.value = String(value);
      element.dispatchEvent(new element.ownerDocument.defaultView.Event("input", { bubbles: true }));
      element.dispatchEvent(new element.ownerDocument.defaultView.Event("change", { bubbles: true }));
    };
    const answerTask = async (wrong = false) => {
      const task = pack.tasks.find(task => task.id === view.data.task.id);
      for (const item of task.items) {
        let value = item.kind === "explain_self" ? "Synthetic verification response, not learner evidence." : item.accepted[0];
        if (wrong && item.kind !== "explain_self") {
          value = item.kind === "choice" ? item.options.find(option => !item.accepted.includes(option.value))?.value
            : item.kind === "number" ? Number(item.accepted[0]) + 999 : "synthetic-invalid";
          if (value === undefined) value = item.accepted[0];
        }
        enter(view.contentEl.querySelector(`[data-item-id="${item.id}"]`), value);
      }
      await click("submit");
    };
    const selfGrade = async () => {
      while (view.data.feedback.some(entry => entry.needs_self_grade)) {
        const entry = view.data.feedback.find(entry => entry.needs_self_grade);
        const card = view.contentEl.querySelector(`[data-feedback-task="${entry.task.id}"]`);
        card.querySelectorAll("[data-rating-item]").forEach(select => enter(select, "yes"));
        await click("self-grade", card);
      }
    };
    await click("start");
    const sessionId = view.data.session.id, firstTask = view.data.task.id;
    checks.fixture = fixture; checks.session = sessionId; checks.pack = pack.lesson_id;
    const firstInput = view.contentEl.querySelector("[data-item-id]");
    const firstValue = firstInput.tagName === "SELECT" ? firstInput.options[1].value : "Synthetic partial draft";
    const firstItem = firstInput.dataset.itemId;
    enter(firstInput, firstValue);
    await view.flush();
    await view.loadCatalog(); await click("start");
    ensure(view.data.session.id === sessionId && view.data.task.id === firstTask, "Reopening resumes the same durable session and task");
    ensure(view.contentEl.querySelector(`[data-item-id="${firstItem}"]`).value === firstValue, "Reopening restores the saved answer draft");
    if (visibleTiming) {
      await view.flush(); const before = view.data.timing.diagnostic;
      await sleep(2200); await view.flush();
      const elapsed = view.data.timing.diagnostic - before;
      ensure(elapsed >= 1.5 && elapsed < 5, "Natural foreground time accumulates in the diagnostic phase");
      checks.diagnosticSeconds = elapsed;
    }
    await click("pause");
    const beforePause = JSON.stringify(view.clock.pending);
    await sleep(1150); view.tick();
    ensure(!view.running && JSON.stringify(view.clock.pending) === beforePause, "Paused visible time does not accumulate");
    ensure((await transport({ op: "get", session_id: sessionId })).paused === true, "Pause is durable in the isolated store");
    await click("pause");
    let questionCount = 0;
    while (view.data.session.stage === "question") {
      await answerTask(questionCount++ === 0);
      if (view.data.session.stage === "question") ensure(view.data.feedback.length === 0 && !view.contentEl.querySelector("[data-feedback-task]"), "No answer feedback appears before the diagnostic round ends");
    }
    await selfGrade();
    ensure(view.data.actions.can_repair.length > 0, "A wrong deterministic answer produces a repair route");
    await click("repair");
    const repairTask = view.data.repair.task_id;
    const note = "Synthetic repair draft to preserve across PDF navigation.";
    enter(view.contentEl.querySelector("textarea"), note);
    const pdfRoute = view.data.repair.routes.find(route => route.source?.path.endsWith(".pdf") && route.page);
    if (!pdfRoute) throw new Error("The copied real pack needs a PDF page repair route for this check.");
    const sourceIndex = view.data.repair.routes.indexOf(pdfRoute);
    const sourceButton = view.contentEl.querySelectorAll('[data-review-action="source"]')[sourceIndex];
    sourceButton.click();
    await until(() => view.sourceLeaves.size > 0, "Source did not create its dedicated split");
    for (const leaf of view.sourceLeaves) ownedLeaves.add(leaf);
    const sourceLeaf = [...view.sourceLeaves][0];
    await until(() => sourceLeaf.view.containerEl.querySelector(".pdf-page-input")?.value === String(pdfRoute.page), "PDF failed to open at the requested page");
    ensure(sourceLeaf.view.file?.path === pdfRoute.source.path, "Source button opens the exact original PDF at the requested page");
    if (visibleTiming) {
      await view.flush(); const before = view.data.timing.repair;
      await sleep(2200); await view.flush();
      const saved = await transport({ op: "get", session_id: sessionId });
      const elapsed = saved.timing.repair - before;
      ensure(elapsed >= 1.5 && elapsed < 5, "Natural time in the source PDF is durably recorded as repair");
      checks.repairSeconds = elapsed;
    }
    await until(() => sourceLeaf.view.containerEl.querySelector(".sor-return"), "PDF return-to-review action is missing");
    sourceLeaf.view.containerEl.querySelector(".sor-return").click();
    await until(() => app.workspace.activeLeaf === view.leaf, "Return did not restore the review leaf");
    ensure(view.data.session.id === sessionId && view.data.repair.task_id === repairTask && view.draft.repair_notes === note, "PDF return preserves session, repair phase and draft");
    ensure(sourceLeaf.view.containerEl.querySelector(".pdf-page-input")?.value === String(pdfRoute.page), "Returning to review preserves the PDF page");
    await click("retest");
    ensure(view.data.task.id !== repairTask && view.data.session.phase === "retest", "Repair uses an unseen task for retesting");
    ensure(view.sourceLeaves.size === 0 && !sourceLeaf.parent, "Retest closes only the review-owned material split");
    await answerTask(true); await selfGrade();
    ensure(view.data.feedback.some(entry => entry.family_status === "needs_study"), "A failed retest stays as an explicit learning gap");
    while (view.data.actions.can_defer.length) await click("defer");
    await click("finish");
    ensure(view.data.session.stage === "complete" && !view.running, "Finish freezes the review and shows its report");
    const reportPath = path.join(fixtureVault, view.data.report_path), report = fs.readFileSync(reportPath, "utf8");
    ensure(report.includes("仍需学习") && report.includes("本人解释自评不等于独立核验"), "Private report preserves unresolved gaps and evidence limits");
    ensure(["diagnostic", "feedback", "repair", "retest", "delayed", "system_wait"].every(phase => Number.isFinite(view.data.timing[phase]) && report.includes(phase)), "Report contains every phase and finite timing totals");
    ensure(view.contentEl.querySelectorAll("table tbody tr").length === 7, "Visible report shows six timing categories plus total");
    ensure(fs.readFileSync(path.join(fixtureVault, "canonical-sentinel.md"), "utf8") === "- [ ] Synthetic canonical task; must remain unchanged. #student-os/task\n", "Review does not change the fixture canonical task");
    ensure([...originalHashes].every(([file, hash]) => sha(fs.readFileSync(file)) === hash), "All original source files remain byte-identical");
    const due = view.data.schedule.due_at;
    ensure(Boolean(due), "First review records a future single review point");
    if (due) {
      await transport({ op: "start", lesson_id: pack.lesson_id }).then(() => ensure(false, "Future review is blocked before its due time"), () => ensure(true, "Future review is blocked before its due time"));
      fs.writeFileSync(clockFile, JSON.stringify(new Date(new Date(due).getTime() + 1000).toISOString()));
      const delayed = await transport({ op: "start", lesson_id: pack.lesson_id });
      ensure(delayed.session.id !== sessionId && delayed.session.phase === "delayed" && delayed.task.id !== firstTask, "Advancing only the fixture clock selects a new delayed task");
    }
    checks.report = reportPath;
  } catch (error) { checks.failures.push(error.stack || error.message); }
  finally {
    if (injected) {
      try {
        // Never flush fixture drafts through the production transport.
        if (view.flushing) await view.flushing.catch(() => {});
        view.running = false; view.data = null; view.draft = {}; view.pendingCheckpoints = null;
        view.transport = null; view.plugin = originalPlugin;
        for (const leaf of view.sourceLeaves) ownedLeaves.add(leaf);
        view.sourceLeaves.clear();
        for (const leaf of ownedLeaves) if (leaf.parent) leaf.detach();
        await view.loadCatalog();
        await app.workspace.revealLeaf(view.leaf); app.workspace.setActiveLeaf(view.leaf, { focus: true });
      } catch (error) { checks.failures.push(`Restoring real landing page: ${error.message}`); }
    }
    checks.pending = false;
  }
  return JSON.stringify(checks);
})()
