const { ItemView, Modal, FuzzySuggestModal, Notice, MarkdownRenderer, Component, setIcon } = require("obsidian");
const { spawn } = require("child_process");
const { existsSync } = require("fs");
const { join } = require("path");
const { randomUUID } = require("crypto");

const STEWARD_VIEW = "student-os-steward-view";
const WORKBENCH = "99_学习情况记录/workbench.md";
const DEADLINES = "99_学习情况记录/deadlines.md";
const OBLIGATIONS = "99_学习情况记录/截止事项记录.md";
const SECTIONS = { overview: "学习总览", workbench: "Workbench · 任务", deadlines: "Deadlines · 期限", today: "今日安排" };
const id = () => `request-${randomUUID()}`;
const labels = { queued: "等待备课", creating: "建立课堂", preparing: "正在备课", result_saved: "保存开场",
  prepared: "等待接入", linking: "正在接入", registered: "课堂已准备", uncertain: "需要恢复", cancelled: "已取消" };

function homeModel(data) {
  const task = data.focus?.task;
  const job = (data.jobs || []).find(j => j.request?.task?.id === task?.id && !j.handed_off && j.state !== "cancelled");
  return { task, job, phase: displayText(data.focus?.stage || "读取当前安排"),
    next: data.focus?.next_step || "从实际停点继续；课堂会先核对你已经做到哪里。",
    obligations: (data.obligations || []).filter(t => t.status === "completion_unknown" && t.days_until !== null && t.days_until <= 14),
    notices: (data.commitments || []).filter(c => c.due) };
}

function positionFor(fact = {}) {
  const outdated = !!(fact.historical || fact.needs_review);
  const display = fact.display || {};
  const week = Number.isInteger(display.week) && display.week > 0 && display.week <= 52 ? display.week : null;
  return { ...display, week: outdated ? null : week, units: outdated ? undefined : display.units, outdated,
    label: display.label || fact.label || "学习记录",
    topic: display.topic || fact.summary || "尚无位置记录",
    date: fact.as_of || "", freshness: outdated ? (fact.historical ? "上次定位 · 已过核验期" : "上次定位 · 来源有更新") : "" };
}

function londonClock(now = new Date()) {
  const parts = Object.fromEntries(new Intl.DateTimeFormat("en-GB", { timeZone: "Europe/London",
    year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23" })
    .formatToParts(now).map(p => [p.type, p.value]));
  return { date: `${parts.year}-${parts.month}-${parts.day}`, time: `${parts.hour}:${parts.minute}` };
}

function agendaFor(data, now = new Date()) {
  const clock = londonClock(now);
  if (!data.today?.path?.includes(clock.date)) return { slots: [], current: null, upcoming: [], date: clock.date };
  const spine = data.today.text?.match(/<!-- student-os:day-spine:start -->([\s\S]*?)<!-- student-os:day-spine:end -->/)?.[1] || "";
  const slots = [];
  for (const line of spine.split("\n")) {
    const heading = line.match(/^\s*-\s*\*\*(.+?)\*\*/)?.[1];
    if (!heading) continue;
    const times = [...heading.matchAll(/(\d{2}:\d{2})[–—-](\d{2}:\d{2})[｜|]/g)];
    for (let i = 0; i < times.length; i++) {
      const m = times[i];
      const raw = heading.slice(m.index + m[0].length, times[i + 1]?.index).replace(/[。；;\s]+$/, "");
      let label = displayText(raw), location = "";
      if (label.startsWith("LSE Library，")) { location = "LSE Library"; label = label.slice(12); }
      else {
        const room = label.match(/，([A-Z]{2,}\.[\d.]+)$/);
        if (room) { location = room[1]; label = label.slice(0, room.index); }
      }
      slots.push({ start: m[1], end: m[2], label, location, ref: raw.match(/\[\[([^\]|]+)/)?.[1] || "" });
    }
  }
  return { slots, date: clock.date, current: slots.find(s => s.start <= clock.time && clock.time < s.end) || null,
    upcoming: slots.filter(s => s.start > clock.time) };
}

function todayNotices(data, now = new Date()) {
  if (!data.today?.path?.includes(londonClock(now).date)) return [];
  const section = data.today.text?.match(/<!-- student-os:deadline-radar:start -->([\s\S]*?)<!-- student-os:deadline-radar:end -->/)?.[1] || "";
  const result = [];
  for (const line of section.split("\n")) {
    const heading = line.match(/^\s*-\s*\*\*(.+?)\*\*/)?.[1];
    if (!heading) continue;
    const parts = displayText(heading).split("｜"), days = parts[0].match(/D-(\d+)/)?.[1];
    if (days === undefined || parts.length < 3) continue;
    const ref = heading.match(/\[\[([^\]|]+)/)?.[1];
    const canonical = (data.obligations || []).find(t => sameRef(ref, t.ref));
    if (canonical && ["recorded_complete", "cancelled"].includes(canonical.status)) continue;
    result.push({ days: Number(days), time: parts[1], title: parts.slice(2).join("｜").replace(/。$/, ""),
      optional: parts[0].includes("可选"), ref: ref || "", detail: displayText(line.slice(line.indexOf("**", 4) + 2)).trim() });
  }
  return result.sort((a, b) => a.days - b.days || (a.time.match(/\d{2}:\d{2}/)?.[0] || "").localeCompare(b.time.match(/\d{2}:\d{2}/)?.[0] || ""));
}

function sameRef(left, right) {
  if (!left || !right) return false;
  const [a, anchor] = left.split("#"), [b, other] = right.split("#");
  const clean = p => p.replace(/\.md$/, "");
  return anchor === other && (clean(a) === clean(b) || (!a.includes("/") && clean(a) === clean(b).split("/").pop())
    || (!b.includes("/") && clean(b) === clean(a).split("/").pop()));
}

function deadlineNotices(data, now = new Date()) {
  const date = londonClock(now).date, scheduled = agendaFor(data, now).slots;
  const originals = data.obligations || [];
  const canonical = originals.filter(t => t.status === "completion_unknown").map(t => {
    const days = t.date ? Math.round((Date.parse(t.date) - Date.parse(date)) / 86400000) : NaN;
    return { days, time: `${days === 0 ? "今天" : t.date?.slice(5).replace("-", "/")} ${deadlineTime(t)}`.trim(),
      title: titleForDeadline(t.raw), ref: t.ref || t.path, task: t };
  }).filter(t => t.days >= 0 && (t.days <= 14 || scheduled.some(s => sameRef(s.ref, t.ref))));
  // Today contributes optional events and appointments; dated obligations always come from their originals.
  const extra = todayNotices(data, now).filter(t => {
    if (originals.some(o => sameRef(t.ref, o.ref))) return false;
    const path = t.ref.split("#")[0].replace(/\.md$/, "");
    if (t.ref.includes("#^") && path.split("/").pop() === "截止事项记录") return false;
    return !(data.tasks || []).some(o => sameRef(t.ref, o.ref) && ["recorded_complete", "cancelled"].includes(o.status));
  });
  return [...canonical, ...extra].sort((a, b) => a.days - b.days || (a.time.match(/\d{2}:\d{2}/)?.[0] || "").localeCompare(b.time.match(/\d{2}:\d{2}/)?.[0] || ""));
}

function sectionForPath(path, todayPath) {
  const base = path?.split("#")[0];
  return base === WORKBENCH ? "workbench" : base === DEADLINES ? "deadlines" : base && base === todayPath ? "today" : null;
}

class StewardBridge {
  constructor(plugin) {
    this.plugin = plugin;
    this.children = new Set();
    this.listeners = new Set();
    this.busy = false;
    this.status = "正在读取记录…";
    this.activeRequest = null;
    this.activeTask = null;
    this.lastState = null;
  }
  notify() { for (const fn of this.listeners) fn(); }
  isSource(path) {
    if (!path) return false;
    if ([WORKBENCH, DEADLINES, OBLIGATIONS, "99_学习情况记录/Overview & Study Record.md", "99_学习情况记录/Student OS 规则.md"].includes(path)) return true;
    if (/^99_学习情况记录\/\d{4}-\d{2}-\d{2}——\w+\.md$/.test(path)) return true;
    const data = this.lastState;
    if (!data) return false;
    const paths = [data.today?.path, ...(data.tasks || []).map(t => t.path), ...(data.obligations || []).map(t => t.path),
      ...(data.curriculum?.source_paths || []), ...(data.curriculum?.courses || []).flatMap(c =>
        [c.plan, ...(c.school?.sources || []).map(r => r.path), ...(c.personal?.sources || []).map(r => r.path)])];
    return paths.some(p => p?.split("#")[0] === path);
  }
  sourceChanged(path) {
    if (this.stopped || !this.isSource(path)) return;
    clearTimeout(this.refreshTimer);
    this.refreshTimer = setTimeout(() => { void this.refresh(true).catch(() => {}); }, 700);
  }
  call(command, input = {}, onStatus = null) {
    const vault = this.plugin.app.vault.adapter.getBasePath();
    const script = join(vault, "05_tools", "StudentOS", "steward", "service.py");
    const python = ["/opt/homebrew/bin/python3", "/usr/bin/python3"].find(existsSync);
    if (!python || !existsSync(script)) return Promise.reject(new Error("管家运行文件尚不可用"));
    return new Promise((resolve, reject) => {
      const child = spawn(python, ["-B", script, "--vault", vault, command], {
        cwd: vault, env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" }, stdio: ["pipe", "pipe", "pipe"]
      });
      this.children.add(child);
      let buffer = "", result, failure = "", stderr = "";
      const consume = line => {
        if (!line.trim()) return;
        try {
          const event = JSON.parse(line);
          if (event.event === "result") result = event.data;
          if (event.event === "error") failure = event.data;
          if (event.event === "status") onStatus?.(event.data);
        } catch { failure = "管家回执格式不正确；原请求保留。"; }
      };
      child.stdout.on("data", chunk => {
        buffer += chunk.toString();
        const lines = buffer.split("\n"); buffer = lines.pop();
        lines.forEach(consume);
      });
      child.stderr.on("data", chunk => { stderr = (stderr + chunk).slice(-4000); });
      child.on("error", error => { this.children.delete(child); reject(error); });
      child.on("close", code => {
        this.children.delete(child);
        consume(buffer);
        if (result !== undefined) resolve(result);
        else reject(new Error(failure || stderr || `管家进程已退出 (${code})；请继续原请求。`));
      });
      child.stdin.on("error", () => {});
      child.stdin.end(JSON.stringify(input));
    });
  }
  async refresh(invalidated = false) {
    if (this.stopped) return this.lastState;
    if (this.refreshing) {
      this.refreshQueued ||= invalidated;
      return this.refreshing;
    }
    this.refreshing = this.call("view").then(data => {
      this.lastState = data;
      if (!this.busy) this.status = "记录已同步";
      this.notify(); return data;
    }).catch(error => { this.status = `读取暂未成功${this.lastState ? "，保留上次记录" : ""}：${error.message}`; this.notify(); throw error; })
      .finally(() => {
        this.refreshing = null;
        if (this.refreshQueued && !this.stopped) { this.refreshQueued = false; void this.refresh().catch(() => {}); }
      });
    return this.refreshing;
  }
  async run(command, input) {
    if (this.busy) throw new Error("当前请求正在处理，可以停止它或稍后继续。");
    this.busy = true;
    this.activeRequest = ["chat", "resume"].includes(command) ? input?.key : null;
    this.activeCompletion = command === "completion-check" ? input?.key : null;
    this.activeTask = command === "prepare-focus" ? this.lastState?.focus?.task?.id
      : input?.action?.kind === "prepare_classroom" ? input.action.task_id : null;
    this.status = "正在处理…"; this.notify();
    try {
      const result = await this.call(command, input, value => {
        this.status = value; this.notify();
        void this.refresh().catch(() => {});
      });
      this.status = result.error || result.message || "已保存";
      if (result.error) new Notice(result.error);
      return result;
    } finally {
      this.busy = false; this.activeRequest = null; this.activeTask = null; this.activeCompletion = null;
      await this.refresh().catch(() => {});
    }
  }
  async act(action, userText, key = id()) {
    return this.run("act", { key, action: { ...action, source_quote: userText }, user_text: userText });
  }
  async cancelClassroom(job) {
    const userText = "取消这次课堂派工";
    const result = await this.call("act", { key: id(), action: { kind: "cancel_classroom", job_key: job.key, source_quote: userText }, user_text: userText });
    await this.refresh(); return result;
  }
  async cancelActive() {
    if (this.activeCompletion) await this.call("completion-cancel", { key: this.activeCompletion });
    else if (this.activeRequest) await this.call("cancel-request", { key: this.activeRequest });
    else if (this.activeTask) {
      const data = await this.refresh();
      const job = data.jobs.find(j => j.request?.task?.id === this.activeTask && !j.handed_off && j.state !== "cancelled");
      if (job) await this.cancelClassroom(job);
    }
    await this.refresh();
  }
  async start() {
    const data = await this.refresh();
    if (this.stopped || !data) return;
    const model = homeModel(data);
    const pending = data.chat.some(r => !["done", "cancelled"].includes(r.phase));
    const known = data.jobs.some(j => j.request?.task?.id === model.task?.id);
    if (model.task && !known && !pending && !this.busy) {
      // New Today tasks may be prepared; a handoff or cancellation needs an explicit next request.
      await this.run("prepare-focus").catch(error => { this.status = error.message; this.notify(); });
    }
  }
  stop() {
    this.stopped = true; this.refreshQueued = false;
    clearTimeout(this.refreshTimer);
    for (const child of this.children) child.kill("SIGTERM");
    this.children.clear();
  }
}

class EvidencePicker extends FuzzySuggestModal {
  constructor(app, choose) { super(app); this.choose = choose; this.setPlaceholder("搜索笔记、代码、作业或提交凭证…"); }
  getItems() { return this.app.vault.getFiles().filter(f => /^(md|txt|pdf|r|rmd|py|ipynb|html|csv|json|png|jpg|jpeg|heic|webp)$/i.test(f.extension)); }
  getItemText(file) { return file.path; }
  onChooseItem(file) { this.choose(file.path); }
}

class CompletionModal extends Modal {
  constructor(plugin, task, report = "") { super(plugin.app); this.plugin = plugin; this.task = task; this.report = report; this.files = []; }
  async onOpen() {
    this.titleEl.setText("任务处理"); this.contentEl.addClass("sos-completion-modal");
    this.contentEl.createEl("p", { cls: "sos-completion-task", text: displayText(this.task.title || this.task.text || this.task.raw) });
    const intro = this.contentEl.createEl("p", { cls: "sos-muted", text: "完成后核验并记录；不再需要的任务可直接关闭，保留原因并退出待办。" });
    const reportLabel = this.contentEl.createEl("label", { cls: "sos-field", text: "完成情况／关闭原因" });
    this.reportInput = reportLabel.createEl("textarea", { attr: { rows: "3", placeholder: "例如：完成 Exercise 1 第 1–4 题；或：这节课已经上完，不再需要这项预习。部分进度可用“记录停点”。" } });
    this.reportInput.value = this.report;
    const dateLabel = this.contentEl.createEl("label", { cls: "sos-field", text: "实际完成日期（不确定就留空）" });
    this.dateInput = dateLabel.createEl("input", { attr: { type: "date", max: londonClock().date } });
    const today = this.todayButton = dateLabel.createEl("button", { text: "今天", cls: "sos-quiet", attr: { type: "button" } });
    today.addEventListener("click", () => { this.dateInput.value = londonClock().date; });
    this.fileList = this.contentEl.createDiv({ cls: "sos-evidence-files" });
    const add = this.addFile = this.contentEl.createEl("button", { text: "关联笔记或凭证…" });
    add.addEventListener("click", () => new EvidencePicker(this.app, path => {
      if (!this.files.includes(path) && this.files.length < 6) this.files.push(path);
      this.renderFiles();
    }).open());
    this.contentEl.createEl("p", { cls: "sos-muted", text: "核验会读取关联文件的内容；作业文件与提交成功凭证分别判断。" });
    this.resultEl = this.contentEl.createDiv({ cls: "sos-check-result", attr: { role: "status", "aria-live": "polite" } });
    const actions = this.contentEl.createDiv({ cls: "sos-actions" });
    this.submit = actions.createEl("button", { cls: "mod-cta", text: "核验并记录完成" });
    this.submit.disabled = true;
    this.closeTask = actions.createEl("button", { text: "不再需要，关闭任务" }); this.closeTask.disabled = true;
    this.closeTask.addEventListener("click", () => this.check("close"));
    this.stop = actions.createEl("button", { text: "停止核验" }); this.stop.hidden = true;
    this.stop.addEventListener("click", () => this.plugin.steward.cancelActive().catch(e => new Notice(e.message)));
    this.restartButton = actions.createEl("button", { text: "补充后另开核验", cls: "sos-quiet" }); this.restartButton.hidden = true;
    this.restartButton.addEventListener("click", async () => {
      try {
        const result = await this.plugin.steward.call("completion-cancel", { key: this.resumeKey });
        if (result.phase !== "cancelled") { new Notice(result.message); return; }
        this.resumeKey = null; this.restartButton.hidden = true; this.lockForm(false); this.renderFiles(); this.submit.setText("核验并记录完成");
      } catch (error) { new Notice(error.message); }
    });
    this.submit.addEventListener("click", () => this.check());
    try {
      const context = await this.plugin.steward.call("completion-context", { path: this.task.path, raw: this.task.raw });
      if (!this.contentEl.isConnected) return;
      if (context.required_external) intro.setText("记录提交成功需要匹配的凭证；也可按本人决定关闭任务，关闭不会记作已提交。");
      const latest = context.latest;
      this.files = latest?.request.evidence || context.suggested_evidence;
      if (latest && !this.report) this.reportInput.value = latest.request.report;
      this.dateInput.value = latest?.request.date || "";
      if (latest) this.showResult({ ...latest, path: latest.export_path });
      if (latest && !["complete", "closed", "insufficient", "cancelled"].includes(latest.phase)) {
        this.resumeKey = latest.key; this.submit.setText("继续原核验");
        this.restartButton.hidden = false;
        this.reportInput.disabled = this.dateInput.disabled = add.disabled = today.disabled = true;
      }
      this.renderFiles(); this.submit.disabled = this.closeTask.disabled = false;
    } catch (error) { this.resultEl.setText(error.message); }
  }
  renderFiles() {
    this.fileList.empty();
    for (const path of this.files) {
      const row = this.fileList.createDiv(); row.createSpan({ text: path });
      if (!this.resumeKey) {
        const remove = row.createEl("button", { text: "移除", cls: "sos-quiet" });
        remove.addEventListener("click", () => { this.files = this.files.filter(p => p !== path); this.renderFiles(); });
      }
    }
  }
  showResult(result) {
    this.resultEl.empty();
    const labels = { complete: "✓ 完成已记录", closed: "已关闭 · 不再需要", insufficient: "待核验 · 需要补充", cancelled: "核验已停止" };
    this.resultEl.createEl("strong", { text: labels[result.phase] || "核验尚未完成，可继续原请求" });
    const verdict = result.verdict;
    if (verdict) {
      this.resultEl.createEl("p", { text: verdict.reason });
      if (verdict.missing) this.resultEl.createEl("p", { text: "还缺：" + verdict.missing });
      this.resultEl.createEl("small", { text: result.phase === "closed" ? "依据：本人决定。已退出待办，未计为完成。" : { user_report: "依据：本人明确报告", artifact: "依据：已读取的产物内容", external: "依据：已读取的外部结果凭证" }[verdict.basis] });
    }
    if (result.error) this.resultEl.createEl("p", { text: result.error });
    if (result.path) {
      const link = this.resultEl.createEl("button", { text: "查看处理记录 ↗", cls: "sos-quiet" });
      link.addEventListener("click", () => this.app.workspace.openLinkText(result.path, "", true));
    }
  }
  async check(intent = "complete") {
    const resumeKey = intent === "close" ? null : this.resumeKey;
    if (intent !== "close" && !resumeKey && !this.reportInput.value.trim()) { new Notice("请说明这项任务实际完成的范围。"); this.reportInput.focus(); return; }
    this.submit.disabled = this.closeTask.disabled = true; this.stop.hidden = false; this.lockForm(true);
    this.restartButton.hidden = true;
    const key = resumeKey || id();
    const request = intent === "close"
      ? { path: this.task.path, raw: this.task.raw, report: this.reportInput.value.trim() || "不再需要，关闭这项任务。", date: "", evidence: [], intent: "close" }
      : { path: this.task.path, raw: this.task.raw, report: this.reportInput.value.trim(), date: this.dateInput.value, evidence: this.files };
    this.resultEl.setText(intent === "close" ? "正在保存关闭决定…" : "正在核对任务意图与证据内容…可关闭面板，结果会保存在任务中。");
    try {
      const result = await this.plugin.steward.run("completion-check", resumeKey ? { key } : { key, request });
      await this.plugin.refreshViews(); this.showResult(result);
      if (["complete", "closed"].includes(result.phase)) { this.terminalPhase = result.phase; this.submit.setText(result.phase === "closed" ? "已关闭" : "已记录完成"); new Notice(result.message); }
      else { this.resumeKey = null; this.submit.disabled = false; this.submit.setText("补充后重新核验"); this.lockForm(false); this.renderFiles(); }
    } catch (error) {
      this.resultEl.setText(error.message + "；请刷新查看原任务或继续原请求。");
      const current = (this.plugin.steward.lastState?.completion_checks || []).find(c => c.key === key);
      const resumable = current && !["complete", "closed", "insufficient", "cancelled"].includes(current.phase);
      this.resumeKey = resumable ? key : null;
      this.restartButton.hidden = !resumable;
      this.submit.disabled = false; this.submit.setText(resumable ? "继续原核验" : "重新核验");
      this.lockForm(!!resumable);
    } finally { this.stop.hidden = true; this.closeTask.disabled = !!this.terminalPhase; }
  }
  lockForm(locked) {
    for (const control of [this.reportInput, this.dateInput, this.addFile, this.todayButton, ...this.fileList.querySelectorAll("button")]) control.disabled = locked;
  }
  onClose() { this.contentEl.empty(); }
}

function renderedTask(node, data) {
  const source = node.querySelector(".task-description [data-code-space-source-path]")?.dataset.codeSpaceSourcePath;
  if (!source) return null;
  const tasks = [...(data?.tasks || []), ...(data?.obligations || [])].filter(t => t.path === source);
  const anchor = node.querySelector(".task-block-link")?.textContent.trim();
  const plain = value => displayText(value).replace(/#student-os\/(?:task|deadline)/g, "").replace(/\s+/g, " ").trim();
  const description = plain(node.querySelector(".task-description")?.textContent || "");
  const matches = anchor ? tasks.filter(t => t.ref?.endsWith("#" + anchor)) : tasks.filter(t => plain(t.title) === description);
  return matches.length === 1 ? matches[0] : null;
}

function installCompletionGuard(plugin) {
  plugin.openCompletion = (task, report = "") => new CompletionModal(plugin, task, report).open();
  const guard = event => {
    if (!event.target?.matches?.("input.task-list-item-checkbox")) return;
    const row = event.target.closest(".plugin-tasks-list-item");
    if (!row || row.dataset.taskStatusType === "DONE" || row.dataset.taskStatusType === "CANCELLED") return;
    const isStudentTask = row.querySelector('[data-tag-name="#student-os/task"], [data-tag-name="#student-os/deadline"]') || row.closest(".sos-management-note");
    if (!isStudentTask) return;
    event.preventDefault(); event.stopImmediatePropagation();
    const task = renderedTask(row, plugin.steward.lastState);
    if (task?.status === "completion_unknown") plugin.openCompletion(task);
    else new Notice("尚未唯一定位任务原件，请刷新后从每周任务发起核验。");
  };
  plugin.registerDomEvent(document, "click", guard, true);
  plugin.registerDomEvent(document, "contextmenu", guard, true);
}

class ProgressModal extends Modal {
  constructor(plugin, task, job) { super(plugin.app); this.plugin = plugin; this.task = task; this.job = job; }
  onOpen() {
    this.titleEl.setText("保存这次学习的停点");
    this.contentEl.createEl("p", { text: courseTerms(this.task.title) });
    const field = (name, placeholder) => {
      const label = this.contentEl.createEl("label", { cls: "sos-field", text: name });
      return label.createEl("textarea", { attr: { rows: "3", placeholder } });
    };
    const summary = field("这次实际做到哪里", "具体页码、题目、回答、产物或仍未理解的地方");
    const next = field("下次第一步", "打开哪个文件，从哪道题或哪个问题继续");
    const evidence = field("已有产物（可留空）", "vault 内的相对文件路径，每行一个");
    this.contentEl.createEl("p", { cls: "sos-muted", text: "保存本人报告与证据位置；掌握和正式提交状态分别核验。" });
    const save = this.contentEl.createEl("button", { cls: "mod-cta", text: "保存停点" });
    const operationId = id();
    save.addEventListener("click", async () => {
      if (!summary.value.trim() || !next.value.trim()) { new Notice("请写实际范围和下次第一步。"); return; }
      save.disabled = true;
      const quote = `${summary.value.trim()}\n下次：${next.value.trim()}`;
      try {
        const receipt = await this.plugin.steward.act({ kind: "record_progress", task_id: this.task.id,
          job_key: this.job?.key || "", summary: summary.value.trim(), next_step: next.value.trim(),
          evidence: evidence.value.split("\n").map(s => s.trim()).filter(Boolean) }, quote, operationId);
        new Notice(receipt.message); this.close();
      } catch (error) { new Notice(error.message); save.disabled = false; }
    });
  }
  onClose() { this.contentEl.empty(); }
}

class StewardView extends ItemView {
  constructor(leaf, plugin) { super(leaf); this.plugin = plugin; this.bridge = plugin.steward; this.draft = ""; }
  getViewType() { return STEWARD_VIEW; }
  getDisplayText() { return "Student OS 管家"; }
  getIcon() { return "bot"; }
  getState() { return { draft: this.input?.value ?? this.draft, course: this.selectedCourse || null, week: this.selectedWeek || null,
    filter: this.taskFilter || "all", section: this.section || "overview", context: this.chatContext || null }; }
  async setState(state) {
    this.draft = state?.draft || ""; this.selectedCourse = state?.course || null; this.selectedWeek = state?.week || null;
    this.taskFilter = state?.filter || "all"; this.section = SECTIONS[state?.section] ? state.section : "overview";
    this.chatContext = state?.context || null;
    if (this.input) this.input.value = this.draft; this.render();
  }
  async onOpen() {
    this.contentEl.empty(); this.contentEl.addClass("sos-home");
    const header = this.contentEl.createDiv({ cls: "sos-header" });
    const heading = header.createDiv();
    heading.createEl("h1", { text: "我的学习" });
    this.termEl = heading.createEl("p", { cls: "sos-header-context" });
    const nav = header.createDiv({ cls: "sos-nav" });
    this.button(nav, "刷新", () => this.bridge.refresh(), "sos-quiet");
    this.statusEl = nav.createDiv({ cls: "sos-status", attr: { role: "status", "aria-live": "polite" } });
    const tabs = this.contentEl.createEl("nav", { cls: "sos-workspace-nav", attr: { "aria-label": "Student OS 工作区" } });
    for (const [key, label] of Object.entries(SECTIONS)) {
      const button = this.button(tabs, label, () => this.selectSection(key), "sos-section-tab");
      button.dataset.section = key;
    }
    const grid = this.contentEl.createDiv({ cls: "sos-grid" });
    this.board = grid.createDiv({ cls: "sos-board" });
    const side = grid.createDiv({ cls: "sos-side" });
    this.classroom = side.createDiv({ cls: "sos-classroom" });
    const chat = side.createDiv({ cls: "sos-chat" });
    const chatHeading = chat.createDiv({ cls: "sos-chat-heading" });
    setIcon(chatHeading.createSpan({ cls: "sos-icon" }), "bot");
    chatHeading.createEl("h2", { text: "学习管家" });
    chat.createEl("p", { cls: "sos-muted", text: "安排、备课和记录进度，都在这里。" });
    const shortcuts = chat.createDiv({ cls: "sos-shortcuts" });
    for (const [label, value] of [["下一步做什么", "我现在先做什么？"], ["准备课堂", "帮我准备当前课堂"]]) {
      this.button(shortcuts, label, () => this.send(value));
    }
    const composer = chat.createDiv({ cls: "sos-composer" });
    this.contextEl = composer.createDiv({ cls: "sos-chat-context" });
    this.input = composer.createEl("textarea", { attr: { rows: "3", placeholder: "告诉管家你的安排、进度或变更…", "aria-label": "给 Student OS 管家的消息" } });
    this.input.value = this.draft;
    this.input.addEventListener("input", () => this.app.workspace.requestSaveLayout());
    this.input.addEventListener("keydown", event => {
      if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) { event.preventDefault(); event.stopPropagation(); void this.send(); }
    });
    const row = composer.createDiv({ cls: "sos-actions" });
    this.sendButton = this.button(row, "发送", () => this.send(), "mod-cta");
    this.stopButton = this.button(row, "停止处理", () => this.bridge.cancelActive());
    this.messages = chat.createDiv({ cls: "sos-messages", attr: { role: "log", "aria-label": "管家对话记录" } });
    this.radar = side.createDiv({ cls: "sos-radar" });
    this.renderCallback = () => this.render();
    this.bridge.listeners.add(this.renderCallback);
    // Native Tasks queries render asynchronously; translate their labels without touching source text or links.
    this.termsObserver = new MutationObserver(records => {
      for (const record of records) {
        if (record.type === "characterData") renderCourseTerms(record.target);
        else record.addedNodes.forEach(renderCourseTerms);
      }
    });
    this.termsObserver.observe(this.contentEl, { childList: true, subtree: true, characterData: true });
    this.render();
    await this.bridge.refresh().catch(() => {});
  }
  async onClose() { this.termsObserver?.disconnect(); this.bridge.listeners.delete(this.renderCallback); this.clearManagement(); }
  button(parent, label, action, cls = "") {
    const button = parent.createEl("button", { text: label, cls });
    button.addEventListener("click", () => Promise.resolve().then(action).catch(error => new Notice(error.message)));
    return button;
  }
  async openNote(path) {
    if (!path) return;
    const section = sectionForPath(path, this.bridge.lastState?.today?.path);
    if (section) { this.selectSection(section, path.split("#")[1]); return; }
    return this.openOriginal(path);
  }
  async openOriginal(path) {
    if (!path) return;
    const file = this.app.vault.getAbstractFileByPath(path.split("#")[0]);
    if (!file) { new Notice("记录文件尚不存在"); return; }
    const existing = this.app.workspace.getLeavesOfType("markdown").find(l => l.view.file?.path === file.path);
    const leaf = existing || this.app.workspace.getLeaf("tab");
    await leaf.openLinkText(path, "", {});
    await this.app.workspace.revealLeaf(leaf);
    this.plugin.addStewardReturn?.(leaf);
  }
  selectSection(section, heading = null) {
    if (!SECTIONS[section]) return;
    this.section = section; this.managementHeading = heading;
    this.app.workspace.requestSaveLayout(); this.render();
    this.contentEl.scrollTop = 0;
  }
  clearManagement() {
    this.managementGeneration = (this.managementGeneration || 0) + 1;
    if (this.managementComponent) this.removeChild(this.managementComponent);
    this.managementComponent = null; this.managementSource = null;
  }
  async renderManagement(data) {
    const path = this.section === "workbench" ? WORKBENCH : this.section === "deadlines" ? DEADLINES : data.today?.path;
    let generation = this.managementGeneration = (this.managementGeneration || 0) + 1;
    const file = path && this.app.vault.getAbstractFileByPath(path);
    try {
      if (!file) throw new Error("记录文件尚未生成，请稍后刷新。");
      const text = await this.app.vault.read(file);
      if (generation !== this.managementGeneration || this.section === "overview") return;
      if (this.managementSource !== path + text) {
        this.clearManagement(); this.board.empty();
        generation = this.managementGeneration;
        this.managementSource = path + text;
        const actions = this.board.createDiv({ cls: "sos-management-actions" });
        actions.createSpan({ text: "点击勾选发起完成核验；通过后自动同步原任务与总览。" });
        this.button(actions, "打开原笔记 ↗", () => this.openOriginal(path), "sos-quiet");
        const note = this.board.createDiv({ cls: "sos-management-note markdown-rendered student-os-task-view" });
        note.dataset.source = path;
        note.addEventListener("click", event => {
          const link = event.target.closest("a.internal-link");
          if (!link || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
          const target = link.getAttribute("data-href") || link.getAttribute("href");
          const dest = this.app.metadataCache.getFirstLinkpathDest(target.split("#")[0] || path, path);
          const section = sectionForPath(dest?.path, data.today?.path);
          if (section) { event.preventDefault(); event.stopPropagation(); this.selectSection(section, target.split("#")[1]); }
        }, true);
        this.managementComponent = this.addChild(new Component());
        await MarkdownRenderer.render(this.app, text.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n/, ""), note, path, this.managementComponent);
      }
      if (generation !== this.managementGeneration) return;
      this.board.querySelector(".sos-management-error")?.remove();
      if (this.managementHeading) {
        const target = Array.from(this.board.querySelectorAll("h1,h2,h3,h4,h5,h6,.callout-title-inner")).find(el =>
          el.textContent === this.managementHeading || el.textContent.startsWith(this.managementHeading + " ·"));
        if (target?.closest(".callout")?.classList.contains("is-collapsed")) target.closest(".callout-title").click();
        target?.scrollIntoView({ block: "start" }); this.managementHeading = null;
      }
    } catch (error) {
      if (generation !== this.managementGeneration) return;
      this.managementSource = null;
      this.board.querySelector(".sos-management-error")?.remove();
      this.board.createEl("p", { cls: "sos-error sos-management-error", text: `当前页面未更新：${error.message}` });
    }
  }
  discussTask(task) {
    this.chatContext = { ref: task.ref || task.path, title: taskCaption(task) };
    this.renderChatContext(); this.app.workspace.requestSaveLayout();
    this.input.scrollIntoView({ block: "nearest" }); this.input.focus();
  }
  renderChatContext() {
    this.contextEl.empty(); this.contextEl.hidden = !this.chatContext;
    if (!this.chatContext) return;
    this.contextEl.createSpan({ text: "正在讨论：" + this.chatContext.title });
    this.button(this.contextEl, "移除", () => { this.chatContext = null; this.renderChatContext(); this.app.workspace.requestSaveLayout(); }, "sos-quiet");
  }
  async send(body = null) {
    body = (body || this.input.value).trim();
    if (!body || this.bridge.busy) return;
    const draftBody = body, context = this.chatContext;
    if (context) body += `\n\n讨论的原任务：${context.title}\n原记录：[[${context.ref}]]`;
    const unfinished = this.bridge.lastState?.chat.find(r => r.body === body && !["done", "cancelled"].includes(r.phase));
    const key = unfinished?.key || id();
    const result = await this.bridge.run("chat", { key, body });
    if (result.phase === "done" && this.input.value.trim() === draftBody) {
      this.input.value = ""; this.draft = ""; this.app.workspace.requestSaveLayout();
    }
    if (result.phase === "done" && this.chatContext === context) { this.chatContext = null; this.renderChatContext(); this.app.workspace.requestSaveLayout(); }
  }
  async openClassroom(job) {
    let entry = this.openingClassroom;
    if (!entry || entry.key !== job.key) {
      entry = { key: job.key, work: this.loadClassroom(job) };
      this.openingClassroom = entry;
      entry.work.finally(() => { if (this.openingClassroom === entry) this.openingClassroom = null; }).catch(() => {});
    }
    let timer;
    try {
      await Promise.race([entry.work, new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("原课堂已找到；Claudian 仍在加载。恢复窗口显示后会继续加载，不会重复创建课堂。")), 15000); })]);
    } finally { clearTimeout(timer); }
  }
  async loadClassroom(job) {
    const plugin = this.app.plugins.plugins.realclaudian;
    if (!plugin?.chatHost) throw new Error("请先启用 Claudian，原课堂已保留。");
    await plugin.chatHost.ensureConversationMetadataLoaded?.([job.conversation_id]);
    const conversation = await plugin.chatHost.getConversationById(job.conversation_id);
    if (conversation?.sessionId !== job.thread_id || !conversation.messages?.length) throw new Error("课堂历史尚未读回，未创建替代对话。");
    if (plugin.views?.activateView) await plugin.views.activateView();
    else await this.app.commands.executeCommandById("realclaudian:open-view");
    const leaf = this.app.workspace.getLeavesOfType("claudian-view")[0];
    if (!leaf) throw new Error("Claudian 视图尚不可用，课堂已保存在会话列表。");
    await this.app.workspace.revealLeaf(leaf);
    // A restored, hidden pane may have delivered its state before layout was ready.
    // Re-deliver that exact state through the view API; do not replace its tabs.
    if (leaf.view.tabWorkspace?.isReady === false) await leaf.view.setState(leaf.view.getState(), {});
    await leaf.view.getTabManager().openConversation(conversation.id, { activate: true, preferNewTab: true });
    const tab = leaf.view.getActiveTab();
    if (tab?.state.currentConversationId !== job.thread_id || !tab.state.messages?.length) throw new Error("课堂打开尚未确认；原会话保持不变。");
    this.plugin?.addStewardReturn?.(leaf);
  }
  render() {
    if (!this.board) return;
    this.statusEl.setText(this.bridge.status === "记录已同步" ? "自动同步" : this.bridge.status);
    this.statusEl.setAttribute("title", "记录变化时自动刷新；每分钟检查换日和更新。读取与校验不调用大模型。" + (this.bridge.lastState?.at ? ` 上次读取：${this.bridge.lastState.at}` : ""));
    this.sendButton.disabled = this.bridge.busy;
    this.stopButton.hidden = !this.bridge.activeRequest && !this.bridge.activeTask;
    const data = this.bridge.lastState;
    if (!data) { this.board.setText("正在读取已有的 Student OS 记录…"); return; }
    const home = homeModel(data), agenda = agendaFor(data);
    const section = this.section || "overview";
    this.contentEl.querySelectorAll(".sos-section-tab").forEach(button => button.setAttribute("aria-current", button.dataset.section === section ? "page" : "false"));
    this.termEl.setText(`${data.curriculum?.stage || "课程与学习位置"}  ·  ${agenda.date.slice(5).replace("-", "月")}日`);
    const expanded = new Set(Array.from(this.contentEl.querySelectorAll("details[data-key]")).filter(d => d.open).map(d => d.dataset.key));
    if (section === "overview") {
      this.clearManagement(); this.board.empty();
      this.renderDueToday(data);
      this.renderAgenda(data, agenda);
      this.renderCurriculum(data, home);
      this.renderDetails(data);
    } else {
      if (this.renderedSection !== section) { this.clearManagement(); this.board.empty(); }
      void this.renderManagement(data);
    }
    this.renderedSection = section;
    this.renderClassroom(data, home, agenda);
    this.renderRadar(data, home);
    this.contentEl.querySelectorAll("details[data-key]").forEach(d => { d.open = expanded.has(d.dataset.key); });
    this.renderChat(data);
    this.renderChatContext();
    this.lastRender = { taskId: home.task?.id || null, classroomId: home.job?.thread_id || null,
      classroomReady: home.job?.state === "registered", chatCount: data.chat.length,
      progressCount: data.progress.length, sourceStatus: data.source_status,
      courseCount: data.curriculum?.courses.length || 0,
      visibleCourseIds: (data.curriculum?.courses || []).filter(c => ["degree", "foundation"].includes(c.group)).map(c => c.id),
      schoolStage: data.curriculum?.stage, scheduledNow: agenda.current?.label || null };
  }
  renderDueToday(data) {
    const items = deadlineNotices(data).filter(t => t.task && t.days === 0);
    if (!items.length) return;
    const banner = this.board.createEl("section", { cls: "sos-due-today", attr: { "aria-label": "今天到期的未关闭事项" } });
    banner.createEl("strong", { text: "今天到期" });
    for (const item of items) {
      const button = this.button(banner, "", () => this.selectSection("deadlines"), "sos-due-item");
      button.createSpan({ cls: "sos-due-time", text: deadlineTime(item.task) || "今天" });
      button.createSpan({ text: item.title });
    }
  }
  renderAgenda(data, agenda) {
    const section = this.board.createEl("section", { cls: "sos-day", attr: { "aria-label": "按今日安排" } });
    const heading = section.createDiv({ cls: "sos-section-heading" });
    heading.createEl("h2", { text: "现在与接下来" });
    this.button(heading, "按今日安排 ↗", () => this.openNote(data.today.path + "#今日骨架"), "sos-quiet");
    const steps = [agenda.current, ...agenda.upcoming].filter(Boolean).slice(0, 3);
    if (!steps.length) { section.createEl("p", { cls: "sos-muted", text: "当前没有可读取的后续时段，打开今日安排查看。" }); return; }
    const strip = section.createDiv({ cls: "sos-agenda" });
    for (const [i, slot] of steps.entries()) {
      const active = slot === agenda.current;
      const step = strip.createDiv({ cls: `sos-agenda-step${active ? " is-now" : ""}` });
      const top = step.createDiv({ cls: "sos-agenda-top" });
      top.createSpan({ cls: "sos-agenda-state", text: active ? "现在" : i === 0 || (agenda.current && i === 1) ? "接下来" : "随后" });
      top.createSpan({ cls: "sos-agenda-time", text: `${slot.start}–${slot.end}` });
      step.createEl("h3", { text: slot.label });
      if (slot.location) step.createDiv({ cls: "sos-agenda-location", text: slot.location });
    }
  }
  renderClassroom(data, home, agenda) {
    const expanded = this.classroom.querySelector("details")?.open;
    this.classroom.empty();
    const ready = home.job?.state === "registered";
    const head = this.classroom.createDiv({ cls: "sos-classroom-heading" });
    setIcon(head.createSpan({ cls: "sos-icon" }), ready ? "book-open-check" : "book-open");
    head.createEl("h2", { text: ready ? "已备好的课堂" : "学习课堂" });
    if (!home.task) { this.classroom.createEl("p", { text: "跟管家确定下一件学习任务，再准备课堂。" }); return; }
    const slot = agenda.slots.find(s => s.ref?.split("#^")[1] && home.task.ref?.endsWith("#^" + s.ref.split("#^")[1]));
    const parts = home.task.title.split("：");
    this.classroom.createDiv({ cls: "sos-classroom-unit", text: parts.length > 1 ? parts[0] : "当前学习任务" });
    this.classroom.createEl("h3", { text: (parts.length > 1 ? parts.slice(1).join("：") : home.task.title).replace(/（原课时[^）]*）/, "").replace(/^依据课堂记录/, "") });
    if (slot) this.classroom.createDiv({ cls: "sos-classroom-time", text: `今日安排 ${slot.start}–${slot.end}${slot.location ? " · " + slot.location : ""}` });
    else this.classroom.createDiv({ cls: "sos-classroom-time", text: "准备就绪后可随时进入" });
    if (ready) this.button(this.classroom, "进入已准备课堂 →", () => this.openClassroom(home.job), "mod-cta sos-enter");
    else this.button(this.classroom, home.job ? "继续原备课" : "准备课堂", () => this.prepareTask(home.task), "mod-cta sos-enter");
    const actions = this.classroom.createDiv({ cls: "sos-classroom-actions" });
    this.button(actions, "学完记录停点", () => new ProgressModal(this.plugin, home.task, home.job).open(), "sos-quiet");
    this.button(actions, "核验完成", () => this.plugin.openCompletion(home.task), "sos-quiet");
    const more = this.classroom.createEl("details", { cls: "sos-current-more", attr: { "data-key": "classroom-details" } });
    more.open = !!expanded;
    more.createEl("summary", { text: "接续说明" });
    more.createEl("p", { text: home.next });
    if (data.focus?.last_progress) more.createEl("p", { text: "上次停点：" + data.focus.last_progress });
    if (home.job) this.button(more, "取消这次派工", () => this.bridge.cancelClassroom(home.job), "sos-quiet");
    if (home.job?.error) this.classroom.createEl("p", { cls: "sos-error", text: home.job.error });
  }
  renderCurriculum(data, home) {
    const curriculum = data.curriculum;
    const section = this.board.createEl("section", { cls: "sos-curriculum" });
    const heading = section.createDiv({ cls: "sos-section-heading" });
    heading.createEl("h2", { text: "课程进度" });
    this.button(heading, "课程与依据 ↗", () => this.openNote(curriculum?.note_path), "sos-quiet");
    if (!curriculum?.courses.length) { section.createEl("p", { text: "课程记录暂未读到，刷新后再查看。" }); return; }
    const weekly = curriculum.weekly;
    if (!weekly) { section.createEl("p", { text: "正在整理原计划的每周任务…" }); return; }
    const overall = heading.createDiv({ cls: "sos-progress-totals" });
    this.button(overall, `✓ 已完成 ${weekly.counts.done} 项`, () => this.selectTasks(null, "all", "done"), "sos-total-done");
    this.button(overall, `◐ 有学习记录 ${weekly.counts.recorded} 项`, () => this.selectTasks(null, "all", "recorded"), "sos-quiet");
    const cards = section.createDiv({ cls: "sos-progress-cards" });
    for (const course of curriculum.courses.filter(c => c.group === "degree")) {
      const count = course.task_progress, personal = positionFor(course.personal);
      const card = this.button(cards, "", () => this.selectTasks(this.selectedCourse === course.id ? null : course.id, this.selectedWeek, "all"), "sos-progress-card");
      card.dataset.course = course.id;
      card.setAttribute("aria-pressed", String(this.selectedCourse === course.id));
      const top = card.createDiv({ cls: "sos-progress-card-top" });
      top.createSpan({ text: course.id });
      top.createSpan({ text: `本周 ${weekly.tasks.filter(t => t.course_id === course.id && t.week === String(curriculum.week) && t.state !== "cancelled").length} 项` });
      card.createEl("h3", { text: course.title });
      const number = card.createDiv({ cls: "sos-progress-number" });
      number.createEl("strong", { text: String(count.done) });
      number.createSpan({ text: ` / ${count.total} 项已记录完成` });
      const bar = card.createDiv({ cls: "sos-completion-bar", attr: { role: "img", "aria-label": `原计划 ${count.total} 项：已完成 ${count.done}，有学习记录 ${count.recorded}，待确认 ${count.unknown}，后续计划 ${count.planned}` } });
      for (const key of ["done", "recorded", "unknown", "planned"]) {
        if (count[key]) bar.createSpan({ cls: `is-${key}`, attr: { style: `flex:${count[key]}` } });
      }
      const states = card.createDiv({ cls: "sos-progress-states" });
      if (count.recorded) states.createSpan({ cls: "is-recorded", text: `${count.recorded} 项有记录` });
      if (count.unknown) states.createSpan({ text: `${count.unknown} 项待确认` });
      if (count.planned) states.createSpan({ text: `${count.planned} 项后续计划` });
      card.createDiv({ cls: "sos-progress-stop", text: `最近：${personal.topic}` });
    }
    this.renderWeekly(section, curriculum);
    const foundation = curriculum.courses.filter(c => c.group === "foundation");
    if (foundation.length) {
      const support = section.createDiv({ cls: "sos-foundation" });
      for (const course of foundation) {
        const personal = positionFor(course.personal);
        const card = support.createDiv({ cls: "sos-foundation-card" });
        const top = card.createDiv({ cls: "sos-foundation-top" });
        this.button(top, course.title, () => this.openCourse(course), "sos-foundation-title");
        top.createSpan({ cls: "sos-foundation-state", text: personal.label });
        if (personal.units?.length) {
          const units = card.createDiv({ cls: "sos-unit-dots", attr: { "aria-label": "本人确认已学的内容" } });
          for (const unit of personal.units) units.createSpan({ text: "✓ " + unit });
        } else card.createDiv({ cls: "sos-course-meta", text: personal.topic });
        if (personal.freshness) card.createDiv({ cls: "sos-course-meta", text: personal.freshness });
      }
    }
    const future = curriculum.courses.filter(c => c.group === "future");
    if (future.length) {
      const strip = section.createDiv({ cls: "sos-course-future" });
      strip.createSpan({ cls: "sos-future-label", text: "冬季学期" });
      for (const c of future) this.button(strip, `${c.title} · ${c.id}`, () => this.openCourse(c), "sos-quiet");
    }
    const others = curriculum.courses.filter(c => ["other", "queued"].includes(c.group));
    if (others.length) {
      const details = section.createEl("details", { cls: "sos-details", attr: { "data-key": "other-courses" } });
      details.createEl("summary", { text: `其他学习 · ${others.length} 条学习线` });
      for (const c of others) {
        const row = details.createDiv({ cls: "sos-history-row" });
        const text = row.createDiv({ cls: "sos-other-position" });
        text.createEl("strong", { text: `${c.id} · ${c.title}` });
        text.createDiv({ cls: "sos-course-meta", text: c.personal.summary });
        if (c.unit_progress?.total) {
          text.createEl("progress", { attr: { max: String(c.unit_progress.total), value: String(c.unit_progress.completed), "aria-label": `${c.id} 原计划已记录完成 ${c.unit_progress.completed} / ${c.unit_progress.total} 个单元` } });
          text.createSpan({ cls: "sos-unit-count", text: `原计划已记录 ${c.unit_progress.completed} / ${c.unit_progress.total}` });
        }
        this.button(row, "查看 ↗", () => this.openCourse(c), "sos-quiet");
      }
    }
    if (curriculum.issues.length) section.createEl("p", { cls: "sos-error", text: curriculum.cached_at
      ? `课程来源暂不可用，显示 ${curriculum.cached_at.slice(5, 10).replace("-", "/")} 保存的记录；恢复后自动更新。`
      : "部分课程来源暂不可用；已保留可读取的记录。" });
  }
  selectTasks(course, week, filter = "all") {
    this.selectedCourse = course; this.selectedWeek = week; this.taskFilter = filter;
    this.app.workspace.requestSaveLayout(); this.render();
  }
  renderWeekly(parent, curriculum) {
    const plan = curriculum.weekly;
    const key = this.selectedWeek || String(curriculum.week || "unassigned");
    const course = this.selectedCourse, filter = this.taskFilter || "all";
    const panel = parent.createEl("section", { cls: "sos-weekly", attr: { "aria-label": "每周具体任务与完成记录" } });
    const heading = panel.createDiv({ cls: "sos-section-heading" });
    heading.createEl("h2", { text: "每周任务" });
    this.button(heading, "回到本周", () => this.selectTasks(null, null), "sos-quiet");
    const weeks = panel.createDiv({ cls: "sos-week-selector", attr: { "aria-label": "选择周次" } });
    for (const week of plan.weeks) {
      const button = this.button(weeks, "", () => this.selectTasks(course, week.key), "sos-week-button");
      button.setAttribute("aria-pressed", String(key === week.key));
      button.classList.toggle("is-current", week.number === curriculum.week);
      button.createEl("strong", { text: `第 ${week.number} 周` });
      button.createSpan({ text: week.start.slice(5).replace("-", "/") });
    }
    const filters = panel.createDiv({ cls: "sos-week-filters" });
    const courseSelect = filters.createEl("select", { attr: { "aria-label": "筛选课程" } });
    courseSelect.createEl("option", { value: "", text: "所有学习课程" });
    for (const c of curriculum.courses.filter(c => c.task_progress?.total)) courseSelect.createEl("option", { value: c.id, text: `${c.id} ${c.title}` });
    courseSelect.value = course || "";
    courseSelect.addEventListener("change", () => this.selectTasks(courseSelect.value || null, this.selectedWeek, filter));
    for (const [value, label] of [["all", "全部状态"], ["verification", "待核验"], ["done", "已完成"], ["recorded", "有学习记录"], ["unknown", "待确认"]]) {
      const button = this.button(filters, label, () => this.selectTasks(course, this.selectedWeek, value), "sos-filter-button");
      button.setAttribute("aria-pressed", String(filter === value));
    }
    const extra = panel.createDiv({ cls: "sos-week-extra" });
    for (const [value, label] of [["all", "全部周次"], ["unassigned", "未分周"], ["outside", "其他日期"]]) {
      const n = plan.tasks.filter(t => (!course || t.course_id === course) && (value === "all" || t.week === value) && t.state !== "cancelled").length;
      const button = this.button(extra, `${label} ${n}`, () => this.selectTasks(course, value, filter), "sos-quiet");
      button.setAttribute("aria-pressed", String(key === value));
    }
    const inWeek = plan.tasks.filter(t => (!course || t.course_id === course) && (key === "all" || t.week === key) && t.state !== "cancelled");
    const tasks = inWeek.filter(t => filter === "all" || (filter === "verification" ? t.completion_check && !["complete", "closed", "cancelled"].includes(t.completion_check.phase) && t.state !== "done" : t.state === filter));
    const selected = plan.weeks.find(w => w.key === key);
    const caption = panel.createDiv({ cls: "sos-week-caption" });
    caption.createEl("strong", { text: selected ? `第 ${selected.number} 周 · ${selected.start.slice(5).replace("-", "/")}–${selected.end.slice(5).replace("-", "/")}` : key === "all" ? "全部周次" : key === "unassigned" ? "未分周的任务" : "其他日期的记录" });
    caption.createSpan({ text: `${inWeek.filter(t => t.state === "done").length} / ${inWeek.length} 项已记录完成${filter !== "all" ? ` · 当前显示 ${tasks.length} 项` : ""}` });
    if (!tasks.length) panel.createEl("p", { cls: "sos-empty", text: filter === "done" ? "这周还没有任务登记为完成。上方“已完成”总数可查看全部历史完成项。" : "这个筛选下没有任务记录。" });
    const states = { done: ["✓", "已完成"], recorded: ["◐", "有学习记录"], unknown: ["○", "待确认"], planned: ["·", "后续计划"] };
    for (const c of curriculum.courses) {
      const own = tasks.filter(t => t.course_id === c.id);
      if (!own.length) continue;
      const group = panel.createDiv({ cls: "sos-week-course" });
      const title = group.createDiv({ cls: "sos-week-course-heading" });
      title.createEl("h3", { text: `${c.id} · ${c.title}` });
      this.button(title, "课程位置与依据 ↗", () => this.openCourse(c), "sos-quiet");
      for (const task of own) {
        const row = group.createEl("details", { cls: `sos-week-task is-${task.state}`, attr: { "data-key": "task-" + task.id } });
        const summary = row.createEl("summary");
        const check = task.completion_check;
        if (task.state === "done") summary.createSpan({ cls: "sos-task-indicator", text: "✓", attr: { "aria-hidden": "true" } });
        else {
          const control = summary.createEl("button", { cls: "sos-task-indicator sos-check-task", text: "○", attr: { "aria-label": "核验完成：" + taskCaption(task), title: "核验完成" } });
          control.addEventListener("click", event => { event.preventDefault(); event.stopPropagation(); this.plugin.openCompletion(task); });
        }
        const text = summary.createDiv({ cls: "sos-task-caption" });
        text.createSpan({ text: taskCaption(task) });
        const meta = [task.kind === "obligation" ? "截止／必到" : "", task.date ? `${task.date.slice(5).replace("-", "/")} · ${task.date_basis}` : ""].filter(Boolean).join(" · ");
        if (meta) text.createEl("small", { text: meta });
        summary.createSpan({ cls: "sos-task-state", text: task.state !== "done" && check && !["complete", "closed", "cancelled"].includes(check.phase) ? "待核验" : states[task.state][1] });
        const detail = row.createDiv({ cls: "sos-week-task-detail" });
        detail.createEl("p", { text: displayText(task.title || task.raw) });
        if (task.record) detail.createEl("p", { cls: "sos-recorded-evidence", text: "已有记录：" + task.record });
        if (check?.verdict) detail.createEl("p", { cls: "sos-check-summary", text: (check.verdict.basis === "user_report" ? "本人确认：" : "核验：") + check.verdict.reason + (check.verdict.missing ? " 还缺：" + check.verdict.missing : "") });
        const actions = detail.createDiv({ cls: "sos-actions" });
        if (task.state !== "done") this.button(actions, check ? "查看／继续核验" : "核验完成", () => this.plugin.openCompletion(task));
        if (check) this.button(actions, "核验依据 ↗", () => this.openOriginal(check.export_path), "sos-quiet");
        if (task.state !== "done") this.button(actions, task.kind === "obligation" ? "在 Deadlines 管理" : "在 Workbench 管理",
          () => this.selectSection(task.kind === "obligation" ? "deadlines" : "workbench", task.kind === "obligation" ? null : c.id), "sos-quiet");
        this.button(actions, "就这项问管家", () => this.discussTask(task), "sos-quiet");
        this.button(actions, "打开原任务 ↗", () => this.openOriginal(task.ref || task.path), "sos-quiet");
      }
    }
    panel.createEl("p", { cls: "sos-plan-basis", text: "点圆框核验完成，点任务文字展开要求与依据；只学到一半可向管家记录停点。" });
  }
  renderRadar(data, home) {
    this.radar.empty();
    const header = this.radar.createDiv({ cls: "sos-course-heading" });
    header.createEl("h2", { text: "需要留意" });
    this.button(header, "Deadlines →", () => this.selectSection("deadlines"), "sos-quiet");
    const near = deadlineNotices(data);
    let more;
    for (const [index, t] of near.entries()) {
      if (index === 3) {
        more = this.radar.createEl("details", { cls: "sos-radar-more", attr: { "data-key": "radar-more" } });
        more.createEl("summary", { text: `还有 ${near.length - 3} 项近期安排` });
      }
      const row = (index >= 3 ? more : this.radar).createDiv({ cls: `sos-event${t.days === 0 ? " is-today" : ""}` });
      const date = row.createDiv({ cls: "sos-event-date" });
      date.createEl("strong", { text: t.time });
      if (t.optional) date.createSpan({ cls: "sos-event-optional", text: "可选" });
      const title = this.button(row, t.title, () => {
        if (t.task) return this.openOriginal(t.ref);
        return this.app.workspace.openLinkText(t.ref || data.today.path + "#临近 Deadline", data.today.path, "tab");
      }, "sos-event-title");
      if (t.detail) title.setAttribute("title", t.detail);
    }
    const past = home.obligations.filter(t => t.days_until < 0);
    if (past.length) this.button(this.radar, `${past.length} 项历史记录待确认 ↗`, () => this.openNote("99_学习情况记录/deadlines.md"), "sos-past-link");
    for (const n of home.notices) {
      const row = this.radar.createDiv({ cls: "sos-notice" }); row.createEl("p", { text: n.summary });
      this.button(row, "今天知道了", async () => { await this.bridge.call("acknowledge", { id: n.notice_id }); await this.bridge.refresh(); });
    }
    if (!near.length && !past.length && !home.notices.length) this.radar.createEl("p", { cls: "sos-muted", text: "已读取记录中暂无近期事项。" });
    if (data.source_issues.length) this.radar.createEl("p", { cls: "sos-error", text: `${data.source_issues.length} 处来源需要核对。` });
  }
  openCourse(course) {
    const modal = new Modal(this.app);
    const content = modal.contentEl;
    content.addClass("sos-course-modal");
    content.createEl("h2", { text: `${course.id} · ${course.title}` });
    for (const [key, label] of [["school", "学校进度／课程范围"], ["personal", "我的学习位置"]]) {
      const fact = course[key];
      content.createEl("h3", { text: label });
      content.createEl("p", { text: fact.summary });
      if (fact.detail) content.createEl("p", { text: fact.detail });
      if (fact.as_of) content.createEl("p", { cls: "sos-muted", text: `记录截至 ${fact.as_of}` });
      if (fact.historical || fact.needs_review) content.createEl("p", { cls: "sos-evidence-warning", text: "上次定位保留作参考，当前来源需要重新对账。" });
      const links = content.createDiv({ cls: "sos-actions" });
      for (const ref of fact.sources || []) this.button(links, ref.label || "来源", () => this.openNote(ref.path));
    }
    content.createEl("h3", { text: "下一步" });
    content.createEl("p", { text: course.next_step });
    const actions = content.createDiv({ cls: "sos-actions" });
    if (course.next_task) {
      const task = course.next_task;
      this.button(actions, "设为当前任务", async () => {
        await this.bridge.act({ kind: "set_focus", task_id: task.id, summary: task.title, next_step: course.next_step }, "把这个任务设为当前安排");
        modal.close();
      }, "mod-cta");
      this.button(actions, "打开任务原件", () => this.openNote(task.ref || task.path));
    }
    const tasks = (this.bridge.lastState?.tasks || []).filter(t => course.task_ids.includes(t.id));
    const details = content.createEl("details", { cls: "sos-details" });
    details.createEl("summary", { text: `本课任务原件 · ${tasks.length} 项` });
    for (const task of tasks) {
      const row = details.createDiv({ cls: "sos-history-row" });
      row.createEl("span", { text: `${task.status === "recorded_complete" ? "已记录完成" : "完成待核对"} · ${task.title}` });
      this.button(row, "原件", () => this.openNote(task.ref || task.path));
    }
    renderCourseTerms(content);
    modal.open();
  }
  async prepareTask(task) {
    const cancelled = this.bridge.lastState.jobs.find(j => j.state === "cancelled" && j.request?.task?.id === task.id);
    return this.bridge.act({ kind: "prepare_classroom", task_id: task.id, job_key: cancelled?.key || "" }, cancelled ? "重新准备这个任务的课堂" : "准备这个任务的课堂");
  }
  renderDetails(data) {
    const details = (label) => { const d = this.board.createEl("details", { cls: "sos-details", attr: { "data-key": label.split(" · ")[0] } }); d.createEl("summary", { text: label }); return d; };
    const tasks = details(`完整任务与状态 · ${data.tasks.length} 条原始记录`);
    for (const task of data.tasks) {
      const row = tasks.createDiv({ cls: "sos-task-row" });
      row.createEl("span", { text: `${task.status === "recorded_complete" ? "已记录完成" : task.status === "completion_unknown" ? "完成待核对" : task.status} · ${task.title}` });
      if (!["recorded_complete", "cancelled"].includes(task.status)) this.button(row, "设为当前", () => this.bridge.act({ kind: "set_focus", task_id: task.id,
        summary: task.title, next_step: "进入课堂，先核对实际停点。" }, "把这个任务设为当前安排"));
    }
    const history = details(`学习停点与课堂历史 · ${data.progress.length} 条交接`);
    for (const record of data.progress) {
      const row = history.createDiv({ cls: "sos-history-row" });
      row.createEl("p", { text: `${record.payload.task_title}：${record.payload.summary}` });
      this.button(row, "查看原记录", () => this.openNote(record.export_path));
    }
    for (const job of data.jobs) {
      const row = history.createDiv({ cls: "sos-history-row" });
      row.createEl("span", { text: `${job.title} · ${job.handed_off ? "已保存学习交接" : (labels[job.state] || job.state)}` });
      if (job.state === "registered") this.button(row, "打开课堂", () => this.openClassroom(job));
    }
    const commitments = details(`约定与提示 · ${data.commitments.length} 项`);
    for (const item of data.commitments) {
      const row = commitments.createDiv(); row.createEl("p", { text: `${item.summary} · ${item.status} · 至 ${item.ends_at}` });
      if (item.status === "active") this.button(row, "取消约定", () => this.bridge.act({ kind: "cancel_commitment", commitment_id: item.id }, "取消这项约定"));
    }
    commitments.createEl("p", { cls: "sos-muted", text: "管家随 Obsidian 运行。到期停止，漏过的提示不会集中补发；现有系统提醒与 Telegram 保持原入口。" });
    const links = this.board.createDiv({ cls: "sos-record-links" });
    this.button(links, "本周事实记录 ↗", () => this.openNote(data.weekly_report.path), "sos-quiet");
    this.button(links, "完整期限 ↗", () => this.openNote("99_学习情况记录/deadlines.md"), "sos-quiet");
    this.button(links, "成长看板 ↗", () => this.plugin.openDashboard(), "sos-quiet");
    if (data.weekly_report.error) this.board.createEl("p", { cls: "sos-error", text: data.weekly_report.error });
  }
  renderChat(data) {
    const day = londonClock().date;
    const signature = JSON.stringify([day, data.chat]);
    if (signature === this.chatSignature) return;
    this.chatSignature = signature;
    this.messages.empty();
    
    const latest = data.chat[data.chat.length - 1];
    const previous = data.chat.filter(item => ["done", "cancelled"].includes(item.phase)
      && (item !== latest || !item.updated_at || londonClock(new Date(item.updated_at)).date !== day));
    let history;
    if (previous.length) {
      history = this.messages.createEl("details", { cls: "sos-chat-history" });
      history.createEl("summary", { text: `查看历史对话 · ${previous.length} 条` });
    }
    for (const item of data.chat) {
      const parent = previous.includes(item) ? history : this.messages;
      parent.createDiv({ cls: "sos-message sos-user", text: item.body });
      if (item.reply) {
        const bubble = parent.createDiv({ cls: "sos-message sos-assistant" });
        let target = bubble;
        if (item.reply.length > 480) {
          const preview = item.reply.split("\n").filter(line => line.trim() && !/^\s*[|#>-]/.test(line)).slice(0, 2).join(" ");
          bubble.createEl("p", { text: displayText(preview).slice(0, 180) + (preview.length > 180 ? "…" : "") || "课程位置和依据已整理。" });
          const full = bubble.createEl("details", { cls: "sos-chat-full" });
          full.createEl("summary", { text: "展开完整回复" });
          target = full.createDiv();
        }
        MarkdownRenderer.render(this.app, item.reply, target, "", this).catch(() => target.setText(item.reply));
      }
      if (item.error) parent.createDiv({ cls: "sos-message sos-error", text: "尚未全部完成：" + item.error });
      if (!["done", "cancelled"].includes(item.phase)) {
        const row = this.messages.createDiv({ cls: "sos-actions" });
        this.button(row, "继续原请求", () => this.bridge.run("resume", { key: item.key }));
        this.button(row, "停止这条请求", async () => { await this.bridge.call("cancel-request", { key: item.key }); await this.bridge.refresh(); });
      }
      if (item.phase === "cancelled") parent.createDiv({ cls: "sos-muted", text: "这条请求已停止。" });
    }
    this.messages.scrollTop = this.messages.scrollHeight;
  }
}

function taskCaption(task) {
  if (task.kind === "obligation") return titleForDeadline(task.raw);
  let text = displayText(task.title || task.raw).replace(/^[^｜]+｜/, "").replace(/（原课时[^）]*）/, "").replace(/依据课堂记录/g, "");
  text = text.replace(/(\d{4})-(\d{2})-(\d{2})/, "$2/$3").split(/[；;]/)[0];
  return text.length > 95 ? text.slice(0, 95) + "…" : text;
}

function titleForDeadline(raw) {
  return displayText(raw.replace(/\[\[[^\]]+\]\]/g, "")
    .replace(/^\s*(?:>\s*)*[-*+]\s+\[[^\]]*\]\s*/, "")
    .replace(/^\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?(?:\s*前)?\s*[|｜·]?\s*/, "")
    .replace(/\s+(?:#student-os\/\S+|\^[\w-]+|(?:📅|⏳|✅|🛫|➕)\s*\d{4}-\d{2}-\d{2})/g, "")).trim();
}

function deadlineTime(task) {
  const match = task.raw?.match(/\d{4}-\d{2}-\d{2}\s+(\d{2}:\d{2})(\s*前)?/);
  return task.local_time || (match ? match[1] + (match[2] ? "前" : "") : "");
}

function courseTerms(value) { return value.replaceAll("大课", "Lecture").replaceAll("小班", "Seminar"); }

function renderCourseTerms(root) {
  const walker = root.ownerDocument.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let node = root.nodeType === Node.TEXT_NODE ? root : walker.nextNode();
  while (node) {
    if (!node.parentElement?.closest("textarea,input,[contenteditable],pre,code")) {
      const text = courseTerms(node.nodeValue);
      if (text !== node.nodeValue) node.nodeValue = text;
    }
    node = walker.nextNode();
  }
}

function displayText(value) { return courseTerms(value.replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, "$2").replace(/\[\[([^\]]+)\]\]/g, "$1").replace(/\*\*/g, "")); }

module.exports = { STEWARD_VIEW, StewardBridge, StewardView, CompletionModal, installCompletionGuard, renderedTask, homeModel, positionFor, agendaFor, todayNotices, deadlineNotices, sameRef, sectionForPath, titleForDeadline, deadlineTime };
