const { ItemView, Modal, Notice, Plugin, TFile, normalizePath } = require("obsidian");

const PLUGIN_ID = "learning-progress-dashboard";
const TASKS_PLUGIN_ID = "obsidian-tasks-plugin";
const VIEW_TYPE = "learning-progress-dashboard-view";

const OVERVIEW_PATH = "99_学习情况记录/Overview & Study Record.md";
const WORKBENCH_PATH = "99_学习情况记录/workbench.md";
const PLAN_FOLDER = "99_学习情况记录/学习计划";
const DAILY_FOLDER = "99_学习情况记录";
const DAILY_TEMPLATE = "00_inbox/日记模版.md";
const CANONICAL_TASK_TAG = "#student-os/task";

function pad(value) {
  return String(value).padStart(2, "0");
}

function formatDate(date = new Date()) {
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

function formatTime(date = new Date()) {
  return `${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function todayNotePath(date = new Date()) {
  const weekdays = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  return `${DAILY_FOLDER}/${formatDate(date)}——${weekdays[date.getDay()]}.md`;
}

function materializeTodayTemplate(template, date = new Date()) {
  return String(template).replace(/^date:\s*pending\s*$/m, `date: ${formatDate(date)}`);
}

function headingLevel(line) {
  const match = line.match(/^(#{1,6})\s+/);
  return match ? match[1].length : 0;
}

function findSection(lines, heading) {
  const start = lines.findIndex((line) => line.trim() === heading);
  if (start === -1) return null;
  const level = headingLevel(lines[start]);
  let end = lines.length;
  for (let index = start + 1; index < lines.length; index += 1) {
    const nextLevel = headingLevel(lines[index]);
    if (nextLevel && nextLevel <= level) {
      end = index;
      break;
    }
  }
  return { start, contentStart: start + 1, end };
}

function ensureSection(text, heading) {
  if (text.split("\n").some((line) => line.trim() === heading)) return text;
  return `${text.trimEnd()}\n\n${heading}\n`;
}

function insertIntoSection(text, heading, entry) {
  const withSection = ensureSection(text, heading);
  const lines = withSection.split("\n");
  const section = findSection(lines, heading);
  lines.splice(section.end, 0, entry);
  return lines.join("\n");
}

function stripQuotes(value) {
  const trimmed = String(value || "").trim();
  if (
    trimmed.length >= 2 &&
    ((trimmed.startsWith('"') && trimmed.endsWith('"')) ||
      (trimmed.startsWith("'") && trimmed.endsWith("'")))
  ) {
    return trimmed.slice(1, -1);
  }
  return trimmed;
}

function parseFrontmatter(text) {
  const lines = String(text || "").split("\n");
  if (lines[0]?.trim() !== "---") return {};
  const data = {};
  for (let index = 1; index < lines.length; index += 1) {
    const line = lines[index];
    if (line.trim() === "---") break;
    const match = line.match(/^([A-Za-z0-9_-]+):\s*(.*?)\s*$/);
    if (match) data[match[1]] = stripQuotes(match[2]);
  }
  return data;
}

function hasCanonicalTaskTag(text) {
  return new RegExp(`(?:^|\\s)${CANONICAL_TASK_TAG.replace("/", "\\/")}(?=\\s|$)`).test(text);
}

function parseTaskLine(line, path = "", lineNumber = 0) {
  const match = String(line || "").match(/^(\s*)[-*+]\s+\[([^\]])\]\s+(.+?)\s*$/);
  if (!match) return null;
  const doneDate = match[3].match(/✅\s*(\d{4}-\d{2}-\d{2})/)?.[1] || "";
  const scheduledDate = match[3].match(/⏳\s*(\d{4}-\d{2}-\d{2})/)?.[1] || "";
  return {
    path,
    line: lineNumber,
    raw: line,
    indent: match[1],
    status: match[2],
    body: match[3],
    text: cleanTaskText(match[3]),
    doneDate,
    scheduledDate,
    isDone: /[xX]/.test(match[2]),
    isCancelled: match[2] === "-",
  };
}

function cleanTaskText(value) {
  return String(value || "")
    .replace(new RegExp(`(?:^|\\s)${CANONICAL_TASK_TAG.replace("/", "\\/")}(?=\\s|$)`, "g"), " ")
    .replace(/\[\[([^\]|#]+)(?:#[^\]|]+)?\|([^\]]+)\]\]/g, "$2")
    .replace(/\[\[([^\]|#]+)(?:#[^\]]+)?\]\]/g, (_match, target) => target.split("/").pop())
    .replace(/\s+(?:✅|⏳|📅|🛫|➕)\s*\d{4}-\d{2}-\d{2}/g, "")
    .replace(/\s+🆔\s*\S+/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeStatus(value) {
  const status = String(value || "").toLowerCase();
  if (status === "completed" || status.includes("已完成")) return "completed";
  if (status === "paused" || status.includes("暂停")) return "paused";
  if (
    status === "queued" ||
    status.includes("确定待学") ||
    status.includes("尚未开始") ||
    status.includes("待规划")
  ) {
    return "queued";
  }
  return "active";
}

function parsePlanText(text, path) {
  const frontmatter = parseFrontmatter(text);
  if (frontmatter.student_os !== "learning-plan") return null;
  const lines = String(text || "").split("\n");
  const tasks = [];
  for (let index = 0; index < lines.length; index += 1) {
    if (!hasCanonicalTaskTag(lines[index])) continue;
    const task = parseTaskLine(lines[index], path, index);
    if (task && !task.isCancelled) tasks.push(task);
  }
  const completed = tasks.filter((task) => task.isDone).length;
  const next = tasks.find((task) => !task.isDone) || null;
  let status = normalizeStatus(frontmatter.status);
  if (tasks.length > 0 && completed === tasks.length) status = "completed";
  return {
    title: frontmatter.title || path.split("/").pop().replace(/\.md$/, ""),
    path,
    track: frontmatter.track || "self-directed",
    kind: frontmatter.kind || "finite-course",
    status,
    tasks,
    completed,
    total: tasks.length,
    next,
  };
}

function parseOverviewTracks(text) {
  const lines = String(text || "").split("\n");
  const tracks = [];
  let section = "";
  for (let index = 0; index < lines.length; index += 1) {
    const match = lines[index].match(/^(#{2,6})\s+(.+?)\s*$/);
    if (!match) continue;
    const level = match[1].length;
    const title = match[2].trim();
    if (level === 2) {
      section = title;
      continue;
    }
    let end = lines.length;
    for (let cursor = index + 1; cursor < lines.length; cursor += 1) {
      const nextLevel = headingLevel(lines[cursor]);
      if (nextLevel) {
        end = cursor;
        break;
      }
    }
    const block = lines.slice(index + 1, end);
    const status = block.find((line) => /^-\s*状态：/.test(line.trim()))?.replace(/^\s*-\s*状态：\s*/, "");
    if (!status) continue;
    const type = block.find((line) => /^-\s*类型：/.test(line.trim()))?.replace(/^\s*-\s*类型：\s*/, "") || "";
    const planTargets = block
      .filter((line) => /^-\s*(?:真实计划|计划)：/.test(line.trim()))
      .flatMap((line) =>
        [...line.matchAll(/\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]/g)].map((link) =>
          link[1].trim()
        )
      );
    tracks.push({
      title,
      section,
      status: normalizeStatus(status),
      statusLabel: status,
      type,
      planTarget: planTargets[0] || "",
      planTargets,
      track: section.includes("学校") ? "school" : "self-directed",
    });
  }
  return tracks;
}

function parseCompletedRecords(text, path = WORKBENCH_PATH) {
  const lines = String(text || "").split("\n");
  const section = findSection(lines, "## 计划外记录");
  if (!section) return [];
  const tasks = [];
  for (let index = section.contentStart; index < section.end; index += 1) {
    if (!hasCanonicalTaskTag(lines[index])) continue;
    const task = parseTaskLine(lines[index], path, index);
    if (task?.isDone && !task.isCancelled) {
      tasks.push({ ...task, source: "计划外记录", track: "unplanned" });
    }
  }
  return tasks;
}

function isOngoingKind(kind) {
  const normalized = String(kind || "").toLowerCase();
  return (
    normalized.includes("ongoing") ||
    normalized.includes("continuous") ||
    normalized.includes("capability") ||
    normalized.includes("practice")
  );
}

function trackLabel(track) {
  if (track === "school") return "学校责任";
  if (track === "unplanned") return "计划外完成";
  return "自主成长";
}

function locateTaskLine(lines, expectedLine, expectedRaw) {
  if (lines[expectedLine] === expectedRaw) return { index: expectedLine, reason: "exact" };
  const matches = [];
  lines.forEach((line, index) => {
    if (line === expectedRaw) matches.push(index);
  });
  if (matches.length === 1) return { index: matches[0], reason: "moved" };
  return { index: -1, reason: matches.length > 1 ? "ambiguous" : "missing" };
}

function recentCutoff(days = 28, today = new Date()) {
  const cutoff = new Date(today.getFullYear(), today.getMonth(), today.getDate());
  cutoff.setDate(cutoff.getDate() - days + 1);
  return cutoff;
}

function isRecentDate(value, cutoff = recentCutoff()) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(String(value || ""))) return false;
  return new Date(`${value}T00:00:00`) >= cutoff;
}

class RecordProgressModal extends Modal {
  constructor(app, tasks, onComplete, onSaveInput, onRecordCompleted) {
    super(app);
    this.tasks = tasks;
    this.onComplete = onComplete;
    this.onSaveInput = onSaveInput;
    this.onRecordCompleted = onRecordCompleted;
  }

  onOpen() {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.addClass("lpd-modal");
    contentEl.createEl("h2", { text: "快速记录" });

    const completeForm = contentEl.createEl("form", { cls: "lpd-modal-form" });
    const taskField = completeForm.createEl("label", { cls: "lpd-field" });
    taskField.createSpan({ text: "完成一个当前任务" });
    const select = taskField.createEl("select");
    this.tasks.forEach((task) => {
      select.createEl("option", {
        text: task.source ? `${task.source}｜${task.text}` : task.text,
        attr: { value: task.id },
      });
    });
    if (this.tasks.length === 0) {
      select.createEl("option", { text: "当前没有可完成任务", attr: { value: "" } });
      select.disabled = true;
    }

    const noteField = completeForm.createEl("label", { cls: "lpd-field" });
    noteField.createSpan({ text: "完成说明（可选）" });
    const noteInput = noteField.createEl("input", { attr: { type: "text", placeholder: "一句话即可" } });
    const completeButton = completeForm.createEl("button", {
      cls: "lpd-primary",
      text: "标记完成",
      attr: { type: "submit" },
    });
    completeButton.disabled = this.tasks.length === 0;
    completeForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const task = this.tasks.find((item) => item.id === select.value);
      if (!task) return;
      const completed = await this.onComplete(task, noteInput.value.trim());
      if (completed) this.close();
    });

    contentEl.createDiv({ cls: "lpd-modal-divider" });
    const captureForm = contentEl.createDiv({ cls: "lpd-modal-form" });
    const captureField = captureForm.createEl("label", { cls: "lpd-field" });
    captureField.createSpan({ text: "输入一件事" });
    const captureInput = captureField.createEl("textarea", {
      attr: { placeholder: "还没想清楚就存到输入箱；刚做完的事直接记为完成" },
    });
    const captureActions = captureForm.createDiv({ cls: "lpd-capture-actions" });
    const saveInputButton = captureActions.createEl("button", {
      cls: "lpd-ghost",
      text: "保存到输入箱",
      attr: { type: "button" },
    });
    const recordCompletedButton = captureActions.createEl("button", {
      cls: "lpd-primary",
      text: "记录为刚完成",
      attr: { type: "button" },
    });
    saveInputButton.addEventListener("click", async () => {
      const captured = await this.onSaveInput(captureInput.value);
      if (captured) this.close();
    });
    recordCompletedButton.addEventListener("click", async () => {
      const captured = await this.onRecordCompleted(captureInput.value);
      if (captured) this.close();
    });

    window.setTimeout(() => (this.tasks.length > 0 ? select : captureInput).focus(), 0);
  }

  onClose() {
    this.contentEl.empty();
  }
}

class WorkflowStore {
  constructor(app) {
    this.app = app;
  }

  getFile(path) {
    return this.app.vault.getFileByPath(normalizePath(path));
  }

  async readPath(path) {
    const file = this.getFile(path);
    return file ? this.app.vault.cachedRead(file) : "";
  }

  async ensureFolder(path) {
    const parts = normalizePath(path).split("/");
    let current = "";
    for (let index = 0; index < parts.length - 1; index += 1) {
      current = current ? `${current}/${parts[index]}` : parts[index];
      if (!this.app.vault.getAbstractFileByPath(current)) await this.app.vault.createFolder(current);
    }
  }

  async ensureTodayNote() {
    const path = todayNotePath();
    const existing = this.getFile(path);
    if (existing) return existing;
    const template = (await this.readPath(DAILY_TEMPLATE)) || "# 今日\n";
    await this.ensureFolder(path);
    const content = materializeTodayTemplate(template);
    return this.app.vault.create(path, content.endsWith("\n") ? content : `${content}\n`);
  }

  resolveOverviewLink(target) {
    if (!target) return null;
    const cached = this.app.metadataCache.getFirstLinkpathDest(target, OVERVIEW_PATH);
    if (cached) return cached;
    const suffix = target.endsWith(".md") ? target : `${target}.md`;
    return this.getFile(suffix) || this.getFile(`${DAILY_FOLDER}/${suffix}`);
  }

  async readPlans(overviewTracks = []) {
    const fileByPath = new Map(
      this.app.vault
        .getMarkdownFiles()
        .filter((file) => file.path.startsWith(`${PLAN_FOLDER}/`))
        .map((file) => [file.path, file])
    );
    for (const entry of overviewTracks) {
      for (const target of entry.planTargets || []) {
        const file = this.resolveOverviewLink(target);
        if (file?.extension === "md") fileByPath.set(file.path, file);
      }
    }
    const plans = [];
    for (const file of fileByPath.values()) {
      const plan = parsePlanText(await this.app.vault.cachedRead(file), file.path);
      if (plan) plans.push(plan);
    }
    return plans;
  }

  async readDashboardData() {
    const [overviewText, workbenchText] = await Promise.all([
      this.readPath(OVERVIEW_PATH),
      this.readPath(WORKBENCH_PATH),
    ]);
    const overviewTracks = parseOverviewTracks(overviewText);
    const plans = await this.readPlans(overviewTracks);
    const planByPath = new Map(plans.map((plan) => [plan.path, plan]));
    const usedPlans = new Set();
    const tracks = overviewTracks.flatMap((entry) => {
      const linkedPlans = [...new Set(entry.planTargets || [])]
        .map((target) => this.resolveOverviewLink(target))
        .map((file) => (file ? planByPath.get(file.path) : null))
        .filter(Boolean);
      if (linkedPlans.length === 0) {
        return [{ ...entry, path: OVERVIEW_PATH, tasks: [], completed: 0, total: 0, next: null }];
      }
      return linkedPlans.map((plan) => {
        usedPlans.add(plan.path);
        return {
          ...entry,
          ...plan,
          track: entry.track || plan.track,
          type: entry.type,
          status: plan.status,
        };
      });
    });
    plans.forEach((plan) => {
      if (!usedPlans.has(plan.path)) {
        tracks.push({
          ...plan,
          type: "",
          section: plan.track === "school" ? "学校责任" : "自主成长",
        });
      }
    });

    const recent = [
      ...tracks.flatMap((track) =>
        (track.tasks || [])
          .filter((task) => task.isDone && isRecentDate(task.doneDate))
          .map((task) => ({ ...task, source: track.title, track: track.track }))
      ),
      ...parseCompletedRecords(workbenchText).filter((task) => isRecentDate(task.doneDate)),
    ].sort((a, b) => b.doneDate.localeCompare(a.doneDate));
    return {
      tracks,
      recent,
      active: tracks.filter((track) => track.status === "active"),
      queued: tracks.filter((track) => ["queued", "paused"].includes(track.status)),
      completed: tracks.filter((track) => track.status === "completed"),
    };
  }

  async taskChoices() {
    const data = await this.readDashboardData();
    const choices = [];
    const today = formatDate();
    for (const track of data.tracks) {
      const open = (track.tasks || []).filter((task) => !task.isDone && !task.isCancelled);
      const selected = new Set();
      open.filter((task) => task.scheduledDate === today).forEach((task) => selected.add(task));
      open.slice(0, track.status === "active" ? 2 : 1).forEach((task) => selected.add(task));
      selected.forEach((task) => {
        choices.push({
          ...task,
          source: track.title,
          rank: task.scheduledDate === today ? 0 : track.status === "active" ? 2 : 3,
        });
      });
    }
    const unique = new Map();
    choices.forEach((task) => unique.set(`${task.path}:${task.line}`, task));
    return [...unique.values()]
      .sort((a, b) => a.rank - b.rank || a.path.localeCompare(b.path) || a.line - b.line)
      .map((task) => ({ ...task, id: `${task.path}:${task.line}:${task.raw}` }));
  }

  tasksApi() {
    const plugin =
      this.app.plugins.getPlugin?.(TASKS_PLUGIN_ID) || this.app.plugins.plugins?.[TASKS_PLUGIN_ID];
    return plugin?.apiV1 || null;
  }

  async completeCanonicalTask(task, note = "") {
    const file = this.getFile(task.path);
    if (!file) {
      new Notice("任务源文件不存在，未作修改");
      return false;
    }
    const api = this.tasksApi();
    if (!api?.executeToggleTaskDoneCommand) {
      new Notice("Tasks 插件尚未就绪，未作修改");
      return false;
    }
    let outcome = "missing";
    await this.app.vault.process(file, (current) => {
      const lines = current.split("\n");
      const located = locateTaskLine(lines, task.line, task.raw);
      if (located.index < 0) {
        outcome = located.reason;
        return current;
      }
      const currentTask = parseTaskLine(lines[located.index], task.path, located.index);
      if (!currentTask || currentTask.isDone || currentTask.isCancelled) {
        outcome = "stale";
        return current;
      }
      const toggled = api.executeToggleTaskDoneCommand(lines[located.index], task.path);
      const replacement = String(toggled).split("\n");
      if (note) replacement.splice(1, 0, `${currentTask.indent}  - ${formatTime()}｜${note}`);
      lines.splice(located.index, 1, ...replacement);
      outcome = "completed";
      return lines.join("\n");
    });
    if (outcome !== "completed") {
      const reason = outcome === "ambiguous" ? "任务出现重复，无法安全定位" : "任务已变化或已完成";
      new Notice(`${reason}，请重新打开面板`);
      return false;
    }
    new Notice("已完成；原任务与完成日期已同步");
    return true;
  }

  normalizeInput(text) {
    return String(text || "").replace(/\s+/g, " ").trim();
  }

  async saveInput(text) {
    const input = this.normalizeInput(text);
    if (!input) {
      new Notice("没有输入内容");
      return false;
    }
    const file = this.getFile(WORKBENCH_PATH);
    if (!file) {
      new Notice("未找到 Workbench，输入未保存");
      return false;
    }
    const entry = `- ${formatDate()} ${formatTime()}｜${input}`;
    await this.app.vault.process(file, (current) => insertIntoSection(current, "## 输入箱", entry));
    new Notice("已保存到 Workbench → 输入箱");
    return true;
  }

  async recordCompleted(text) {
    const input = this.normalizeInput(text);
    if (!input) {
      new Notice("没有输入内容");
      return false;
    }
    const file = this.getFile(WORKBENCH_PATH);
    if (!file) {
      new Notice("未找到 Workbench，完成记录未保存");
      return false;
    }
    const entry = `- [x] ${input} ${CANONICAL_TASK_TAG} ✅ ${formatDate()}`;
    await this.app.vault.process(file, (current) => insertIntoSection(current, "## 计划外记录", entry));
    new Notice("已记录为刚完成；Today 与看板会自动汇总");
    return true;
  }
}

class LearningProgressDashboardView extends ItemView {
  constructor(leaf, plugin) {
    super(leaf);
    this.plugin = plugin;
    this.data = null;
  }

  getViewType() {
    return VIEW_TYPE;
  }

  getDisplayText() {
    return "Learning Progress Dashboard";
  }

  getIcon() {
    return "sprout";
  }

  async onOpen() {
    await this.refresh();
  }

  async refresh() {
    this.data = await this.plugin.store.readDashboardData();
    this.render();
  }

  render() {
    const container = this.containerEl.children[1];
    container.empty();
    container.addClass("lpd-root");
    const shell = container.createDiv({ cls: "lpd-shell" });
    this.renderSidebar(shell.createDiv({ cls: "lpd-sidebar" }));
    this.renderMain(shell.createDiv({ cls: "lpd-main" }));
  }

  renderSidebar(sidebar) {
    sidebar.createDiv({ cls: "lpd-kicker", text: "Student OS" });
    sidebar.createEl("h1", { text: "Learning Progress" });
    sidebar.createDiv({
      cls: "lpd-muted",
      text: "只显示已经确认的学习承诺与真实完成记录。这里不催办，也不处理任务。",
    });
    const stats = sidebar.createDiv({ cls: "lpd-stats" });
    this.renderStat(stats, this.data.active.length, "当前推进");
    this.renderStat(stats, this.data.queued.length, "确定待学");
    this.renderStat(stats, this.data.completed.length, "完整成果");

    const actions = sidebar.createDiv({ cls: "lpd-nav" });
    actions.createEl("button", { cls: "lpd-primary", text: "打开 Overall" }).addEventListener("click", () => {
      this.plugin.openPath(OVERVIEW_PATH);
    });
    actions.createEl("button", { cls: "lpd-ghost", text: "刷新看板" }).addEventListener("click", () => this.refresh());

    sidebar.createDiv({ cls: "lpd-sidebar-note", text: "只累计前进，不记录断签。" });
  }

  renderStat(parent, value, label) {
    const stat = parent.createDiv({ cls: "lpd-stat" });
    stat.createDiv({ cls: "lpd-stat-value", text: String(value) });
    stat.createDiv({ cls: "lpd-muted", text: label });
  }

  renderMain(main) {
    const head = main.createDiv({ cls: "lpd-main-head" });
    const title = head.createDiv();
    title.createEl("h2", { text: "稳定生长" });
    title.createDiv({ cls: "lpd-muted", text: "来自 Overall 与真实学习计划" });

    this.renderRecent(main.createDiv({ cls: "lpd-panel" }));
    this.renderActive(main.createDiv({ cls: "lpd-panel" }));
    this.renderQueue(main.createDiv({ cls: "lpd-panel" }));
    this.renderCompleted(main.createDiv({ cls: "lpd-panel" }));
  }

  renderRecent(panel) {
    panel.createEl("h3", { text: "最近推进" });
    if (this.data.recent.length === 0) {
      panel.createDiv({ cls: "lpd-empty", text: "新的完成记录会从真实任务自动汇总到这里。" });
      return;
    }
    const list = panel.createDiv({ cls: "lpd-recent-list" });
    this.data.recent.slice(0, 10).forEach((task) => {
      const row = list.createDiv({ cls: "lpd-recent-row" });
      row.createDiv({ cls: "lpd-recent-date", text: task.doneDate.slice(5) });
      const body = row.createDiv();
      body.createDiv({ cls: "lpd-card-title", text: task.text });
      body.createDiv({
        cls: "lpd-muted",
        text: `${trackLabel(task.track)} · ${task.source}`,
      });
    });
  }

  renderActive(panel) {
    panel.createEl("h3", { text: "当前推进" });
    if (this.data.active.length === 0) {
      panel.createDiv({ cls: "lpd-empty", text: "目前没有处于推进状态的学习线。" });
      return;
    }
    const grid = panel.createDiv({ cls: "lpd-card-grid" });
    this.data.active.forEach((track) => this.renderTrackCard(grid, track));
  }

  renderTrackCard(parent, track) {
    const card = parent.createDiv({ cls: "lpd-card" });
    const header = card.createDiv({ cls: "lpd-card-head" });
    header.createDiv({ cls: "lpd-badge", text: trackLabel(track.track) });
    if (track.path && track.path !== OVERVIEW_PATH) {
      header.createEl("button", { cls: "lpd-link-button", text: track.title }).addEventListener("click", () => {
        this.plugin.openPath(track.path);
      });
    } else {
      header.createDiv({ cls: "lpd-card-title", text: track.title });
    }
    if (track.total > 0 && !isOngoingKind(track.kind)) {
      const percent = Math.round((track.completed / track.total) * 100);
      const progress = card.createDiv({ cls: "lpd-progress" });
      progress.setAttribute("role", "progressbar");
      progress.setAttribute("aria-label", `${track.title} progress`);
      progress.setAttribute("aria-valuemin", "0");
      progress.setAttribute("aria-valuemax", String(track.total));
      progress.setAttribute("aria-valuenow", String(track.completed));
      progress.createDiv({ cls: "lpd-progress-fill", attr: { style: `width:${percent}%` } });
      const unit = track.kind === "multi-stage-project" ? "个真实阶段" : "个真实单元";
      card.createDiv({ cls: "lpd-progress-label", text: `${track.completed} / ${track.total} ${unit}` });
      card.createDiv({
        cls: "lpd-next",
        text: track.next ? `下一项：${track.next.text}` : "这条学习线已经完成。",
      });
    } else if (isOngoingKind(track.kind)) {
      const recentCount = (track.tasks || []).filter(
        (task) => task.isDone && isRecentDate(task.doneDate)
      ).length;
      card.createDiv({ cls: "lpd-next", text: track.type || "持续能力训练" });
      card.createDiv({ cls: "lpd-progress-label", text: `最近 28 天完成 ${recentCount} 次练习` });
    } else {
      card.createDiv({ cls: "lpd-next", text: track.type || "计划仍在确认中；不伪造百分比。" });
    }
  }

  renderQueue(panel) {
    panel.createEl("h3", { text: "确定待学" });
    if (this.data.queued.length === 0) {
      panel.createDiv({ cls: "lpd-empty", text: "目前没有等待启动的学习线。" });
      return;
    }
    const list = panel.createDiv({ cls: "lpd-queue" });
    this.data.queued.forEach((track) => {
      const row = list.createDiv({ cls: "lpd-queue-row" });
      const body = row.createDiv();
      if (track.path && track.path !== OVERVIEW_PATH) {
        body.createEl("button", { cls: "lpd-link-button", text: track.title }).addEventListener("click", () => {
          this.plugin.openPath(track.path);
        });
      } else {
        body.createDiv({ cls: "lpd-card-title", text: track.title });
      }
      body.createDiv({ cls: "lpd-muted", text: track.type || "具体计划待确定" });
      const ongoing = isOngoingKind(track.kind);
      row.createDiv({
        cls: "lpd-queue-progress",
        text:
          track.total > 0 && !ongoing
            ? `${track.completed}/${track.total}`
            : ongoing
              ? "持续练习"
              : "待规划",
      });
    });
  }

  renderCompleted(panel) {
    panel.createEl("h3", { text: "已完成成果" });
    if (this.data.completed.length === 0) {
      panel.createDiv({ cls: "lpd-empty lpd-empty-quiet", text: "完成的课程和阶段成果会一直保留在这里。" });
      return;
    }
    const list = panel.createDiv({ cls: "lpd-completed" });
    this.data.completed.forEach((track) => {
      const result =
        track.total > 0 && !isOngoingKind(track.kind)
          ? `${track.completed}/${track.total}`
          : track.completed > 0
            ? `${track.completed} 次记录`
            : "已完成";
      list.createDiv({ cls: "lpd-completed-item", text: `${track.title} · ${result}` });
    });
  }
}

class LearningProgressDashboardPlugin extends Plugin {
  async onload() {
    this.store = new WorkflowStore(this.app);
    this.registerView(VIEW_TYPE, (leaf) => new LearningProgressDashboardView(leaf, this));
    this.addRibbonIcon("sprout", "Learning Progress Dashboard", () => this.openDashboard());
    this.addCommand({
      id: "open-learning-progress-dashboard",
      name: "Open Learning Progress Dashboard",
      callback: () => this.openDashboard(),
    });
    this.addCommand({
      id: "record-learning-progress",
      name: "Record learning progress",
      callback: () => this.openRecordModal(),
    });
    this.addCommand({
      id: "open-today-note",
      name: "Open today note",
      callback: () => this.openToday(),
    });
    this.registerEvent(
      this.app.vault.on("modify", (file) => {
        if (this.isDashboardSource(file)) this.refreshViews();
      })
    );
    this.app.workspace.onLayoutReady(() => {
      void this.openToday();
    });
  }

  isDashboardSource(file) {
    const frontmatter = file instanceof TFile ? this.app.metadataCache.getFileCache(file)?.frontmatter : null;
    return (
      file instanceof TFile &&
      file.extension === "md" &&
      (file.path === OVERVIEW_PATH ||
        file.path === WORKBENCH_PATH ||
        file.path.startsWith(`${PLAN_FOLDER}/`) ||
        frontmatter?.student_os === "learning-plan")
    );
  }

  onunload() {
    this.app.workspace.detachLeavesOfType(VIEW_TYPE);
  }

  async openDashboard() {
    const existing = this.app.workspace.getLeavesOfType(VIEW_TYPE);
    if (existing.length > 0) {
      this.app.workspace.revealLeaf(existing[0]);
      return;
    }
    await this.app.workspace.getLeaf("tab").setViewState({ type: VIEW_TYPE, active: true });
  }

  async refreshViews() {
    for (const leaf of this.app.workspace.getLeavesOfType(VIEW_TYPE)) {
      if (leaf.view instanceof LearningProgressDashboardView) await leaf.view.refresh();
    }
  }

  async openToday() {
    const file = await this.store.ensureTodayNote();
    await this.app.workspace.getLeaf(false).openFile(file);
  }

  async openRecordModal() {
    const tasks = await this.store.taskChoices();
    new RecordProgressModal(
      this.app,
      tasks,
      async (task, note) => {
        const completed = await this.store.completeCanonicalTask(task, note);
        if (completed) await this.refreshViews();
        return completed;
      },
      (text) => this.store.saveInput(text),
      async (text) => {
        const recorded = await this.store.recordCompleted(text);
        if (recorded) await this.refreshViews();
        return recorded;
      }
    ).open();
  }

  async openPath(path) {
    let file = this.app.vault.getFileByPath(normalizePath(path));
    if (!file && !path.endsWith(".md")) file = this.app.vault.getFileByPath(normalizePath(`${path}.md`));
    if (!file) {
      new Notice(`Note not found: ${path}`);
      return;
    }
    await this.app.workspace.getLeaf(false).openFile(file);
  }
}

module.exports = LearningProgressDashboardPlugin;
module.exports.__test = {
  isOngoingKind,
  locateTaskLine,
  materializeTodayTemplate,
  parseCompletedRecords,
  parseFrontmatter,
  parseOverviewTracks,
  parsePlanText,
  parseTaskLine,
};
