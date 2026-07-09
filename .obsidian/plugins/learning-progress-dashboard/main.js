const { ItemView, Modal, Notice, Plugin, TFile, normalizePath } = require("obsidian");

const PLUGIN_ID = "learning-progress-dashboard";
const VIEW_TYPE = "learning-progress-dashboard-view";

const OVERVIEW_PATH = "99_学习情况记录/Overview & Study Record.md";
const WORKBENCH_PATH = "99_学习情况记录/workbench.md";
const DEADLINES_PATH = "99_学习情况记录/deadlines.md";
const DAILY_FOLDER = "99_学习情况记录";
const DAILY_TEMPLATE = "00_inbox/日记模版.md";
const WEEKLY_REVIEW_FOLDER = "99_学习情况记录/week-review";
const COURSE_ROOTS = {
  "01_Math": "Math",
  "02_Economy": "Economy",
  "03_Computer_Science": "Computer Science",
};
const ROOT_ORDER = ["Math", "Economy", "Computer Science"];
const STATES = ["raw", "active", "learned", "organized", "mapped"];
const STATE_META = {
  raw: { depth: 0, label: "未开始" },
  active: { depth: 0.5, label: "进行中" },
  learned: { depth: 1, label: "已学完" },
  organized: { depth: 2, label: "已整理" },
  mapped: { depth: 3, label: "已成图" },
};
const LESSON_RE = /^(\d{1,2})[_-](.+)$/;
const EXCLUDE_KEYWORDS = ["作业", "考试", "划重点", "补充", "course map", "exam", "review", "roadmap", "index", "main", "零散"];

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

function normalizeState(value) {
  return STATES.includes(value) ? value : "raw";
}

function stateDepth(value) {
  return STATE_META[normalizeState(value)].depth;
}

function hashText(value) {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash << 5) - hash + value.charCodeAt(index);
    hash |= 0;
  }
  return Math.abs(hash).toString(36);
}

function cleanTaskText(line) {
  return line.replace(/^\s*-\s+\[[ xX]\]\s+/, "").trim();
}

function taskIndent(line) {
  return (line.match(/^\s*/) || [""])[0].length;
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

function wikilinkTarget(raw) {
  const match = String(raw || "").match(/\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]/);
  return match ? match[1].trim() : String(raw || "").trim();
}

function lessonFileTitle(file) {
  const match = file.basename.match(LESSON_RE);
  if (!match) return file.basename;
  return match[2].replace(/[_-]+/g, " ").trim();
}

function isLessonFile(file) {
  const basename = file.basename.toLowerCase();
  if (!LESSON_RE.test(file.basename)) return false;
  if (file.path.split("/").length === 3) return true;
  return !EXCLUDE_KEYWORDS.some((keyword) => basename.includes(keyword.toLowerCase()));
}

function isCourseFile(file) {
  if (!(file instanceof TFile) || file.extension !== "md") return false;
  const root = file.path.split("/")[0];
  return Boolean(COURSE_ROOTS[root]);
}

function isWorkflowFile(file) {
  if (!(file instanceof TFile) || file.extension !== "md") return false;
  if ([OVERVIEW_PATH, WORKBENCH_PATH, DEADLINES_PATH].includes(file.path)) return true;
  if (file.path.startsWith(`${DAILY_FOLDER}/`) && /^\d{4}-\d{2}-\d{2}——[A-Za-z]{3}\.md$/.test(file.name)) return true;
  if (file.path.startsWith(`${WEEKLY_REVIEW_FOLDER}/`)) return true;
  return isCourseFile(file);
}

class RecordProgressModal extends Modal {
  constructor(app, tasks, onSubmit) {
    super(app);
    this.tasks = tasks;
    this.onSubmit = onSubmit;
  }

  onOpen() {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.addClass("lpd-modal");
    contentEl.createEl("h2", { text: "记录今日推进" });

    const form = contentEl.createEl("form", { cls: "lpd-modal-form" });
    const taskField = form.createEl("label", { cls: "lpd-field" });
    taskField.createSpan({ text: "任务" });
    const select = taskField.createEl("select");
    this.tasks.forEach((task) => {
      select.createEl("option", {
        text: task.project ? `${task.project}｜${task.text}` : task.text,
        attr: { value: task.id },
      });
    });
    select.createEl("option", { text: "计划外完成（不绑定任务）", attr: { value: "unplanned" } });

    const noteField = form.createEl("label", { cls: "lpd-field" });
    noteField.createSpan({ text: "记录" });
    const noteInput = noteField.createEl("textarea", {
      attr: { placeholder: "写一句推进或完成说明；只勾完成可留空" },
    });

    const doneLabel = form.createEl("label", { cls: "lpd-field" });
    const done = doneLabel.createEl("input", { attr: { type: "checkbox" } });
    doneLabel.appendText(" 标记为完成");

    const actions = form.createDiv({ cls: "lpd-detail-actions" });
    const cancel = actions.createEl("button", {
      cls: "lpd-ghost",
      text: "Cancel",
      attr: { type: "button" },
    });
    cancel.addEventListener("click", () => this.close());
    actions.createEl("button", { cls: "lpd-primary", text: "Write", attr: { type: "submit" } });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      await this.onSubmit({
        taskId: select.value,
        note: noteInput.value.trim(),
        done: done.checked,
      });
      this.close();
    });

    window.setTimeout(() => noteInput.focus(), 0);
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
    if (!file) return "";
    return this.app.vault.cachedRead(file);
  }

  async writePath(path, text) {
    const file = this.getFile(path);
    if (file) {
      await this.app.vault.modify(file, text);
      return file;
    }
    await this.ensureFolder(path);
    return this.app.vault.create(normalizePath(path), text);
  }

  async ensureFolder(path) {
    const parts = normalizePath(path).split("/");
    let current = "";
    for (let index = 0; index < parts.length - 1; index += 1) {
      current = current ? `${current}/${parts[index]}` : parts[index];
      if (!this.app.vault.getAbstractFileByPath(current)) {
        await this.app.vault.createFolder(current);
      }
    }
  }

  async ensureTodayNote() {
    const path = todayNotePath();
    const existing = this.getFile(path);
    if (existing) return existing;
    const template = (await this.readPath(DAILY_TEMPLATE)) || [
      "# 今日",
      "",
      "## 即将到期",
      "",
      "## 历史未完成任务",
      "",
      "## 工作台",
      "![[workbench#当前项目]]",
      "",
      "## 今日完成",
      "",
      "## 计划外完成",
      "",
      "## 收尾复盘",
      "- 未完成：",
      "- 转移：",
      "- 需要同步回 Overview/Workbench/Deadline/项目文件：",
      "",
    ].join("\n");
    await this.ensureFolder(path);
    return this.app.vault.create(path, template.endsWith("\n") ? template : `${template}\n`);
  }

  parseWorkbench(text) {
    const lines = text.split("\n");
    const current = findSection(lines, "## 当前项目");
    const done = findSection(lines, "## 今日完成");
    const tasks = [];
    const completed = [];

    if (current) {
      let project = "";
      for (let index = current.contentStart; index < current.end; index += 1) {
        const line = lines[index];
        const heading = line.match(/^###\s+(.+?)\s*$/);
        if (heading) {
          project = heading[1].trim();
          continue;
        }
        if (!/^\s*-\s+\[\s\]\s+/.test(line) || taskIndent(line) !== 0) continue;

        const start = index;
        let end = index + 1;
        while (end < current.end) {
          const next = lines[end];
          if (/^###\s+/.test(next)) break;
          if (/^\s*-\s+\[[ xX]\]\s+/.test(next) && taskIndent(next) === 0) break;
          end += 1;
        }

        const block = lines.slice(start, end);
        const metadata = {};
        for (const item of block.slice(1)) {
          const meta = item.trim().match(/^-\s*([A-Za-z_]+)::\s*(.+?)\s*$/);
          if (meta) metadata[meta[1]] = meta[2];
        }
        const textValue = cleanTaskText(line);
        tasks.push({
          id: hashText(`${project}|${textValue}|${metadata.lesson || ""}|${start}`),
          project,
          text: textValue,
          metadata,
          start,
          end,
          block,
        });
        index = end - 1;
      }
    }

    if (done) {
      for (let index = done.contentStart; index < done.end; index += 1) {
        const line = lines[index].trim();
        if (/^-\s+\[x\]\s+/.test(line)) completed.push(line);
      }
    }

    return { lines, tasks, completed };
  }

  pruneWorkbenchDoneText(text, today = formatDate()) {
    const lines = text.split("\n");
    const done = findSection(lines, "## 今日完成");
    if (!done) return ensureSection(text, "## 今日完成");

    const keep = [];
    for (let index = done.contentStart; index < done.end; index += 1) {
      const line = lines[index];
      if (/^\s*-\s+\[x\]\s+\d{4}-\d{2}-\d{2}/.test(line) && !line.includes(today)) {
        while (index + 1 < done.end && /^\s+-\s+/.test(lines[index + 1])) index += 1;
        continue;
      }
      keep.push(line);
    }
    return [...lines.slice(0, done.contentStart), ...keep, ...lines.slice(done.end)].join("\n");
  }

  async pruneWorkbenchDone() {
    const text = await this.readPath(WORKBENCH_PATH);
    if (!text) return;
    const pruned = this.pruneWorkbenchDoneText(text);
    if (pruned !== text) await this.writePath(WORKBENCH_PATH, pruned);
  }

  async appendDaily(section, entry) {
    const file = await this.ensureTodayNote();
    const text = await this.app.vault.read(file);
    await this.app.vault.modify(file, insertIntoSection(text, section, entry));
  }

  async recordUnplanned(note) {
    if (!note) {
      new Notice("未输入内容，已取消");
      return;
    }
    await this.appendDaily("## 计划外完成", `- ${formatTime()}｜${note}`);
  }

  async recordTaskProgress(taskId, note) {
    if (!note) {
      new Notice("未输入内容，已取消");
      return;
    }
    const text = await this.readPath(WORKBENCH_PATH);
    const parsed = this.parseWorkbench(text);
    const task = parsed.tasks.find((item) => item.id === taskId);
    if (!task) {
      new Notice("任务不存在，可能已经被同步");
      return;
    }
    const lines = parsed.lines;
    lines.splice(task.end, 0, `  - ${formatTime()}｜${note}`);
    await this.writePath(WORKBENCH_PATH, lines.join("\n"));
  }

  async completeTask(taskId, note = "") {
    const raw = await this.readPath(WORKBENCH_PATH);
    const pruned = this.pruneWorkbenchDoneText(raw);
    const parsed = this.parseWorkbench(pruned);
    const task = parsed.tasks.find((item) => item.id === taskId);
    if (!task) {
      new Notice("任务不存在，可能已经完成");
      return;
    }

    const lines = parsed.lines;
    const doneLine = `- [x] ${formatDate()} ${formatTime()}｜${task.project ? `${task.project}｜` : ""}${task.text}`;
    const doneBlock = [doneLine];
    task.block
      .slice(1)
      .filter((line) => !/^\s*-\s*(lesson|done_state)::/.test(line.trim()))
      .forEach((line) => doneBlock.push(line));
    if (note) doneBlock.push(`  - note:: ${note}`);
    if (task.metadata.lesson) doneBlock.push(`  - lesson:: ${task.metadata.lesson}`);
    if (task.metadata.done_state) doneBlock.push(`  - done_state:: ${task.metadata.done_state}`);

    lines.splice(task.start, task.end - task.start);
    let nextText = lines.join("\n");
    nextText = insertIntoSection(nextText, "## 今日完成", doneBlock.join("\n"));
    await this.writePath(WORKBENCH_PATH, nextText);

    await this.appendDaily(
      "## 今日完成",
      `- ${formatTime()}｜${task.project ? `${task.project}｜` : ""}${task.text}${note ? `｜${note}` : ""}`
    );

    if (task.metadata.lesson && task.metadata.done_state) {
      await this.updateLessonState(task.metadata.lesson, task.metadata.done_state);
    }
  }

  async updateLessonState(rawLesson, rawState) {
    const state = normalizeState(rawState);
    const target = wikilinkTarget(rawLesson);
    let file = this.app.metadataCache.getFirstLinkpathDest(target, "");
    if (!file && target.endsWith(".md")) file = this.getFile(target);
    if (!file && !target.endsWith(".md")) file = this.getFile(`${target}.md`);
    if (!file) {
      new Notice(`未找到课节笔记：${target}`);
      return;
    }

    await this.app.fileManager.processFrontMatter(file, (frontmatter) => {
      frontmatter.learning_state = state;
      frontmatter.learning_updated = formatDate();
    });
  }

  parseDeadlines(text) {
    const today = new Date(`${formatDate()}T00:00:00`);
    return text
      .split("\n")
      .map((line) => line.match(/^\s*-\s+\[\s\]\s+(\d{4}-\d{2}-\d{2})\s*[｜|]\s*(.+?)\s*$/))
      .filter(Boolean)
      .map((match) => {
        const due = new Date(`${match[1]}T00:00:00`);
        const days = Math.round((due - today) / 86400000);
        return { due: match[1], body: match[2], days };
      })
      .filter((item) => item.days >= 0)
      .sort((a, b) => a.days - b.days);
  }

  parseDailySection(text, heading) {
    const lines = text.split("\n");
    const section = findSection(lines, heading);
    if (!section) return [];
    return lines.slice(section.contentStart, section.end).filter((line) => line.trim());
  }

  latestWeeklyReviews() {
    return this.app.vault
      .getMarkdownFiles()
      .filter((file) => file.path.startsWith(`${WEEKLY_REVIEW_FOLDER}/`))
      .sort((a, b) => b.basename.localeCompare(a.basename))
      .slice(0, 4);
  }

  scanCourses() {
    const courses = new Map();
    for (const file of this.app.vault.getMarkdownFiles()) {
      if (!isCourseFile(file) || !isLessonFile(file)) continue;
      const parts = file.path.split("/");
      if (parts.length < 3) continue;
      const root = COURSE_ROOTS[parts[0]];
      const title = parts[1];
      const path = `${parts[0]}/${parts[1]}`;
      const course =
        courses.get(path) ||
        {
          title,
          root,
          path,
          lessons: [],
        };
      const cache = this.app.metadataCache.getFileCache(file) || {};
      const frontmatter = cache.frontmatter || {};
      const labelMatch = file.basename.match(LESSON_RE);
      course.lessons.push({
        label: labelMatch ? labelMatch[1].padStart(2, "0") : "?",
        title: lessonFileTitle(file),
        path: file.path,
        state: normalizeState(frontmatter.learning_state),
      });
      courses.set(path, course);
    }

    return [...courses.values()]
      .sort((a, b) => {
        const rootDiff = ROOT_ORDER.indexOf(a.root) - ROOT_ORDER.indexOf(b.root);
        if (rootDiff) return rootDiff;
        return a.title.localeCompare(b.title);
      })
      .map((course) => ({
        ...course,
        lessons: course.lessons.sort((a, b) => Number(a.label) - Number(b.label) || a.title.localeCompare(b.title)),
      }));
  }

  async readDashboardData() {
    await this.pruneWorkbenchDone();
    const [overview, workbench, deadlines, todayText] = await Promise.all([
      this.readPath(OVERVIEW_PATH),
      this.readPath(WORKBENCH_PATH),
      this.readPath(DEADLINES_PATH),
      this.readPath(todayNotePath()),
    ]);
    const parsedWorkbench = this.parseWorkbench(workbench);
    return {
      overview,
      workbench: parsedWorkbench,
      deadlines: this.parseDeadlines(deadlines),
      today: {
        path: todayNotePath(),
        completed: this.parseDailySection(todayText, "## 今日完成"),
        unplanned: this.parseDailySection(todayText, "## 计划外完成"),
      },
      weeklyReviews: this.latestWeeklyReviews(),
      courses: this.scanCourses(),
    };
  }
}

class LearningProgressDashboardView extends ItemView {
  constructor(leaf, plugin) {
    super(leaf);
    this.plugin = plugin;
    this.data = null;
    this.query = "";
  }

  getViewType() {
    return VIEW_TYPE;
  }

  getDisplayText() {
    return "Learning Progress Dashboard";
  }

  getIcon() {
    return "map";
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
    this.renderDetail(shell.createDiv({ cls: "lpd-detail" }));
  }

  renderSidebar(sidebar) {
    sidebar.createDiv({ cls: "lpd-kicker", text: "Academic Vault" });
    sidebar.createEl("h1", { text: "Learning Progress" });
    sidebar.createDiv({ cls: "lpd-muted", text: "Dashboard 是显示和同步器；Markdown 才是事实源。" });

    const actions = sidebar.createDiv({ cls: "lpd-detail-actions" });
    actions.createEl("button", { cls: "lpd-primary", text: "Open Today" }).addEventListener("click", () => {
      this.plugin.openToday();
    });
    actions.createEl("button", { cls: "lpd-ghost", text: "Record" }).addEventListener("click", () => {
      this.plugin.openRecordModal();
    });
    actions.createEl("button", { cls: "lpd-ghost", text: "Refresh" }).addEventListener("click", () => this.refresh());

    sidebar.createEl("input", {
      cls: "lpd-search",
      attr: { type: "search", placeholder: "Filter courses..." },
    }).addEventListener("input", (event) => {
      this.query = event.target.value;
      this.render();
    });

    const legend = sidebar.createDiv({ cls: "lpd-legend" });
    STATES.forEach((state) => {
      const row = legend.createDiv({ cls: "lpd-legend-row" });
      row.createSpan({ cls: `lpd-legend-swatch lpd-state-${state}` });
      row.createSpan({ text: `${STATE_META[state].label} · ${state}` });
    });
  }

  renderMain(main) {
    const head = main.createDiv({ cls: "lpd-main-head" });
    head.createEl("h2", { text: "Today Desk" });
    head.createDiv({
      cls: "lpd-muted",
      text: `${this.data.workbench.tasks.length} current tasks · ${this.data.deadlines.length} upcoming deadlines`,
    });

    this.renderTasks(main.createDiv({ cls: "lpd-panel" }));
    this.renderDeadlines(main.createDiv({ cls: "lpd-panel" }));
    this.renderToday(main.createDiv({ cls: "lpd-panel" }));
    this.renderCourses(main.createDiv({ cls: "lpd-panel" }));
  }

  renderTasks(panel) {
    panel.createEl("h3", { text: "Workbench" });
    if (this.data.workbench.tasks.length === 0) {
      panel.createDiv({ cls: "lpd-empty", text: "Workbench 当前没有未完成任务。" });
      return;
    }
    for (const task of this.data.workbench.tasks) {
      const row = panel.createDiv({ cls: "lpd-task-row" });
      row.createEl("button", { cls: "lpd-mini", text: "✓" }).addEventListener("click", async () => {
        await this.plugin.store.completeTask(task.id);
        await this.refresh();
        new Notice("任务已同步完成。");
      });
      const body = row.createDiv();
      body.createDiv({ cls: "lpd-course-title", text: task.text });
      body.createDiv({ cls: "lpd-muted", text: task.project || "No project" });
    }
  }

  renderDeadlines(panel) {
    panel.createEl("h3", { text: "Deadlines" });
    const items = this.data.deadlines.slice(0, 6);
    if (items.length === 0) {
      panel.createDiv({ cls: "lpd-empty", text: "暂无即将到来的截止日期。" });
      return;
    }
    for (const item of items) {
      panel.createDiv({ cls: "lpd-muted", text: `D-${item.days}｜${item.due}｜${item.body}` });
    }
  }

  renderToday(panel) {
    panel.createEl("h3", { text: "Today" });
    panel.createDiv({ cls: "lpd-muted", text: this.data.today.path });
    const completed = this.data.today.completed.slice(-5);
    if (completed.length === 0) {
      panel.createDiv({ cls: "lpd-empty", text: "今天还没有完成记录。" });
    } else {
      completed.forEach((line) => panel.createDiv({ cls: "lpd-muted", text: line.replace(/^\s*-\s*/, "") }));
    }
  }

  renderCourses(panel) {
    panel.createEl("h3", { text: "Courses" });
    const query = this.query.trim().toLowerCase();
    const courses = this.data.courses
      .filter((course) => !query || [course.title, course.root, course.path].some((value) => value.toLowerCase().includes(query)))
      .slice(0, 12);
    if (courses.length === 0) {
      panel.createDiv({ cls: "lpd-empty", text: "No courses match the filter." });
      return;
    }
    for (const course of courses) {
      const row = panel.createDiv({ cls: "lpd-course-row" });
      const info = row.createDiv({ cls: "lpd-course-info" });
      info.createDiv({ cls: "lpd-course-title", text: course.title });
      info.createDiv({ cls: "lpd-course-path", text: `${course.root} · ${course.path}` });
      const progress = this.courseProgress(course);
      info.createDiv({
        cls: "lpd-muted",
        text: `${progress.learned}/${progress.total} 已学 · ${progress.organized} 已整理 · ${progress.mapped} 已成图`,
      });
      const nodes = row.createDiv({ cls: "lpd-track-nodes" });
      course.lessons.slice(0, 30).forEach((lesson) => {
        const node = nodes.createEl("button", {
          cls: `lpd-lesson-node lpd-state-${lesson.state}`,
          attr: { title: `${lesson.label} ${lesson.title} · ${STATE_META[lesson.state].label}` },
        });
        node.createSpan({ cls: "lpd-node-label", text: lesson.label });
        node.addEventListener("click", () => this.plugin.openPath(lesson.path));
      });
    }
  }

  renderDetail(detail) {
    detail.createDiv({ cls: "lpd-kicker", text: "Context" });
    detail.createEl("h2", { text: "Sources" });
    [
      ["Overview", OVERVIEW_PATH],
      ["Workbench", WORKBENCH_PATH],
      ["Deadlines", DEADLINES_PATH],
      ["Today", this.data.today.path],
    ].forEach(([label, path]) => {
      detail.createEl("button", { cls: "lpd-ghost", text: label }).addEventListener("click", () => this.plugin.openPath(path));
    });

    detail.createEl("h2", { text: "Weekly Reviews" });
    if (this.data.weeklyReviews.length === 0) {
      detail.createDiv({ cls: "lpd-empty", text: "还没有周复盘。" });
    } else {
      this.data.weeklyReviews.forEach((file) => {
        detail.createEl("button", { cls: "lpd-ghost", text: file.basename }).addEventListener("click", () => this.plugin.openPath(file.path));
      });
    }
  }

  courseProgress(course) {
    const lessons = course.lessons || [];
    return {
      total: lessons.length,
      learned: lessons.filter((lesson) => stateDepth(lesson.state) >= 1).length,
      organized: lessons.filter((lesson) => stateDepth(lesson.state) >= 2).length,
      mapped: lessons.filter((lesson) => stateDepth(lesson.state) >= 3).length,
    };
  }
}

module.exports = class LearningProgressDashboardPlugin extends Plugin {
  async onload() {
    this.store = new WorkflowStore(this.app);
    this.registerView(VIEW_TYPE, (leaf) => new LearningProgressDashboardView(leaf, this));

    this.addRibbonIcon("map", "Learning Progress Dashboard", () => this.openDashboard());
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
        if (isWorkflowFile(file)) this.refreshViews();
      })
    );
    this.app.workspace.onLayoutReady(async () => {
      await this.store.pruneWorkbenchDone();
      await this.openDashboard();
    });
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
    const data = await this.store.readDashboardData();
    new RecordProgressModal(this.app, data.workbench.tasks, async ({ taskId, note, done }) => {
      if (taskId === "unplanned") {
        await this.store.recordUnplanned(note);
      } else if (done) {
        await this.store.completeTask(taskId, note);
      } else {
        await this.store.recordTaskProgress(taskId, note);
      }
      await this.refreshViews();
    }).open();
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
};
