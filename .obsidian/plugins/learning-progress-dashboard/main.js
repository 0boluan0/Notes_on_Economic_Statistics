const { ItemView, Modal, Notice, Plugin, TFile, normalizePath } = require("obsidian");

const PLUGIN_ID = "learning-progress-dashboard";
const VIEW_TYPE = "learning-progress-dashboard-view";
const DATA_PATH = "98_attachment/vault-home/learning-board.md";
const PLUGIN_SOURCE_PATHS = new Set([
  `.obsidian/plugins/${PLUGIN_ID}/main.js`,
  `.obsidian/plugins/${PLUGIN_ID}/styles.css`,
  `.obsidian/plugins/${PLUGIN_ID}/manifest.json`,
]);

const COURSE_ROOTS = {
  "01_Math": "Math",
  "02_Economy": "Economy",
  "03_Computer_Science": "Computer Science",
};

const ROOT_ORDER = ["Math", "Economy", "Computer Science"];
const LESSON_RE = /^(\d{1,2})[_-](.+)$/;
const SECTION_RE = /^(\d+)[_-]/;
const EXCLUDE_KEYWORDS = [
  "作业",
  "考试",
  "划重点",
  "补充",
  "course map",
  "exam",
  "review",
  "roadmap",
  "index",
  "main",
  "零散",
];

const STATE_META = {
  raw: {
    depth: 0,
    label: "未开始",
    description: "还没有开始学习加工",
  },
  learned: {
    depth: 1,
    label: "已学完",
    description: "已经完成学习或听课",
  },
  organized: {
    depth: 2,
    label: "已整理",
    description: "课程笔记已经整理到可回顾",
  },
  mapped: {
    depth: 3,
    label: "已成图",
    description: "已经进入 Big Picture 或画图",
  },
};

function hashText(value) {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash << 5) - hash + value.charCodeAt(index);
    hash |= 0;
  }
  return Math.abs(hash).toString(36);
}

function makeId(value) {
  const safe = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return `${safe || "item"}-${hashText(value)}`;
}

function readJsonBlock(markdown) {
  const match = markdown.match(/```learning-board-json\s*\n([\s\S]*?)\n```/);
  if (!match) return null;
  return JSON.parse(match[1]);
}

function courseSortKey(course) {
  const rootIndex = ROOT_ORDER.indexOf(course.root);
  return [
    rootIndex === -1 ? 99 : rootIndex,
    Number.isFinite(Number(course.order)) ? Number(course.order) : 9999,
    course.title || "",
  ];
}

function compareSortKey(a, b) {
  const left = courseSortKey(a);
  const right = courseSortKey(b);
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] < right[index]) return -1;
    if (left[index] > right[index]) return 1;
  }
  return 0;
}

function lessonSortValue(lesson) {
  const label = String(lesson.label || "");
  const dotted = label.match(/^(\d+)\.(\d+)$/);
  if (dotted) return Number(dotted[1]) * 100 + Number(dotted[2]);
  const integer = label.match(/^(\d+)$/);
  if (integer) return Number(integer[1]);
  return 9999;
}

function compareLessons(a, b) {
  const left = lessonSortValue(a);
  const right = lessonSortValue(b);
  if (left !== right) return left - right;
  return String(a.title || "").localeCompare(String(b.title || ""));
}

function normalizeState(state) {
  return STATE_META[state] ? state : "raw";
}

function stateDepth(state) {
  return STATE_META[normalizeState(state)].depth;
}

function percentage(count, total) {
  if (!total) return "0%";
  return `${Math.max(0, Math.min(100, (count / total) * 100)).toFixed(2)}%`;
}

function nextLessonLabel(course) {
  const labels = (course.lessons || [])
    .map((lesson) => String(lesson.label || "").match(/\d+$/))
    .filter(Boolean)
    .map((match) => Number(match[0]));
  const next = labels.length > 0 ? Math.max(...labels) + 1 : 1;
  return String(next).padStart(2, "0");
}

function normalizeLessonLabel(value) {
  const digits = String(value || "").trim().match(/\d+$/);
  if (!digits) return "";
  return digits[0].padStart(2, "0");
}

function safeFileNamePart(value) {
  return String(value || "")
    .trim()
    .replace(/[\\/:*?"<>|#^[\]]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function buildLessonNotePath(course, label, title) {
  const safeTitle = safeFileNamePart(title);
  return normalizePath(`${course.path}/${label}_${safeTitle}.md`);
}

function yamlString(value) {
  return JSON.stringify(String(value || ""));
}

function buildLessonNoteContent(course, title) {
  return [
    "---",
    "aliases: []",
    "tags:",
    "  - course-note",
    `科目: ${yamlString(course.root)}`,
    `course: ${yamlString(course.title)}`,
    "---",
    "",
    `# ${title}`,
    "",
  ].join("\n");
}

function serializeData(data) {
  const payload = {
    ...data,
    version: 1,
    updatedAt: new Date().toISOString(),
    courses: [...data.courses].sort(compareSortKey),
  };
  return [
    "---",
    "learningBoard: true",
    "version: 1",
    `updated: "${payload.updatedAt}"`,
    "---",
    "",
    "# Learning Progress Board",
    "",
    "This file is the single source of truth for the Learning Progress Dashboard plugin.",
    "You can hand-edit it, but keep the fenced JSON block valid.",
    "",
    "```learning-board-json",
    JSON.stringify(payload, null, 2),
    "```",
    "",
  ].join("\n");
}

function isLessonFile(file) {
  const basename = file.basename.toLowerCase();
  if (!LESSON_RE.test(file.basename)) return false;
  if (file.path.split("/").length === 3) return true;
  return !EXCLUDE_KEYWORDS.some((keyword) => basename.includes(keyword.toLowerCase()));
}

function isCourseRootPath(path) {
  const rootName = String(path || "").split("/")[0];
  return Boolean(COURSE_ROOTS[rootName]);
}

function shouldSyncCourseFile(file) {
  return file instanceof TFile && file.extension === "md" && isCourseRootPath(file.path);
}

function isPluginSourceFile(file) {
  return file instanceof TFile && PLUGIN_SOURCE_PATHS.has(file.path);
}

function lessonTitle(file) {
  const match = file.basename.match(LESSON_RE);
  if (!match) return file.basename;
  return match[2].replace(/[_-]+/g, " ").trim();
}

function lessonLabelAndSort(file, coursePath) {
  const match = file.basename.match(LESSON_RE);
  const lessonNumber = match ? Number(match[1]) : 999;
  const lessonLabel = match ? match[1].padStart(2, "0") : "?";
  const relative = file.path.slice(coursePath.length + 1);
  const parentParts = relative.split("/").slice(0, -1);
  if (parentParts.length > 0) {
    const sectionMatch = parentParts[0].match(SECTION_RE);
    if (sectionMatch) {
      const sectionNumber = Number(sectionMatch[1]);
      return {
        label: `${sectionNumber}.${lessonLabel}`,
        sort: sectionNumber + lessonNumber / 100,
      };
    }
  }
  return { label: lessonLabel, sort: lessonNumber };
}

class LearningBoardStore {
  constructor(app) {
    this.app = app;
  }

  shouldDropStaleLesson(lesson, coursePath) {
    const notePath = normalizePath(String(lesson.notePath || "").trim());
    const normalizedCoursePath = normalizePath(String(coursePath || ""));
    if (!notePath || !normalizedCoursePath) return false;
    if (!notePath.endsWith(".md")) return false;
    if (!notePath.startsWith(`${normalizedCoursePath}/`)) return false;
    if (this.app.vault.getAbstractFileByPath(notePath)) return false;
    if (normalizeState(lesson.state) !== "raw") return false;
    if (String(lesson.remark || "").trim()) return false;
    return true;
  }

  async ensureFolder(path) {
    const parts = path.split("/");
    let current = "";
    for (let index = 0; index < parts.length - 1; index += 1) {
      current = current ? `${current}/${parts[index]}` : parts[index];
      if (!this.app.vault.getAbstractFileByPath(current)) {
        await this.app.vault.createFolder(current);
      }
    }
  }

  async ensureDataFile() {
    const normalized = normalizePath(DATA_PATH);
    let file = this.app.vault.getFileByPath(normalized);
    if (file) return file;
    await this.ensureFolder(normalized);
    const data = await this.scanData(null);
    file = await this.app.vault.create(normalized, serializeData(data));
    return file;
  }

  async read() {
    const file = await this.ensureDataFile();
    const markdown = await this.app.vault.cachedRead(file);
    const parsed = readJsonBlock(markdown);
    if (!parsed || !Array.isArray(parsed.courses)) {
      const data = await this.scanData(null);
      await this.write(data);
      return data;
    }
    return {
      version: 1,
      updatedAt: parsed.updatedAt || new Date().toISOString(),
      courses: parsed.courses.map((course) => ({
        ...course,
        lessons: Array.isArray(course.lessons)
          ? course.lessons.map((lesson) => ({ ...lesson, state: normalizeState(lesson.state) }))
          : [],
      })),
    };
  }

  async write(data) {
    const file = await this.ensureDataFile();
    await this.app.vault.modify(file, serializeData(data));
  }

  async sync(existingData) {
    const data = await this.scanData(existingData);
    const changed = JSON.stringify(existingData?.courses || []) !== JSON.stringify(data.courses);
    if (changed) await this.write(data);
    return { data, changed };
  }

  async scanData(existingData) {
    const existing = existingData && Array.isArray(existingData.courses) ? existingData : { courses: [] };
    const scannedCourses = new Map();

    for (const file of this.app.vault.getMarkdownFiles()) {
      const parts = file.path.split("/");
      if (parts.length < 3) continue;
      const rootName = parts[0];
      const rootLabel = COURSE_ROOTS[rootName];
      if (!rootLabel || !isLessonFile(file)) continue;

      const courseTitle = parts[1];
      const coursePath = `${rootName}/${courseTitle}`;
      const labelData = lessonLabelAndSort(file, coursePath);
      const course =
        scannedCourses.get(coursePath) ||
        {
          id: makeId(coursePath),
          title: courseTitle,
          root: rootLabel,
          path: coursePath,
          visible: true,
          order: scannedCourses.size + 1,
          lessons: [],
        };
      course.lessons.push({
        id: makeId(file.path),
        label: labelData.label,
        title: lessonTitle(file),
        notePath: file.path,
        state: "raw",
        remark: "",
        sort: labelData.sort,
      });
      scannedCourses.set(coursePath, course);
    }

    const resultByPath = new Map();
    let maxOrder = 0;
    for (const existingCourse of existing.courses) {
      const cloned = {
        ...existingCourse,
        visible: true,
        lessons: Array.isArray(existingCourse.lessons)
          ? existingCourse.lessons
              .map((lesson) => ({ ...lesson, state: normalizeState(lesson.state) }))
              .filter((lesson) => !this.shouldDropStaleLesson(lesson, existingCourse.path))
          : [],
      };
      maxOrder = Math.max(maxOrder, Number(cloned.order) || 0);
      resultByPath.set(cloned.path, cloned);
    }

    const sortedScanned = [...scannedCourses.values()].sort(compareSortKey);
    for (const scanned of sortedScanned) {
      const current = resultByPath.get(scanned.path);
      if (!current) {
        maxOrder += 1;
        scanned.order = maxOrder;
        scanned.lessons.sort((a, b) => a.sort - b.sort || a.title.localeCompare(b.title));
        scanned.lessons.forEach((lesson) => delete lesson.sort);
        resultByPath.set(scanned.path, scanned);
        continue;
      }

      const existingLessons = new Map();
      for (const lesson of current.lessons) {
        existingLessons.set(lesson.notePath || lesson.id, lesson);
      }
      for (const lesson of scanned.lessons.sort((a, b) => a.sort - b.sort || a.title.localeCompare(b.title))) {
        const key = lesson.notePath || lesson.id;
        if (!existingLessons.has(key)) {
          delete lesson.sort;
          current.lessons.push(lesson);
        }
      }
      current.lessons = current.lessons.sort(compareLessons);
    }

    return {
      version: 1,
      updatedAt: new Date().toISOString(),
      courses: [...resultByPath.values()].sort(compareSortKey),
    };
  }
}

class ManualLessonModal extends Modal {
  constructor(app, course, onSubmit) {
    super(app);
    this.course = course;
    this.onSubmit = onSubmit;
  }

  onOpen() {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.addClass("lpd-modal");
    contentEl.createEl("h2", { text: "Add lesson" });
    contentEl.createDiv({ cls: "lpd-muted", text: this.course.title });

    const form = contentEl.createEl("form", { cls: "lpd-modal-form" });
    const labelField = form.createEl("label", { cls: "lpd-field" });
    labelField.createSpan({ text: "课节点编号" });
    const labelInput = labelField.createEl("input", {
      attr: { type: "text", placeholder: "例如 08" },
    });
    labelInput.value = nextLessonLabel(this.course);

    const titleField = form.createEl("label", { cls: "lpd-field" });
    titleField.createSpan({ text: "课节标题" });
    const titleInput = titleField.createEl("input", {
      attr: { type: "text", placeholder: "例如 Unit root and ARIMA" },
    });

    const actions = form.createDiv({ cls: "lpd-detail-actions" });
    const cancel = actions.createEl("button", {
      cls: "lpd-ghost",
      text: "Cancel",
      attr: { type: "button" },
    });
    cancel.addEventListener("click", () => this.close());
    actions.createEl("button", {
      cls: "lpd-primary",
      text: "Add",
      attr: { type: "submit" },
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const label = labelInput.value.trim();
      const title = titleInput.value.trim();
      if (!label || !title) {
        new Notice("Lesson label and title are required.");
        return;
      }
      const submitted = await this.onSubmit({
        label,
        title,
      });
      if (submitted !== false) this.close();
    });

    window.setTimeout(() => titleInput.focus(), 0);
  }
}

class LearningProgressDashboardView extends ItemView {
  constructor(leaf, plugin) {
    super(leaf);
    this.plugin = plugin;
    this.data = null;
    this.query = "";
    this.rootFilter = "All";
    this.selected = null;
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
    this.data = await this.plugin.store.read();
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

  visibleCourses() {
    if (!this.data) return [];
    const query = this.query.trim().toLowerCase();
    return [...this.data.courses]
      .sort(compareSortKey)
      .filter((course) => this.rootFilter === "All" || course.root === this.rootFilter)
      .filter((course) => {
        if (!query) return true;
        return [course.title, course.path, course.root].some((value) =>
          String(value || "").toLowerCase().includes(query)
        );
      });
  }

  renderSidebar(sidebar) {
    sidebar.createDiv({ cls: "lpd-kicker", text: "Academic Vault" });
    sidebar.createEl("h1", { text: "Learning Progress" });
    sidebar.createDiv({
      cls: "lpd-muted",
      text: "一门课一条轨道，一节课一个点。",
    });

    const search = sidebar.createEl("input", {
      cls: "lpd-search",
      attr: { type: "search", placeholder: "Filter courses..." },
    });
    search.value = this.query;
    search.addEventListener("input", () => {
      this.query = search.value;
      this.render();
    });

    const roots = sidebar.createDiv({ cls: "lpd-filter-list" });
    ["All", ...ROOT_ORDER].forEach((root) => {
      const button = roots.createEl("button", {
        cls: `lpd-filter ${this.rootFilter === root ? "is-active" : ""}`,
        text: root,
      });
      button.addEventListener("click", () => {
        this.rootFilter = root;
        this.render();
      });
    });

    const legend = sidebar.createDiv({ cls: "lpd-legend" });
    Object.entries(STATE_META).forEach(([state, meta]) => {
      const row = legend.createDiv({ cls: "lpd-legend-row" });
      const swatch = row.createSpan({ cls: `lpd-legend-swatch lpd-state-${state}` });
      for (let index = 1; index <= 3; index += 1) {
        swatch.createSpan({ cls: `lpd-legend-segment ${meta.depth >= index ? "is-filled" : ""}` });
      }
      row.createSpan({ text: `${meta.label}：${meta.description}` });
    });
  }

  renderMain(main) {
    const top = main.createDiv({ cls: "lpd-main-head" });
    top.createEl("h2", { text: "Course Rails" });
    const stats = this.computeStats();
    top.createDiv({
      cls: "lpd-muted",
      text: `${stats.courses} courses · ${stats.lessons} lessons · ${stats.mapped} mapped`,
    });

    const list = main.createDiv({ cls: "lpd-rail-list" });
    const courses = this.visibleCourses();
    if (courses.length === 0) {
      list.createDiv({ cls: "lpd-empty", text: "No courses match the current filter." });
      return;
    }

    courses.forEach((course) => {
      const row = list.createDiv({ cls: "lpd-course-row" });
      const courseInfo = row.createDiv({ cls: "lpd-course-info" });
      courseInfo.createDiv({ cls: "lpd-course-title", text: course.title });
      courseInfo.createDiv({ cls: "lpd-course-path", text: `${course.root} · ${course.path}` });

      const progress = this.courseProgress(course);
      const track = row.createDiv({ cls: "lpd-track" });
      this.applyProgressStyles(track, progress);
      const layers = track.createDiv({ cls: "lpd-track-layers", attr: { "aria-hidden": "true" } });
      layers.createDiv({ cls: "lpd-track-layer lpd-track-layer-learned" });
      layers.createDiv({ cls: "lpd-track-layer lpd-track-layer-organized" });
      layers.createDiv({ cls: "lpd-track-layer lpd-track-layer-mapped" });
      const nodes = track.createDiv({ cls: "lpd-track-nodes" });
      const lessons = [...(course.lessons || [])].sort(compareLessons);
      lessons.forEach((lesson) => {
        const state = normalizeState(lesson.state);
        const meta = STATE_META[state];
        const node = nodes.createEl("button", {
          cls: `lpd-lesson-node lpd-state-${state} ${
            this.selected && this.selected.courseId === course.id && this.selected.lessonId === lesson.id ? "is-selected" : ""
          }`,
          attr: {
            title: `${lesson.label} ${lesson.title} · ${meta.label}`,
          },
        });
        const capsule = node.createSpan({ cls: "lpd-capsule" });
        for (let index = 1; index <= 3; index += 1) {
          capsule.createSpan({ cls: `lpd-capsule-segment ${meta.depth >= index ? "is-filled" : ""}` });
        }
        node.createSpan({ cls: "lpd-node-label", text: lesson.label });
        node.addEventListener("click", () => {
          this.selected = { courseId: course.id, lessonId: lesson.id };
          this.render();
        });
      });

      const courseActions = row.createDiv({ cls: "lpd-course-actions" });
      courseActions.createDiv({
        cls: "lpd-course-progress",
        text: `${progress.learned}/${progress.total} 已学 · ${progress.organized} 已整理 · ${progress.mapped} 已成图`,
      });
      const moveUp = courseActions.createEl("button", { cls: "lpd-mini", text: "↑" });
      moveUp.addEventListener("click", () => this.plugin.moveCourse(course.id, -1));
      const moveDown = courseActions.createEl("button", { cls: "lpd-mini", text: "↓" });
      moveDown.addEventListener("click", () => this.plugin.moveCourse(course.id, 1));
      const addLesson = courseActions.createEl("button", { cls: "lpd-mini", text: "+ Lesson" });
      addLesson.addEventListener("click", () => this.plugin.addManualLesson(course.id));
    });
  }

  renderDetail(detail) {
    detail.createDiv({ cls: "lpd-kicker", text: "Lesson Detail" });
    const selected = this.getSelectedLesson();
    if (!selected) {
      detail.createEl("h2", { text: "Select a lesson" });
      detail.createDiv({
        cls: "lpd-muted",
        text: "点击轨道上的课节点，在这里编辑状态、链接和备注。",
      });
      return;
    }

    const { course, lesson } = selected;
    detail.createEl("h2", { text: `${lesson.label} ${lesson.title}` });
    detail.createDiv({ cls: "lpd-detail-course", text: course.title });

    const stateField = detail.createEl("label", { cls: "lpd-field" });
    stateField.createSpan({ text: "状态" });
    const stateSelect = stateField.createEl("select");
    Object.entries(STATE_META).forEach(([state, meta]) => {
      const option = stateSelect.createEl("option", {
        text: `${meta.label} · ${meta.description}`,
        attr: { value: state },
      });
      option.selected = normalizeState(lesson.state) === state;
    });

    const pathField = detail.createEl("label", { cls: "lpd-field" });
    pathField.createSpan({ text: "课节笔记链接" });
    const notePath = pathField.createEl("input", {
      attr: { type: "text", placeholder: "01_Math/..." },
    });
    notePath.value = lesson.notePath || "";

    const remarkField = detail.createEl("label", { cls: "lpd-field" });
    remarkField.createSpan({ text: "人工备注" });
    const remark = remarkField.createEl("textarea", {
      attr: { placeholder: "哪里卡住、下一步要回看什么、Big Picture 里怎么定位..." },
    });
    remark.value = lesson.remark || "";

    const actions = detail.createDiv({ cls: "lpd-detail-actions" });
    const save = actions.createEl("button", { cls: "lpd-primary", text: "Save" });
    save.addEventListener("click", async () => {
      await this.plugin.updateLesson(course.id, lesson.id, {
        state: stateSelect.value,
        notePath: notePath.value.trim(),
        remark: remark.value.trim(),
      });
      new Notice("Learning progress saved.");
    });

    const open = actions.createEl("button", { cls: "lpd-ghost", text: "Open note" });
    open.addEventListener("click", () => this.plugin.openPath(notePath.value.trim()));
  }

  getSelectedLesson() {
    if (!this.selected || !this.data) return null;
    const course = this.data.courses.find((item) => item.id === this.selected.courseId);
    if (!course) return null;
    const lesson = (course.lessons || []).find((item) => item.id === this.selected.lessonId);
    if (!lesson) return null;
    return { course, lesson };
  }

  computeStats() {
    const courses = this.data ? this.data.courses : [];
    const lessons = courses.flatMap((course) => course.lessons || []);
    return {
      courses: courses.length,
      lessons: lessons.length,
      mapped: lessons.filter((lesson) => normalizeState(lesson.state) === "mapped").length,
    };
  }

  courseProgress(course) {
    const lessons = course.lessons || [];
    const total = lessons.length;
    const learned = lessons.filter((lesson) => stateDepth(lesson.state) >= 1).length;
    const organized = lessons.filter((lesson) => stateDepth(lesson.state) >= 2).length;
    const mapped = lessons.filter((lesson) => stateDepth(lesson.state) >= 3).length;
    return {
      total,
      learned,
      organized,
      mapped,
    };
  }

  applyProgressStyles(track, progress) {
    track.style.setProperty("--lpd-learned-width", percentage(progress.learned, progress.total));
    track.style.setProperty("--lpd-organized-width", percentage(progress.organized, progress.total));
    track.style.setProperty("--lpd-mapped-width", percentage(progress.mapped, progress.total));
  }
}

module.exports = class LearningProgressDashboardPlugin extends Plugin {
  async onload() {
    this.store = new LearningBoardStore(this.app);
    this.syncTimer = null;
    this.reloadTimer = null;
    this.isReloading = false;
    this.registerView(VIEW_TYPE, (leaf) => new LearningProgressDashboardView(leaf, this));

    this.addRibbonIcon("map", "Learning Progress Dashboard", () => this.openDashboard());
    this.addCommand({
      id: "open-learning-progress-dashboard",
      name: "Open Learning Progress Dashboard",
      callback: () => this.openDashboard(),
    });
    this.addCommand({
      id: "reload-learning-progress-dashboard",
      name: "Reload Learning Progress Dashboard plugin",
      callback: () => this.reloadSelf(),
    });

    this.registerEvent(
      this.app.vault.on("modify", (file) => {
        if (file instanceof TFile && file.path === DATA_PATH) {
          this.refreshViews();
        }
        if (isPluginSourceFile(file)) {
          this.scheduleSelfReload();
        }
      })
    );
    this.registerEvent(
      this.app.vault.on("create", (file) => {
        if (shouldSyncCourseFile(file)) this.scheduleCourseSync();
      })
    );
    this.registerEvent(
      this.app.vault.on("rename", (file, oldPath) => {
        if (shouldSyncCourseFile(file) || isCourseRootPath(oldPath)) this.scheduleCourseSync();
      })
    );

    this.app.workspace.onLayoutReady(async () => {
      await this.store.ensureDataFile();
      await this.syncCoursesQuietly();
      await this.openDashboard();
    });
  }

  onunload() {
    if (this.syncTimer) window.clearTimeout(this.syncTimer);
    if (this.reloadTimer) window.clearTimeout(this.reloadTimer);
    this.app.workspace.detachLeavesOfType(VIEW_TYPE);
  }

  async openDashboard() {
    const existing = this.app.workspace.getLeavesOfType(VIEW_TYPE);
    if (existing.length > 0) {
      this.app.workspace.revealLeaf(existing[0]);
      return;
    }
    await this.app.workspace.getLeaf("tab").setViewState({
      type: VIEW_TYPE,
      active: true,
    });
  }

  async refreshViews() {
    for (const leaf of this.app.workspace.getLeavesOfType(VIEW_TYPE)) {
      if (leaf.view instanceof LearningProgressDashboardView) {
        await leaf.view.refresh();
      }
    }
  }

  scheduleCourseSync() {
    if (this.syncTimer) window.clearTimeout(this.syncTimer);
    this.syncTimer = window.setTimeout(async () => {
      this.syncTimer = null;
      await this.syncCoursesQuietly();
    }, 800);
  }

  scheduleSelfReload() {
    if (this.reloadTimer) window.clearTimeout(this.reloadTimer);
    this.reloadTimer = window.setTimeout(() => {
      this.reloadTimer = null;
      this.reloadSelf();
    }, 600);
  }

  async reloadSelf() {
    if (this.isReloading) return;
    const plugins = this.app.plugins;
    const pluginId = this.manifest?.id || PLUGIN_ID;
    if (!plugins?.disablePlugin || !plugins?.enablePlugin) {
      new Notice("Plugin reload is unavailable; use Obsidian reload.");
      return;
    }

    this.isReloading = true;
    try {
      await plugins.disablePlugin(pluginId);
      await plugins.enablePlugin(pluginId);
      new Notice("Learning Progress Dashboard reloaded.");
    } catch (error) {
      console.error(error);
      new Notice("Plugin reload failed; use Obsidian reload.");
    }
  }

  async syncCoursesQuietly() {
    const existing = await this.store.read();
    const result = await this.store.sync(existing);
    if (result.changed) await this.refreshViews();
  }

  async updateLesson(courseId, lessonId, patch) {
    const data = await this.store.read();
    const course = data.courses.find((item) => item.id === courseId);
    if (!course) return;
    const lesson = (course.lessons || []).find((item) => item.id === lessonId);
    if (!lesson) return;
    Object.assign(lesson, patch);
    await this.store.write(data);
    await this.refreshViews();
  }

  async addManualLesson(courseId) {
    const data = await this.store.read();
    const course = data.courses.find((item) => item.id === courseId);
    if (!course) return;

    new ManualLessonModal(this.app, course, async ({ label, title }) => {
      const normalizedLabel = normalizeLessonLabel(label);
      if (!normalizedLabel) {
        new Notice("Lesson label must contain a number.");
        return false;
      }

      const latest = await this.store.read();
      const latestCourse = latest.courses.find((item) => item.id === courseId);
      if (!latestCourse) return;
      const notePath = buildLessonNotePath(latestCourse, normalizedLabel, title);
      if (this.app.vault.getAbstractFileByPath(notePath)) {
        new Notice(`Note already exists: ${notePath}`);
        return false;
      }

      await this.store.ensureFolder(notePath);
      const file = await this.app.vault.create(notePath, buildLessonNoteContent(latestCourse, title));
      latestCourse.lessons = Array.isArray(latestCourse.lessons) ? latestCourse.lessons : [];
      latestCourse.lessons.push({
        id: makeId(notePath),
        label: normalizedLabel,
        title,
        notePath,
        state: "raw",
        remark: "",
      });
      latestCourse.lessons.sort(compareLessons);
      await this.store.write(latest);
      await this.refreshViews();
      await this.app.workspace.getLeaf(false).openFile(file);
      new Notice("Lesson note created.");
      return true;
    }).open();
  }

  async moveCourse(courseId, delta) {
    const data = await this.store.read();
    const courses = [...data.courses].sort(compareSortKey);
    const index = courses.findIndex((course) => course.id === courseId);
    const targetIndex = index + delta;
    if (index < 0 || targetIndex < 0 || targetIndex >= courses.length) return;
    const current = courses[index];
    const target = courses[targetIndex];
    const currentOrder = Number(current.order) || index + 1;
    current.order = Number(target.order) || targetIndex + 1;
    target.order = currentOrder;
    await this.store.write(data);
    await this.refreshViews();
  }

  async openPath(path) {
    if (!path) {
      new Notice("No note path set.");
      return;
    }
    let file = this.app.vault.getFileByPath(path);
    if (!file && !path.endsWith(".md")) {
      file = this.app.vault.getFileByPath(`${path}.md`);
    }
    if (!file) {
      new Notice(`Note not found: ${path}`);
      return;
    }
    await this.app.workspace.getLeaf(false).openFile(file);
  }
};
