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

function classroomMaterial(app, job, weeklyTasks = []) {
  const task = job.request?.task || {};
  const weekly = weeklyTasks.find(t => t.id === task.id && t.raw === task.raw);
  const usable = file => file && ["md", "pdf"].includes(file.extension) && file.path !== task.path
    && !file.path.startsWith("99_学习情况记录/")
    && !/课程总览|资料索引|Previous_Versions|transcript|solution|(?:^|[_ -])sol[_. -]|textbook|答案|解答/i.test(file.path);
  const resolve = (link, from = task.path) => {
    const path = link?.split("#")[0];
    if (!path || /^(?:[a-z]+:|\/\/)/i.test(path)) return null;
    const file = app.metadataCache.getFirstLinkpathDest(path, from || "");
    return usable(file) ? file : null;
  };
  const unique = files => [...new Map(files.filter(Boolean).map(file => [file.path, file])).values()];
  const linksIn = file => {
    const cache = app.metadataCache.getFileCache(file);
    return [...(cache?.links || []), ...(cache?.embeds || [])];
  };
  const primary = files => {
    files = unique(files);
    if (!files.length) return null;
    const pdfs = files.filter(file => file.extension === "pdf");
    if (pdfs.length === 1) return { file: pdfs[0] };
    const notes = files.filter(file => file.extension === "md");
    if (notes.length !== 1) return { message: "这项任务有多份材料，尚未明确本次讲义。" };
    const note = notes[0];
    // Only follow teaching-material links in the exact lesson, never every link in a course index.
    const slides = unique(linksIn(note).filter(l => /slides?|handout|讲义|课件|幻灯片|lecture.notes/i.test(`${l.link} ${l.displayText || ""}`)
      && !/transcript|solution|textbook|参考答案|上一讲|下一讲|previous|next/i.test(`${l.link} ${l.displayText || ""}`))
      .map(l => resolve(l.link, note.path)).filter(file => file?.extension === "pdf"));
    return slides.length === 1 ? { file: slides[0] } : { file: note,
      message: slides.length > 1 ? "已打开本课材料页，其中包含多份讲义。" : "已打开本课笔记；独立讲义尚未定位。" };
  };
  const source = task.path && app.vault.getAbstractFileByPath(task.path);
  const direct = source ? linksIn(source).filter(l => task.raw?.includes(l.original)).map(l => resolve(l.link)) : [];
  // Saved tasks also work before Obsidian has indexed their source note.
  for (const match of (task.raw || "").matchAll(/\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g)) direct.push(resolve(match[1]));
  let result = primary(direct);
  if (result) return result;
  result = primary((weekly?.detail_sources || []).filter(s => s.status === "unchanged").map(s => resolve(s.path)));
  if (result) return result;
  const lecture = task.title?.match(/\bLecture\s*0*(\d+)\b/i)?.[1];
  if (lecture) {
    const number = new RegExp(`(?:^0*${Number(lecture)}[_ .-]|(?:lecture|lec)[_ -]*0*${Number(lecture)}(?=[^0-9]|$))`, "i");
    result = primary((job.request?.materials || []).map(m => resolve(m.path)).filter(file => file && number.test(file.basename)));
    if (result) return result;
  }
  // School weeks are not chapter numbers. Only use an explicitly week-labelled lecture file.
  if (weekly?.group === "lecture" && /^\d+$/.test(weekly.week) && /^(ST|EC)\d{3}$/.test(weekly.course_id)) {
    const week = new RegExp(`(?:^|[/ _-])Week[_ -]*0*${Number(weekly.week)}(?=[^0-9]|$)`, "i");
    result = primary(app.vault.getFiles().filter(file => usable(file) && file.extension === "pdf"
      && file.basename.startsWith(weekly.course_id + " ") && /\/02_Lecture_Notes\//.test(file.path) && week.test(file.path)));
    if (result) return result;
  }
  return { message: "这项任务还没有明确匹配的本地讲义。" };
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

function deadlineBoard(data, now = new Date()) {
  const clock = londonClock(now), groups = { upcoming: [], later: [], past: [], undated: [], done: [] };
  for (const task of data.obligations || []) {
    if (task.status === "cancelled") continue;
    const time = deadlineTime(task).replace(/前$/, "");
    const distance = task.date ? Math.round((Date.parse(task.date) - Date.parse(clock.date)) / 86400000) : NaN;
    const days = Number.isFinite(distance) ? distance : null;
    const late = days !== null && (days < 0 || days === 0 && time && time <= clock.time);
    const group = task.status === "recorded_complete" ? "done" : days === null || !Number.isFinite(days) ? "undated" : late ? "past" : days <= 14 ? "upcoming" : "later";
    const text = titleForDeadline(task.raw).replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
    const parts = text.split(/[｜|]/).map(s => s.trim());
    const course = parts.length > 1 ? parts.shift() : text.match(/\b(?:ST|EC)\d{3}\b/)?.[0] || "其他事项";
    const body = parts.join(" · ") || text;
    const compound = /[）)]\s*[＋+]/.test(body);
    const title = (compound ? body : body.replace(/[（(][^）)]+[）)]/g, "")).split(/[，；：]/)[0].replace(/^参加/, "").replace(/^提交/, "").trim();
    const detail = compound ? "" : [...body.matchAll(/[（(]([^）)]+)[）)]/g)].map(m => m[1]).join(" · ");
    groups[group].push({ task, course, title, detail, days, time,
      kind: /参加|出席|说明会/.test(body) ? "到场" : /登记|注册|报名/.test(body) ? "登记" : /提交|上传/.test(body) ? "提交" : "事项",
      timing: group === "done" ? "已完成" : group === "past" ? "时间已过 · 待核对" : days === null ? "日期待确认" : days === 0 ? "今天" : days === 1 ? "明天" : `${days} 天后` });
  }
  for (const items of Object.values(groups)) items.sort((a, b) => `${a.task.date || "9999"} ${a.time}`.localeCompare(`${b.task.date || "9999"} ${b.time}`));
  groups.done.reverse();
  return groups;
}

function deadlineSupplements(text = "") {
  // This callout is the existing read-only list of optional and unresolved dates.
  const section = text.match(/^> \[![^\]]+\][+-]?\s*其他已公布提交节点与待定日期[^\n]*\n((?:>[^\n]*(?:\n|$))*)/m)?.[1] || "";
  return section.split("\n").filter(line => /^>\s*-\s/.test(line)).map((line, index) => {
    const ref = line.match(/\[\[([^\]|]+)/)?.[1] || DEADLINES;
    const plain = displayText(line.replace(/^>\s*-\s*/, "").replace(/\[\[[^\]]+\]\]/g, "")).trim();
    const head = plain.split(/[，。]/)[0];
    const date = head.match(/^(\d+)\s*月\s*(\d+)\s*日(?:\s+(\d{2}:\d{2}))?/);
    return { key: `deadline-note-${index}`, ref, text: plain,
      date: date ? `${date[1]}/${date[2]}` : "待定", time: date?.[3] || (date ? "时刻待确认" : ""),
      title: head.replace(date?.[0] || "", "").replace(/^\s*[｜|]\s*/, "").replace(/\s*[｜|]\s*(?:可选提交|形成性反馈节点|日期待学校确认).*$/, ""),
      kind: /可选/.test(head) ? "可选提交" : /日期待/.test(head) ? "日期待确认" : "反馈节点" };
  });
}

function selfStudyFor(course, weekly) {
  const tasks = (weekly?.tasks || []).filter(t => t.course_id === course.id && t.state !== "cancelled");
  const units = tasks.filter(t => !t.title.includes("本轮重看"));
  const replay = tasks.filter(t => t.title.includes("本轮重看"));
  const next = tasks.find(t => course.next_task && t.path === course.next_task.path && t.raw === course.next_task.raw);
  const observed = next?.record ? next : tasks.find(t => t.record);
  const position = course.replay_progress ? course.personal.summary : observed?.record || course.personal.summary;
  return { units, replay, nextTask: next, defaultTask: next || replay.at(-1) || units.find(t => t.state === "recorded") || units[0],
    finite: course.mode === "finite" && !!course.unit_progress?.total,
    recorded: units.filter(t => t.state === "recorded").length,
    position: position.replace(/^(?:原计划连续完成至\s*|本轮重看已完成：)/, "").replace(course.id + "｜", ""),
    next: next ? taskCaption(next).split(/[；，]/)[0].replace(/^完成\s*/, "").replace(/这一组.*$/, "")
      : course.continuation ? displayText(course.continuation.title).replace(course.id + "｜", "") : course.next_step };
}

function learningContext(data, original) {
  const task = (data.curriculum?.weekly?.tasks || []).find(t => t.path === original?.path && t.raw === original?.raw) || original;
  if (!task) return null;
  const course = data.curriculum?.courses?.find(c => c.id === task.course_id);
  const fact = positionFor(course?.personal);
  const ownFocus = data.focus?.task?.id === original.id;
  const next = task.record_next || (ownFocus && data.focus.next_step)
    || task.checkpoints?.[0]?.condition || task.checkpoints?.[0]?.label
    || (course?.next_task?.id === original.id && !fact.outdated && course.next_step)
    || "进入课堂，先从已确认的停点接上。";
  return { task, course, next: displayText(next), condition: studyUnitFor(task).condition,
    stop: displayText(task.record || (course ? fact.topic : "这项的实际学习停点尚未记录。")),
    stopLabel: task.record ? "上次停点" : `课程最近记录${fact.date ? " · " + fact.date.slice(5).replace("-", "/") : ""}${fact.outdated ? " · 待更新" : ""}` };
}

function nearbyUnits(units, selected) {
  const index = Math.max(0, units.findIndex(t => t.id === selected.id));
  return units.slice(Math.max(0, index - 1), Math.min(units.length, index + 2));
}

function studyUnitFor(task, index = 0) {
  const full = displayText(task.title || task.raw).replace(/^[^｜]+｜/, "").replace(/^本轮重看\s*/, "");
  const lecture = full.match(/^Lecture\s+(\d+)\s*[:：]?\s*/i);
  const range = full.match(/第\s*(\d+\s*[–—-]\s*\d+)\s*单元/);
  const label = lecture ? `Lecture ${lecture[1].padStart(2, "0")}` : range ? `第 ${range[1]} 单元` : `学习项 ${index + 1}`;
  const title = lecture ? full.slice(lecture[0].length).split(/[，；]/)[0]
    : range && full.includes("：") ? full.split("：")[1].split(/[；;]/)[0]
      : full.match(/^完成\s+(.+?)\s+这一组/)?.[1] || full.replace(/^完成\s*/, "").split(/[：；，]/)[0];
  // Unstructured plans may expose only the text after a colon as done_when. Keep the full requirement.
  const proposed = displayText(task.done_when || task.title || task.raw).replace(/^[^｜]+｜/, "").replace(/^本轮重看\s*/, "");
  const condition = task.short_title || task.checkpoints?.length || !full.endsWith(proposed) ? proposed : full;
  const explicitCondition = !!task.checkpoints?.length || !lecture || condition !== full || /[，；]/.test(condition);
  // Student OS rules give MIT video lectures a default closure: watched and understood.
  const videoDefault = !explicitCondition && /^MIT\s*\d/.test(task.course_id || task.title || "");
  return { label, number: lecture?.[1].padStart(2, "0") || String(index + 1).padStart(2, "0"),
    title: title || label, full, condition: videoDefault ? "看完本讲视频，并确认已理解。" : condition, lecture: !!lecture,
    hasCondition: explicitCondition || videoDefault,
    state: { done: "已完成", recorded: "有学习停点", unknown: "待确认", planned: "后续计划" }[task.state] || "待确认",
    symbol: { done: "✓", recorded: "◐", unknown: "○", planned: "○" }[task.state] || "○" };
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
  constructor(plugin, task, report = "") {
    super(plugin.app); this.plugin = plugin; this.task = taskPresentation(plugin, task); this.report = report; this.files = [];
  }
  mount(host, close) {
    this.inlineHost = host; this.afterClose = close;
    host.addClass("sos-task-dialog", "sos-task-detail-panel");
    const heading = host.createDiv({ cls: "sos-inline-heading" });
    heading.appendChild(this.titleEl);
    heading.createEl("button", { text: "收起", cls: "sos-quiet" }).addEventListener("click", () => this.close());
    host.appendChild(this.contentEl);
    void this.onOpen().catch(error => new Notice(error.message));
  }
  close() {
    if (!this.inlineHost) return super.close();
    this.onClose(); this.inlineHost.remove(); this.afterClose?.();
  }
  async onOpen() {
    this.titleEl.setText("任务详情"); this.modalEl.addClass("sos-task-dialog"); this.contentEl.addClass("sos-completion-modal");
    const task = this.task, points = task.checkpoints || [];
    const header = this.contentEl.createDiv({ cls: "sos-detail-header" });
    const group = { lecture: "Lecture", seminar: "Seminar · 课后练习", recommended: "推荐内容", other: "准备与其他要求" }[task.group];
    header.createDiv({ cls: "sos-detail-eyebrow", text: [task.course_id, /^\d+$/.test(task.week || "") ? `Week ${task.week}` : "", group].filter(Boolean).join("  /  ") });
    header.createEl("h2", { text: taskCaption(task) });
    const meta = header.createDiv({ cls: "sos-detail-meta" });
    meta.createSpan({ cls: "sos-detail-status", text: task.state === "done" ? "✓ 已记录完成" : task.record ? "已有停点 · 整项待核验" : "完成状态待核验" });
    if (points.length) meta.createSpan({ text: `${points.length} 个检查点` });
    if (task.group === "recommended") meta.createSpan({ text: "不影响本周核心收工" });
    if (task.record) {
      const stop = this.contentEl.createDiv({ cls: "sos-detail-stop" });
      stop.createEl("small", { text: "上次停在这里 · 本人报告" }); stop.createEl("p", { text: task.record });
      if (task.record_next) stop.createDiv({ text: "接着做：" + task.record_next });
    }
    const route = this.contentEl.createDiv({ cls: "sos-detail-route" });
    const steps = route.createDiv({ cls: "sos-detail-steps" });
    steps.createEl("h3", { text: points.length ? "分步推进" : "本次任务" });
    this.canRecord = task.kind !== "obligation" && !["done", "cancelled"].includes(task.state) && !["recorded_complete", "cancelled"].includes(task.status);
    if (points.length) {
      const list = steps.createEl("ol", { cls: "sos-detail-checkpoints" });
      points.forEach((point, index) => {
        const row = list.createEl("li"); row.createSpan({ cls: "sos-step-number", text: String(index + 1) });
        const text = row.createDiv(); text.createEl("strong", { text: point.label }); text.createEl("p", { text: point.condition });
        if (this.canRecord) {
          const record = row.createEl("button", { cls: "sos-step-record", text: "记停点", attr: { "aria-label": "记录停点：" + point.label } });
          record.addEventListener("click", () => {
            this.showMode("progress");
            this.progressForm.summary.placeholder = `关于“${point.label}”，已做到哪里、还卡在哪里？`;
            this.progressForm.next.value = points[index + 1] ? points[index + 1].label + "：" + points[index + 1].condition : "核对本项收工条件与实际产物";
            this.progressForm.summary.focus();
          });
        }
      });
      steps.createEl("small", { cls: "sos-detail-hint", text: "按实际停点接续；检查点编号不表示已经完成。" });
    } else steps.hidden = true;
    const finish = route.createDiv({ cls: "sos-detail-finish" });
    finish.createEl("small", { text: "收工条件" });
    finish.createEl("p", { text: /^MIT\s*\d/.test(task.course_id || task.title || "")
      ? studyUnitFor(task).condition : displayText(task.done_when || task.title || task.raw) });
    finish.createDiv({ cls: "sos-detail-boundary", text: "达到整项要求后申请核验，核验通过才会同步完成状态。" });
    if (task.detail_needs_review) finish.createEl("p", { cls: "sos-evidence-warning", text: "要求有更新，分步路线待管家重新整理。" });
    const modes = this.contentEl.createDiv({ cls: "sos-detail-actions" }); this.modeButtons = {};
    if (this.canRecord) {
      modes.createEl("button", { text: "学这一项", cls: "sos-primary" }).addEventListener("click", () => {
        const view = this.app.workspace.getLeavesOfType(STEWARD_VIEW)[0]?.view;
        if (view) void view.beginTask(task).catch(error => new Notice(error.message));
      });
    }
    for (const [mode, label] of [["progress", "记录进度"], ["complete", "申请完成核验"], ["close", "不再需要…"]]) {
      if (mode === "progress" && !this.canRecord) continue;
      if (mode === "close") modes.createEl("button", { text: "问管家", cls: "sos-quiet" }).addEventListener("click", () => this.app.workspace.getLeavesOfType(STEWARD_VIEW)[0]?.view.discussTask(task));
      const button = modes.createEl("button", { text: label, cls: mode === "progress" ? "sos-primary" : mode === "close" ? "sos-quiet sos-close-option" : "" });
      button.addEventListener("click", () => this.showMode(mode)); this.modeButtons[mode] = button;
    }
    this.progressPanel = this.contentEl.createDiv({ cls: "sos-detail-editor" }); this.progressPanel.hidden = true;
    if (this.canRecord) this.progressForm = progressForm(this.progressPanel, this.plugin, task, null, () => this.close());
    this.completionPanel = this.contentEl.createDiv({ cls: "sos-detail-editor" }); this.completionPanel.hidden = true;
    const intro = this.completionPanel.createEl("p", { cls: "sos-muted", text: "说明完成的范围，并关联可核对的产物。" });
    const reportLabel = this.completionPanel.createEl("label", { cls: "sos-field", text: "这一整项实际完成了什么" });
    this.reportInput = reportLabel.createEl("textarea", { attr: { rows: "2", placeholder: "写具体完成范围、结果，以及仍需说明的情况。" } });
    this.reportInput.value = this.report;
    const dateLabel = this.completionPanel.createEl("label", { cls: "sos-field sos-detail-date", text: "完成日期（可留空）" });
    this.dateInput = dateLabel.createEl("input", { attr: { type: "date", max: londonClock().date } });
    const today = this.todayButton = dateLabel.createEl("button", { text: "今天", cls: "sos-quiet", attr: { type: "button" } });
    today.addEventListener("click", () => { this.dateInput.value = londonClock().date; });
    this.fileList = this.completionPanel.createDiv({ cls: "sos-evidence-files" });
    const add = this.addFile = this.completionPanel.createEl("button", { text: "＋ 关联笔记、代码或凭证" });
    add.addEventListener("click", () => new EvidencePicker(this.app, path => {
      if (!this.files.includes(path) && this.files.length < 6) this.files.push(path);
      this.renderFiles();
    }).open());
    this.completionPanel.createEl("p", { cls: "sos-muted", text: "读取文件内容核验；正式提交还需提交成功凭证。" });
    const actions = this.completionPanel.createDiv({ cls: "sos-actions" });
    this.submit = actions.createEl("button", { cls: "sos-primary", text: "开始核验" });
    this.submit.disabled = true;
    this.closePanel = this.contentEl.createDiv({ cls: "sos-detail-editor" }); this.closePanel.hidden = true;
    this.closePanel.createEl("p", { cls: "sos-muted", text: "关闭后退出待办，保留原因，不计入学习完成。" });
    this.closeReason = this.closePanel.createEl("textarea", { attr: { rows: "2", placeholder: "这项任务为什么不再需要？" } });
    this.closeTask = this.closePanel.createEl("button", { text: "确认关闭任务" }); this.closeTask.disabled = true;
    this.closeTask.addEventListener("click", () => this.check("close"));
    this.resultEl = this.contentEl.createDiv({ cls: "sos-check-result", attr: { role: "status", "aria-live": "polite" } });
    this.stop = this.contentEl.createEl("button", { text: "停止核验" }); this.stop.hidden = true;
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
    if (this.report) this.showMode("complete", false);
    try {
      const context = await this.plugin.steward.call("completion-context", { path: this.task.path, raw: this.task.raw });
      if (!this.contentEl.isConnected) return;
      if (context.required_external) intro.setText("这项任务包含正式提交，请关联匹配的提交成功凭证。");
      const latest = context.latest;
      this.files = latest?.request.evidence || context.suggested_evidence;
      if (latest && !this.report) this.reportInput.value = latest.request.report;
      this.dateInput.value = latest?.request.date || "";
      if (latest) this.showResult({ ...latest, path: latest.export_path });
      if (latest && !["complete", "closed", "insufficient", "cancelled"].includes(latest.phase)) {
        this.showMode("complete", false);
        this.resumeKey = latest.key; this.submit.setText("继续原核验");
        this.restartButton.hidden = false;
        this.reportInput.disabled = this.dateInput.disabled = add.disabled = today.disabled = true;
      }
      this.renderFiles();
      const finished = ["recorded_complete", "cancelled"].includes(task.status);
      this.submit.disabled = this.closeTask.disabled = finished;
      if (finished) for (const button of Object.values(this.modeButtons)) button.hidden = true;
    } catch (error) { this.resultEl.setText(error.message); }
  }
  showMode(mode, scroll = true) {
    this.progressPanel.hidden = mode !== "progress"; this.completionPanel.hidden = mode !== "complete"; this.closePanel.hidden = mode !== "close";
    for (const [key, button] of Object.entries(this.modeButtons)) button.setAttribute("aria-pressed", String(key === mode));
    if (scroll) ({ progress: this.progressPanel, complete: this.completionPanel, close: this.closePanel })[mode].scrollIntoView({ block: "nearest" });
  }
  renderFiles() {
    this.fileList.empty();
    for (const path of this.files) {
      const row = this.fileList.createDiv(); row.createSpan({ text: path.split("/").pop(), attr: { title: path } });
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
  }
  async check(intent = "complete") {
    const resumeKey = intent === "close" ? null : this.resumeKey;
    if (intent !== "close" && !resumeKey && !this.reportInput.value.trim()) { new Notice("请说明这项任务实际完成的范围。"); this.reportInput.focus(); return; }
    this.submit.disabled = this.closeTask.disabled = true; this.stop.hidden = false; this.lockForm(true);
    this.restartButton.hidden = true;
    const key = resumeKey || id();
    const request = intent === "close"
      ? { path: this.task.path, raw: this.task.raw, report: this.closeReason.value.trim() || "不再需要，关闭这项任务。", date: "", evidence: [], intent: "close" }
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
    for (const control of [this.reportInput, this.dateInput, this.addFile, this.todayButton, this.closeReason, ...Object.values(this.modeButtons), ...this.fileList.querySelectorAll("button")]) control.disabled = locked;
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
  plugin.openCompletion = async (task, report = "", mode = null) => {
    await plugin.openSteward();
    const view = plugin.app.workspace.getLeavesOfType(STEWARD_VIEW)[0]?.view;
    view?.openTask(task, report, mode);
  };
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

function taskPresentation(plugin, task) {
  const projected = plugin.steward.lastState?.curriculum?.weekly?.tasks?.find(t => t.path === task.path && t.raw === task.raw);
  return projected ? { ...task, ...projected } : task;
}

function progressForm(parent, plugin, task, job, saved) {
    const field = (name, placeholder) => {
      const label = parent.createEl("label", { cls: "sos-field", text: name });
      return label.createEl("textarea", { attr: { rows: "2", placeholder } });
    };
    const summary = field("这次实际做到哪里", "具体页码、题目、回答、产物或仍未理解的地方");
    const next = field("下次第一步", "打开哪个文件，从哪道题或哪个问题继续");
    next.value = task.record_next || "";
    const files = [], evidence = parent.createDiv({ cls: "sos-evidence-files" });
    const add = parent.createEl("button", { text: "＋ 关联产物（可选）" });
    const render = () => {
      evidence.empty(); files.forEach((path, index) => {
        const row = evidence.createDiv(); row.createSpan({ text: path.split("/").pop(), attr: { title: path } });
        row.createEl("button", { text: "移除", cls: "sos-quiet" }).addEventListener("click", () => { files.splice(index, 1); render(); });
      });
    };
    add.addEventListener("click", () => new EvidencePicker(plugin.app, path => { if (!files.includes(path) && files.length < 6) files.push(path); render(); }).open());
    parent.createEl("p", { cls: "sos-muted", text: "保存停点后可接着学；整项完成另行核验。" });
    const save = parent.createEl("button", { cls: "sos-primary", text: "保存这次进度" });
    const operationId = id();
    save.addEventListener("click", async () => {
      if (!summary.value.trim() || !next.value.trim()) { new Notice("请写实际范围和下次第一步。"); return; }
      save.disabled = true;
      const quote = `${summary.value.trim()}\n下次：${next.value.trim()}`;
      try {
        const receipt = await plugin.steward.act({ kind: "record_progress", task_id: task.id,
          job_key: job?.key || "", summary: summary.value.trim(), next_step: next.value.trim(), evidence: [...files] }, quote, operationId);
        new Notice(receipt.message); saved();
      } catch (error) { new Notice(error.message); save.disabled = false; }
    });
    return { summary, next };
}

class StewardView extends ItemView {
  constructor(leaf, plugin) { super(leaf); this.plugin = plugin; this.bridge = plugin.steward; this.draft = ""; }
  getViewType() { return STEWARD_VIEW; }
  getDisplayText() { return "Student OS · 我的学习"; }
  getIcon() { return "bot"; }
  getState() { return { draft: this.input?.value ?? this.draft, course: this.selectedCourse || null, week: this.selectedWeek || null,
    filter: this.taskFilter || "all", section: this.section || "overview", context: this.chatContext || null, manager: !!this.managerOpen,
    study: { units: { ...this.studyUnits }, routes: { ...this.studyRoutes }, open: this.board?.querySelector(".sos-self-study")
      ? [...this.board.querySelectorAll(".sos-learning-track[open]")].map(row => row.dataset.courseDetails) : this.studyOpen || [] } }; }
  async setState(state) {
    this.draft = state?.draft || ""; this.selectedCourse = state?.course || null; this.selectedWeek = state?.week || null;
    this.taskFilter = state?.filter || "all"; this.section = SECTIONS[state?.section] ? state.section : "overview";
    this.chatContext = state?.context || null;
    this.managerOpen = !!state?.manager;
    this.studyUnits = { ...state?.study?.units }; this.studyRoutes = { ...state?.study?.routes }; this.studyOpen = state?.study?.open || [];
    if (this.input) this.input.value = this.draft; this.render();
    this.board?.querySelectorAll(".sos-learning-track").forEach(row => { row.open = this.studyOpen.includes(row.dataset.courseDetails); });
    this.board?.querySelectorAll(".sos-study-allunits").forEach(row => { row.open = !!this.studyRoutes?.[row.closest(".sos-learning-track").dataset.courseDetails]; });
  }
  saveWorkspace() {
    this.app.workspace.requestSaveLayout();
    clearTimeout(this.workspaceSaveTimer);
    this.workspaceSaveTimer = setTimeout(() => { void this.saveWorkspaceNow().catch(error => new Notice("界面位置尚未保存：" + error.message)); }, 300);
  }
  async saveWorkspaceNow() {
    clearTimeout(this.workspaceSaveTimer);
    await this.plugin.saveStewardState?.(this.getState());
  }
  async onOpen() {
    this.contentEl.empty(); this.contentEl.addClass("sos-home");
    const header = this.contentEl.createDiv({ cls: "sos-header" });
    const heading = header.createDiv();
    heading.createEl("h1", { text: "我的学习" });
    this.termEl = heading.createEl("p", { cls: "sos-header-context" });
    const nav = header.createDiv({ cls: "sos-nav" });
    this.managerButton = this.button(nav, "询问 AI", () => { this.managerOpen = !this.managerOpen; this.saveWorkspace(); this.render(); if (this.managerOpen) this.input.focus(); });
    this.button(nav, "刷新", () => this.bridge.refresh(), "sos-quiet");
    this.statusEl = nav.createDiv({ cls: "sos-status", attr: { role: "status", "aria-live": "polite" } });
    const tabs = this.contentEl.createEl("nav", { cls: "sos-workspace-nav", attr: { "aria-label": "Student OS 工作区" } });
    for (const [key, label] of Object.entries(SECTIONS)) {
      const button = this.button(tabs, label, () => this.selectSection(key), "sos-section-tab");
      button.dataset.section = key;
    }
    this.taskDock = this.contentEl.createDiv({ cls: "sos-inline-task-slot" }); this.taskDock.hidden = true;
    const grid = this.contentEl.createDiv({ cls: "sos-grid" });
    this.board = grid.createDiv({ cls: "sos-board" });
    const side = this.side = grid.createDiv({ cls: "sos-side" });
    this.classroom = side.createDiv({ cls: "sos-classroom" });
    const chat = side.createDiv({ cls: "sos-chat" });
    const chatHeading = chat.createDiv({ cls: "sos-chat-heading" });
    setIcon(chatHeading.createSpan({ cls: "sos-icon" }), "bot");
    chatHeading.createEl("h2", { text: "学习助理" });
    chat.createEl("p", { cls: "sos-muted", text: "讨论学习安排，或说明需要调整的地方。" });
    const composer = chat.createDiv({ cls: "sos-composer" });
    this.contextEl = composer.createDiv({ cls: "sos-chat-context" });
    this.input = composer.createEl("textarea", { attr: { rows: "3", placeholder: "告诉管家你的安排、进度或变更…", "aria-label": "给 Student OS 管家的消息" } });
    this.input.value = this.draft;
    this.input.addEventListener("input", () => this.saveWorkspace());
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
  async onClose() { this.termsObserver?.disconnect(); this.bridge.listeners.delete(this.renderCallback); this.clearManagement(); await this.saveWorkspaceNow().catch(error => new Notice("界面位置尚未保存：" + error.message)); }
  button(parent, label, action, cls = "") {
    const button = parent.createEl("button", { text: label, cls });
    button.addEventListener("click", () => Promise.resolve().then(action).catch(error => new Notice(error.message)));
    return button;
  }
  openTask(task, report = "", mode = null, rowKey = null) {
    task = taskPresentation(this.plugin, task);
    if (this.taskDetail?.controller.task.id !== task.id) this.taskDetail?.controller.close();
    if (!this.taskDetail) {
      const host = this.taskDock.createDiv();
      const controller = new CompletionModal(this.plugin, task, report);
      this.taskDetail = { host, controller };
      controller.mount(host, () => {
        const row = [...this.board.querySelectorAll("[data-task-id]")].find(node => node.dataset.key === this.taskDetail?.rowKey);
        if (row) row.open = false;
        this.taskDetail = null; this.taskDock.hidden = true; this.render();
      });
    }
    if (rowKey) this.taskDetail.rowKey = rowKey;
    if (mode) this.taskDetail.controller.showMode(mode, false);
    this.placeTaskDetail(); this.taskDetail.host.scrollIntoView({ block: "nearest" });
  }
  placeTaskDetail() {
    if (!this.taskDetail) return;
    const matches = [...this.board.querySelectorAll("[data-task-id]")].filter(node => node.dataset.taskId === this.taskDetail.controller.task.id);
    const row = matches.find(node => node.dataset.key === this.taskDetail.rowKey) || matches[0];
    const target = row?.querySelector(".sos-task-detail") || this.taskDock;
    if (row) {
      this.taskDetail.rowKey = row.dataset.key;
      target.empty(); row.open = true;
      for (let parent = row.parentElement; parent; parent = parent.parentElement) if (parent.tagName === "DETAILS") parent.open = true;
    }
    target.appendChild(this.taskDetail.host); this.taskDock.hidden = target !== this.taskDock;
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
    this.saveWorkspace(); this.render();
    this.contentEl.scrollTop = 0;
  }
  clearManagement() {
    this.managementGeneration = (this.managementGeneration || 0) + 1;
    if (this.managementComponent) this.removeChild(this.managementComponent);
    this.managementComponent = null; this.managementSource = null;
  }
  renderDeadlines(data) {
    const groups = deadlineBoard(data), extras = deadlineSupplements(data.deadline_notes?.text);
    const incomplete = (data.source_issues || []).some(issue => issue.path === OBLIGATIONS || /deadline/.test(issue.code || ""));
    const page = this.board.createEl("section", { cls: "sos-deadline-board", attr: { "aria-label": "截止日期与待核对事项" } });
    const heading = page.createDiv({ cls: "sos-section-heading" });
    heading.createEl("h2", { text: "Deadlines" });
    heading.createSpan({ cls: "sos-muted", text: "伦敦当地时间" });
    if (incomplete) page.createEl("p", { cls: "sos-error", text: "部分截止记录尚未同步完整，以下数量可能不完整。" });
    const metrics = page.createDiv({ cls: "sos-deadline-metrics" });
    for (const [key, label] of [["upcoming", "未来 14 天"], ["later", "之后的安排"], ["past", "已过时间 · 待核对"]]) {
      const card = metrics.createDiv({ cls: `sos-deadline-metric is-${key}` });
      card.createEl("strong", { text: String(groups[key].length) }); card.createSpan({ text: label });
    }
    const section = (key, label, items, folded = false) => {
      const wrap = page.createEl(folded ? "details" : "section", { cls: "sos-deadline-section", attr: { "data-key": "deadlines-" + key } });
      const head = wrap.createEl(folded ? "summary" : "div", { cls: "sos-deadline-section-heading" });
      head.createEl("strong", { text: label }); head.createSpan({ text: `${items.length} 项` });
      const list = wrap.createDiv({ cls: "sos-deadline-list" });
      for (const item of items) this.renderDeadlineTask(list, item, key);
      return wrap;
    };
    section("upcoming", "接下来 14 天", groups.upcoming);
    if (!groups.upcoming.length) page.createEl("p", { cls: "sos-empty", text: incomplete ? "截止记录尚待同步，请稍后刷新。" : "未来 14 天内没有已登记且尚未关闭的截止或必到事项。" });
    if (extras.length || groups.undated.length) {
      const wrap = page.createEl("section", { cls: "sos-deadline-section" });
      const head = wrap.createDiv({ cls: "sos-deadline-section-heading" });
      head.createEl("strong", { text: "可选与待确认" }); head.createSpan({ text: `${extras.length + groups.undated.length} 项` });
      const list = wrap.createDiv({ cls: "sos-deadline-list" });
      for (const item of extras) {
        const row = list.createEl("details", { cls: "sos-deadline-note", attr: { "data-key": item.key } });
        const summary = row.createEl("summary");
        const date = summary.createDiv({ cls: "sos-deadline-date" });
        date.createEl("strong", { text: item.date }); date.createEl("span", { text: item.time });
        const title = summary.createDiv({ cls: "sos-deadline-description" });
        title.createEl("small", { text: item.kind }); title.createEl("strong", { text: item.title });
        const body = row.createDiv({ cls: "sos-deadline-note-body" }); body.createEl("p", { text: item.text });
        this.button(body, "问管家", () => this.discussTask({ ref: item.ref, title: item.title }), "sos-quiet");
      }
      for (const item of groups.undated) this.renderDeadlineTask(list, item, "undated");
    }
    if (groups.later.length) section("later", "之后的安排", groups.later, true);
    if (groups.past.length) section("past", "已过时间 · 完成情况待核对", groups.past, true);
    if (groups.done.length) section("done", "已完成", groups.done, true);
    if (data.deadline_notes?.available === false || data.deadline_notes?.truncated) page.createEl("p", { cls: "sos-error", text: "可选及待定节点的说明暂未完整读到，待同步。" });
  }
  renderDeadlineTask(parent, item, group) {
    const task = taskPresentation(this.plugin, item.task);
    const row = parent.createEl("details", { cls: `sos-deadline-task is-${group}`, attr: { "data-key": "deadline-" + task.id, "data-task-id": task.id } });
    const summary = row.createEl("summary");
    summary.addEventListener("click", event => {
      if (event.target.closest("button")) return;
      event.preventDefault();
      if (this.taskDetail?.controller.task.id === task.id) this.taskDetail.controller.close();
      else this.openTask(task, "", null, row.dataset.key);
    });
    const date = summary.createDiv({ cls: "sos-deadline-date" });
    date.createEl("strong", { text: item.task.date ? item.task.date.slice(5).replace("-", "/") : "待定" });
    if (item.task.date && !item.task.date.startsWith(londonClock().date.slice(0, 4))) date.createEl("small", { text: item.task.date.slice(0, 4) });
    date.createSpan({ text: item.time || "时刻待确认" });
    const description = summary.createDiv({ cls: "sos-deadline-description" });
    const tags = description.createDiv({ cls: "sos-deadline-tags" });
    tags.createSpan({ text: item.course }); tags.createSpan({ text: item.kind });
    description.createEl("strong", { text: item.title });
    if (item.detail) description.createEl("small", { text: item.detail });
    const state = summary.createDiv({ cls: "sos-deadline-state" });
    state.createSpan({ text: item.timing });
    if (group === "done") state.createSpan({ cls: "sos-recorded-evidence", text: "✓" });
    else this.button(state, "核验完成", () => this.openTask(task, "", "complete", row.dataset.key), "sos-quiet sos-deadline-check");
    row.createDiv({ cls: "sos-task-detail" });
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
    this.managerOpen = true;
    this.chatContext = { ref: task.ref || task.path, title: taskCaption(task) };
    this.render(); this.saveWorkspace();
    this.input.scrollIntoView({ block: "nearest" }); this.input.focus();
  }
  renderChatContext() {
    this.contextEl.empty(); this.contextEl.hidden = !this.chatContext;
    if (!this.chatContext) return;
    this.contextEl.createSpan({ text: "正在讨论：" + this.chatContext.title });
    this.button(this.contextEl, "移除", () => { this.chatContext = null; this.renderChatContext(); this.saveWorkspace(); }, "sos-quiet");
  }
  async send(body = null) {
    body = (body || this.input.value).trim();
    if (!body || this.bridge.busy) return;
    this.managerOpen = true;
    const draftBody = body, context = this.chatContext;
    if (context) body += `\n\n讨论的原任务：${context.title}\n原记录：[[${context.ref}]]`;
    const unfinished = this.bridge.lastState?.chat.find(r => r.body === body && !["done", "cancelled"].includes(r.phase));
    const key = unfinished?.key || id();
    const result = await this.bridge.run("chat", { key, body });
    if (result.phase === "done" && this.input.value.trim() === draftBody) {
      this.input.value = ""; this.draft = ""; this.saveWorkspace();
    }
    if (result.phase === "done" && this.chatContext === context) { this.chatContext = null; this.renderChatContext(); this.saveWorkspace(); }
  }
  async openClassroom(job) {
    let entry = this.openingClassroom;
    if (entry && entry.key !== job.key) throw new Error("正在打开另一间课堂，请稍后继续。");
    this.classroomEntry = { key: job.key, phase: "opening" };
    this.updateClassroomEntry();
    if (!entry) {
      entry = { key: job.key, work: Promise.resolve().then(() => this.loadClassroom(job)).then(message => {
        this.classroomEntry = { key: job.key, phase: "opened", message };
        this.updateClassroomEntry();
        return message;
      }) };
      this.openingClassroom = entry;
      entry.work.finally(() => { if (this.openingClassroom === entry) this.openingClassroom = null; }).catch(() => {});
    }
    let timer;
    try {
      await Promise.race([entry.work, new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("课堂仍在加载，原对话已保留。可以点击重新进入。")), 15000); })]);
    } catch (error) {
      this.classroomEntry = { key: job.key, phase: "error", message: error.message };
      this.updateClassroomEntry();
      throw error;
    } finally { clearTimeout(timer); }
  }
  updateClassroomEntry() {
    const button = this.classroom?.querySelector(".sos-enter"), feedback = this.classroom?.querySelector(".sos-entry-feedback");
    if (!button || !feedback) return;
    const entry = this.classroomEntry?.key === this.classroom.dataset.jobKey ? this.classroomEntry : null;
    button.disabled = entry?.phase === "opening";
    button.setAttribute("aria-busy", String(button.disabled));
    if (entry) button.setText(entry.phase === "opening" ? "正在打开课堂…" : entry.phase === "opened" ? "回到课堂" : "重新进入课堂");
    feedback.hidden = !entry || entry.phase === "opening";
    feedback.dataset.phase = entry?.phase || "";
    feedback.setText(entry?.message || "");
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
    if (tab?.state.currentConversationId !== conversation.id || !tab.state.messages?.length) throw new Error("课堂打开尚未确认；原会话保持不变。");
    let material;
    try { material = await this.openClassroomMaterial(job, leaf); }
    catch (error) { material = { message: `讲义打开失败：${error.message}` }; }
    this.app.workspace.setActiveLeaf(leaf, { focus: true });
    tab.renderer?.scrollToBottom();
    leaf.view.focusActiveInput();
    this.plugin?.addStewardReturn?.(leaf);
    const place = leaf.getRoot() === this.app.workspace.rightSplit ? "右侧" : "旁边";
    if (material.message) new Notice(`课堂已打开。${material.message}`, 8000);
    return material.file && !material.message ? `讲义已打开，${place}是对应课堂，可以开始学习。` : `课堂已打开。${material.message}`;
  }
  async openClassroomMaterial(job, classroom) {
    const material = classroomMaterial(this.app, job, this.bridge?.lastState?.curriculum?.weekly?.tasks);
    if (!material.file) return material;
    const workspace = this.app.workspace;
    const existing = workspace.getLeavesOfType(material.file.extension === "pdf" ? "pdf" : "markdown")
      .find(l => l.getRoot() === workspace.rootSplit && (l.view.file?.path || l.getViewState?.().state?.file) === material.file.path
        && (classroom.getRoot() !== workspace.rootSplit || l.parent !== classroom.parent));
    const leaf = existing || workspace.getLeaf(classroom.getRoot() === workspace.rootSplit ? "split" : "tab", "vertical");
    // Revealing an existing leaf preserves its PDF page, scroll position and unsaved note edits.
    if (!existing) await leaf.openFile(material.file, { active: false });
    await workspace.revealLeaf(leaf);
    this.plugin?.addStewardReturn?.(leaf);
    return material;
  }
  render() {
    if (!this.board) return;
    if (this.taskDetail) this.taskDock.appendChild(this.taskDetail.host);
    this.statusEl.setText(this.bridge.status === "记录已同步" ? "自动同步" : this.bridge.status);
    this.statusEl.setAttribute("title", "记录变化时自动刷新；每分钟检查换日和更新。读取与校验不调用大模型。" + (this.bridge.lastState?.at ? ` 上次读取：${this.bridge.lastState.at}` : ""));
    this.sendButton.disabled = this.bridge.busy;
    this.stopButton.hidden = !this.bridge.activeRequest && !this.bridge.activeTask;
    const data = this.bridge.lastState;
    if (!data) { this.board.setText("正在读取已有的 Student OS 记录…"); return; }
    const home = homeModel(data), agenda = agendaFor(data);
    const section = this.section || "overview";
    this.side.hidden = !this.managerOpen;
    this.side.parentElement.classList.toggle("has-steward", !!this.managerOpen);
    this.managerButton.setText(this.managerOpen ? "收起助理" : "询问 AI");
    this.managerButton.setAttribute("aria-expanded", String(!!this.managerOpen));
    this.contentEl.querySelectorAll(".sos-section-tab").forEach(button => button.setAttribute("aria-current", button.dataset.section === section ? "page" : "false"));
    this.termEl.setText(`${data.curriculum?.stage || "课程与学习位置"}  ·  ${agenda.date.slice(5).replace("-", "月")}日`);
    const expanded = new Map(Array.from(this.contentEl.querySelectorAll("details[data-key]")).map(d => [d.dataset.key, d.open]));
    if (section === "overview") {
      this.clearManagement(); this.board.empty();
      this.renderDueToday(data);
      this.board.appendChild(this.classroom);
      this.renderCurriculum(data, home);
      this.renderDetails(data);
    } else if (section === "deadlines") {
      this.clearManagement(); this.board.empty(); this.renderDeadlines(data);
    } else {
      if (this.renderedSection !== section) { this.clearManagement(); this.board.empty(); }
      void this.renderManagement(data);
    }
    this.renderedSection = section;
    if (section !== "overview") this.side.prepend(this.classroom);
    this.classroom.classList.toggle("is-launch", section === "overview");
    this.renderClassroom(data, home, agenda);
    this.renderRadar(data, home);
    this.classroom.hidden = section !== "overview";
    this.radar.hidden = section === "deadlines";
    this.contentEl.querySelectorAll("details[data-key]").forEach(d => { if (expanded.has(d.dataset.key)) d.open = expanded.get(d.dataset.key); });
    this.placeTaskDetail();
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
    const upcoming = deadlineNotices(data).filter(t => t.task);
    const today = upcoming.filter(t => t.days === 0), items = today.length ? today : upcoming.slice(0, 1);
    if (!items.length) return;
    const banner = this.board.createEl("section", { cls: `sos-due-today${today.length ? "" : " is-upcoming"}`, attr: { "aria-label": today.length ? "今天到期的未关闭事项" : "最近的未关闭事项" } });
    banner.createEl("strong", { text: today.length ? "今天到期" : "最近事项" });
    for (const item of items) {
      const button = this.button(banner, "", () => this.selectSection("deadlines"), "sos-due-item");
      button.createSpan({ cls: "sos-due-time", text: today.length ? deadlineTime(item.task) || "今天" : item.time });
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
    this.classroom.dataset.jobKey = home.job?.key || "";
    const ready = home.job?.state === "registered";
    const head = this.classroom.createDiv({ cls: "sos-classroom-heading" });
    setIcon(head.createSpan({ cls: "sos-icon" }), "book-open");
    head.createEl("h2", { text: "接着学" });
    if (!home.task) {
      this.classroom.createEl("p", { text: "选择下面的一门课，就能从已有进度接着学。" });
      return;
    }
    const context = learningContext(data, home.task), task = context.task;
    head.createSpan({ cls: "sos-classroom-unit", text: [task.course_id, /^\d+$/.test(task.week || "") ? `Week ${task.week}` : ""].filter(Boolean).join(" · ") });
    const main = this.classroom.createDiv({ cls: "sos-launch-main" });
    const body = main.createDiv();
    body.createEl("h3", { text: taskCaption(task) });
    body.createEl("p", { cls: "sos-launch-position", text: `${context.stopLabel}：${context.stop}`, attr: { title: context.stop } });
    const enter = this.button(main, ready ? "进入课堂" : "开始学习", () => ready ? this.openClassroom(home.job) : this.beginTask(home.task), "mod-cta sos-enter");
    enter.setAttribute("title", "一起打开本课讲义和学习对话");
    this.classroom.createDiv({ cls: "sos-entry-feedback", attr: { role: "status", "aria-live": "polite", "aria-atomic": "true" } });
    this.updateClassroomEntry();
    const next = this.classroom.createDiv({ cls: "sos-launch-next" });
    next.createEl("small", { text: "下一步" }); next.createEl("span", { text: context.next });
    const more = this.classroom.createEl("details", { cls: "sos-current-more", attr: { "data-key": "classroom-details" } });
    more.open = !!expanded;
    more.createEl("summary", { text: "进度与收工条件" });
    const handoff = more.createDiv({ cls: "sos-launch-handoff" });
    handoff.createEl("small", { text: "做到这里就可以收工" });
    handoff.createEl("p", { text: context.condition });
    if (task.checkpoints?.length) {
      const steps = more.createEl("ol"); task.checkpoints.forEach(point => steps.createEl("li", { text: point.label }));
    }
    const slot = agenda.slots.find(s => s.ref?.split("#^")[1] && home.task.ref?.endsWith("#^" + s.ref.split("#^")[1]));
    if (slot) more.createDiv({ cls: "sos-classroom-time", text: `今日安排 ${slot.start}–${slot.end}${slot.end <= londonClock().time ? " · 时段已过，完成情况以实际记录为准" : ""}` });
    const actions = more.createDiv({ cls: "sos-classroom-actions" });
    this.button(actions, "记录进度", () => this.openTask(home.task, "", "progress"), "sos-quiet");
    this.button(actions, "核验完成", () => this.plugin.openCompletion(home.task), "sos-quiet");
    if (home.job) this.button(actions, "取消这次派工", () => this.bridge.cancelClassroom(home.job), "sos-quiet");
    if (home.job?.error) this.classroom.createEl("p", { cls: "sos-error", text: "课堂暂未接通，点击开始学习继续恢复。" });
  }
  renderCurriculum(data, home) {
    const curriculum = data.curriculum;
    const section = this.board.createEl("section", { cls: "sos-curriculum" });
    const heading = section.createDiv({ cls: "sos-section-heading" });
    heading.createEl("h2", { text: "学习全景" });
    if (!curriculum?.courses.length) { section.createEl("p", { text: "课程记录暂未读到，刷新后再查看。" }); return; }
    const weekly = curriculum.weekly;
    if (!weekly) { section.createEl("p", { text: "正在整理原计划的每周任务…" }); return; }
    const overall = heading.createDiv({ cls: "sos-progress-totals" });
    this.button(overall, `✓ 已完成 ${weekly.counts.done} 项`, () => this.selectTasks(null, "all", "done"), "sos-total-done");
    this.button(overall, `◐ 有学习记录 ${weekly.counts.recorded} 项`, () => this.selectTasks(null, "all", "recorded"), "sos-quiet");
    section.createDiv({ cls: "sos-course-group-label", text: "学校课程 · 跟随学期推进" });
    const cards = section.createDiv({ cls: "sos-progress-cards" });
    for (const course of curriculum.courses.filter(c => c.group === "degree")) {
      const count = course.core_progress || course.task_progress, personal = positionFor(course.personal);
      const card = this.button(cards, "", () => this.selectTasks(this.selectedCourse === course.id ? null : course.id, this.selectedWeek, "all"), "sos-progress-card");
      card.dataset.course = course.id;
      card.setAttribute("aria-pressed", String(this.selectedCourse === course.id));
      const top = card.createDiv({ cls: "sos-progress-card-top" });
      top.createSpan({ text: course.id });
      top.createSpan({ text: `本周 ${weekly.tasks.filter(t => t.course_id === course.id && t.week === String(curriculum.week) && t.state !== "cancelled").length} 项` });
      card.createEl("h3", { text: course.title });
      const number = card.createDiv({ cls: "sos-progress-number" });
      number.createEl("strong", { text: String(count.done) });
      number.createSpan({ text: ` / ${count.total} 项核心已记录完成` });
      const bar = card.createDiv({ cls: "sos-completion-bar", attr: { role: "img", "aria-label": `原计划 ${count.total} 项：已完成 ${count.done}，有学习记录 ${count.recorded}，待确认 ${count.unknown}，后续计划 ${count.planned}` } });
      for (const key of ["done", "recorded", "unknown", "planned"]) {
        if (count[key]) bar.createSpan({ cls: `is-${key}`, attr: { style: `flex:${count[key]}` } });
      }
      const states = card.createDiv({ cls: "sos-progress-states" });
      if (count.recorded) states.createSpan({ cls: "is-recorded", text: `${count.recorded} 项有记录` });
      if (count.unknown) states.createSpan({ text: `${count.unknown} 项待确认` });
      if (count.planned) states.createSpan({ text: `${count.planned} 项后续计划` });
      if (course.recommended_progress?.total) states.createSpan({ text: `另有 ${course.recommended_progress.total} 项推荐` });
      const positions = card.createDiv({ cls: "sos-course-positions" });
      for (const [key, label] of [["school", "学校"], ["personal", "我学到"]]) {
        const fact = positionFor(course[key]);
        const line = positions.createDiv();
        const stage = key === "school" && fact.label !== "学习记录" ? " · " + fact.label.split(" · ")[0] : "";
        line.createEl("small", { text: label + stage + (fact.outdated ? " · 上次定位" : "") });
        line.createEl("span", { text: taskCaption({ title: fact.topic.replace(/^最近完成：/, "") }), attr: { title: fact.topic } });
      }
    }
    this.renderSelfStudy(section, curriculum);
    this.renderWeekly(section, curriculum);
    const queued = curriculum.courses.filter(c => c.group === "future" || ["foundation", "other", "queued"].includes(c.group) && !c.active);
    if (queued.length) {
      const later = section.createEl("details", { cls: "sos-details", attr: { "data-key": "queued-courses" } });
      later.createEl("summary", { text: `后续课程 · ${queued.length} 门` });
      for (const c of queued) later.createDiv({ cls: "sos-course-meta", text: `${c.id} · ${c.title}${c.group === "future" ? " · 冬季学期" : ""}` });
    }
    if (curriculum.issues.length) section.createEl("p", { cls: "sos-error", text: curriculum.cached_at
      ? `进度暂未同步，正在显示 ${curriculum.cached_at.slice(5, 10).replace("-", "/")} 的记录。`
      : "部分进度暂未同步。" });
  }
  renderSelfStudy(section, curriculum) {
    const courses = curriculum.courses.filter(c => ["foundation", "other", "queued"].includes(c.group) && c.active);
    if (!courses.length) return;
    const details = section.createEl("section", { cls: "sos-self-study" });
    const heading = details.createDiv({ cls: "sos-section-heading" });
    heading.createEl("h2", { text: "自主学习" });
    heading.createSpan({ cls: "sos-muted", text: "按学习单元推进" });
    const cards = details.createDiv({ cls: "sos-learning-tracks" });
    for (const c of [...courses.filter(c => c.mode === "finite"), ...courses.filter(c => c.mode !== "finite")]) {
      const model = selfStudyFor(c, curriculum.weekly);
      const row = cards.createEl("details", { cls: `sos-learning-track${model.finite ? "" : " is-practice"}${c.replay_progress ? " is-replay" : ""}`, attr: { "data-key": "course-" + c.id, "data-course-details": c.id } });
      row.open = (this.studyOpen || []).includes(c.id);
      const summary = row.createEl("summary");
      summary.addEventListener("click", event => {
        event.preventDefault(); row.open = !row.open;
        this.studyOpen = [...cards.querySelectorAll(".sos-learning-track[open]")].map(node => node.dataset.courseDetails);
        this.saveWorkspace();
      });
      const top = summary.createDiv({ cls: "sos-track-top" });
      top.createSpan({ cls: "sos-track-code", text: c.id });
      top.createSpan({ cls: `sos-track-badge${model.recorded ? " has-record" : ""}`, text: c.replay_progress ? "本轮重看" : model.recorded ? "已有学习记录" : model.finite ? "按单元接续" : "持续实践" });
      summary.createEl("h3", { text: c.title });
      const position = summary.createDiv({ cls: "sos-track-position" });
      position.createEl("small", { text: "我学到" });
      position.createEl("p", { text: model.position });
      if (model.finite) {
        const units = c.replay_progress ? model.replay : model.units;
        const done = c.replay_progress ? c.replay_progress.completed : c.unit_progress.completed;
        const grain = units.every(t => studyUnitFor(t).lecture) ? "讲" : "组任务";
        const progress = summary.createDiv({ cls: "sos-track-progress" });
        const count = progress.createDiv({ cls: "sos-track-count" });
        count.createEl("strong", { text: String(done) });
        count.createSpan({ text: c.replay_progress ? " 项本轮已完成" : ` / ${units.length} ${grain}已记录完成` });
        if (c.replay_progress) summary.createEl("p", { cls: "sos-study-map-hint", text: `历史课程记录：${c.unit_progress.completed} / ${c.unit_progress.total} 讲已完成` });
        else {
          const bar = summary.createDiv({ cls: "sos-track-segments", attr: { role: "img", "aria-label": `${c.id}：已登记 ${units.length} 项，完成 ${done} 项` } });
          for (const unit of units) bar.createSpan({ cls: `is-${unit.state}`, attr: { title: taskCaption(unit), "aria-hidden": "true" } });
        }
      }
      const next = summary.createDiv({ cls: "sos-track-next" });
      next.createEl("small", { text: "接下来" }); next.createEl("span", { text: model.next });
      const footer = summary.createDiv({ cls: "sos-track-footer" });
      footer.createSpan({ cls: "sos-expand-label", text: "展开单元与收工条件" });
      setIcon(footer.createSpan({ cls: "sos-track-chevron", attr: { "aria-hidden": "true" } }), "chevron-down");
      this.renderCourseDetail(row.createDiv({ cls: "sos-inline-course" }), c, model);
    }
  }
  selectTasks(course, week, filter = "all") {
    this.selectedCourse = course; this.selectedWeek = week; this.taskFilter = filter;
    this.saveWorkspace(); this.render();
    this.board.querySelector(".sos-weekly")?.scrollIntoView({ block: "start", behavior: "smooth" });
  }
  renderWeekly(parent, curriculum) {
    const plan = curriculum.weekly;
    const key = this.selectedWeek || String(curriculum.week || "unassigned");
    const course = this.selectedCourse, filter = this.taskFilter || "all";
    const panel = parent.createEl("section", { cls: "sos-weekly", attr: { "aria-label": "每周具体任务与完成记录" } });
    const heading = panel.createDiv({ cls: "sos-section-heading" });
    heading.createEl("h2", { text: "每周学习与收尾" });
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
    courseSelect.createEl("option", { value: "", text: /^\d+$/.test(key) ? "本学期课程" : "所有学习课程" });
    for (const c of curriculum.courses.filter(c => c.task_progress?.total)) courseSelect.createEl("option", { value: c.id, text: `${c.id} ${c.title}` });
    courseSelect.value = course || "";
    courseSelect.addEventListener("change", () => this.selectTasks(courseSelect.value || null, this.selectedWeek, filter));
    for (const [value, label] of [["all", "全部状态"], ["verification", "待核验"], ["done", "已完成"], ["recorded", "有学习记录"], ["unknown", "待确认"]]) {
      const button = this.button(filters, label, () => this.selectTasks(course, this.selectedWeek, value), "sos-filter-button");
      button.setAttribute("aria-pressed", String(filter === value));
    }
    const extra = panel.createEl("details", { cls: "sos-week-record-tools", attr: { "data-key": "week-record-tools" } });
    extra.createEl("summary", { text: "其他周次与历史" });
    const more = extra.createDiv({ cls: "sos-week-extra" });
    for (const [value, label] of [["all", "全部周次"], ["unassigned", "未分周"], ["outside", "其他日期"]]) {
      const button = this.button(more, label, () => this.selectTasks(course, value, filter), "sos-quiet");
      button.setAttribute("aria-pressed", String(key === value));
    }
    const included = id => course ? id === course : !/^\d+$/.test(key) || curriculum.courses.some(c => c.id === id && c.group === "degree");
    const inWeek = plan.tasks.filter(t => included(t.course_id) && (key === "all" || t.week === key) && t.state !== "cancelled");
    const tasks = inWeek.filter(t => filter === "all" || (filter === "verification" ? t.completion_check && !["complete", "closed", "cancelled"].includes(t.completion_check.phase) && t.state !== "done" : t.state === filter));
    const selected = plan.weeks.find(w => w.key === key);
    const caption = panel.createDiv({ cls: "sos-week-caption" });
    caption.createEl("strong", { text: selected ? `第 ${selected.number} 周 · ${selected.start.slice(5).replace("-", "/")}–${selected.end.slice(5).replace("-", "/")}` : key === "all" ? "全部周次" : key === "unassigned" ? "未分周的任务" : "其他日期的记录" });
    const core = inWeek.filter(t => t.group !== "recommended"), recommended = inWeek.filter(t => t.group === "recommended");
    caption.createSpan({ text: `核心 ${core.filter(t => t.state === "done").length}/${core.length} · 推荐 ${recommended.filter(t => t.state === "done").length}/${recommended.length}${filter !== "all" ? ` · 筛选显示 ${tasks.length} 项` : ""}` });
    const bundles = (plan.bundles || []).filter(b => included(b.course_id) && (key === "all" || b.week === key));
    if (!tasks.length && filter !== "all") panel.createEl("p", { cls: "sos-empty", text: "这个筛选下没有任务记录。收尾状态仍按本周全部核心事项计算。" });
    if (!bundles.length) panel.createEl("p", { cls: "sos-empty", text: "尚未读到这一周的学习要求；没有任务记录不代表已经完成。" });
    for (const entry of bundles) {
      const c = curriculum.courses.find(c => c.id === entry.course_id);
      const own = tasks.filter(t => t.course_id === c.id && t.week === entry.week);
      if (filter !== "all" && !own.length) continue;
      const group = panel.createDiv({ cls: "sos-week-course", attr: { "data-course": c.id, "data-week": entry.week } });
      const title = group.createDiv({ cls: "sos-week-course-heading" });
      title.createEl("h3", { text: `${c.id} · ${c.title}${key === "all" ? ` · ${entry.week.match(/^\d+$/) ? "Week " + entry.week : entry.week === "unassigned" ? "未分周" : "其他日期"}` : ""}` });
      if (course) {
        const position = group.createDiv({ cls: "sos-week-position" });
        for (const [key, label] of [["school", "学校最近记录"], ["personal", "我的最近停点"]]) {
          const fact = positionFor(c[key]);
          const cell = position.createDiv(); cell.createEl("small", { text: label + (fact.outdated ? " · 待更新" : "") });
          cell.createEl("p", { text: fact.topic });
        }
      }
      const closure = group.createDiv({ cls: "sos-week-closure" + (entry.ready ? " is-ready" : "") });
      const outcome = closure.createDiv();
      outcome.createEl("strong", { text: entry.ready ? "本周核心已收工" : entry.core.total ? `${entry.core.done} / ${entry.core.total} 项核心已完成` : "本周核心要求待整理" });
      outcome.createDiv({ cls: "sos-course-meta", text: entry.academic !== false && !entry.scope_verified ? "本周要求仍待补齐，暂不能确认收工" : entry.ready ? "推荐内容按需继续" : "收工条件：下方核心事项全部完成" });
      if (entry.core.total) closure.createEl("progress", { attr: { value: String(entry.core.done), max: String(entry.core.total), "aria-label": `${c.id} 本周已列核心完成 ${entry.core.done}/${entry.core.total}` } });
      const next = plan.tasks.find(t => t.id === entry.next_task_id);
      if (next) {
        const line = group.createDiv({ cls: "sos-week-next" });
        line.createSpan({ text: "下一步" });
        this.button(line, taskCaption(next), () => this.openTask(next), "sos-quiet");
      }
      const lanes = group.createDiv({ cls: "sos-week-lanes" });
      for (const section of entry.groups) {
        const selectedTasks = own.filter(t => section.task_ids.includes(t.id));
        if (filter !== "all" && !selectedTasks.length) continue;
        const optional = section.key === "recommended";
        const lane = lanes.createEl(optional ? "details" : "section", { cls: `sos-week-lane is-${section.key}`, attr: { "data-key": `lane-${c.id}-${entry.week}-${section.key}` } });
        if (optional && filter !== "all") lane.open = true;
        const laneHeading = lane.createEl(optional ? "summary" : "div", { cls: "sos-lane-heading" });
        laneHeading.createEl("strong", { text: section.label });
        laneHeading.createSpan({ text: section.missing ? "待整理" : `${section.counts.done}/${section.counts.total} 已完成` });
        if (section.missing) lane.createEl("p", { cls: "sos-empty", text: "本周任务待管家整理。" });
        for (const task of selectedTasks) this.renderWeeklyTask(lane, task, c);
      }
      if (entry.scope?.recommendations?.length) {
        let resources = lanes.querySelector(".is-recommended");
        if (!resources) {
          resources = lanes.createEl("details", { cls: "sos-week-lane is-recommended", attr: { "data-key": `resources-${c.id}-${entry.week}` } });
          resources.createEl("summary", { cls: "sos-lane-heading", text: "推荐内容" });
        }
        for (const resource of entry.scope.recommendations) this.button(resources, resource.label + " ↗", () => this.openOriginal(resource.path), "sos-quiet");
      }
    }
  }
  renderWeeklyTask(group, task, c, prefix = "task") {
    const states = { done: ["✓", "已完成"], recorded: ["◐", "有学习记录"], unknown: ["○", "待确认"], planned: ["·", "后续计划"] };
        const row = group.createEl("details", { cls: `sos-task-row ${prefix === "unit" ? "sos-unit-task" : "sos-week-task"} is-${task.state}`, attr: { "data-key": prefix + "-" + task.id, "data-task-id": task.id } });
        const summary = row.createEl("summary");
        const check = task.completion_check;
        summary.addEventListener("click", event => {
          if (event.target.closest("button")) return;
          event.preventDefault();
          if (this.taskDetail?.controller.task.id === task.id) { row.open = false; this.taskDetail.controller.close(); }
          else this.openTask(task, "", null, row.dataset.key);
        });
        if (task.state === "done") summary.createSpan({ cls: "sos-task-indicator", text: "✓", attr: { "aria-hidden": "true" } });
        else {
          const control = summary.createEl("button", { cls: "sos-task-indicator sos-check-task", text: "○", attr: { "aria-label": "核验完成：" + taskCaption(task), title: "核验完成" } });
          control.addEventListener("click", event => { event.preventDefault(); event.stopPropagation(); this.openTask(task, "", null, row.dataset.key); });
        }
        const text = summary.createDiv({ cls: "sos-task-caption" });
        text.createSpan({ text: taskCaption(task) });
        const meta = task.kind === "obligation" ? [task.date, deadlineTime(task)].filter(Boolean).join(" · ") : "";
        if (meta) text.createEl("small", { text: meta });
        summary.createSpan({ cls: "sos-task-state", text: task.state !== "done" && check && !["complete", "closed", "cancelled"].includes(check.phase) ? "待核验" : states[task.state][1] });
        row.createDiv({ cls: "sos-week-task-detail sos-task-detail" });
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
        if (t.task) return this.openTask(t.task);
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
    const target = [...this.contentEl.querySelectorAll("[data-course-details]")].find(node => node.dataset.courseDetails === course.id);
    if (target) {
      target.open = !target.open;
      for (let parent = target.parentElement; parent; parent = parent.parentElement) if (parent.tagName === "DETAILS") parent.open = true;
      target.scrollIntoView({ block: "nearest" });
    } else this.selectTasks(course.id, "all");
  }
  selectStudyUnit(course, task, focus = false) {
    this.studyUnits = { ...this.studyUnits, [course.id]: task.id };
    const card = [...this.board.querySelectorAll(".sos-learning-track")].find(node => node.dataset.courseDetails === course.id);
    if (!card) return;
    const content = card.querySelector(".sos-inline-course");
    if (this.taskDetail && content.contains(this.taskDetail.host)) this.taskDock.appendChild(this.taskDetail.host);
    content.empty();
    this.renderCourseDetail(content, course, selfStudyFor(course, this.bridge.lastState.curriculum.weekly));
    this.placeTaskDetail();
    if (focus) [...content.querySelectorAll("[data-unit-id]")].find(button => button.dataset.unitId === task.id)?.focus({ preventScroll: true });
    this.saveWorkspace();
  }
  renderCourseDetail(content, course, model) {
    const all = [...model.replay, ...model.units];
    const selected = all.find(t => t.id === this.studyUnits?.[course.id]) || model.defaultTask;
    if (!selected) { content.createEl("p", { cls: "sos-muted", text: "学习路线尚待安排。" }); return; }
    const isReplay = model.replay.includes(selected), units = isReplay ? model.replay : model.units;
    const toolbar = content.createDiv({ cls: "sos-study-toolbar" });
    const tabs = toolbar.createDiv({ cls: "sos-study-track-tabs", attr: { "aria-label": "选择学习记录" } });
    if (model.replay.length) {
      for (const [label, tasks, active] of [["本轮重看", model.replay, isReplay], ["历史课程清单", model.units, !isReplay]]) {
        if (!tasks.length) continue;
        const tab = this.button(tabs, label, () => this.selectStudyUnit(course, tasks.find(t => t.state === "recorded") || tasks[0]), "sos-study-track-tab");
        tab.setAttribute("aria-pressed", String(active));
      }
    } else tabs.createEl("strong", { text: model.finite ? "学习路线" : "当前实践" });
    this.button(toolbar, model.nextTask ? "回到接续点" : "回到本轮记录", () => this.selectStudyUnit(course, model.defaultTask), "sos-quiet sos-study-return");
    const workspace = content.createDiv({ cls: "sos-study-workspace" });
    const inspector = workspace.createDiv({ cls: "sos-study-inspector", attr: { "data-task-id": selected.id, "data-key": "study-" + selected.id } });
    this.renderStudyUnit(inspector, course, selected, units.indexOf(selected), model);
    inspector.createDiv({ cls: "sos-task-detail" });
    if (units.length > 1) {
      const navigation = workspace.createDiv({ cls: "sos-study-navigation" });
      const nearby = navigation.createDiv({ cls: "sos-route-peek", attr: { "aria-label": "当前单元附近的学习路线" } });
      for (const task of nearbyUnits(units, selected)) this.renderUnitChoice(nearby, course, task, units, selected, model, true);
      const route = navigation.createEl("details", { cls: "sos-study-allunits", attr: { "data-key": "route-" + course.id } });
      route.open = !!this.studyRoutes?.[course.id];
      route.createEl("summary", { text: `${isReplay ? "本轮全部记录" : "完整路线与历史"} · ${units.length} 项 · ${units.filter(t => t.state === "done").length} 项已完成` });
      route.addEventListener("toggle", () => {
        if (!route.isConnected) return;
        this.studyRoutes = { ...this.studyRoutes, [course.id]: route.open };
        this.saveWorkspace();
      });
      const map = route.createDiv({ cls: "sos-unit-map is-list", attr: { "aria-label": course.id + (isReplay ? "本轮重看单元" : "学习单元") } });
      units.forEach(task => this.renderUnitChoice(map, course, task, units, selected, model, false));
    }
    if (model.replay.length) content.createEl("p", { cls: "sos-study-map-hint", text: isReplay
      ? `历史课程清单另存：${model.units.filter(t => t.state === "done").length}/${model.units.length} 讲已记录完成。`
      : "这里是历史课程记录。本轮重看在另一个标签中。" });

    // Browsing other units must not detach or overwrite an unfinished progress/completion form.
    const editing = all.find(t => t.id === this.taskDetail?.controller.task.id);
    if (editing && editing.id !== selected.id) {
      const parked = content.createDiv({ cls: "sos-study-editing", attr: { "data-task-id": editing.id, "data-key": "study-" + editing.id } });
      const heading = parked.createDiv({ cls: "sos-study-editing-heading" });
      heading.createSpan({ text: "正在填写 · " + studyUnitFor(editing, all.indexOf(editing)).label });
      this.button(heading, "回到这一项", () => this.selectStudyUnit(course, editing), "sos-quiet");
      parked.createDiv({ cls: "sos-task-detail" });
    }
  }
  renderUnitChoice(parent, course, task, units, selected, model, nearby) {
    const index = units.indexOf(task), unit = studyUnitFor(task, index), current = task.id === model.nextTask?.id;
    const button = this.button(parent, "", () => this.selectStudyUnit(course, task, true), `${nearby ? "sos-route-peek-item" : "sos-unit-node"} is-${task.state}${current ? " is-next" : ""}`);
    button.dataset.unitId = task.id;
    button.setAttribute("aria-pressed", String(task.id === selected.id));
    button.setAttribute("aria-label", `${unit.label} · ${unit.title} · ${unit.state}${current ? " · 接续点" : ""}`);
    if (nearby) button.createEl("small", { text: task.id === selected.id ? "当前查看" : index < units.indexOf(selected) ? "上一项" : "下一项" });
    else button.createEl("strong", { cls: "sos-unit-order", text: unit.number });
    const text = button.createDiv(); text.createEl("small", { text: unit.label }); text.createEl("span", { text: unit.title });
    button.createEl("small", { cls: "sos-unit-mark", text: current ? "接续点 · " + unit.state : unit.state });
    button.addEventListener("keydown", event => {
      let next;
      if (["ArrowLeft", "ArrowUp"].includes(event.key)) next = Math.max(0, index - 1);
      if (["ArrowRight", "ArrowDown"].includes(event.key)) next = Math.min(units.length - 1, index + 1);
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = units.length - 1;
      if (next !== undefined) { event.preventDefault(); this.selectStudyUnit(course, units[next], true); }
    });
  }
  renderStudyUnit(parent, course, task, index, model) {
    const unit = studyUnitFor(task, index), replay = model.replay.includes(task);
    const preview = parent.createDiv({ cls: "sos-unit-preview" });
    const top = preview.createDiv({ cls: "sos-study-unit-top" });
    top.createSpan({ text: unit.label });
    top.createSpan({ cls: `sos-study-unit-state is-${task.state}`, text: unit.symbol + " " + unit.state });
    preview.createEl("h3", { text: unit.title });
    if (task.record) {
      const stop = preview.createDiv({ cls: "sos-study-stop" });
      stop.createEl("small", { text: "你的学习停点" }); stop.createEl("p", { text: task.record });
      if (task.record_next) stop.createEl("p", { text: "下次从这里开始：" + task.record_next });
    } else if (task.id === model.defaultTask?.id && task.state !== "done") {
      const stop = preview.createDiv({ cls: "sos-study-stop" });
      stop.createEl("small", { text: "这门课的最近记录" }); stop.createEl("p", { text: model.position });
    }
    if (task.state === "done") {
      const date = task.raw.match(/✅\s*(\d{4}-\d{2}-\d{2})/)?.[1];
      const done = preview.createDiv({ cls: "sos-study-complete" });
      setIcon(done.createSpan({ cls: "sos-icon" }), "circle-check");
      const body = done.createDiv(); body.createEl("strong", { text: replay ? "这次重看已完成" : "这项已有完成记录" });
      body.createSpan({ text: [date, replay ? "本轮记录已保存" : "保留原计划中的完成状态"].filter(Boolean).join(" · ") });
    }
    const scope = preview.createEl(task.state === "done" ? "details" : "div", { cls: "sos-study-scope" });
    scope.createEl(task.state === "done" ? "summary" : "small", { text: task.state === "done" ? "本项完成要求" : unit.hasCondition ? "做到这里可以收工" : "收工条件待确认" });
    if (task.checkpoints?.length) {
      const steps = scope.createEl("ol");
      for (const point of task.checkpoints) { const li = steps.createEl("li"); li.createEl("strong", { text: point.label }); li.createEl("p", { text: point.condition }); }
    }
    scope.createEl("p", { text: unit.hasCondition ? unit.condition : "原计划只列了本讲主题。准备课堂时需确认具体练习，以及要保留的学习成果。" });
    const actions = preview.createDiv({ cls: "sos-study-unit-actions" });
    if (this.taskDetail && this.taskDetail.controller.task.id !== task.id) {
      actions.createEl("p", { cls: "sos-study-map-hint", text: "已有一条记录正在填写，保存或收起后可处理这一项。" });
      this.button(actions, "回到正在填写的记录", () => this.taskDetail.host.scrollIntoView({ block: "center" }), "sos-quiet");
    } else if (task.state === "done") {
      this.button(actions, "查看完成记录", () => this.openTask(task, "", "complete", "study-" + task.id), "sos-quiet");
      if (course.continuation?.after_task_id === task.id) {
        const number = course.continuation.title.match(/Lecture\s+(\d+)/)?.[1];
        this.button(actions, number ? `继续 Lecture ${number}` : "继续下一讲", () => this.continueCourse(course), "mod-cta sos-study-continue");
      } else if (model.nextTask) this.button(actions, "继续学习", () => this.beginTask(model.nextTask), "mod-cta sos-study-start");
    } else {
      const canonical = this.bridge.lastState.tasks.find(t => t.path === task.path && t.raw === task.raw) || task;
      const job = this.bridge.lastState.jobs.find(j => j.request?.task?.id === canonical.id && !j.handed_off && j.state !== "cancelled");
      this.button(actions, job?.state === "registered" ? "继续学习" : "开始学习", () => this.beginTask(canonical), "mod-cta sos-study-start");
      this.button(actions, "记录进度", () => this.openTask(task, "", "progress", "study-" + task.id));
      this.button(actions, "核验完成", () => this.openTask(task, "", "complete", "study-" + task.id), "sos-quiet");
    }
    if (course.continuation?.after_task_id === task.id) preview.createEl("p", { cls: "sos-study-map-hint", text: "下一讲：" + displayText(course.continuation.title).replace(course.id + "｜", "") });
  }
  async continueCourse(course) {
    if (this.bridge.busy || this.continuingCourse || this.startingLesson) return;
    if (this.taskDetail) throw new Error("请先保存或收起正在填写的记录。");
    this.continuingCourse = true;
    try {
      const result = await this.bridge.act({ kind: "continue_course", course_id: course.id, task_id: course.continuation.after_task_id }, "接着学习这门课的下一讲，沿用原课程顺序和收工标准。");
      if (result?.error || !result?.task_id) throw new Error(result?.error || "接续点尚未读回。");
      const current = this.bridge.lastState.curriculum.courses.find(c => c.id === course.id);
      const task = this.bridge.lastState.tasks.find(t => t.id === result.task_id);
      if (!task || task.status !== "completion_unknown") throw new Error("这讲已有完成或关闭记录，已同步最新位置。");
      this.selectStudyUnit(current, task, true);
      return await this.beginTask(task);
    } finally { this.continuingCourse = false; }
  }
  async beginTask(task) {
    if (this.taskDetail && this.taskDetail.controller.task.id !== task.id) throw new Error("请先保存或收起正在填写的记录。");
    if (this.bridge.busy || this.startingLesson) throw new Error("正在处理当前请求，请稍后继续。");
    this.startingLesson = true;
    try {
      const data = this.bridge.lastState;
      const canonical = data.tasks.find(t => t.path === task.path && t.raw === task.raw);
      if (!canonical || ["recorded_complete", "cancelled"].includes(canonical.status)) throw new Error("任务记录已变化，请刷新后选择当前学习项。");
      const context = learningContext(data, canonical);
      if (data.focus?.task?.id !== canonical.id) {
        const result = await this.bridge.act({ kind: "set_focus", task_id: canonical.id,
          summary: "从课程中选择当前学习项", next_step: context.next }, "我现在学习这项任务，从已有停点继续。");
        if (result?.error || this.bridge.lastState.focus?.task?.id !== canonical.id) throw new Error(result?.error || "当前学习项尚未保存，请刷新后重试。");
      }
      const job = this.bridge.lastState.jobs.find(j => j.request?.task?.id === canonical.id && !j.handed_off && j.state !== "cancelled");
      if (job?.state === "registered") return await this.openClassroom(job);
      const result = await this.prepareTask(canonical);
      const prepared = this.bridge.lastState.jobs.find(j => j.key === result?.job_key && j.state === "registered" && !j.handed_off);
      if (prepared) await this.openClassroom(prepared);
      return result;
    } finally { this.startingLesson = false; }
  }
  async prepareTask(task) {
    const cancelled = this.bridge.lastState.jobs.find(j => j.state === "cancelled" && j.request?.task?.id === task.id);
    return this.bridge.act({ kind: "prepare_classroom", task_id: task.id, job_key: cancelled?.key || "" }, cancelled ? "重新准备这个任务的课堂" : "准备这个任务的课堂");
  }
  renderDetails(data) {
    const details = (label) => { const d = this.board.createEl("details", { cls: "sos-details", attr: { "data-key": label.split(" · ")[0] } }); d.createEl("summary", { text: label }); return d; };
    if (data.progress.length) {
      const history = details(`学习记录 · ${data.progress.length} 条`);
      for (const record of data.progress) {
        const row = history.createDiv({ cls: "sos-history-row" });
        row.createEl("p", { text: `${record.payload.task_title}：${record.payload.summary}` });
      }
    }
    const commitments = data.commitments.length ? details(`约定与提示 · ${data.commitments.length} 项`) : null;
    for (const item of data.commitments) {
      const row = commitments.createDiv(); row.createEl("p", { text: `${item.summary} · ${item.status} · 至 ${item.ends_at}` });
      if (item.status === "active") this.button(row, "取消约定", () => this.bridge.act({ kind: "cancel_commitment", commitment_id: item.id }, "取消这项约定"));
    }
    if (commitments) commitments.createEl("p", { cls: "sos-muted", text: "到期自动停止；提示在 Obsidian 打开时可用。" });
    if (data.weekly_report.error) this.board.createEl("p", { cls: "sos-error", text: data.weekly_report.error });
  }
  renderChat(data) {
    const day = londonClock().date;
    const signature = JSON.stringify([day, data.chat, data.jobs.map(j => [j.key, j.state]), this.bridge.busy ? this.bridge.status : ""]);
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
      const receipts = item.receipts || [];
      if (receipts.length) {
        const outcome = parent.createDiv({ cls: "sos-message sos-outcome", attr: { role: "status" } });
        outcome.createEl("small", { text: item.phase === "done" ? "处理结果" : "已完成的处理" });
        for (const receipt of receipts) {
          const result = receipt.result;
          const row = outcome.createDiv({ cls: "sos-outcome-row" });
          row.createEl("p", { text: (result.phase === "insufficient" ? "待补充 · " : "✓ ") + result.message });
          if (result.job_key && result.state === "registered") {
            const job = data.jobs.find(j => j.key === result.job_key && j.state === "registered");
            if (job) this.button(row, "进入课堂", () => this.openClassroom(job), "sos-primary");
          }
          if (receipt.kind === "set_focus") {
            const task = data.tasks.find(t => t.id === receipt.task_id);
            if (task) this.button(row, "展开任务", () => this.openTask(task), "sos-quiet");
          }
          if (receipt.kind === "check_completion" && result.phase === "insufficient") {
            const task = [...data.tasks, ...data.obligations].find(t => t.id === receipt.task_id);
            if (task) this.button(row, "补充核验", () => this.openTask(task, "", "complete"), "sos-quiet");
          }
        }
      }
      const reply = receipts.length ? item.explanation : item.reply;
      if (reply) {
        const bubble = parent.createDiv({ cls: "sos-message sos-assistant" });
        let target = bubble;
        if (reply.length > 480) {
          const preview = reply.split("\n").filter(line => line.trim() && !/^\s*[|#>-]/.test(line)).slice(0, 2).join(" ");
          bubble.createEl("p", { text: displayText(preview).slice(0, 180) + (preview.length > 180 ? "…" : "") || "课程位置和依据已整理。" });
          const full = bubble.createEl("details", { cls: "sos-chat-full" });
          full.createEl("summary", { text: "展开完整回复" });
          target = full.createDiv();
        }
        MarkdownRenderer.render(this.app, reply, target, "", this).catch(() => target.setText(reply));
      }
      if (item.error) parent.createDiv({ cls: "sos-message sos-error", text: "尚未全部完成：" + item.error });
      if (!["done", "cancelled"].includes(item.phase)) {
        if (!item.error) parent.createDiv({ cls: "sos-chat-working", text: this.bridge.activeRequest === item.key ? this.bridge.status : "处理尚未结束，可以从原停点继续。", attr: { role: "status" } });
        const row = parent.createDiv({ cls: "sos-actions" });
        this.button(row, "继续原请求", () => this.bridge.run("resume", { key: item.key }));
        this.button(row, "停止这条请求", async () => { await this.bridge.call("cancel-request", { key: item.key }); await this.bridge.refresh(); });
      }
      if (item.phase === "cancelled") parent.createDiv({ cls: "sos-muted", text: "这条请求已停止。" });
    }
    this.messages.scrollTop = this.messages.scrollHeight;
  }
}

function taskCaption(task) {
  if (task.short_title) return task.short_title;
  if (task.kind === "obligation") return titleForDeadline(task.raw);
  let text = displayText(task.title || task.raw).replace(/^[^｜]+｜/, "").replace(/（原课时[^）]*）/, "").replace(/依据课堂记录/g, "").replace(/(\d{4})-(\d{2})-(\d{2})/, "$2/$3");
  if (text.includes("复盘本讲内容")) return text.split("：")[0] + " · 独立复盘";
  if (text.includes("正式双语笔记")) return text.split("：")[0] + " · 整理笔记";
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

module.exports = { STEWARD_VIEW, StewardBridge, StewardView, CompletionModal, installCompletionGuard, renderedTask, homeModel, classroomMaterial, positionFor, agendaFor, todayNotices, deadlineNotices, deadlineBoard, deadlineSupplements, selfStudyFor, studyUnitFor, learningContext, nearbyUnits, sameRef, sectionForPath, titleForDeadline, deadlineTime };
