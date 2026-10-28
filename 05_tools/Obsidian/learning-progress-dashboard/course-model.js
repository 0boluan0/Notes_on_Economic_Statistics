const { createHash } = require("crypto");

const PRIVATE_RECORD = "99_学习情况记录/课程地图/学习记录.md";
const BINDINGS = {
  "mit-1806sc": "MIT 18.06",
  "mit-6041sc": "MIT 6.041SC",
  "mit-1801sc": "MIT 18.01",
  "mit-6100l": "MIT 6.100L",
  "mit-1802sc": "MIT 多元微积分",
};
const isSchoolCourse = course => !!course && (["degree", "future"].includes(course.group) || ["DSL Python", "DSL R"].includes(course.id));
const unique = list => [...new Set(list)];
const unitKey = (course, unit) => `${course.id}|${course.version}|${unit.id}`;
const anchorFor = (course, unit) => "cm-" + createHash("sha256").update(unitKey(course, unit)).digest("hex").slice(0, 24);
const unitOutcomes = unit => unit.mapping === "reviewed-outline" ? unique(unit.outcomes || []) : [];

function catalogFrom(parts) {
  const courses = parts.flatMap(p => p.courses || []);
  const outcomes = new Map(), domains = new Map();
  for (const part of parts) {
    for (const d of part.domains || []) domains.set(d.id, d);
    for (const o of part.outcomes || []) {
      if (!outcomes.has(o.id)) outcomes.set(o.id, o);
      else {
        const old = outcomes.get(o.id);
        if (old.title !== o.title || old.scope !== o.scope || JSON.stringify(old.extends || []) !== JSON.stringify(o.extends || [])) {
          throw new Error("Conflicting knowledge scope: " + o.id);
        }
        outcomes.set(o.id, { ...o, ...old, notes: unique([...(old.notes || []), ...(o.notes || [])]) });
      }
    }
  }
  const courseIds = new Set();
  for (const c of courses) {
    if (courseIds.has(c.id)) throw new Error("Duplicate course: " + c.id);
    courseIds.add(c.id);
    const unitIds = new Set();
    for (const u of c.units || []) {
      if (unitIds.has(u.id)) throw new Error("Duplicate unit: " + c.id + "/" + u.id);
      unitIds.add(u.id);
      if (!["reviewed-outline", "syllabus"].includes(u.mapping)) throw new Error("Unreviewed mapping: " + c.id + "/" + u.id);
      for (const id of unitOutcomes(u)) if (!outcomes.has(id)) throw new Error("Unknown outcome: " + c.id + "/" + id);
    }
  }
  return { courses, outcomes: [...outcomes.values()], domains: [...domains.values()] };
}

function parseRecords(text, catalog) {
  const anchors = new Map();
  for (const c of catalog.courses) for (const u of c.units || []) anchors.set(anchorFor(c, u), unitKey(c, u));
  const result = new Map();
  for (const line of String(text).split("\n")) {
    const m = line.match(/^- \[([ xX])\].*\^(cm-[a-f0-9]{24})\s*$/);
    if (!m || !anchors.has(m[2])) continue;
    const key = anchors.get(m[2]);
    if (result.has(key)) throw new Error("学习记录有重复标识，请先核对原记录。");
    result.set(key, { done: m[1].toLowerCase() === "x", date: line.match(/✅ (\d{4}-\d{2}-\d{2})/)?.[1] || "", raw: line });
  }
  return result;
}

function updateRecord(text, course, unit, completed, date) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) throw new Error("Invalid completion date");
  const anchor = anchorFor(course, unit), lines = String(text).split("\n");
  const hits = lines.map((line, i) => line.trimEnd().endsWith("^" + anchor) ? i : -1).filter(i => i >= 0);
  if (hits.length > 1) throw new Error("学习记录有重复标识，未改写文件。");
  const label = `${course.code || course.title}｜${unit.id} ${unit.title}`.replace(/[\r\n]/g, " ");
  if (hits.length) {
    const i = hits[0];
    if (!/^- \[[ xX]\]/.test(lines[i])) throw new Error("原学习记录格式已改变，未覆盖。");
    lines[i] = lines[i].replace(/^- \[[ xX]\]/, completed ? "- [x]" : "- [ ]").replace(/\s+✅ \d{4}-\d{2}-\d{2}/g, "");
    if (completed) lines[i] = lines[i].replace(" ^" + anchor, ` ✅ ${date} ^${anchor}`);
  } else lines.push(`- [${completed ? "x" : " "}] ${label} #student-os/task${completed ? " ✅ " + date : ""} ^${anchor}`);
  return lines.join("\n").replace(/\n+$/, "") + "\n";
}

function taskFor(course, unit, curriculum) {
  // These courses have verified one-to-one session mappings to canonical plans.
  if (!["mit-1806sc", "mit-6041sc"].includes(course.id)) return null;
  const courseId = BINDINGS[course.id];
  const label = course.id === "mit-1806sc" ? unit.id :
    /^lec\d{2}$/.test(unit.id) ? `Lecture ${unit.id.slice(3)}` :
    ({ "quiz-1": "Quiz 1", "quiz-2": "Quiz 2", "final-exam": "Final Exam" })[unit.id];
  if (!label) return null;
  const prefix = `${courseId}｜${label} `;
  const matches = (curriculum?.weekly?.tasks || []).filter(t => t.course_id === courseId && (t.title || "").startsWith(prefix) && !t.title.includes("本轮重看"));
  return matches.length === 1 ? matches[0] : null;
}

function completionFor(course, unit, curriculum, records) {
  const task = taskFor(course, unit, curriculum);
  if (task) return { done: task.state === "done" || task.status === "recorded_complete", task, source: task.path, date: task.raw?.match(/✅ (\d{4}-\d{2}-\d{2})/)?.[1] || "" };
  if (BINDINGS[course.id]) {
    return { done: false, unavailable: true };
  }
  const record = records.get(unitKey(course, unit));
  return { done: !!record?.done, source: record ? PRIVATE_RECORD : "", date: record?.date || "" };
}

function learnedOutcomes(catalog, curriculum, records, comparison = "personal") {
  const learned = new Map();
  for (const course of catalog.courses) for (const unit of course.units || []) {
    const completion = completionFor(course, unit, curriculum, records);
    const include = comparison === "personal" ? completion.done : comparison === course.id;
    if (!include) continue;
    for (const id of unitOutcomes(unit)) {
      const sources = learned.get(id) || [];
      sources.push({ courseId: course.id, courseTitle: course.code || course.title, unitId: unit.id, unitTitle: unit.title, date: completion.date, source: unit.source || course.syllabus, simulated: comparison !== "personal" });
      learned.set(id, sources);
    }
  }
  return learned;
}

function coverageFor(unit, learned, definitions = new Map()) {
  const ids = unitOutcomes(unit);
  if (!ids.length) return { status: unit.mapping === "reviewed-outline" ? "overview" : "unknown", known: [], missing: [], total: 0, extensions: [] };
  const known = ids.filter(id => learned.has(id)), missing = ids.filter(id => !learned.has(id));
  const extensions = missing.filter(id => (definitions.get(id)?.extends || []).some(parent => learned.has(parent)));
  return { status: known.length === ids.length ? "covered" : known.length ? "partial" : extensions.length ? "extension" : "new", known, missing, total: ids.length, extensions };
}

function courseCoverage(course, learned, definitions) {
  return coverageFor({ mapping: "reviewed-outline", outcomes: unique((course.units || []).flatMap(unitOutcomes)) }, learned, definitions);
}

const KNOWLEDGE_DOMAINS = {
  "linear-algebra": "math", calculus: "math", "multivariable-calculus": "math", "numerical-analysis": "math",
  "real-analysis": "math", "measure-theory": "math", "differential-equations": "math", "matrix-methods": "math",
  probability: "probability", "stochastic-processes": "probability", "statistical-inference": "statistics",
  bayesian: "bayesian", "causal-inference": "causal", "convex-optimization": "optimization", optimization: "optimization",
  "software-engineering": "programming", "data-engineering": "data-engineering", "machine-learning": "ml",
  "reinforcement-learning": "ml", "deep-learning": "deep-learning", "ai-systems": "ai-systems",
};
function knowledgeIndex(courses, definitions = new Map()) {
  const index = new Map();
  for (const course of courses) for (const unit of course.units || []) for (const id of unitOutcomes(unit)) {
    if (!index.has(id)) index.set(id, { id, domain: KNOWLEDGE_DOMAINS[definitions.get(id)?.family] || course.domain, domains: [], courses: [], units: [] });
    const entry = index.get(id);
    if (!entry.domains.includes(course.domain)) entry.domains.push(course.domain);
    if (!entry.courses.includes(course.id)) entry.courses.push(course.id);
    entry.units.push({ courseId: course.id, unitId: unit.id, title: unit.titleZh || unit.title, source: unit.source || course.syllabus });
  }
  return index;
}

function knowledgeSelection(catalog, state = {}, learned = new Map()) {
  const definitions = new Map(catalog.outcomes.map(o => [o.id, o]));
  const index = knowledgeIndex(catalog.courses, definitions), courses = new Map(catalog.courses.map(c => [c.id, c]));
  const q = (state.query || "").trim().toLowerCase();
  return [...index.values()].filter(entry => {
    if (!inDomain(entry, state.domain || "all")) return false;
    const eligible = entry.courses.some(id => {
      const c = courses.get(id);
      return (!state.role || state.role === "all" || c.role === state.role) && (!state.track || state.track === "both" || (c.tracks || []).includes(state.track));
    });
    if (!eligible) return false;
    const o = definitions.get(entry.id), known = learned.has(entry.id);
    const match = !q || [o.title, o.scope].join(" ").toLowerCase().includes(q) || entry.units.some(ref => {
      const c = courses.get(ref.courseId);
      return [c.code, c.school, c.title, c.titleZh, ref.title].join(" ").toLowerCase().includes(q);
    });
    const filter = state.knowledgeFilter || "all";
    return match && (filter === "all" || (filter === "learned" && known) || (filter === "unrecorded" && !known) || (filter === "shared" && entry.courses.length > 1));
  });
}

function comparisonCandidates(target, courses) {
  const wanted = new Set((target.units || []).flatMap(unitOutcomes));
  return courses.filter(c => c.id !== target.id).map(course => {
    const scope = unique((course.units || []).flatMap(unitOutcomes));
    return { course, overlap: scope.filter(id => wanted.has(id)).length, total: scope.length };
  }).filter(c => c.total).sort((a, b) => b.overlap - a.overlap);
}

const PROBABILITY_DOMAINS = ["probability", "statistics", "bayesian"];
function inDomain(course, domain) {
  return domain === "all" || (domain === "probability-bayes" ? PROBABILITY_DOMAINS.includes(course.domain) : course.domain === domain);
}

function graphLayout(courses) {
  const byId = new Map(courses.map(c => [c.id, c]));
  const edges = courses.flatMap(c => (c.prerequisites || []).filter(p => byId.has(p.id) && p.id !== c.id).map(p => ({ ...p, from: p.id, to: c.id })));
  const ranks = new Map(courses.map(c => [c.id, 0]));
  const incoming = new Map(courses.map(c => [c.id, 0]));
  for (const e of edges) incoming.set(e.to, incoming.get(e.to) + 1);
  const queue = courses.filter(c => incoming.get(c.id) === 0).map(c => c.id), ordered = [];
  while (queue.length) {
    const id = queue.shift(); ordered.push(id);
    for (const e of edges.filter(e => e.from === id)) {
      ranks.set(e.to, Math.max(ranks.get(e.to), ranks.get(id) + 1));
      incoming.set(e.to, incoming.get(e.to) - 1);
      if (incoming.get(e.to) === 0) queue.push(e.to);
    }
  }
  // Cycles remain visible for review rather than pretending they are a valid learning order.
  const cycles = courses.filter(c => !ordered.includes(c.id)).map(c => c.id);
  for (const id of cycles) ranks.set(id, Math.max(0, Number(byId.get(id).stage) || 0));
  return { ranks, edges, cycles };
}

module.exports = { PRIVATE_RECORD, BINDINGS, isSchoolCourse, unique, unitKey, anchorFor, unitOutcomes, catalogFrom, parseRecords, updateRecord, taskFor, completionFor, learnedOutcomes, coverageFor, courseCoverage, knowledgeIndex, knowledgeSelection, comparisonCandidates, inDomain, graphLayout };
