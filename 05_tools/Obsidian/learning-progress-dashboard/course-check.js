const assert = require('assert/strict');
const catalog = require('./course-catalog');
const { PRIVATE_RECORD, anchorFor, unitKey, unitOutcomes, catalogFrom, parseRecords, updateRecord, completionFor, learnedOutcomes, coverageFor, knowledgeIndex, knowledgeSelection, comparisonCandidates, inDomain, graphLayout } = require('./course-model');
const course = id => catalog.courses.find(c => c.id === id);
const definitions = new Map(catalog.outcomes.map(o => [o.id, o]));
const ids = new Set(catalog.courses.map(c => c.id));
const domainIds = new Set(catalog.domains.map(d => d.id));
assert.equal(catalog.courses.length, 57);
for (const c of catalog.courses) {
  assert(domainIds.has(c.domain), c.id);
  assert(['core', 'branch', 'alternative'].includes(c.role), c.id);
  assert(['full-outline', 'topic-outline', 'syllabus'].includes(c.coverage), c.id);
  for (const field of ['url', 'syllabus']) assert.match(c[field], /^https?:\/\//, c.id + '/' + field);
  assert.match(c.checked, /^\d{4}-\d{2}-\d{2}$/);
  if (c.outlineVerified) {
    assert.equal(c.outlineCount, c.units.length, c.id + ' must count actual directory entries');
    assert.match(c.outlineSource, /^https?:\/\//, c.id + '/outlineSource');
    assert.match(c.outlineVerified, /^\d{4}-\d{2}-\d{2}$/);
    assert(c.outlineNotes && c.unitLabel && c.outlineUnit, c.id + ' must explain its counting basis');
    if (c.outlineComplete === false) assert.notEqual(c.coverage, 'full-outline', c.id + ' incomplete directory');
  }
  for (const edge of c.prerequisites || []) {
    assert(ids.has(edge.id), c.id + ' <- ' + edge.id);
    assert(['official', 'inferred'].includes(edge.kind));
    assert(edge.reason && edge.source);
  }
  for (const r of c.requirements || []) {
    assert.match(r.source, /^https?:\/\//, c.id + '/requirement');
    for (const id of r.outcomes || []) assert(definitions.has(id), c.id + '/requires/' + id);
  }
  for (const u of c.units) {
    assert(u.source && u.title && u.module);
    assert.match(u.source, /^https?:\/\//, c.id + '/' + u.id);
    if (c.outlineVerified) assert(u.label !== undefined, c.id + '/' + u.id + ' needs its official display label');
    for (const resource of u.resources || []) {
      assert(resource.label); assert.match(resource.url, /^https?:\/\//);
    }
    if (c.coverage === 'syllabus') assert.equal(unitOutcomes(u).length, 0, 'Coarse syllabus cannot grant topic coverage');
  }
  if (c.coverage !== 'syllabus') assert(c.units.some(u => unitOutcomes(u).length), c.id + ' needs reviewed scope');
}
for (const o of catalog.outcomes) {
  assert(o.title && o.scope, o.id + ' needs an explicit scope');
  for (const parent of o.extends || []) assert(definitions.has(parent), o.id + ' extends ' + parent);
}
assert.equal(unitOutcomes({outcomes:['la.svd']}).length, 0, 'Missing review cannot light topics');
const graph = graphLayout(catalog.courses);
assert.deepEqual(graph.cycles, []);
for (const e of graph.edges) assert(graph.ranks.get(e.from) < graph.ranks.get(e.to));
assert.equal(course('mit-1806sc').units.length, 36);
assert.equal(course('stanford-engr108').units.length, 16);
assert.equal(course('mit-6036').units.length, 15);
assert.equal(course('mit-1806sc').unitLabel, '节');
assert.equal(course('stanford-engr108').unitLabel, '章');
assert.equal(course('mit-6036').unitLabel, '教学章节');
const finalReview = course('mit-1806sc').units.at(-1);
assert.equal(finalReview.id, 'final-review');
assert.equal(coverageFor(finalReview, new Map()).status, 'overview');
assert.deepEqual(unitOutcomes(finalReview), [], 'A review session must not grant the course knowledge without its source lessons');
assert(unitOutcomes(course('mit-6041sc').units.find(u => u.id === 'lec03')).includes('prob.conditional-independence'), '6.041SC Lecture 3 explicitly teaches conditional independence');

// Probability lectures and assessments share the canonical plan's completion.
const probability = course('mit-6041sc');
for (const [id, label] of [['lec01', 'Lecture 01'], ['lec10', 'Lecture 10'], ['quiz-1', 'Quiz 1'], ['quiz-2', 'Quiz 2'], ['final-exam', 'Final Exam']]) {
  const unit = probability.units.find(u => u.id === id);
  const legacy = new Map([[unitKey(probability, unit), {done:true}]]);
  assert.equal(completionFor(probability, unit, {}, legacy).unavailable, true, 'Do not fall back to a second progress record');
  const task = {course_id:'MIT 6.041SC', title:`MIT 6.041SC｜${label} topic`, state:'open', path:'canonical-plan.md'};
  const context = {weekly:{tasks:[task]}};
  assert.equal(completionFor(probability, unit, context, legacy).done, false);
  task.state = 'done';
  assert.equal(completionFor(probability, unit, context, legacy).done, true);
  context.weekly.tasks.push({...task});
  assert.equal(completionFor(probability, unit, context, legacy).unavailable, true, 'Ambiguous canonical tasks must not be guessed');
}

// A completed lecture grants only its reviewed scope; it never completes a matching lecture elsewhere.
const aalto = course('aalto-bda'), duke = course('duke-sta601');
const bayesBasics = aalto.units.find(u => u.id === 'lec02');
const basicRecords = new Map([[unitKey(aalto, bayesBasics), { done: true }]]);
const bayesLearned = learnedOutcomes(catalog, {}, basicRecords);
assert(bayesLearned.has('bayes.conjugate-models'));
assert(!bayesLearned.has('bayes.mcmc-diagnostics'), 'Bayesian basics must not grant later computational knowledge');
const dukeBasics = duke.units.find(u => u.id === 'lec01');
assert.equal(coverageFor(dukeBasics, bayesLearned).status, 'covered');
assert.equal(completionFor(duke, dukeBasics, {}, basicRecords).done, false);
const continuation = duke.units.find(u => u.id === 'continuation-hmc');
const continuationRecords = new Map([[unitKey(duke, continuation), { done: true }]]);
assert.equal(completionFor(duke, continuation, {}, continuationRecords).done, true);
assert.equal(learnedOutcomes(catalog, {}, continuationRecords).size, 0, 'Unknown lecture boundaries cannot grant knowledge');
const oldModule = { id: 'm01', title: 'Former broad module' };
const legacyText = updateRecord('', duke, oldModule, true, '2026-10-25');
assert.equal(parseRecords(legacyText, catalog).size, 0, 'Former module completion must not spill into new lectures');

// Search and shared-scope discovery only index reviewed, equivalent knowledge.
const index = knowledgeIndex(catalog.courses, definitions);
assert(index.get('la.linear-combinations').courses.includes('mit-1806sc'));
assert(index.get('la.linear-combinations').courses.includes('stanford-engr108'));
assert.equal(new Set(index.get('la.linear-combinations').courses).size, index.get('la.linear-combinations').courses.length);
const fixture = [{ id: 'a', domain: 'math', units: [
  { id: '1', title: 'Confirmed', mapping: 'reviewed-outline', outcomes: ['x', 'x'] },
  { id: '2', title: 'Unreviewed', mapping: 'syllabus', outcomes: ['y'] },
] }];
assert.equal(knowledgeIndex(fixture).get('x').units.length, 1);
assert(!knowledgeIndex(fixture).has('y'));
assert.equal(knowledgeIndex(fixture, new Map([['x', {family: 'bayesian'}]])).get('x').domain, 'bayesian', 'Knowledge grouping must not depend on which course first teaches it');
const bayesianShared = knowledgeSelection(catalog, {domain: 'bayesian', knowledgeFilter: 'shared'});
const variational = bayesianShared.find(e => e.id === 'bayes.variational-inference');
assert(variational.courses.includes('berkeley-cs285'), 'Domain selection must retain teaching sources in other course domains');
assert(variational.courses.includes('stanford-cs228'));
assert(bayesianShared.every(e => e.domain === 'bayesian'));
assert(knowledgeSelection(catalog, {query: 'CS285', domain: 'bayesian'}).some(e => e.id === 'bayes.variational-inference'));
assert.equal(coverageFor({mapping: 'reviewed-outline', outcomes: []}, new Map()).status, 'overview');
assert(inDomain(course('harvard-stat110'), 'probability-bayes'));
assert(inDomain(course('aalto-bda'), 'probability-bayes'));
assert(!inDomain(course('mit-1806sc'), 'probability-bayes'));
assert(inDomain(course('mit-1806sc'), 'all'));
const equivalent = { id: 'x', title: 'Same scope', scope: 'A precise learning scope' };
assert.equal(catalogFrom([{outcomes: [equivalent]}, {outcomes: [{...equivalent}]}]).outcomes.length, 1);
assert.throws(() => catalogFrom([{outcomes: [equivalent]}, {outcomes: [{...equivalent, scope: 'A deeper scope'}]}]), /Conflicting knowledge scope/);
const candidates = comparisonCandidates(course('stanford-engr108'), catalog.courses);
assert(candidates.find(c => c.course.id === 'mit-1806sc').overlap > 0);
assert(!candidates.some(c => c.course.id === 'stanford-engr108'));
assert(candidates.every((c, i) => i === 0 || candidates[i - 1].overlap >= c.overlap));

// The expanded catalog must support real overlap discovery outside linear algebra.
for (const [a, b] of [['mit-6041sc', 'harvard-stat110'], ['aalto-bda', 'duke-sta601'], ['cmu-15445', 'berkeley-cs186']]) {
  const left = course(a), right = course(b);
  assert(left && right, a + '/' + b);
  assert(comparisonCandidates(right, catalog.courses).find(c => c.course.id === a).overlap > 0, a + '/' + b + ' should share verified scopes');
}

// A source confirmation lights shared knowledge, never another course's attendance.
const mit = course('mit-1806sc'), stanford = course('stanford-engr108');
const curriculum = { courses: [{ id: 'MIT 18.06', plan: 'canonical.md' }, { id: 'MIT 18.01', plan: 'calculus.md' }], weekly: { tasks: mit.units.map((u, i) => ({
  course_id: 'MIT 18.06', title: `MIT 18.06｜${u.id} ${u.title}`, state: i < 8 ? 'done' : 'unknown', status: i < 8 ? 'recorded_complete' : 'completion_unknown', path: 'canonical.md', raw: 'original'
})) }};
let records = new Map(), learned = learnedOutcomes(catalog, curriculum, records);
assert.equal(mit.units.filter(u => completionFor(mit, u, curriculum, records).done).length, 8);
assert.equal(mit.units.find(u => !completionFor(mit, u, curriculum, records).done).id, '1.9');
const reviewOnly = structuredClone(curriculum);
reviewOnly.weekly.tasks.forEach(t => { t.state = 'unknown'; t.status = 'completion_unknown'; });
reviewOnly.weekly.tasks.at(-1).state = 'done';
reviewOnly.weekly.tasks.at(-1).status = 'recorded_complete';
assert.equal(mit.units.filter(u => completionFor(mit, u, reviewOnly, records).done).length, 1);
assert.equal(learnedOutcomes(catalog, reviewOnly, records).size, 0, 'Completing only final review changes lesson count but cannot light prior knowledge');
assert(learned.has('la.linear-combinations'));
assert(stanford.units.some(u => coverageFor(u, learned, definitions).known.length));
assert(stanford.units.every(u => !completionFor(stanford, u, curriculum, records).done));
assert(completionFor(course('mit-1801sc'), course('mit-1801sc').units[0], curriculum, records).unavailable, 'An existing plan must not gain duplicate progress owners');
assert(completionFor(mit, mit.units[0], { weekly: { tasks: [] } }, records).unavailable);
const ambiguous = structuredClone(curriculum); ambiguous.weekly.tasks.push(ambiguous.weekly.tasks[0]);
assert(completionFor(mit, mit.units[0], ambiguous, records).unavailable, 'Ambiguous canonical task must be read-only');

const unit = stanford.units.find(u => unitOutcomes(u).includes('la.linear-combinations'));
const original = '# My record\n\nUser prose must survive.\n';
let text = updateRecord(original, stanford, unit, true, '2026-10-22');
assert(text.startsWith(original));
assert.equal(parseRecords(text, catalog).get(unitKey(stanford, unit)).done, true);
assert.equal(updateRecord(text, stanford, unit, true, '2026-10-22'), text, 'Retry is idempotent');
text = text.trimEnd() + '  \n';
text = updateRecord(text, stanford, unit, false, '2026-10-22');
assert.equal((text.match(new RegExp(anchorFor(stanford, unit), 'g')) || []).length, 1, 'Trailing spaces must not duplicate records');
assert.equal(parseRecords(text, catalog).get(unitKey(stanford, unit)).done, false);
assert(!text.includes('✅'));
assert.throws(() => updateRecord(text + text, stanford, unit, true, '2026-10-22'), /重复/);
assert.throws(() => parseRecords(text + text, catalog), /重复/);
text = updateRecord(text, stanford, unit, true, '2026-10-22'); records = parseRecords(text, catalog);
assert.equal(completionFor(stanford, unit, curriculum, records).source, PRIVATE_RECORD);
assert.equal(completionFor({ ...stanford, version: 'Different edition' }, unit, curriculum, records).done, false, 'A new edition must not inherit old chapter learning');
const both = learnedOutcomes(catalog, curriculum, records);
assert(both.get('la.linear-combinations').some(s => s.courseId === mit.id));
assert(both.get('la.linear-combinations').some(s => s.courseId === stanford.id));
const undone = structuredClone(curriculum); undone.weekly.tasks.forEach(t => { t.state = 'unknown'; t.status = 'completion_unknown'; });
assert(learnedOutcomes(catalog, undone, records).has('la.linear-combinations'), 'Undoing one source retains other confirmed sources');
records = parseRecords(updateRecord(text, stanford, unit, false, '2026-10-22'), catalog);
assert(!learnedOutcomes(catalog, undone, records).has('la.linear-combinations'), 'Undoing the final source extinguishes coverage');

// Full-course comparison is a projection and cannot alter personal data.
const before = JSON.stringify(curriculum);
const simulated = learnedOutcomes(catalog, curriculum, records, mit.id);
assert.equal(JSON.stringify(curriculum), before);
assert(simulated.size > learned.size);
assert(simulated.get('la.svd').every(s => s.simulated));
const contrast = stanford.units.map(u => coverageFor(u, simulated, definitions));
assert(contrast.some(c => c.status === 'partial'));
assert(contrast.some(c => c.status === 'new'));
assert(contrast.some(c => c.status === 'covered'));
assert(!simulated.has('ml.ols-analytic-solution'), 'Geometric least squares does not prove gradient derivation');
const regression = course('mit-6036').units.find(u => unitOutcomes(u).includes('ml.ols-objective'));
assert.equal(coverageFor(regression, simulated, definitions).status, 'partial');
assert.equal(coverageFor({ mapping: 'syllabus', outcomes: ['la.svd'] }, simulated).status, 'unknown');
assert.equal(coverageFor({ mapping: 'reviewed-outline', outcomes: ['la.svd'] }, new Map([['basic', []]]), new Map([['la.svd', { extends: ['basic'] }]])).status, 'extension');
console.log(`course-map check: ok (${catalog.courses.length} courses, ${catalog.outcomes.length} outcomes; shared learning, undo, canonical ownership, comparison and catalog references)`);
