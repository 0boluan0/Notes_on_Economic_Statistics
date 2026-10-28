const assert = require("assert");
const Module = require("module");
const load = Module._load;
Module._load = function(request, ...args) {
  return request === "obsidian" ? { ItemView: class {}, Notice: class {} } : load.call(this, request, ...args);
};
const { ReviewView, ReviewClock, minutes, totalTime } = require("./review-view");
Module._load = load;
const clock = new ReviewClock();
clock.tick(0, "diagnostic", true);
clock.tick(1000, "diagnostic", true);
clock.tick(2000, "diagnostic", false);
clock.tick(3602000, "repair", true);
clock.tick(3603000, "repair", true);
assert.deepEqual(clock.pending, { diagnostic: 1, repair: 1 });
clock.saved("diagnostic", 1);
assert.equal(totalTime(clock.pending), 1);
assert.equal(minutes(125.8), "2:05");
assert.equal(minutes(-1), "0:00");
// An unknown write outcome must retry the original event, then save edits made
// while the connection was unavailable before navigation discards the UI draft.
async function checkpointRetry(initial = { diagnostic: 7 }, lossAt = 1) {
  const view = Object.create(ReviewView.prototype), saved = new Map();
  Object.assign(view, { data: { session: { id: "synthetic", stage: "question" }, timing: {} },
    draft: { task_id: "task", answers: { why: "first draft" } }, running: false,
    clock: new ReviewClock(), tick() {}, phase: () => "diagnostic" });
  view.clock.pending = { ...initial };
  let uncertain = true, durableDraft, durableSeconds = 0;
  view.transport = async event => {
    const value = JSON.stringify(event);
    if (saved.has(event.event_id)) assert.equal(value, saved.get(event.event_id), "Retries must keep the exact checkpoint payload");
    else { saved.set(event.event_id, value); durableDraft = JSON.parse(JSON.stringify(event.draft)); durableSeconds += event.seconds; }
    assert(event.seconds <= 300, "Long recovery intervals must be split into accepted checkpoint sizes");
    if (uncertain && saved.size === lossAt) { uncertain = false; throw new Error("Synthetic lost response after write"); }
    return { timing: { diagnostic: durableSeconds } };
  };
  await assert.rejects(view.flush(), /lost response/);
  view.draft.answers.why = "newer draft after failure";
  view.clock.pending.diagnostic += 3;
  await view.flush();
  assert.equal(durableDraft.answers.why, "newer draft after failure", "Retry discarded the latest in-memory draft");
  assert.equal(totalTime(view.clock.pending), 0, "Retry left recently recorded time unsaved before navigation");
  assert.equal(durableSeconds, totalTime(initial) + 3, "Retry must save time recorded since the uncertain write exactly once");
}

checkpointRetry().then(() => checkpointRetry({ diagnostic: 650, repair: 17 }, 2))
  .then(() => console.log("Review timing and checkpoint checks passed"))
  .catch(error => { console.error(error); process.exitCode = 1; });
