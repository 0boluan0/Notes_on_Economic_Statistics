# Student OS steward

## Course review pilot

The course review pilot is paused; its interface is not registered in the dashboard.
Its source, attempts and reports are retained. The ST445 Lecture 1
pilot is a finite course-level question pack, independent of classroom chats and
canonical task completion. `review.py` supplies the `review` JSON CLI command;
`review-view.js` is retained as pilot source outside the deployed bundle. No online model is used
while answering, checking structured responses, reading material or scheduling.

The learner completes four scenario tasks before feedback. Explanations retain
the original response and require explicit rubric self-assessment. Incorrect
families lead to specific local PDF pages, followed by an unseen retest. A failed
retest or explicit deferral remains a learning gap. The report distinguishes
script checks, self-assessment, repair and delayed attempts; no checkbox or mastery
record is silently changed. Scope limitations remain visible in the entry/report.

Review the entire visible task together before publishing: its title, scenario,
instructions, options and later subquestions must not supply another item's answer
or declare an option wrong. Keep necessary problem constraints, but move diagnosis,
solution mechanisms and explanations to post-submission feedback. This requires
semantic review; stripping answer-key fields alone cannot detect prose hints.
When an existing attempt has documented prompt exposure, preserve its answers and
mark that evidence separately. A correct cued attempt needs an unseen check; it
is not a knowledge error. Deferring that check stays pending verification.

Sessions, drafts, seen questions, timing checkpoints and the single next due date
are stored in the existing private SQLite database. Finished reports use the
ignored `99_学习情况记录/管家/复习/` directory. Source hashes bind each session to
its material version. A resumed attempt keeps its original question pack. Changes
to the source block further answers while retaining old reports and attempts.
Explicit question corrections preserve submitted/seen task history and record
the old version and revised task snapshots in private `question_revisions`.

The first delayed set opens at 07:00 London time on the next calendar date; after
a successful delayed set, the second opens six calendar days later. Remaining
gaps use the next day. These are pilot parameters and in-app eligibility windows,
not a universal memory schedule or a new notification service. Missed days do
not generate catch-up copies. Used questions do not count as unseen transfer
checks; exhausted inventory is explicit and needs another authored pack.

Time is recorded by phase while the review or its dedicated source pane is active
and the application is visible/focused. The learner can pause. Sleep gaps are
discarded, drafts and time flush every ten seconds and on transitions. This is
an estimate of visible time on the task, not a measure of cognitive attention.
An abrupt process kill can lose the most recent unsaved interval. The report also
shows wall-clock session span, including pauses. The final report-save latency
is outside the report's recorded interval.

`test_review.py` exercises durable transitions, replay/rollback, source changes,
finite inventory, self-assessment and London-date scheduling in temporary vaults.
`review-check.js` checks timing and lost-response recovery. `verify-review.js`
uses an isolated temporary vault/runtime through a per-view transport hook for
live Obsidian acceptance; only source-page navigation uses the real course file.
Synthetic attempts must never be created in the learner's runtime.

## Dashboard and background reconciliation

The Obsidian plugin is a local learning dashboard for course position, weekly tasks,
materials, Workbench and Deadlines. Open **Student OS 看板** from the ribbon or
**打开 Student OS 看板** from the command palette. Teaching takes place in Codex or DSH.
The local `dsh-student-os` plugin reads these canonical records and uses the existing
`record_progress` operation for explicitly matched teaching checkpoints; it never
closes tasks or declares independent mastery.
The existing Today/Shutdown, Telegram, Calendar and Computer History workflows
keep their owners and schedules.

The view reads canonical task/deadline Markdown, the private curriculum dossier
and saved learning reports. File-change refresh and the minute timer use Python
without model calls. No embedded chat or automatic classroom preparation is invoked
by opening the dashboard. The bridge permits only deterministic view, progress,
focus, course-continuation, cancellation and manual-status operations.

Task checkboxes mark completion directly and can be unchecked. The optional
inline form adds a report, actual completion date and related files. These are
user confirmations, distinct from independent artifact or platform verification.
Native Tasks checkboxes retain their normal direct editing behavior. A task's
steps and stopping point expand inline; partial progress remains separate from
whole-task completion. Completed and cancelled tasks retain their source anchors.

`completion-record` stores manual complete/reopen/close decisions in the existing
completion ledger, exports private receipts under `管家/状态记录/`, and uses the
same guarded, recoverable canonical write. Retried operation IDs are idempotent.
Pending older checks for the same task are cancelled when a manual decision is
applied. A closed task keeps its original checkbox and replaces its active tag
with `#student-os/archived`; closure never asserts completion.

Background `completion-check` remains available for evidence-based reconciliation.
It checks real task scope, actual artifact contents and matching external receipts.
Computer History helps find relevant work and stopping points; browsing alone
cannot establish understanding or submission. A subsequent user correction takes
precedence over an older automated verdict. Removing Obsidian chat does not remove
existing native sessions, progress, evidence or automation state.

**打开学习材料** selects the canonical task and opens its exact linked lesson file,
unchanged material binding or explicitly numbered/week-labelled lecture handout.
Existing tabs retain their page and edit state. Ambiguous material is reported;
opening a file never records completion. **继续 Lecture …** uses the existing
Python sequence and reuses or creates the appropriate private replay task.

`weekly.py` groups Lecture, Seminar/exercises, other requirements and recommendations.
Counts use canonical tasks, not elapsed time or mastery estimates. A sourced weekly
scope may override a timetable's generic event types (for example, a project
briefing instead of a standard Lecture). Missing or stale scope remains visible.
`curriculum.py` projects live completion and saved stopping points. A course may
define a sourced milestone sequence by canonical task references: the project card
counts those stage nodes, keeps auxiliary preparation separate, and expands the
whole route inline. Missing references stay visible and never reduce the denominator;
manual ticks and deadline changes feed the same projection.
 the private
`管家/课程总览.md` remains readable independently of Obsidian. Source hashes and audit
records stay in the backend rather than cluttering learner navigation.

Private SQLite state remains in `~/Library/Application Support/StudentOSLearning/steward/`.
Existing teaching sessions remain in the sibling `codex/` directory. The former
Claudian plugin is removed using Obsidian's plugin manager, after a private backup.
No Codex account, desktop conversation or Computer History exporter is removed.

## Execution and recovery

`service.py` is the interface backend. It accepts one JSON object on stdin where
needed and emits JSON lines with `status`, `result` or `error` events on stdout.
It supports `view`, `chat`, `resume`, `act`, `cancel-request`, `prepare-focus`,
`acknowledge`, `completion-context`, `completion-check`, `completion-record` and `completion-cancel`.
`--state` and `--codex-home` allow isolated acceptance environments.

```sh
python3 -B service.py --vault /path/to/Academic view
python3 -B service.py --vault /path/to/Academic prepare-focus
```

`steward.py` remains the low-level classroom CLI: `snapshot`, `enqueue`, `run`,
`attach`, `status`, `cancel`. Learning requests carry the canonical task identity,
version and bounded material excerpts. Synthetic requests explicitly use
`scope: synthetic`; they must never be counted as actual learning.

The planner calls the local [Codex App Server](https://learn.chatgpt.com/docs/app-server)
through stdio with a structured output schema. It proposes a small set of allowed
operations. The core validates each operation and requires a quote from the
current user request, then appends real receipts. Preparation and planning use
read-only model sessions; this client rejects tool and approval requests. This
restriction does not change a classroom's existing Claudian permission settings.

The manager can execute `refresh_dashboard` when reported progress and the board
disagree: reread canonical tasks and checks, rebuild the course projection and
verify its private export. This never changes learning facts. `check_completion`
uses the background evidence-check path, with a stable request ID,
the user's actual report, and an explicitly known date only. Chat cancellation
also stops its active check. Submission still requires a matching receipt.
Structured action receipts appear separately from the conversational explanation,
with classroom entry or evidence follow-up where appropriate. A partial failure
keeps successful receipts visible and resumes only the unfinished work.

Repeated request IDs reuse the same result. Different content under an existing
ID is rejected. A crash after a model reply is recovered by reading the original
turn, not by sending a replacement. Partial action batches preserve completed
receipts. Unknown native creation or unfinished/mismatched turns stop visibly.
Cancelling a request stops further actions; it does not undo already saved work.

Classroom preparation releases the App Server writer before registration.
Claudian's internal `chatHost` adapter loads saved metadata before checking its
lazy session list, then registers and reads back the same native ID.
Registration tracks the asynchronous plugin call, polls for up to 20 seconds
plus CLI latency, and keeps the actual asynchronous error. Retrying while the
same registration is pending joins it instead of starting another mutation.
The independent readback must confirm identity, title and readable history.
Opening verifies readable history and the selected conversation; it never silently
creates a replacement. A source-version conflict requires explicit reconciliation.
After cancelling the obsolete job, an explicit prepare action can create its
replacement. Claudian internal APIs and App Server experimental APIs are version
sensitive, so validate them after upgrades.

SQLite snapshots must use the SQLite backup API rather than copying a live file.
Credentials, runtime databases, logs and acceptance outputs stay private and out
of this source tree. Source deployment uses the plugin's build/deploy scripts;
do not hand-edit live Obsidian configuration or session metadata.

## Verification

```sh
python3 -B -m unittest discover -s . -p 'test_*.py' -v
node ../../Obsidian/learning-progress-dashboard/self-check.js
```

The dashboard deployment runs these checks before installing. The reusable
`learning-progress-dashboard/verify-ui.js` checks the live weekly view in Obsidian,
including each week, the completed filter and responsive widths, restoring the
original view state afterward.
`verify-inline.js` checks inline checkpoints, draft retention and finite-course
replay display. `verify-study.js` checks the self-study unit navigator, original
lecture numbering, replay/history separation, saved selection, task binding and
draft retention across unit switches and refreshes. Its classroom actions are
stubbed; it never creates a conversation or changes learning records.
`verify-workspace.js` checks navigation and guarded native
checkboxes. `verify-steward.js` opens an existing classroom, verifies its original
identity/history and checks visible action receipts. These scripts do not send
messages, save learning progress or tick canonical tasks.

Tests exercise interruption recovery, duplicate delivery, cancellation, source
conflicts, bounded material selection, evidence boundaries, private exports and
expiry across the London daylight-saving transition. Provider fixtures do not
certify the live API or GUI. Live acceptance separately checks model planning,
classroom registration/opening, application reload, and progress writeback in an
explicit test vault. This does not certify power-loss, disk-failure, network-loss,
off-device backup recovery or real learning outcomes.
