# Student OS steward

A fixed Obsidian home for planning, classroom preparation and durable learning
handoffs. The existing Learning Progress Dashboard plugin hosts the home and its
chat; Claudian hosts separate teaching conversations. Python runs bounded local
operations behind the interface, with no additional Python packages.

## Entry and daily use

Deploy the dashboard through its normal `deploy.sh`, then reload the plugin.
Open **Student OS 管家** from the ribbon or the command **打开 Student OS 管家**.
The existing `open-today-note` command now opens the steward home; the Today note
remains available in the home navigation and continues to own the daily plan.

The home shows the current Today time slot separately from the prepared classroom.
Course cards summarize canonical task records, with completed, observed, unknown,
and future states. The weekly view lists original tasks and supports course,
week, and state filters. Its completed-count control opens completed learning
records across all weeks. Expanding a task shows its full requirement and source.

Click a weekly task's circle, a native Workbench/Today/Deadlines checkbox, or
**核验完成** beside the classroom to open the same completion form. Describe the
actual scope, optionally choose an actual completion date, and attach evidence.
Unknown dates remain blank. **核验并记录完成** reads the evidence and checks it
against that one task; sufficient evidence updates the original checkbox, while
insufficient evidence stays **待核验** with a specific missing item. **记录停点**
continues to save partial progress without claiming completion.

**不再需要，关闭任务** records the user's decision immediately without a model
or completion evidence. Clear cancellation intent in the form or steward chat
uses the same path. Closure preserves the original checkbox, dates and anchor,
replaces the active task/deadline tag with `#student-os/archived`, and saves a
private `closed` receipt. It removes the task from active views without counting
it as completed or submitted. Elapsed deadlines alone never authorize closure.

`completion.py` stores the original claim, task version, bounded evidence content,
file hashes, native check turn and result. Citations must occur in the supplied
evidence; submission receipts must identify the matching course/assignment and
successful state. Ordinary self-reports are labelled separately from artifact
or receipt checks. PDF text, saved notebook outputs and image OCR can be read;
unreadable or incomplete evidence remains uncertain. The check does not log in
to external platforms. The model judges meaning; deterministic code controls
scope, citations, privacy, deduplication, cancellation and canonical writes.

`weekly.py` derives weeks from explicit Week labels, lesson dates, or original
scheduled/due dates. Undated tasks and dates outside the current term stay in
separate buckets. It never reschedules tasks or changes checkboxes. Progress bars
count task records, not mastery or weighted course completion. Learning evidence
is a separate state and never increments the completed count.

`curriculum.py` projects the saved timetable and private `管家/课程定位.json`
observations. Positions carry evidence hashes and dates. A `file.md#Heading`
reference fingerprints that section, so unrelated edits do not invalidate it.
Expired or changed evidence retains its dated last-known description. New learning
handoffs update their matching course, while earlier observations stay associated
with their original tasks. The generated private `管家/课程总览.md` also remains
readable without the plugin.

Record changes trigger a debounced refresh; the minute timer handles time and day
changes. Refresh, weekly grouping, source checks, counts and direct progress-form
writes do not call a model. The last valid course projection is retained in SQLite:
a temporarily missing or corrupt input keeps that snapshot visible with a warning,
including across process restarts. A later successful read replaces it. Failed
bridge reads retain the in-memory view. Generated exports do not watch themselves.
Semantic changes in school materials still require explicit evidence; the code
does not infer knowledge from browsing or silently advance teaching boundaries.

On startup and once per minute while Obsidian is running, the bridge refreshes
records. It prepares a newly selected Today task when no prior classroom or
unfinished manager request exists. It does not automatically restart cancelled
classrooms or create another lesson after a handoff. **进入已准备课堂** opens the
saved native conversation; **学完记录停点** saves the user's actual range and next step.

## Runtime and boundaries

- Requires macOS, Python 3.9+, the installed Codex CLI and its existing login in
  the separate `~/Library/Application Support/StudentOSLearning/codex/` home.
  Obsidian CLI and enabled Claudian are required for classroom registration.
- Private SQLite state lives in
  `~/Library/Application Support/StudentOSLearning/steward/`, outside the iCloud
  vault. Immutable requests, native IDs, stages, receipts and cancellation state
  survive process exits. The fixed chat is a database-backed transcript; internal
  planning turns use separate native contexts, so a long chat is not the memory
  store.
- Existing task and deadline Markdown files remain their canonical originals.
  Only an explicit completion request passing its evidence check may tick one
  original. Direct Markdown edits remain user-owned. An unchecked task is
  completion-unknown, not proven unfinished. Check receipts are exported under
  the private `管家/完成核验/` folder; resumed requests reuse their original turn.
- Learning handoffs are stored in SQLite and exported as immutable Markdown under
  the Git-ignored `99_学习情况记录/管家/学习交接/` folder. They record user reports,
  source versions, optional artifact hashes and the next step. They do not verify
  mastery or formal submission. External edits to an export are preserved.
- The current week's factual report is refreshed under the private
  `管家/事实周报/` folder. It states source coverage and distinguishes recorded
  completion, user reports and activity observations. Computer History is a
  recent cache, not a full-week archive or learning/submission proof.
- Daily commitments have explicit start/end boundaries and London wall-clock
  times. Acknowledgements are deduplicated by commitment and local date. Expired
  or cancelled commitments cannot revive on request replay. Missed days are not
  accumulated. These are **in-app prompts while Obsidian is open**, not OS or
  Telegram notifications.
- The existing Today/Shutdown, Telegram, Calendar and Computer History automation
  owners are unchanged. No new LaunchAgent, network listener or morning schedule
  is installed. Closing or sleeping the Mac suspends this entry's work.

## Execution and recovery

`service.py` is the interface backend. It accepts one JSON object on stdin where
needed and emits JSON lines with `status`, `result` or `error` events on stdout.
It supports `view`, `chat`, `resume`, `act`, `cancel-request`, `prepare-focus`,
`acknowledge`, `completion-context`, `completion-check` and `completion-cancel`.
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

Repeated request IDs reuse the same result. Different content under an existing
ID is rejected. A crash after a model reply is recovered by reading the original
turn, not by sending a replacement. Partial action batches preserve completed
receipts. Unknown native creation or unfinished/mismatched turns stop visibly.
Cancelling a request stops further actions; it does not undo already saved work.

Classroom preparation releases the App Server writer before registration.
Claudian's internal `chatHost` adapter loads saved metadata before checking its
lazy session list, then registers and reads back the same native ID.
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

Tests exercise interruption recovery, duplicate delivery, cancellation, source
conflicts, bounded material selection, evidence boundaries, private exports and
expiry across the London daylight-saving transition. Provider fixtures do not
certify the live API or GUI. Live acceptance separately checks model planning,
classroom registration/opening, application reload, and progress writeback in an
explicit test vault. This does not certify power-loss, disk-failure, network-loss,
off-device backup recovery or real learning outcomes.
