# Knowledge × Open Courses

Local interactive map connected to the Academic Obsidian vault. Run `打开知识地图.command` in Finder, or `python3 launch.py` from this directory. Python 3.9 or later is required; there are no third-party dependencies. Keep the Terminal window open while using the map. Control-C stops the server.

The map is available at <http://127.0.0.1:8768>. It binds only to the local machine. `python3 launch.py --no-browser --port 8769` starts it without opening a browser and can select another port.

## Using the map

- Select a topic to compare exact course chapters and open existing notes in Obsidian.
- Select a course to highlight the topics mapped to it, or use the course comparison table.
- Select a connection to read its precise meaning and source. Solid lines show preparation for the stated derivation; dashed lines show applications or alternative computational methods. A line never requires completing an entire course.
- Search the existing `00_Knowledge` notes by title, heading, alias, or stable atom ID. Refresh reloads changed files.
- Add a course with its version, connect a chapter to a topic, or connect an existing note. Personal additions are marked as awaiting verification.

## Current coverage

The first curated slice has 15 topics, 17 typed relations, 27 linked notes and 22 chapter mappings across MIT 18.06SC (Fall 2011), Stanford CS229 (fixed SEE edition), and MIT 6.036 (OLL 2019 notes). It covers linear algebra, projections, least squares, PCA, and introductory gradient descent. It is not a comprehensive CS/AI curriculum. An empty table cell means not yet mapped, not absent from the course. Topic/relationship expansion currently requires reviewing and editing `seed.json`; the UI adds courses, chapter mappings and note links.

Horizontal positions follow the displayed knowledge dependency graph, not semesters, difficulty, or learning time. Vertical lanes group topics by field. The graph is a scoped learning interpretation with evidence on each relation. Additional prerequisites can be found in the original notes and course materials.

Note source verification and course checklist progress are displayed separately from learner mastery. The map does not infer mastery. The MIT progress number is a read-only projection of the existing private learning plan.

## Data ownership

- `seed.json`: reviewed topic/relation/course mapping content; course versions and source URLs remain explicit.
- `00_Knowledge/*.md`: canonical source notes, read without modification. The app indexes top-level Markdown only and excludes symlinks.
- `99_学习情况记录/knowledge-map/state.json`: personal additions, created on the first save. This directory is ignored by Git. Each successful change retains the preceding state as `state.previous.json`.
- `99_学习情况记录/knowledge-map/知识地图.md`: private navigation entry.

The app does not rewrite notes, course Atlases, Canvas files, mastery records, or Obsidian settings. Notes are previewed as plain text; full Markdown and formula rendering remains in Obsidian. No remote analytics, assets, or API are used. Official course links open the course sites.

## Verification

Run `python3 -m unittest discover -s 05_tools/KnowledgeMap -p 'test_*.py'` from the vault root, and `node --check 05_tools/KnowledgeMap/static/app.js`. Backend checks cover local-origin/path restrictions, validation, atomic persistence, concurrent writes, corrupt-state protection, seed references and prerequisite cycles.

For isolated UI testing, start `python3 server.py --data-dir /absolute/path/to/temporary-directory`; never add QA records to the default private state.
