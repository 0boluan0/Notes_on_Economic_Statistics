# Student OS Computer History input

`history.py` copies existing Computer History Markdown summaries into one atomic,
private JSON snapshot in the shared vault. It retains summaries whose nominal
windows overlap yesterday or today in Europe/London. It does not interpret
activity or change tasks. Python 3.9+ with timezone data is sufficient.

Run `history.py export --vault /absolute/vault --source /absolute/resources` on
the recording Mac. Run `history.py read --vault /absolute/vault --date YYYY-MM-DD`
on either Mac to inspect freshness and the summary index. Add `--detail` to read
the bodies, optionally selecting repeated `--name FILENAME` arguments.

Only install the exporter on the recording Mac. A user LaunchAgent may run it at
login, when the source directory changes, and every ten minutes while awake.
Install the executable outside iCloud; use an absolute interpreter and source
path. Store the snapshot under the Git-ignored private learning-record directory.
The Student OS heartbeat continues to own planning, reconciliation, and messages.

`checked_at` is the last successful local export, not proof of cloud delivery or
recorder health. File-name windows are approximate and can have gaps. The reader
flags old exports; old snapshots can still supply historical evidence. Check the
remote file hash to verify delivery. A missing source or failed read preserves
the previous snapshot. Re-exporting rebuilds the snapshot, removing copies of
expired or deleted summaries without changing their originals.

Test with `python3 -m unittest discover -s 05_tools/StudentOS/computer_history`.
Deployment paths and service controls belong in the private vault integration
note. Never commit snapshots, logs, or personal example activity.
