# Student OS · Telegram

One private bot chat for morning plans and short progress replies. The existing Student OS heartbeat generates plans and reconciliation questions; this bridge sends them and receives the user's replies on the always-on Mac. It creates no additional schedule.

## Setup

1. Create a bot with Telegram's official verified `@BotFather` using `/newbot`.
2. Run `python3 -B setup_local.py --vault '/absolute/Academic'` on the user's current Mac. Open the printed localhost URL, paste the token, and submit. Never put credentials in the vault, terminal arguments, screenshots or chat.
3. Deploy over the existing SSH alias with `python3 -B deploy_macmini.py --expected-username BOT_USERNAME`. This sends credentials over encrypted stdin and installs the reviewed runtime and `local.student-os.telegram` LaunchAgent. Re-running with the same token preserves the owner binding. A different existing token is refused.
4. The user opens the setup page's unique Telegram link and presses Start. Only this exact pairing command in a private chat can bind the owner. The first response explains usage.

Runtime code, token, bound chat, inbound messages, receipts and health stay in `~/Library/Application Support/StudentOSTelegram/`, with private permissions. The LaunchAgent is in `~/Library/LaunchAgents/local.student-os.telegram.plist`. A logged-in Mac user session, internet connectivity and a valid existing Codex CLI login are required. The scheduled Today task also requires the existing Codex app automation environment to remain available.

## Notification CLI

Run on the host that owns the service, with its Academic vault as the working directory:

```sh
/opt/homebrew/bin/python3 "$HOME/Library/Application Support/StudentOSTelegram/app/bridge.py" --vault "$PWD" status
/opt/homebrew/bin/python3 "$HOME/Library/Application Support/StudentOSTelegram/app/bridge.py" --vault "$PWD" send --kind morning --date YYYY-MM-DD --text-file /private/path/summary.txt
```

Use `evening` for the reconciliation question and `test` for an explicit setup check. Each kind/date key identifies exactly one message. A repeated key with identical content is suppressed; altered content under the same key is rejected. `--retry-rejected` is only for a reviewed, proven rejection and identical content. Ambiguous timeouts are never automatically resent. `sent` means the Telegram API accepted the message, not that a phone displayed it. Do not bypass deduplication by changing the date or kind after a failure.

The bot handles `/today`, `/status`, `/help`, ordinary text, Telegram voice messages and audio uploads. It durably stores authorized incoming messages before advancing Telegram's cursor. Other people and groups cannot trigger classification or task changes. Replies to transient rate limits stay queued and respect Telegram's cooldown.

## Voice replies

Voice is transcribed on the receiving Mac using its existing `/opt/homebrew/bin/whisper` and FFmpeg, with the multilingual `small` model at the private state directory's `models/small.pt`. Provision that model before enabling voice; no speech API key is required. The current Mac mini already has these prerequisites.

Messages up to 10 minutes and 20 MB use the same reconciliation workflow as typed replies. The response echoes the recognized words so the user can correct them. Unclear transcription or download/recognition failure does not change tasks. The raw audio is removed after transcription; its text receipt stays outside the vault in private `transcripts/` to avoid repeated recognition on replay. The reconciliation log labels the input as voice transcription. Task classification receives the resulting text, as it does for ordinary typed progress.

## Reconciliation boundary

The read-only Codex classifier sees only active task candidates, completion requirements, the user's statement and bounded recent conversation. Code applies validated changes only to active, Git-ignored canonical task lines; it preserves scheduling dates and block IDs and checks for intervening edits. Partial completion stays partial. Unknown completion dates stay blank. Formal deadlines and external submissions require separate source verification. Progress belonging to tracked source files is logged privately for review, without changing public records.

The private `99_学习情况记录/Telegram 对账记录.md` records what was actually applied and what needs clarification. Morning and evening planning must read this log before choosing work. Receipts recover interrupted writes and deduplicate replayed updates. Automated classification remains a semantic judgment: the confirmation message states which tasks changed so the user can correct a mistaken mapping.

## Verification and recovery

Run `python3 -m unittest discover -s 05_tools/StudentOS/telegram -p 'test_*.py'` from the vault. Offline tests use disposable vaults and mocked network/model calls. Also check one real `/start` response, one explicit notification, and one natural-language progress reply before claiming end-to-end completion.

Inspect `bridge.py ... status`, private `health.json`, or `launchctl print gui/$(id -u)/local.student-os.telegram`. Do not print `config.json` or raw message databases. Stop the receiver using `launchctl bootout gui/$(id -u)/local.student-os.telegram`; the existing heartbeat is managed separately through Codex's automation tool. Changing credentials is a separate user-led setup action.
