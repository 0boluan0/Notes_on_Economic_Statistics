"""Local Student OS durable classroom preparation and registration."""

import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from app_server import Cancelled, Client
from state import digest, snapshot

DEFAULT_ROOT = Path.home() / "Library/Application Support/StudentOSLearning"
DEFAULT_BINARY = Path("/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex")


def now():
    return dt.datetime.now(ZoneInfo("Europe/London")).isoformat()


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class Store:
    def __init__(self, root, vault, codex_home):
        self.root = Path(root).resolve()
        vault, codex_home = Path(vault).resolve(), Path(codex_home).resolve()
        self.vault, self.codex_home = vault, codex_home
        if self.root == vault or vault in self.root.parents:
            raise ValueError("Runtime state must stay outside the vault")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.db = sqlite3.connect(self.root / "state.sqlite3", timeout=10)
        os.chmod(self.root / "state.sqlite3", 0o600)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs (
                key TEXT PRIMARY KEY, request TEXT NOT NULL, state TEXT NOT NULL,
                thread_id TEXT, turn_id TEXT, turn_attempted INTEGER NOT NULL DEFAULT 0,
                result TEXT, conversation_id TEXT, error TEXT, created_at TEXT, updated_at TEXT);
            CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, job_key TEXT, at TEXT, state TEXT, detail TEXT);
        """)
        binding = packed({"vault": str(vault), "codex_home": str(codex_home)})
        with self.db:
            existing = self.db.execute("SELECT value FROM kv WHERE key='binding'").fetchone()
            if existing and existing[0] != binding:
                raise ValueError("This state directory belongs to another vault or Codex home")
            self.db.execute("INSERT OR IGNORE INTO kv VALUES ('binding', ?)", (binding,))

    def close(self):
        self.db.close()

    def get(self, key):
        row = self.db.execute("SELECT * FROM jobs WHERE key=?", (key,)).fetchone()
        if row is None:
            raise ValueError("Unknown operation ID")
        result = dict(row)
        result["request"] = json.loads(result["request"])
        return result

    def enqueue(self, key, request):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", key):
            raise ValueError("Operation IDs must use letters, numbers, hyphens or underscores")
        base = {"scope", "title", "prompt", "model"}
        if request.get("scope") == "synthetic":
            valid = set(request) == base
        else:
            valid = request.get("scope") == "learning" and set(request) == base | {"task", "materials"}
            if valid:
                task = request["task"]
                valid = isinstance(task, dict) and all(task.get(k) for k in ("id", "path", "version", "raw"))
        if not valid:
            raise ValueError("A preparation needs an explicit synthetic scope or a versioned learning task")
        for field, limit in (("title", 120), ("prompt", 100000), ("model", 100)):
            if not isinstance(request[field], str) or not request[field].strip() or len(request[field]) > limit:
                raise ValueError("Invalid " + field)
        value = packed(request)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO jobs (key,request,state,created_at,updated_at) VALUES (?,?,'queued',?,?)", (key, value, now(), now()))
            if self.db.execute("SELECT request FROM jobs WHERE key=?", (key,)).fetchone()[0] != value:
                raise ValueError("Operation ID already belongs to different content")
        return self.get(key)

    def update(self, key, state, **fields):
        if not set(fields) <= {"thread_id", "turn_id", "turn_attempted", "result", "conversation_id", "error"}:
            raise ValueError("Unknown stored field")
        fields.update(state=state, updated_at=now())
        with self.db:
            changed = self.db.execute("UPDATE jobs SET " + ",".join(f"{name}=?" for name in fields)
                + " WHERE key=? AND state != 'cancelled'", (*fields.values(), key)).rowcount
            if changed:
                self.db.execute("INSERT INTO events(job_key,at,state,detail) VALUES (?,?,?,?)", (key, now(), state, packed(fields)))
            else:
                self.db.execute("INSERT INTO events(job_key,at,state,detail) VALUES (?,?,?,?)", (key, now(), "late_result_ignored", packed(fields)))
        if not changed:
            raise Cancelled("The request is cancelled; late results were discarded")
        return self.get(key)

    def cancel(self, key):
        previous = self.get(key)
        with self.db:
            self.db.execute("UPDATE jobs SET state='cancelled',updated_at=? WHERE key=?", (now(), key))
            self.db.execute("INSERT INTO events(job_key,at,state,detail) VALUES (?,?,?,?)", (key, now(), "cancelled", packed({"previous_state": previous["state"], "existing_conversation_kept": True})))
        return self.get(key)

    def save_snapshot(self, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO kv VALUES ('snapshot', ?)", (packed(value),))

    @contextmanager
    def worker(self):
        with (self.root / "worker.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another preparation worker owns this state directory")
            yield


def completed_turn(thread, expected_id=None):
    turns = thread.get("turns", [])
    if len(turns) != 1 or (expected_id and turns[0]["id"] != expected_id):
        raise RuntimeError("Recovery needs exactly the original preparation turn; no new turn was sent")
    turn = turns[0]
    if turn["status"] != "completed":
        raise RuntimeError("The original preparation turn is not complete; inspect it before retrying")
    if any(item.get("type") not in {"userMessage", "agentMessage", "reasoning"} for item in turn.get("items", [])):
        raise RuntimeError("Preparation history contains tools; inspect before marking prepared")
    text = "\n".join(item.get("text", "") for item in turn.get("items", []) if item.get("type") == "agentMessage")
    if not text.strip():
        raise RuntimeError("Completed turn has no readable preparation result")
    return turn["id"], text


def prepare(store, key, client_factory):
    with store.worker():
        job = store.get(key)
        if job["state"] in {"cancelled", "prepared", "registered", "linking"}:
            return job
        if not job["thread_id"] and job["state"] != "queued":
            store.update(key, "uncertain", error="Creation outcome is unknown; reconcile the original request before creating anything")
            return store.get(key)
        try:
            with client_factory(lambda: store.get(key)["state"] == "cancelled") as client:
                if not job["thread_id"]:
                    store.update(key, "creating", error=None)
                    thread_id = client.create(job["request"]["model"])
                    job = store.update(key, "preparing", thread_id=thread_id)
                else:
                    client.resume(job["thread_id"])
                if not job["result"]:
                    if job["turn_attempted"]:
                        turn_id, text = completed_turn(client.read(job["thread_id"]), job["turn_id"])
                    else:
                        store.update(key, "preparing", turn_attempted=1)
                        scope = "This is a synthetic acceptance classroom. " if job["request"]["scope"] == "synthetic" else "Prepare this source-grounded learning classroom. "
                        prompt = (scope + "Use only the supplied materials. Do not use tools, read or write files, create tasks, or claim learning completion.\n"
                                  + f"Operation ID: {key}\n\n" + job["request"]["prompt"])
                        turn_id = client.start(job["thread_id"], prompt)
                        store.update(key, "preparing", turn_id=turn_id)
                        turn_id, text = completed_turn(client.finish(job["thread_id"], turn_id), turn_id)
                    store.update(key, "result_saved", turn_id=turn_id, result=text, error=None)
            return store.update(key, "prepared", error=None)
        except Cancelled:
            return store.get(key)
        except Exception as error:
            if store.get(key)["state"] != "cancelled":
                store.update(key, "uncertain", error=str(error))
            return store.get(key)


def register_claudian(vault, thread_id, title, model):
    payload = packed({"threadId": thread_id, "title": title, "model": model})
    # Use the plugin's API, never hand-edit Obsidian settings or session metadata.
    code = """(async()=>{
      const p=PAYLOAD;
      const plugin=app.plugins.plugins.realclaudian;
      const h=plugin?.chatHost;
      if(!h?.createConversation || !h?.getConversationById) throw Error('Claudian adapter unavailable');
      await h.ensureConversationMetadataLoaded?.([p.threadId]);
      if(!h.getConversationList().some(c=>c.id===p.threadId)) {
        await h.createConversation({providerId:'codex',sessionId:p.threadId,selectedModel:'openai-codex/'+p.model});
      }
      const c=await h.getConversationById(p.threadId);
      if(c?.sessionId!==p.threadId || !c.messages?.length) throw Error('Native session or history mismatch');
      await h.renameConversation(c.id,p.title);
      return JSON.stringify({conversation_id:c.id,thread_id:c.sessionId,message_count:c.messages.length,plugin_version:plugin.manifest.version});
    })()""".replace("PAYLOAD", payload)
    subprocess.run(["obsidian", "eval", "code=" + code], cwd=vault, capture_output=True, text=True, timeout=30)
    readback = """(()=>{
      const p=PAYLOAD, plugin=app.plugins.plugins.realclaudian;
      const c=plugin?.chatHost?.getConversationSync(p.threadId);
      return JSON.stringify(c?{conversation_id:c.id,thread_id:c.sessionId,title:c.title,message_count:c.messages?.length||0,plugin_version:plugin.manifest.version}:null);
    })()""".replace("PAYLOAD", payload)
    # The CLI can return no stdout after an asynchronous mutation that actually succeeded.
    for attempt in range(3):
        result = subprocess.run(["obsidian", "eval", "code=" + readback], cwd=vault, capture_output=True, text=True, timeout=10)
        lines = [line[3:] for line in result.stdout.splitlines() if line.startswith("=> ")]
        if result.returncode == 0 and lines:
            reply = json.loads(lines[-1])
            if reply and reply.get("thread_id") == thread_id and reply.get("title") == title and reply.get("message_count", 0) >= 2:
                return reply
        if attempt < 2:
            time.sleep(0.1)
    raise RuntimeError("Claudian registration not confirmed by independent readback; retry the same operation ID")


def attach(store, key, vault, register=register_claudian):
    with store.worker():
        job = store.get(key)
        if job["state"] in {"registered", "cancelled"}:
            return job
        if job["state"] not in {"prepared", "linking"}:
            raise ValueError("Only a completed preparation can be registered")
        store.update(key, "linking", error=None)
        try:
            reply = register(vault, job["thread_id"], job["request"]["title"], job["request"]["model"])
            return store.update(key, "registered", conversation_id=reply["conversation_id"], error=None)
        except Cancelled:
            return store.get(key)
        except Exception as error:
            if store.get(key)["state"] != "cancelled":
                store.update(key, "linking", error=str(error))
            return store.get(key)


def public_job(job):
    return {name: job[name] for name in ("key", "state", "thread_id", "turn_id", "conversation_id", "error", "created_at", "updated_at")}


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=DEFAULT_ROOT / "steward")
    parser.add_argument("--codex-home", type=Path, default=DEFAULT_ROOT / "codex")
    parser.add_argument("--codex", type=Path, default=DEFAULT_BINARY)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("snapshot")
    commands.add_parser("status")
    enqueue = commands.add_parser("enqueue")
    enqueue.add_argument("--key", required=True)
    enqueue.add_argument("--request", type=Path, required=True)
    for name in ("run", "attach", "cancel"):
        commands.add_parser(name).add_argument("--key", required=True)
    args = parser.parse_args()
    store = None
    try:
        store = Store(args.state, args.vault, args.codex_home)
        if args.command == "snapshot":
            value = snapshot(args.vault)
            store.save_snapshot(value)
            output = {"status": value["status"], "captured_at": value["captured_at"], "task_count": len(value["tasks"]),
                "obligation_count": len(value["obligations"]), "issues": value["issues"], "not_checked": value["not_checked"]}
        elif args.command == "status":
            output = {"jobs": [public_job(store.get(row[0])) for row in store.db.execute("SELECT key FROM jobs ORDER BY created_at")]}
        elif args.command == "enqueue":
            output = public_job(store.enqueue(args.key, json.loads(args.request.read_text(encoding="utf-8"))))
        elif args.command == "run":
            factory = lambda cancelled: Client(args.codex, args.codex_home, args.vault, store.root / "app-server.stderr.log", cancelled)
            output = public_job(prepare(store, args.key, factory))
        elif args.command == "attach":
            output = public_job(attach(store, args.key, args.vault))
        else:
            output = public_job(store.cancel(args.key))
        print(packed(output))
        return 1 if output.get("error") else 0
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        print(packed({"error": str(error)}), file=sys.stderr)
        return 1
    finally:
        if store:
            store.close()


if __name__ == "__main__":
    sys.exit(main())
