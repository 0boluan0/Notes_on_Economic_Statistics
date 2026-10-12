"""Bounded, no-tool preparation turns over the local App Server stdio API."""

from collections import deque
import json
import os
from pathlib import Path
import select
import subprocess
import time


class Cancelled(RuntimeError):
    pass


class Client:
    def __init__(self, binary, codex_home, vault, log, cancelled=lambda: False):
        self.binary, self.home, self.vault, self.log = binary, codex_home, vault, log
        self.cancelled = cancelled
        self.sequence, self.buffer, self.events = 0, b"", deque()

    def __enter__(self):
        self.err = Path(self.log).open("ab")
        try:
            self.proc = subprocess.Popen(
                [str(self.binary), "app-server", "--listen", "stdio://"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err,
                env={**os.environ, "CODEX_HOME": str(self.home)}, cwd=self.vault,
            )
            self.request("initialize", {"clientInfo": {"name": "student_os_steward", "version": "0.3.0"},
                                       "capabilities": {"experimentalApi": True}})
            self.send({"method": "initialized"})
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        try:
            if hasattr(self, "proc"):
                try:
                    self.proc.stdin.close()
                except (OSError, BrokenPipeError):
                    pass
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                        self.proc.wait(timeout=5)
                self.proc.stdout.close()
        finally:
            self.err.close()

    def send(self, message):
        self.proc.stdin.write((json.dumps(message) + "\n").encode())
        self.proc.stdin.flush()

    def receive(self, timeout=90):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            if self.cancelled():
                raise Cancelled("Preparation was cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("App Server response timed out")
            if not select.select([self.proc.stdout], [], [], min(remaining, 0.5))[0]:
                continue
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("App Server closed stdout")
            self.buffer += chunk
            if len(self.buffer) > 16 * 1024 * 1024:
                raise RuntimeError("App Server message exceeded the bounded preparation limit")
        line, self.buffer = self.buffer.split(b"\n", 1)
        message = json.loads(line)
        if "method" in message and "id" in message:
            self.send({"id": message["id"], "error": {"code": -32601, "message": "This planner and preparation client does not execute tools"}})
            raise RuntimeError("Unexpected server request: " + message["method"])
        return message

    def request(self, method, params):
        self.sequence += 1
        rid = self.sequence
        self.send({"id": rid, "method": method, "params": params})
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            message = self.receive(max(0.01, deadline - time.monotonic()))
            if message.get("id") == rid:
                if "error" in message:
                    raise RuntimeError(json.dumps(message["error"], ensure_ascii=False))
                return message["result"]
            self.events.append(message)
        raise TimeoutError("App Server request timed out")

    def create(self, model, instructions=None):
        params = {"cwd": str(self.vault), "model": model,
            "sandbox": "read-only", "approvalPolicy": "on-request", "ephemeral": False}
        if instructions:
            params["developerInstructions"] = instructions
        return self.request("thread/start", params)["thread"]["id"]

    def read(self, thread_id):
        # ponytail: recovery only reads a single preparation turn; paginate before supporting long classroom histories.
        return self.request("thread/read", {"threadId": thread_id, "includeTurns": True})["thread"]

    def resume(self, thread_id):
        thread = self.request("thread/resume", {"threadId": thread_id, "sandbox": "read-only", "approvalPolicy": "on-request"})["thread"]
        if thread["id"] != thread_id:
            raise RuntimeError("Resume changed the native thread ID")

    def start(self, thread_id, prompt, output_schema=None):
        params = {"threadId": thread_id,
            "input": [{"type": "text", "text": prompt}], "effort": "low"}
        if output_schema:
            params["outputSchema"] = output_schema
        return self.request("turn/start", params)["turn"]["id"]

    def finish(self, thread_id, turn_id):
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            message = self.events.popleft() if self.events else self.receive(min(90, deadline - time.monotonic()))
            params = message.get("params", {})
            if params.get("threadId") != thread_id:
                continue
            if message.get("method") == "item/started" and params.get("item", {}).get("type") not in {"userMessage", "agentMessage", "reasoning"}:
                raise RuntimeError("Preparation attempted a tool; review before continuing")
            if message.get("method") == "turn/completed" and params.get("turn", {}).get("id") == turn_id:
                if params["turn"]["status"] != "completed":
                    raise RuntimeError("Preparation turn did not complete")
                return self.read(thread_id)
        raise TimeoutError("Preparation turn timed out")
