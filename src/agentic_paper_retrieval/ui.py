"""Local web server and research sessions."""

import copy
import json
import logging
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from uuid import uuid4

from langgraph.types import Command

from agentic_paper_retrieval import __version__
from agentic_paper_retrieval.agent import ResearchAgent
from agentic_paper_retrieval.library import ChromaPaperLibrary, LibraryError, OllamaEmbedder
from agentic_paper_retrieval.ollama import OllamaPlanner
from agentic_paper_retrieval.sources.arxiv import ArxivSource

TERMINAL = {"complete", "stopped", "error"}
LOG = logging.getLogger(__name__)


class SessionConflict(ValueError):
    """A browser action no longer matches the current review step."""


class ResearchSession:
    def __init__(self, question, model, max_rounds, limit, source, planner, executor, library=None):
        initial = ResearchAgent.initial_state(question, max_rounds=max_rounds, limit=limit)
        self.id = uuid4().hex
        self.model = model
        self.max_rounds = max_rounds
        self.limit = limit
        self.executor = executor
        self.lock = threading.RLock()
        self.state = initial
        self.status = "working"
        self.message = "Improving your research question"
        self.pending = None
        self.error = None
        self.revision = 0
        self.events = []
        self.busy = False
        self.stop_requested = False
        self.library = library
        self.library_save = {"status": "offered", "saved_paper_ids": [], "error": None}
        self.graph = ResearchAgent(source, planner, library).build_graph(on_results=self._on_results)
        self.config = {
            "configurable": {"thread_id": self.id},
            "recursion_limit": 8 * max_rounds + 15,
        }
        self._event("Research started", "Improving wording and finding useful search terms.")

    def _event(self, title, detail=""):
        self.events.append({
            "title": title, "detail": detail,
            "time": datetime.now(timezone.utc).isoformat(),
        })
        self.revision += 1

    def _on_results(self, step, papers):
        with self.lock:
            catalog = {paper["id"]: paper for paper in self.state["papers"]}
            for paper in papers:
                catalog.setdefault(paper.id, paper.to_dict())
            self.state["papers"] = list(catalog.values())
            self.state["steps"] = [*self.state["steps"], asdict(step)]
            origins = self.state.setdefault("paper_origins", {})
            for origin, ids in (("arxiv", step.paper_ids), ("library", step.library_paper_ids)):
                for paper_id in ids:
                    values = origins.setdefault(paper_id, [])
                    if origin not in values:
                        values.append(origin)
            self.message = "Reviewing the retrieved evidence"
            self._event(f"Retrieved {len(papers)} papers", step.executed_query)
            if step.library_error or step.source_error:
                self._event("A search source was unavailable", "; ".join(filter(None, (step.library_error, step.source_error))))

    def start(self):
        with self.lock:
            self.busy = True
            self.executor.submit(self._advance, self.state.copy())

    def _advance(self, value):
        try:
            while True:
                state = self.graph.invoke(value, config=self.config)
                with self.lock:
                    self.state = {key: item for key, item in state.items() if key != "__interrupt__"}
                    interruptions = state.get("__interrupt__", ())
                    if interruptions and self.stop_requested:
                        value = Command(resume={"query": None})
                        continue
                    if interruptions:
                        event = interruptions[0].value
                        self.pending = {"id": uuid4().hex, **event}
                        if event["kind"] == "query_review":
                            self.status = "awaiting_query"
                            self.message = "Your improved question is ready to review"
                            self._event("Question refined", "Review the question, concepts, and suggested searches.")
                        else:
                            self.status = "awaiting_search"
                            self.message = "A search is waiting for your approval"
                            self._event("Search proposed", event["decision"]["query"])
                    else:
                        result = self.state["result"]
                        if self.stop_requested:
                            result["stop_reason"] = "user_stopped"
                            result["rationale"] = "Stopped by the researcher after the current step."
                        self.status = "stopped" if result["stop_reason"] == "user_stopped" else "complete"
                        self.message = "Research stopped" if self.status == "stopped" else "Your reading list is ready"
                        self._event(self.message, result["rationale"])
                    self.busy = False
                    self.revision += 1
                    return
        except Exception as exc:
            LOG.exception("Research session %s failed", self.id)
            with self.lock:
                self.status = "error"
                self.error = str(exc) or "The research step failed. Please try again."
                self.message = "This research step could not finish"
                self.busy = False
                self.pending = None
                self._event("Research paused by an error", self.error)

    def respond(self, pending_id, query):
        with self.lock:
            if self.busy or not self.pending or self.pending["id"] != pending_id:
                raise SessionConflict("This review was already submitted. Refresh the session and try again.")
            if query is not None:
                maximum = 4000 if self.pending["kind"] == "query_review" else 500
                if not isinstance(query, str) or not query.strip() or len(query) > maximum:
                    raise ValueError(f"Enter a question or query of 1–{maximum} characters.")
                query = query.strip()
            kind = self.pending["kind"]
            self.pending = None
            self.busy = True
            self.status = "working"
            self.message = "Preparing your next search" if kind == "query_review" else "Searching the literature"
            title = "Research stopped" if query is None else ("Question reviewed" if kind == "query_review" else "Search approved")
            self._event(title, query or "Stopped by researcher")
            self.executor.submit(self._advance, Command(resume={"query": query}))

    def cancel(self):
        with self.lock:
            if self.status in TERMINAL:
                return
            self.stop_requested = True
            self.message = "Stopping after the current step"
            self._event("Stop requested")
            if self.pending and not self.busy:
                self.respond(self.pending["id"], None)

    def save_papers(self, paper_ids):
        with self.lock:
            if self.status not in TERMINAL or self.library_save["status"] == "saving":
                raise SessionConflict("Wait for the current operation to finish before saving papers.")
            if self.library is None:
                raise LibraryError("The saved library is not configured.")
            if not isinstance(paper_ids, list) or len(paper_ids) > 100 or any(not isinstance(value, str) for value in paper_ids):
                raise ValueError("Choose up to 100 retrieved paper IDs.")
            known = {paper["id"]: paper for paper in self.state["papers"]}
            if any(paper_id not in known for paper_id in paper_ids):
                raise ValueError("Only papers retrieved in this session can be saved.")
            if not paper_ids:
                self.library_save = {"status": "skipped", "saved_paper_ids": [], "error": None}
                self._event("Library save skipped")
                return
            from agentic_paper_retrieval.agent import _papers

            papers = _papers([known[paper_id] for paper_id in dict.fromkeys(paper_ids)])
            question = self.state.get("effective_question", self.state["question"])
            self.library_save = {"status": "saving", "saved_paper_ids": [], "error": None}
            self._event("Saving selected papers")
            self.executor.submit(self._save_papers, papers, question)

    def _save_papers(self, papers, question):
        try:
            result = self.library.save(papers, question)
            with self.lock:
                self.library_save = {"status": "saved", "error": None, **result}
                self._event("Papers saved to your library", f"{result['added_count']} new papers saved for future questions.")
        except Exception as exc:
            LOG.exception("Saving session %s papers failed", self.id)
            with self.lock:
                self.library_save = {"status": "error", "saved_paper_ids": [], "error": str(exc)}
                self._event("Papers could not be saved", str(exc))

    def snapshot(self):
        with self.lock:
            result = self.state.get("result")
            if result is not None:
                result = {**result, "library_save": self.library_save}
            return copy.deepcopy({
                "id": self.id, "revision": self.revision, "status": self.status,
                "message": self.message, "model": self.model,
                "max_rounds": self.max_rounds, "limit": self.limit,
                "question": self.state["question"],
                "refinement": self.state.get("refinement"),
                "effective_question": self.state.get("effective_question"),
                "pending": self.pending, "papers": self.state["papers"],
                "steps": self.state["steps"], "result": result,
                "error": self.error, "events": self.events,
                "paper_origins": self.state.get("paper_origins", {}),
                "library_save": self.library_save,
            })


class UIRuntime:
    def __init__(self, ollama_url="http://localhost:11434", cache_dir=Path("data/cache/arxiv"),
                 planner_factory=None, source_factory=None, library=None,
                 library_dir=Path("data/library"), embedding_model=None):
        self.ollama_url = ollama_url.rstrip("/")
        self.planner_factory = planner_factory or (lambda model: OllamaPlanner(model, self.ollama_url))
        self.source_factory = source_factory or (lambda: ArxivSource(cache_dir=cache_dir))
        self.embedding_model = embedding_model or os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")
        self.library = library if library is not None else ChromaPaperLibrary(
            library_dir, OllamaEmbedder(self.embedding_model, self.ollama_url),
        )
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="paper-research")
        self.sessions = {}
        self.lock = threading.RLock()

    def configuration(self):
        models, error = [], None
        try:
            request = urllib.request.Request(self.ollama_url + "/api/tags")
            with urllib.request.urlopen(request, timeout=3) as response:
                data = json.load(response)
            installed = [item["name"] for item in data.get("models", []) if isinstance(item.get("name"), str)]
            models = [name for name in installed if name.removesuffix(":latest") != self.embedding_model.removesuffix(":latest")]
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            error = "Cannot reach Ollama. Start it with ollama serve, then refresh the connection."
            LOG.debug("Ollama connection check: %s", exc)
        preferred = os.getenv("OLLAMA_MODEL", "qwen3:4b")
        return {
            "version": __version__, "models": models, "ollama_available": error is None,
            "default_model": preferred if preferred in models else (models[0] if models else ""),
            "error": error or (None if models else "No local model found. Run ollama pull qwen3:4b, then refresh."),
            "embedding_model": self.embedding_model,
            "library_note": f"Saved papers use {self.embedding_model} for semantic search. Install it with ollama pull {self.embedding_model}.",
        }

    def create(self, data):
        question, model = data.get("question"), data.get("model")
        if not isinstance(question, str) or not question.strip() or len(question) > 4000:
            raise ValueError("Enter a research question of 1–4000 characters.")
        if not isinstance(model, str) or not model.strip() or len(model) > 200:
            raise ValueError("Choose an installed local model.")
        rounds, limit = data.get("max_rounds", 3), data.get("limit", 5)
        if type(rounds) is not int or type(limit) is not int:
            raise ValueError("Search limits must be integers.")
        ResearchAgent.initial_state(question, max_rounds=rounds, limit=limit)
        with self.lock:
            if sum(item.status not in TERMINAL for item in self.sessions.values()) >= 2:
                raise SessionConflict("Two research sessions are already active. Finish or stop one first.")
            if len(self.sessions) >= 30:
                finished = next((key for key, item in self.sessions.items() if item.status in TERMINAL), None)
                if finished:
                    del self.sessions[finished]
            session = ResearchSession(
                question, model.strip(), rounds, limit, self.source_factory(),
                self.planner_factory(model.strip()), self.executor,
                self.library,
            )
            self.sessions[session.id] = session
            session.start()
            return session

    def get(self, session_id):
        with self.lock:
            if session_id not in self.sessions:
                raise LookupError("This research session is no longer available. Start a new one.")
            return self.sessions[session_id]

    def close(self):
        for session in list(self.sessions.values()):
            session.cancel()
        self.executor.shutdown(wait=True)


class UIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, runtime):
        self.runtime = runtime
        super().__init__(address, UIHandler)


class UIHandler(BaseHTTPRequestHandler):
    server_version = "PaperRetriever"

    def log_message(self, format, *args):
        LOG.debug(format, *args)

    def _send(self, status, body, content_type="application/json; charset=utf-8", download=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False, indent=2).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(body)

    def _local_request(self):
        port = self.server.server_address[1]
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in allowed:
            self._send(403, {"error": "Use the localhost URL printed by the server."})
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{host}" for host in allowed}:
            self._send(403, {"error": "This action must come from the local research page."})
            return False
        return True

    def do_GET(self):
        if not self._local_request():
            return
        path = urllib.parse.urlsplit(self.path).path
        if path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        assets = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/styles.css": "styles.css"}
        if path in assets:
            name = assets[path]
            mime = {"html": "text/html", "js": "text/javascript", "css": "text/css"}[name.rsplit(".", 1)[1]]
            return self._send(200, files("agentic_paper_retrieval").joinpath("static", name).read_bytes(), mime + "; charset=utf-8")
        if path == "/api/config":
            return self._send(200, self.server.runtime.configuration())
        if path == "/api/library":
            try:
                params = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                limit, offset = int(params.get("limit", ["50"])[0]), int(params.get("offset", ["0"])[0])
                return self._send(200, self.server.runtime.library.list_papers(limit, offset))
            except ValueError as exc:
                return self._send(400, {"error": str(exc)})
            except LibraryError as exc:
                return self._send(503, {"error": str(exc)})
        match = re.fullmatch(r"/api/sessions/([a-f0-9]{32})(/export)?", path)
        if match:
            try:
                snapshot = self.server.runtime.get(match[1]).snapshot()
            except LookupError as exc:
                return self._send(404, {"error": str(exc)})
            if match[2]:
                return self._send(200, snapshot["result"] or snapshot, download=f"paper-research-{match[1][:8]}.json")
            return self._send(200, snapshot)
        self._send(404, {"error": "Page not found."})

    def do_POST(self):
        if not self._local_request():
            return
        if self.headers.get_content_type() != "application/json":
            return self._send(415, {"error": "Send this action as JSON."})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 16384:
                return self._send(413, {"error": "The request is empty or too large."})
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("The request must be a JSON object.")
            path = urllib.parse.urlsplit(self.path).path
            if path == "/api/sessions":
                return self._send(202, self.server.runtime.create(data).snapshot())
            if path == "/api/library/remove":
                self.server.runtime.library.remove(data.get("paper_id"))
                return self._send(200, {"removed": True})
            match = re.fullmatch(r"/api/sessions/([a-f0-9]{32})/(respond|cancel|save)", path)
            if not match:
                return self._send(404, {"error": "Action not found."})
            session = self.server.runtime.get(match[1])
            if match[2] == "cancel":
                session.cancel()
            elif match[2] == "save":
                session.save_papers(data.get("paper_ids"))
            else:
                if "query" not in data or not isinstance(data.get("pending_id"), str):
                    raise ValueError("Provide the current review ID and an approved query.")
                session.respond(data["pending_id"], data["query"])
            self._send(202, session.snapshot())
        except SessionConflict as exc:
            self._send(409, {"error": str(exc)})
        except LookupError as exc:
            self._send(404, {"error": str(exc)})
        except (ValueError, UnicodeDecodeError) as exc:
            self._send(400, {"error": str(exc)})
        except LibraryError as exc:
            self._send(503, {"error": str(exc)})


def serve(port=8000, ollama_url="http://localhost:11434", cache_dir=Path("data/cache/arxiv"),
          library_dir=Path("data/library"), embedding_model=None):
    runtime = UIRuntime(ollama_url, cache_dir, library_dir=library_dir, embedding_model=embedding_model)
    try:
        server = UIServer(("127.0.0.1", port), runtime)
    except OSError:
        runtime.close()
        raise
    print(f"Paper Retriever is ready at http://127.0.0.1:{server.server_address[1]}", flush=True)
    print("Press Ctrl+C to stop. Research sessions stay in memory.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the local UI.", flush=True)
    finally:
        server.server_close()
        runtime.close()
