"use strict";

const $ = (id) => document.getElementById(id);
let session = null;
let configuration = null;
let pendingKey = null;
let posting = false;
let polling = false;
let paperView = "shortlist";
let renderedPapersKey = null;
const expandedAbstracts = new Set();
const seenPaperIds = new Set();
let savePanelKey = null;
let saveSelectionSession = null;
const papersToSave = new Set();
let libraryOffset = 0;
let libraryLoading = false;
const terminal = (value) => ["complete", "stopped", "error"].includes(value?.status);

function element(tag, options = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(options)) {
    if (key === "text") node.textContent = value;
    else if (key === "className") node.className = value;
    else node.setAttribute(key, value);
  }
  node.append(...children);
  return node;
}

async function api(path, data) {
  let response;
  try {
    response = await fetch(path, data === undefined ? { cache: "no-store" } : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
    });
  } catch {
    throw new Error("The local server is not responding. Check that paper-retriever ui is still running.");
  }
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "This action could not finish. Please try again.");
  return body;
}

function showNotice(message) {
  $("notice").textContent = message || "";
  $("notice").hidden = !message;
}

function updateQuestionMeta() {
  const question = $("question").value;
  $("question-count").textContent = `${question.length.toLocaleString("en-US")} / 4,000`;
  $("clear-question").disabled = Boolean(posting || (session && !terminal(session)) || !question.length);
  document.querySelectorAll(".example").forEach((button) => {
    const picked = question.trim() === button.dataset.question;
    button.classList.toggle("picked", picked);
    button.setAttribute("aria-pressed", String(picked));
  });
}

function controls() {
  const active = session && !terminal(session);
  for (const input of $("research-form").querySelectorAll("textarea, select, button")) {
    input.disabled = Boolean(active || posting);
  }
  $("start").disabled = Boolean(active || posting || !configuration?.models?.length || !$("question").value.trim());
  $("start").firstElementChild.textContent = active ? "Research in progress" : session ? "Start new research" : "Improve my question";
  $("stop").hidden = !active;
  $("stop").disabled = posting || session?.message?.startsWith("Stopping");
  $("export").disabled = !session?.refinement && !session?.papers?.length;
  for (const input of $("review-panel").querySelectorAll("button, textarea, input")) input.disabled = posting;
  updateQuestionMeta();
}

async function refreshConnection() {
  $("connection").disabled = true;
  $("connection-text").textContent = "Checking model…";
  try {
    configuration = await api("/api/config");
    const previous = $("model").value;
    $("model").replaceChildren();
    for (const name of configuration.models) $("model").append(element("option", { value: name, text: name }));
    if (!configuration.models.length) $("model").append(element("option", { value: "", text: "No local model available" }));
    $("model").value = configuration.models.includes(previous) ? previous : configuration.default_model;
    const online = configuration.ollama_available && configuration.models.length > 0;
    $("connection").classList.toggle("online", online);
    $("connection-text").textContent = online ? "Ollama connected" : "Model unavailable";
    showNotice(configuration.error);
  } catch (error) {
    $("connection-text").textContent = "Server unavailable";
    $("connection").classList.remove("online");
    showNotice(error.message);
  } finally {
    $("connection").disabled = false;
    controls();
  }
}

function queryList(queries) {
  return element("ol", { className: "query-list" }, queries.map((query, index) =>
    element("li", {}, [element("span", { text: `0${index + 1}` }), element("div", { text: query })])));
}

function clarificationBox(questions, title = "A little more context?") {
  return element("div", { className: "clarifications" }, [
    element("strong", { text: title }),
    element("ul", {}, questions.map((question) => element("li", { text: question }))),
  ]);
}

async function sendReview(query) {
  if (posting || !session?.pending) return;
  posting = true;
  controls();
  showNotice("");
  try {
    applySession(await api(`/api/sessions/${session.id}/respond`, { pending_id: session.pending.id, query }));
  } catch (error) {
    showNotice(error.message);
  } finally {
    posting = false;
    controls();
  }
}

function renderReview(data) {
  const key = data.pending?.id || `${data.status}:${data.steps.length}`;
  if (key === pendingKey) return;
  pendingKey = key;
  const root = $("review-panel");
  root.replaceChildren();
  if (terminal(data)) return;
  if (!data.pending) {
    root.append(element("div", { className: "card review-card" }, [
      element("p", { className: "eyebrow", text: "A THOUGHTFUL START" }),
      element("h2", { text: data.steps.length ? "Following the evidence." : "Finding the right words." }),
      element("p", { className: "review-description", text: data.steps.length
        ? "Reviewing the papers to decide what would be useful to explore next."
        : "Your local model is clarifying the question and finding useful search terms. You’ll review the plan before any search." }),
    ]));
    return;
  }
  const isQuestion = data.pending.kind === "query_review";
  const plan = data.refinement;
  const card = element("form", { className: "card review-card" });
  const heading = element("h2", { text: isQuestion ? "Make the question yours." : "Ready to explore the literature?", tabindex: "-1" });
  card.append(
    element("p", { className: "eyebrow", text: isQuestion ? "01 / REVIEW YOUR QUESTION" : `02 / SEARCH ${data.steps.length + 1} OF ${data.max_rounds}` }),
    heading,
    element("p", { className: "review-description", text: isQuestion
      ? "Review the wording below. Add any scope or clarification you want the search to reflect."
      : "Approve these keywords, or adjust them before searching arXiv." }),
  );
  const input = element(isQuestion ? "textarea" : "input", {
    id: "review-input", required: "", maxlength: isQuestion ? "4000" : "500",
    "aria-label": isQuestion ? "Improved research question" : "Proposed search query",
  });
  input.value = isQuestion ? plan.refined_question : data.pending.decision.query;
  card.append(element("label", { for: "review-input", text: isQuestion ? "Improved research question" : "Search keywords" }), input);
  if (isQuestion) {
    card.append(element("div", { className: "concepts" }, plan.concepts.map((concept) => element("span", { className: "concept", text: concept }))));
    card.append(element("label", { text: "Suggested search directions" }), queryList(plan.search_queries));
    if (plan.assumptions.length) card.append(clarificationBox(plan.assumptions, "Assumptions to review"));
    if (plan.clarification_questions.length) card.append(clarificationBox(plan.clarification_questions));
  } else card.append(element("p", { className: "search-reason", text: data.pending.decision.reason }));
  const approve = element("button", { className: "button primary", type: "submit", text: isQuestion ? "Approve question →" : "Approve & search →" });
  input.addEventListener("input", () => {
    if (isQuestion) approve.textContent = input.value.trim() === plan.refined_question ? "Approve question →" : "Update question & continue →";
  });
  card.append(element("div", { className: "review-actions" }, [
    element("small", { text: isQuestion ? "An edited question gets a fresh search plan." : "Results include abstracts and links to the original papers." }), approve,
  ]));
  card.addEventListener("submit", (event) => { event.preventDefault(); if (card.reportValidity()) sendReview(input.value.trim()); });
  root.append(card);
  heading.focus({ preventScroll: true });
}

function renderOverview(data) {
  const root = $("overview");
  const wasOpen = root.querySelector("details")?.open;
  root.replaceChildren();
  if (!data.refinement || data.status === "awaiting_query") return;
  const details = element("details", { className: "overview-card" }, [
    element("summary", { text: "Your research question & search plan" }),
    element("p", { className: "overview-question", text: data.effective_question || data.refinement.refined_question }),
    queryList(data.refinement.search_queries),
  ]);
  details.open = Boolean(wasOpen);
  root.append(details);
}

function safeURL(value) {
  try { const url = new URL(value); return ["https:", "http:"].includes(url.protocol) ? url.href : null; }
  catch { return null; }
}

function paperLink(label, url) {
  const href = safeURL(url);
  return href ? element("a", { text: label, href, target: "_blank", rel: "noopener noreferrer" }) : element("span", { text: label });
}

function renderPapers(data) {
  const papers = data.result?.papers || data.papers;
  const ids = data.result?.selected_paper_ids || [];
  const done = terminal(data);
  $("papers-section").hidden = !papers.length && !done;
  $("papers-title").textContent = done ? (ids.length ? "Your reading list" : "Retrieved papers") : "Papers so far";
  $("papers-summary").textContent = `${papers.length} ${papers.length === 1 ? "paper" : "papers"} collected across ${data.steps.length} ${data.steps.length === 1 ? "search" : "searches"}.`;
  $("paper-tabs").hidden = !ids.length;
  $("tab-shortlist").textContent = `Shortlist ${ids.length}`;
  $("tab-all").textContent = `All papers ${papers.length}`;
  $("tab-shortlist").setAttribute("aria-pressed", String(paperView === "shortlist"));
  $("tab-all").setAttribute("aria-pressed", String(paperView === "all"));
  const byId = new Map(papers.map((paper) => [paper.id, paper]));
  const visible = ids.length && paperView === "shortlist" ? ids.map((id) => byId.get(id)).filter(Boolean) : papers;
  // Rebuilding unchanged cards would reset keyboard focus.
  const origins = data.result?.paper_origins || data.paper_origins || {};
  const renderKey = JSON.stringify([visible, ids, origins, paperView, done, done ? data.status : null]);
  if (renderKey === renderedPapersKey) return;
  renderedPapersKey = renderKey;
  const list = $("paper-list");
  list.replaceChildren();
  if (!visible.length && done) {
    const title = data.status === "stopped" ? "Research paused here." : data.status === "error" ? "Let’s try that step again." : "A broader question may help.";
    const description = data.status === "stopped" ? "The session stopped before papers were collected. Start a new question whenever you’re ready."
      : data.status === "error" ? "Review the message above, check your model connection, then start a new session."
      : "Try fewer concepts or a different phrase, then start a new research session.";
    list.append(element("div", { className: "card no-papers" }, [element("h3", { text: title }), element("p", { text: description })]));
  }
  visible.forEach((paper, index) => {
    const meta = element("div", { className: "paper-meta" }, [
      element("span", { text: String(index + 1).padStart(2, "0") }),
      element("span", { className: "paper-source", text: (paper.source || "arxiv").toUpperCase() }),
      element("span", { text: (paper.published || "").slice(0, 4) }),
    ]);
    if (ids.includes(paper.id)) meta.append(element("span", { className: "paper-suggested", text: "Suggested" }));
    if (origins[paper.id]?.includes("library")) meta.append(element("span", { className: "paper-origin", text: "From your library" }));
    if (origins[paper.id]?.includes("arxiv")) meta.append(element("span", { className: "paper-origin", text: "Found on arXiv" }));
    const authors = paper.authors || [];
    const abstract = element("p", { className: `paper-abstract${expandedAbstracts.has(paper.id) ? " expanded" : ""}`, text: paper.abstract || "No abstract available." });
    const toggle = element("button", { type: "button", text: expandedAbstracts.has(paper.id) ? "Less detail ↑" : "Read abstract ↓", "aria-expanded": String(expandedAbstracts.has(paper.id)) });
    toggle.addEventListener("click", () => {
      if (expandedAbstracts.has(paper.id)) expandedAbstracts.delete(paper.id); else expandedAbstracts.add(paper.id);
      const open = expandedAbstracts.has(paper.id);
      abstract.classList.toggle("expanded", open);
      toggle.textContent = open ? "Less detail ↑" : "Read abstract ↓";
      toggle.setAttribute("aria-expanded", String(open));
    });
    const links = element("div", { className: "paper-actions" }, [toggle, paperLink("View paper ↗", paper.url)]);
    if (paper.pdf_url) links.append(paperLink("PDF ↗", paper.pdf_url));
    const isNew = !seenPaperIds.has(paper.id);
    const card = element("article", { className: `card paper-card${isNew ? " paper-new" : ""}` }, [
      meta, element("h3", {}, [paperLink(paper.title, paper.url)]),
      element("p", { className: "paper-authors", text: authors.slice(0, 3).join(", ") + (authors.length > 3 ? ` +${authors.length - 3} authors` : "") }),
      abstract, links,
    ]);
    seenPaperIds.add(paper.id);
    card.addEventListener("animationend", () => card.classList.remove("paper-new"), { once: true });
    list.append(card);
  });
}

async function loadLibrary() {
  if (libraryLoading) return;
  libraryLoading = true;
  $("library-status").textContent = "Loading saved papers…";
  $("library-prev").disabled = $("library-next").disabled = true;
  try {
    const data = await api(`/api/library?limit=10&offset=${libraryOffset}`);
    if (!data.papers.length && libraryOffset > 0) {
      libraryOffset = Math.max(0, libraryOffset - 10);
      libraryLoading = false;
      return loadLibrary();
    }
    $("library-status").textContent = data.total
      ? `${libraryOffset + 1}–${libraryOffset + data.papers.length} of ${data.total} saved papers`
      : "No papers saved yet. You can save papers after a search finishes.";
    $("library-list").replaceChildren(...data.papers.map((paper) => {
      const remove = element("button", { type: "button", className: "button quiet small", text: "Remove from library", "aria-label": `Remove ${paper.title} from library` });
      remove.addEventListener("click", async () => {
        remove.disabled = true;
        try { await api("/api/library/remove", { paper_id: paper.id }); await loadLibrary(); }
        catch (error) { showNotice(error.message); remove.disabled = false; }
      });
      return element("article", { className: "library-paper" }, [
        element("h3", {}, [paperLink(paper.title, paper.url)]),
        element("p", { text: `Published ${(paper.published || "").slice(0, 10)}` }),
        element("p", { className: "library-question", text: `Saved for: ${paper.saved_from_question}` }), remove,
      ]);
    }));
    $("library-prev").disabled = libraryOffset === 0;
    $("library-next").disabled = libraryOffset + data.papers.length >= data.total;
  } catch (error) {
    $("library-status").textContent = error.message;
  } finally { libraryLoading = false; }
}

async function savePapers(skip = false) {
  if (posting || !session || session.library_save?.status === "saving") return;
  const id = session.id;
  posting = true;
  controls();
  $("save-panel").querySelectorAll("button, input").forEach((input) => { input.disabled = true; });
  try {
    const data = await api(`/api/sessions/${id}/save`, { paper_ids: skip ? [] : [...papersToSave] });
    if (session?.id === id) applySession(data);
  } catch (error) { showNotice(error.message); }
  finally {
    posting = false;
    savePanelKey = null;
    if (session) renderSave(session);
    controls();
  }
}

function renderSave(data) {
  const root = $("save-panel");
  const papers = data.result?.papers || data.papers;
  root.hidden = !terminal(data) || !papers.length;
  if (root.hidden) return;
  const save = data.library_save || { status: "offered" };
  const key = `${data.id}:${save.status}`;
  if (key === savePanelKey) return;
  savePanelKey = key;
  root.replaceChildren();
  if (save.status === "saved" || save.status === "skipped") {
    root.append(element("h2", { text: save.status === "saved" ? "Saved for your next question." : "No papers saved." }),
      element("p", { text: save.status === "saved"
        ? `${save.saved_paper_ids.length} papers are in your library. Future searches will look for relevant matches.`
        : "Your results remain available in this session. You can still export them as JSON." }));
    if (save.status === "saved" && !$("library-panel").hidden) loadLibrary();
    return;
  }
  root.append(element("p", { className: "eyebrow", text: "KEEP THE USEFUL PAPERS" }),
    element("h2", { text: "Save papers for future research?" }),
    element("p", { text: "Choose papers to keep in your local library. Each new question searches their titles and abstracts for relevant matches." }));
  if (save.status === "saving") {
    root.append(element("p", { role: "status", text: "Saving your selected papers…" }));
    return;
  }
  if (saveSelectionSession !== data.id) {
    saveSelectionSession = data.id;
    papersToSave.clear();
    (data.result?.selected_paper_ids || []).forEach((id) => papersToSave.add(id));
  }
  if (save.error) root.append(element("p", { className: "notice", role: "alert", text: save.error }));
  const saveButton = element("button", { type: "button", className: "button primary" });
  const updateButton = () => {
    saveButton.textContent = `Save selected (${papersToSave.size})`;
    saveButton.disabled = !papersToSave.size || papersToSave.size > 100 || posting;
  };
  const choices = element("div", { className: "save-choices" }, papers.map((paper) => {
    const checkbox = element("input", { type: "checkbox", value: paper.id });
    checkbox.checked = papersToSave.has(paper.id);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) papersToSave.add(paper.id); else papersToSave.delete(paper.id);
      updateButton();
    });
    return element("label", { className: "save-choice" }, [checkbox, element("span", { text: paper.title })]);
  }));
  root.append(element("details", {}, [element("summary", { text: `Choose from ${papers.length} retrieved papers (up to 100). Shortlist preselected.` }), choices]));
  const skipButton = element("button", { type: "button", className: "button quiet", text: "Skip saving" });
  skipButton.addEventListener("click", () => savePapers(true));
  saveButton.addEventListener("click", () => savePapers());
  root.append(element("div", { className: "save-actions" }, [saveButton, skipButton]));
  updateButton();
}

function renderTrail(data) {
  $("trail-card").hidden = false;
  $("trail-count").textContent = `${data.steps.length}/${data.max_rounds} SEARCHES`;
  $("trail").replaceChildren(...data.events.slice(-10).map((event) => element("li", {}, [
    element("time", { datetime: event.time, text: new Date(event.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) }),
    element("span", { text: event.title }), ...(event.detail ? [element("p", { text: event.detail })] : []),
  ])));
}

function applySession(data, force = false) {
  if (!force && session?.id === data.id && session.revision >= data.revision) return;
  session = data;
  sessionStorage.setItem("paper-retriever-session", data.id);
  $("empty").hidden = true;
  $("live-status").hidden = terminal(data);
  $("live-message").textContent = data.message;
  $("live-status").querySelector(".spinner").hidden = data.status !== "working";
  $("live-status").classList.toggle("is-paused", data.status !== "working");
  const badges = {
    working: ["is-working", "Thinking it through"],
    awaiting_query: ["is-review", "Your question, your call"],
    awaiting_search: ["is-review", "Ready for your approval"],
    complete: ["is-complete", "Reading list ready"],
    stopped: ["", "Stopped here"],
    error: ["is-error", "Needs your attention"],
  };
  const [badgeClass, badgeText] = badges[data.status] || ["", "Ready when you are"];
  $("session-badge").className = `session-badge ${badgeClass}`;
  $("session-label").textContent = badgeText;
  const paperCount = (data.result?.papers || data.papers).length;
  $("workspace-count").textContent = paperCount
    ? `${paperCount} ${paperCount === 1 ? "paper" : "papers"} · ${data.steps.length}/${data.max_rounds} searches`
    : "Making room for new perspectives.";
  const stage = terminal(data) ? 2 : data.status === "awaiting_query" || !data.refinement ? 0 : 1;
  ["refine", "search", "review"].forEach((name, index) => {
    $(`stage-${name}`).classList.toggle("active", index === stage);
    $(`stage-${name}`).classList.toggle("done", index < stage);
    if (index === stage) $(`stage-${name}`).setAttribute("aria-current", "step");
    else $(`stage-${name}`).removeAttribute("aria-current");
  });
  renderReview(data);
  renderOverview(data);
  renderPapers(data);
  renderTrail(data);
  renderSave(data);
  const warnings = [...new Set(data.steps.flatMap((step) => [step.library_error, step.source_error]).filter(Boolean))];
  $("source-notice").hidden = !warnings.length;
  $("source-notice").textContent = warnings.length ? `Search coverage is incomplete. ${warnings.join(" ")}` : "";
  $("completion").hidden = !terminal(data);
  $("completion").replaceChildren();
  if (data.result) {
    $("completion").append(element("p", { text: data.result.rationale }));
    if (data.result.unresolved_clarifications?.length) $("completion").append(clarificationBox(data.result.unresolved_clarifications, "Open clarifications for your next search"));
  }
  if (data.error) showNotice(data.error);
  controls();
}

$("research-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (posting || (session && !terminal(session))) return;
  posting = true;
  showNotice("");
  controls();
  try {
    const data = await api("/api/sessions", { question: $("question").value.trim(), model: $("model").value, max_rounds: Number($("rounds").value), limit: Number($("limit").value) });
    pendingKey = null;
    expandedAbstracts.clear();
    seenPaperIds.clear();
    renderedPapersKey = null;
    paperView = "shortlist";
    applySession(data, true);
    if (window.innerWidth <= 760) document.querySelector(".results-workspace").scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  } catch (error) { showNotice(error.message); }
  finally { posting = false; controls(); }
});

document.querySelectorAll(".example").forEach((button) => button.addEventListener("click", () => {
  $("question").value = button.dataset.question;
  controls();
  $("question").focus();
}));
$("question").addEventListener("input", controls);
$("clear-question").addEventListener("click", () => {
  $("question").value = "";
  controls();
  $("question").focus();
});
$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && !event.isComposing && !$("start").disabled) {
    event.preventDefault();
    $("research-form").requestSubmit();
  }
});
$("connection").addEventListener("click", refreshConnection);
$("stop").addEventListener("click", async () => {
  if (!session || posting) return;
  posting = true;
  controls();
  try { applySession(await api(`/api/sessions/${session.id}/cancel`, {})); }
  catch (error) { showNotice(error.message); }
  finally { posting = false; controls(); }
});
$("export").addEventListener("click", () => {
  if (!session) return;
  const link = element("a", { href: `/api/sessions/${session.id}/export`, download: "paper-research.json" });
  document.body.append(link); link.click(); link.remove();
});
$("tab-shortlist").addEventListener("click", () => { paperView = "shortlist"; if (session) renderPapers(session); });
$("tab-all").addEventListener("click", () => { paperView = "all"; if (session) renderPapers(session); });
$("library-toggle").addEventListener("click", () => {
  const open = $("library-panel").hidden;
  $("library-panel").hidden = !open;
  $("library-toggle").setAttribute("aria-expanded", String(open));
  if (open) loadLibrary();
});
$("library-prev").addEventListener("click", () => { if (!libraryLoading) { libraryOffset = Math.max(0, libraryOffset - 10); loadLibrary(); } });
$("library-next").addEventListener("click", () => { if (!libraryLoading) { libraryOffset += 10; loadLibrary(); } });

setInterval(async () => {
  if (!session || (terminal(session) && session.library_save?.status !== "saving") || polling) return;
  polling = true;
  const id = session.id;
  try { const data = await api(`/api/sessions/${id}`); if (session?.id === id) applySession(data); }
  catch (error) { showNotice(error.message); }
  finally { polling = false; }
}, 1200);

async function initialize() {
  const isMac = /Mac|iPhone|iPad/.test(navigator.platform || "");
  $("start-shortcut").textContent = isMac ? "⌘ Enter" : "Ctrl Enter";
  $("start").setAttribute("aria-keyshortcuts", isMac ? "Meta+Enter" : "Control+Enter");
  updateQuestionMeta();
  await refreshConnection();
  const previous = sessionStorage.getItem("paper-retriever-session");
  if (!previous) return;
  try {
    const data = await api(`/api/sessions/${previous}`);
    if (session) return;
    $("question").value = data.question;
    $("rounds").value = String(data.max_rounds);
    $("limit").value = String(data.limit);
    if (configuration?.models.includes(data.model)) $("model").value = data.model;
    applySession(data, true);
  } catch {
    sessionStorage.removeItem("paper-retriever-session");
    showNotice("The previous session has ended. You can start a new research question.");
  }
}
initialize();
