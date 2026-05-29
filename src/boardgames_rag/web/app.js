// boardgames-rag — frontend logic.
//
// EventSource can't POST, so we fetch /ask/stream and parse the SSE wire
// format by hand off a ReadableStream. Each event updates the trace
// timeline, the streaming answer text, the sources panel, or the error
// banner depending on its name.

const STATE = {
  selectedGame: null,
  games: [],
  sampleQuestions: {},
};

// ---------------------------------------------------------------------------
// Initial load: games + sample questions
// ---------------------------------------------------------------------------

async function loadConfig() {
  const [games, samples] = await Promise.all([
    fetch("/data/games.json").then((r) => r.json()),
    fetch("/data/sample_questions.json").then((r) => r.json()),
  ]);
  STATE.games = games;
  STATE.sampleQuestions = samples;
  renderGameGrid();
}

// ---------------------------------------------------------------------------
// Game grid
// ---------------------------------------------------------------------------

function renderGameGrid() {
  const grid = document.getElementById("game-grid");
  grid.innerHTML = "";
  STATE.games.forEach((game) => {
    const button = document.createElement("button");
    button.className = [
      "game-card",
      "flex flex-col items-center justify-center gap-1",
      "bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg",
      "px-3 py-4 text-sm font-medium transition-all duration-150",
      "hover:border-emerald-700 hover:scale-[1.02]",
    ].join(" ");
    button.dataset.slug = game.slug;
    button.innerHTML = `
      <span class="text-2xl">${game.emoji}</span>
      <span class="text-slate-200">${game.name}</span>
    `;
    button.addEventListener("click", () => selectGame(game.slug));
    grid.appendChild(button);
  });
}

function selectGame(slug) {
  STATE.selectedGame = slug;
  document.querySelectorAll(".game-card").forEach((el) => {
    if (el.dataset.slug === slug) {
      el.classList.add("border-emerald-500", "bg-slate-800");
      el.classList.remove("border-slate-800");
    } else {
      el.classList.remove("border-emerald-500", "bg-slate-800");
      el.classList.add("border-slate-800");
    }
  });
  renderChips(slug);
}

function renderChips(slug) {
  const section = document.getElementById("chips-section");
  const label = document.getElementById("chips-game-label");
  const container = document.getElementById("chips");
  const game = STATE.games.find((g) => g.slug === slug);
  const questions = STATE.sampleQuestions[slug] || [];
  label.textContent = game ? `· ${game.name}` : "";
  container.innerHTML = "";
  questions.forEach((q) => {
    const chip = document.createElement("button");
    chip.className = [
      "bg-slate-800 hover:bg-slate-700 border border-slate-700",
      "rounded-full px-4 py-2 text-sm text-slate-200",
      "hover:border-emerald-700 transition-colors",
    ].join(" ");
    chip.textContent = q;
    chip.addEventListener("click", () => {
      document.getElementById("question-input").value = q;
      submitQuestion(q);
    });
    container.appendChild(chip);
  });
  section.classList.remove("hidden");
}

// ---------------------------------------------------------------------------
// Asking — fetch /ask/stream, parse SSE, render
// ---------------------------------------------------------------------------

document.getElementById("ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = document.getElementById("question-input").value.trim();
  if (question) submitQuestion(question);
});

async function submitQuestion(question) {
  resetResultArea();
  const button = document.getElementById("ask-button");
  button.disabled = true;
  button.textContent = "Thinking…";

  try {
    const response = await fetch("/ask/stream", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) {
      showError(`Request failed: HTTP ${response.status}`);
      return;
    }
    await readSSE(response.body, handleEvent);
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    button.disabled = false;
    button.textContent = "Ask";
  }
}

function resetResultArea() {
  document.getElementById("result").classList.remove("hidden");
  document.getElementById("trace").innerHTML = "";
  document.getElementById("answer").textContent = "";
  document.getElementById("answer-card").classList.add("hidden");
  document.getElementById("sources").innerHTML = "";
  document.getElementById("sources-card").classList.add("hidden");
  document.getElementById("error-card").classList.add("hidden");
}

// SSE frame format: `event: <name>\ndata: <json>\n\n`. Read the response
// body as a stream and dispatch each complete frame to `onEvent`.
async function readSSE(body, onEvent) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const event = parseFrame(frame);
      if (event) onEvent(event.name, event.data);
    }
  }
}

function parseFrame(frame) {
  let name = null;
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event: ")) name = line.slice(7);
    else if (line.startsWith("data: ")) data += line.slice(6);
  }
  if (!name) return null;
  try {
    return { name, data: data ? JSON.parse(data) : {} };
  } catch {
    return { name, data: {} };
  }
}

// ---------------------------------------------------------------------------
// Event handlers — one per SSE event name
// ---------------------------------------------------------------------------

function handleEvent(name, data) {
  switch (name) {
    case "plan":
      addTraceStep("✏️", `Planned search query: <code>${escapeHtml(data.query)}</code>`);
      break;
    case "retrieve":
      addTraceStep(
        "📚",
        `Retrieved ${data.n_chunks} chunks <span class="text-slate-500">(attempt ${data.attempt})</span>`,
      );
      break;
    case "critique": {
      const verdict = data.verdict;
      const icon = verdict === "sufficient" ? "✅" : "🔁";
      const detail = data.reformulated_query
        ? ` → reformulating: <code>${escapeHtml(data.reformulated_query)}</code>`
        : "";
      addTraceStep(icon, `Critic: <strong>${verdict}</strong>${detail}`);
      break;
    }
    case "web_fallback":
      addTraceStep("🌐", `Web fallback fired — ${data.n_chunks} result(s)`);
      break;
    case "token":
      appendAnswerToken(data.text);
      break;
    case "sources":
      renderSources(data);
      break;
    case "done":
      addTraceStep(
        "🏁",
        `Done <span class="text-slate-500">· ${data.attempts} retrieval attempt(s)${data.used_web ? " · used web" : ""}</span>`,
      );
      // Drop the streaming caret on the answer body.
      document.getElementById("answer").classList.add("complete");
      break;
    case "error":
      showError(data.message || "The pipeline failed.");
      break;
    default:
      console.warn("Unknown SSE event:", name, data);
  }
}

function addTraceStep(icon, html) {
  const ol = document.getElementById("trace");
  const li = document.createElement("li");
  li.className = "flex items-start gap-2 trace-step";
  li.innerHTML = `<span class="text-base leading-none mt-0.5">${icon}</span><span class="text-slate-300">${html}</span>`;
  ol.appendChild(li);
}

function appendAnswerToken(text) {
  const card = document.getElementById("answer-card");
  if (card.classList.contains("hidden")) {
    card.classList.remove("hidden");
  }
  const node = document.getElementById("answer");
  node.textContent += text;
}

function renderSources(sources) {
  const card = document.getElementById("sources-card");
  const ol = document.getElementById("sources");
  ol.innerHTML = "";
  sources.forEach((src, i) => {
    const li = document.createElement("li");
    li.className = "border-l-2 border-emerald-700 pl-3";
    li.innerHTML = `
      <div class="text-emerald-400 text-xs font-mono mb-1">[${i + 1}] ${escapeHtml(src.source_file)}</div>
      <div class="text-slate-300 font-medium mb-1">${escapeHtml(src.heading || "—")}</div>
      <div class="text-slate-400 text-xs leading-relaxed line-clamp-4">${escapeHtml(src.text || "")}</div>
    `;
    ol.appendChild(li);
  });
  card.classList.remove("hidden");
}

function showError(message) {
  const card = document.getElementById("error-card");
  document.getElementById("error-message").textContent = message;
  card.classList.remove("hidden");
}

function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

loadConfig().catch((err) => {
  console.error(err);
  showError("Could not load game list — is the service running?");
});
