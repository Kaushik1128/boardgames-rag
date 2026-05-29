// Evaluation dashboard — reads /data/eval_summary.json and renders one
// Chart.js bar chart per experiment, plus the narrative and a delta table.

const COLORS = {
  primary: "rgba(52, 211, 153, 0.85)",     // emerald-400
  primaryBorder: "rgb(16, 185, 129)",      // emerald-500
  secondary: "rgba(148, 163, 184, 0.75)",  // slate-400
  secondaryBorder: "rgb(100, 116, 139)",   // slate-500
};

(async function init() {
  const summary = await fetch("/data/eval_summary.json").then((r) => r.json());
  renderMeta(summary);
  summary.experiments.forEach((exp, i) => renderExperiment(exp, summary.metrics, i));
})();

function renderMeta(summary) {
  const el = document.getElementById("eval-meta");
  el.innerHTML = `
    <span>Judge: <span class="text-slate-300 font-mono">${esc(summary.judge)}</span></span>
    <span>N = <span class="text-slate-300">${summary.n_samples}</span> questions</span>
    <span>Test set: <span class="text-slate-300">${esc(summary.test_set)}</span></span>
  `;
}

function renderExperiment(exp, metrics, index) {
  const container = document.getElementById("experiments");
  const card = document.createElement("article");
  card.className = "bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-5";
  card.id = `exp-${exp.id}`;

  const chartId = `chart-${exp.id}`;
  card.innerHTML = `
    <header>
      <div class="flex items-baseline gap-3 flex-wrap">
        <span class="text-xs text-slate-500 font-mono">#${index + 1}</span>
        <h3 class="text-xl font-semibold text-slate-100">${esc(exp.title)}</h3>
      </div>
      <p class="text-sm text-slate-400 mt-1">${esc(exp.subtitle)}${
        exp.date ? ` · <span class="text-slate-500">${esc(exp.date)}</span>` : ""
      }</p>
    </header>
    <div class="grid grid-cols-1 lg:grid-cols-5 gap-6 items-start">
      <div class="lg:col-span-3">
        <div class="bg-slate-950/50 rounded-lg p-4">
          <canvas id="${chartId}" height="220"></canvas>
        </div>
      </div>
      <div class="lg:col-span-2 space-y-4">
        ${renderDeltaTable(exp, metrics)}
        <p class="text-sm text-slate-300 leading-relaxed">${esc(exp.narrative)}</p>
      </div>
    </div>
  `;
  container.appendChild(card);

  // Build the chart after the canvas is attached so Chart.js can size it.
  drawChart(chartId, exp, metrics);
}

function renderDeltaTable(exp, metrics) {
  if (exp.arms.length !== 2) return "";
  const [a, b] = exp.arms;
  const rows = metrics
    .map((m) => {
      const av = a.scores[m.key];
      const bv = b.scores[m.key];
      const delta = av - bv;
      const sign = delta > 0 ? "+" : "";
      const color =
        Math.abs(delta) < 0.005
          ? "text-slate-400"
          : delta > 0
          ? "text-emerald-400"
          : "text-amber-400";
      return `
        <tr class="border-t border-slate-800">
          <td class="py-1.5 text-slate-400">${esc(m.label)}</td>
          <td class="py-1.5 text-right text-slate-200 font-mono">${av.toFixed(3)}</td>
          <td class="py-1.5 text-right text-slate-200 font-mono">${bv.toFixed(3)}</td>
          <td class="py-1.5 text-right font-mono ${color}">${sign}${delta.toFixed(3)}</td>
        </tr>`;
    })
    .join("");
  return `
    <table class="w-full text-xs">
      <thead>
        <tr class="text-slate-500 uppercase tracking-wider">
          <th class="text-left pb-2 font-medium">Metric</th>
          <th class="text-right pb-2 font-medium">${esc(a.label)}</th>
          <th class="text-right pb-2 font-medium">${esc(b.label)}</th>
          <th class="text-right pb-2 font-medium">Δ</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function drawChart(canvasId, exp, metrics) {
  const ctx = document.getElementById(canvasId);
  const labels = metrics.map((m) => m.label);
  const datasets = exp.arms.map((arm, i) => ({
    label: arm.label,
    data: metrics.map((m) => arm.scores[m.key]),
    backgroundColor: i === 0 ? COLORS.primary : COLORS.secondary,
    borderColor: i === 0 ? COLORS.primaryBorder : COLORS.secondaryBorder,
    borderWidth: 1,
    borderRadius: 4,
  }));
  // eslint-disable-next-line no-undef
  new Chart(ctx, {
    type: "bar",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "rgb(203, 213, 225)", font: { size: 12 } },
          position: "bottom",
        },
        tooltip: {
          backgroundColor: "rgb(15, 23, 42)",
          borderColor: "rgb(51, 65, 85)",
          borderWidth: 1,
          titleColor: "rgb(226, 232, 240)",
          bodyColor: "rgb(203, 213, 225)",
          callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.y.toFixed(3)}` },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 1.0,
          grid: { color: "rgb(30, 41, 59)" },
          ticks: { color: "rgb(148, 163, 184)", stepSize: 0.2 },
        },
        x: {
          grid: { display: false },
          ticks: { color: "rgb(148, 163, 184)", font: { size: 11 } },
        },
      },
    },
  });
}

function esc(s) {
  if (s == null) return "";
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
