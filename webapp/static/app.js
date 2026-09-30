/* HealthGuard Phase-1 dashboard. Every number is fetched from the API, which
   serves what healthguard/experiment.py produced — nothing is hardcoded. */

const $ = (id) => document.getElementById(id);
// Colours are CSS variables, so switching theme recolours every chart
// without re-rendering. (SVG attributes cannot read var(), so charts use style.)
const MODEL_COLOR = { logreg: "var(--m-logreg)", xgboost: "var(--m-xgboost)", mlp: "var(--m-mlp)" };
const CLASS_COLOR = { 0: "var(--c-healthy)", 1: "var(--c-disease)" };
const CLASS_NAME = { 0: "No Disease", 1: "Disease" };
const METRICS = [
  ["accuracy", "Accuracy"], ["precision", "Precision"], ["recall", "Recall"],
  ["f1", "F1"], ["roc_auc", "ROC-AUC"],
];

const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
const fix = (v, d = 3) => Number(v).toFixed(d);
const ms = (v) => (v >= 1000 ? (v / 1000).toFixed(2) + " s" : v.toFixed(v < 10 ? 1 : 0) + " ms");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const range = (xs) => {
  const lo = Math.min(...xs), hi = Math.max(...xs);
  return lo === hi ? String(lo) : `${lo}–${hi}`;
};

async function api(path, body) {
  const res = await fetch(path, body ? {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  } : {});
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

const cache = {};
async function get(path) { return (cache[path] ??= api(path)); }

/* ─── Tooltip ───────────────────────────────────────────────────────────── */
const tip = $("tooltip");
document.addEventListener("mousemove", (e) => {
  const t = e.target.closest?.("[data-tip]");
  if (!t) { tip.classList.add("hidden"); return; }
  tip.textContent = t.dataset.tip;
  tip.classList.remove("hidden");
  const x = Math.min(e.clientX + 12, window.innerWidth - tip.offsetWidth - 8);
  tip.style.left = x + "px";
  tip.style.top = e.clientY + 14 + "px";
});

/* ─── Theme ─────────────────────────────────────────────────────────────── */
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("hg-theme", theme); } catch (e) { /* private mode */ }
  document.querySelectorAll(".theme-btn").forEach((b) => b.classList.toggle("active", b.dataset.theme === theme));
}
document.querySelectorAll(".theme-btn").forEach((b) => b.addEventListener("click", () => setTheme(b.dataset.theme)));
setTheme(document.documentElement.dataset.theme || "dark");

/* ─── Navigation ────────────────────────────────────────────────────────── */
const rendered = {};
const renderers = { overview: renderOverview, dataset: renderDataset, models: renderModels, fhe: renderFhe };

function show(section) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.section === section));
  document.querySelectorAll(".section").forEach((s) => s.classList.toggle("hidden", s.id !== section));
  if (!rendered[section]) {
    rendered[section] = true;
    renderers[section]().catch((err) => console.error(err));
  }
  const nav = document.querySelector(".tabs");
  if (window.scrollY > nav.offsetTop) window.scrollTo({ top: nav.offsetTop - 12 });
}
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => show(t.dataset.section)));

/* ─── SVG helpers ───────────────────────────────────────────────────────── */
const scale = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);

function niceMax(v) {
  const p = Math.pow(10, Math.floor(Math.log10(v || 1)));
  return Math.ceil(v / p) * p;
}

function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const f = raw / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p;
}

function yAxis(y, max, x0, x1, ticks = 4, fmt = (v) => v) {
  let out = "";
  for (let i = 0; i <= ticks; i++) {
    const v = (max / ticks) * i, py = y(v);
    out += `<line class="grid" x1="${x0}" x2="${x1}" y1="${py}" y2="${py}" style="stroke:var(--chart-grid)"/>`;
    out += `<text x="${x0 - 6}" y="${py + 4}" text-anchor="end">${fmt(v)}</text>`;
  }
  return out;
}

/* Grouped bars: groups = [{label, values:{0:n,1:n}}] */
function groupedBars(groups, { width = 760, height = 260, xTitle = "" } = {}) {
  const m = { t: 10, r: 10, b: 44, l: 40 };
  const max = niceMax(Math.max(1, ...groups.flatMap((g) => [g.values[0], g.values[1]])));
  const x0 = m.l, x1 = width - m.r, y = scale(0, max, height - m.b, m.t);
  const gw = (x1 - x0) / groups.length, bw = Math.min(26, gw * 0.36);
  let bars = "";
  groups.forEach((g, i) => {
    const cx = x0 + gw * i + gw / 2;
    [0, 1].forEach((c) => {
      const v = g.values[c], bx = cx + (c === 0 ? -bw - 1 : 1);
      bars += `<rect x="${bx}" y="${y(v)}" width="${bw}" height="${y(0) - y(v)}" rx="3" style="fill:${CLASS_COLOR[c]}"
        data-tip="${esc(g.label)} · ${CLASS_NAME[c]}: ${v}"/>`;
    });
    bars += `<text x="${cx}" y="${height - m.b + 16}" text-anchor="middle">${esc(g.label)}</text>`;
  });
  return `<svg viewBox="0 0 ${width} ${height}" role="img">
    ${yAxis(y, max, x0, x1, 4, (v) => Math.round(v))}
    <line x1="${x0}" x2="${x1}" y1="${y(0)}" y2="${y(0)}" style="stroke:var(--chart-axis)"/>
    ${bars}
    <text x="${(x0 + x1) / 2}" y="${height - 6}" text-anchor="middle" style="font-weight:600">${esc(xTitle)}</text>
    <text x="12" y="${m.t + (height - m.b - m.t) / 2}" text-anchor="middle" transform="rotate(-90 12 ${m.t + (height - m.b - m.t) / 2})">patients</text>
  </svg>`;
}

function lineChart(series, { width = 520, height = 260, xLabel = "", yLabel = "", marker = null, square = false, diagonal = false } = {}) {
  const m = { t: 12, r: 14, b: 40, l: 46 };
  const xs = series.flatMap((s) => s.x), ys = series.flatMap((s) => s.y);
  const yStep = square ? 0.25 : niceStep(Math.max(...ys) / 4);
  const xMax = square ? 1 : Math.max(...xs), yMax = square ? 1 : Math.ceil((Math.max(...ys) * 1.02) / yStep) * yStep;
  const xMin = square ? 0 : Math.min(...xs);
  const x = scale(xMin, xMax, m.l, width - m.r), y = scale(0, yMax, height - m.b, m.t);
  const path = (s) => s.x.map((v, i) => `${i ? "L" : "M"}${x(v).toFixed(1)},${y(s.y[i]).toFixed(1)}`).join("");
  let body = yAxis(y, yMax, m.l, width - m.r, Math.round(yMax / yStep), (v) => v.toFixed(2));
  const xt = square ? [0, 0.25, 0.5, 0.75, 1] : [xMin, Math.round((xMin + xMax) / 2), xMax];
  xt.forEach((v) => (body += `<text x="${x(v)}" y="${height - m.b + 16}" text-anchor="middle">${square ? v.toFixed(2) : v}</text>`));
  body += `<line x1="${m.l}" x2="${width - m.r}" y1="${y(0)}" y2="${y(0)}" style="stroke:var(--chart-axis)"/>`;
  if (diagonal) body += `<line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" style="stroke:var(--chart-axis)" stroke-dasharray="4 4"/>`;
  if (marker) {
    body += `<line x1="${x(marker.x)}" x2="${x(marker.x)}" y1="${m.t}" y2="${y(0)}" style="stroke:var(--text)" stroke-dasharray="3 3"/>
      <text x="${x(marker.x) + 5}" y="${m.t + 10}" style="font-weight:600;fill:var(--text)">${esc(marker.label)}</text>`;
  }
  series.forEach((s) => {
    body += `<path d="${path(s)}" fill="none" style="stroke:${s.color}" stroke-width="${s.width || 2.2}"
      ${s.dash ? `stroke-dasharray="${s.dash}"` : ""} data-tip="${esc(s.tip || s.name)}"/>`;
  });
  body += `<text x="${(m.l + width - m.r) / 2}" y="${height - 4}" text-anchor="middle" style="font-weight:600">${esc(xLabel)}</text>
    <text x="12" y="${(m.t + height - m.b) / 2}" text-anchor="middle" style="font-weight:600" transform="rotate(-90 12 ${(m.t + height - m.b) / 2})">${esc(yLabel)}</text>`;
  return `<svg viewBox="0 0 ${width} ${height}" role="img" style="max-width:${width * 1.15}px">${body}</svg>`;
}

function legend(items) {
  return `<div class="legend">${items.map((i) =>
    `<span class="legend-item"><span class="swatch" style="background:${i.color}"></span>${esc(i.label)}</span>`).join("")}</div>`;
}

function divergingColor(v) {
  // −1 → blue, 0 → card surface, +1 → coral. Mixing towards the surface (not
  // white) keeps weak correlations quiet on both the dark and light themes.
  const a = Math.round(Math.min(1, Math.abs(v)) * 100);
  return `color-mix(in srgb, var(${v >= 0 ? "--heat-pos" : "--heat-neg"}) ${a}%, var(--heat-base))`;
}

function heatmap(labels, matrix, names) {
  const n = labels.length, cell = 38, left = 150, top = 110;
  const w = left + n * cell + 4, h = top + n * cell + 4;
  let body = "";
  labels.forEach((l, i) => {
    const nm = names[l] || l;
    body += `<text x="${left - 8}" y="${top + i * cell + cell / 2 + 4}" text-anchor="end">${esc(nm)}</text>`;
    body += `<text transform="translate(${left + i * cell + cell / 2 + 4},${top - 8}) rotate(-50)">${esc(nm)}</text>`;
    matrix[i].forEach((v, j) => {
      const strong = Math.abs(v) >= 0.35;
      body += `<rect x="${left + j * cell}" y="${top + i * cell}" width="${cell - 2}" height="${cell - 2}" rx="3"
        style="fill:${divergingColor(v)}" data-tip="${esc(names[labels[i]] || labels[i])} × ${esc(names[labels[j]] || labels[j])}: r = ${v.toFixed(2)}"/>`;
      if (i !== j) body += `<text x="${left + j * cell + cell / 2 - 1}" y="${top + i * cell + cell / 2 + 3}" text-anchor="middle"
        style="font-size:9.5px;pointer-events:none;fill:${strong ? "#fff" : "var(--text-2)"}">${v.toFixed(2)}</text>`;
    });
  });
  return `<svg viewBox="0 0 ${w} ${h}" role="img" style="max-width:${w}px">${body}</svg>`;
}

// [value, label, sub, tint] → stat card. Tints mirror meaning:
// disease = coral, no disease = green, neutral counts = blue / teal / violet.
const statCards = (rows) => rows.map(([v, l, s, t]) =>
  `<div class="stat ${t ? "tint-" + t : ""}"><div class="stat-label">${esc(l)}</div>` +
  `<div class="stat-value">${v}</div><div class="stat-sub">${esc(s)}</div></div>`).join("");

/* ═══════════════════ 1 · OVERVIEW ═══════════════════ */
async function renderOverview() {
  const o = await get("/api/overview");
  $("ovStats").innerHTML = statCards([
    [o.n_samples, "Samples", "Heart Disease · Cleveland, UCI", "blue"],
    [o.n_attributes, "Attributes", `${o.n_features} clinical features + target`, "teal"],
    [o.n_models, "ML models", "LR · XGBoost · MLP", "violet"],
    [o.n_folds, "CV folds", "stratified", "amber"],
  ]);

  $("ovPipeline").innerHTML = [
    ["Data", `${o.n_samples} patients · ${o.n_features} features`],
    ["Clean", "relabel · invalid codes → missing"],
    ["Preprocess", "scale · one-hot · impute (per fold)"],
    ["5-fold CV", "tuning inside each fold"],
    ["Models", "LR · XGBoost · MLP"],
    ["Evaluate", "Acc · Prec · Recall · F1 · AUC"],
    ["FHE", "LR via Concrete ML", "fhe"],
  ].map(([t, s, c]) => `<li class="${c || ""}"><strong>${t}</strong><span>${s}</span></li>`).join("");

  const keys = o.model_order;
  const best = (k) => Math.max(...keys.map((m) => o.models[m].pooled[k]));
  $("ovResults").innerHTML = `<thead><tr><th>Model</th><th>Accuracy</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr></thead><tbody>` +
    keys.map((k) => {
      const p = o.models[k].pooled;
      const cell = (m, f) => `<td class="${p[m] === best(m) ? "best" : ""}">${f(p[m])}</td>`;
      return `<tr><td><span class="model-name"><span class="swatch" style="background:${MODEL_COLOR[k]}"></span>${esc(o.models[k].name)}</span></td>
        ${cell("accuracy", pct)}${cell("recall", pct)}${cell("f1", pct)}${cell("roc_auc", (v) => fix(v))}</tr>`;
    }).join("") + "</tbody>";
  $("ovSelection").textContent = `Model selection: ${o.selection_metric} — our Phase-1 choice, consistent with the synopsis' emphasis on sensitivity (a missed disease case is the costlier error). Bold = best per column.`;

  const f = o.fhe, agreeN = o.n_samples - f.n_disagree;
  $("ovFhe").innerHTML = `
    <div class="agg">
      <div class="agg-item tint-healthy"><b>${agreeN}/${o.n_samples}</b><span>encrypted = plaintext decision (${pct(f.agreement, 2)})</span></div>
      <div class="agg-item tint-blue"><b>${ms(f.real_exec_ms_per_sample)}</b><span>per patient, genuine encrypted execution</span></div>
      <div class="agg-item"><b>${f.n_bits}-bit</b><span>quantisation of inputs &amp; weights</span></div>
    </div>
    <p class="note">${o.fhe_status.logreg.ready
      ? "The server evaluates the model on ciphertext only — it never sees a feature value or holds the decryption key."
      : esc(o.fhe_status.logreg.reason)}</p>`;
}

/* ═══════════════════ 2 · DATASET ═══════════════════ */
async function renderDataset() {
  const [hd, bc] = await Promise.all([get("/api/eda/heart"), get("/api/eda/breast")]);
  const names = Object.fromEntries(Object.entries(hd.features).map(([k, v]) => [k, v.label]));
  names.disease = "Disease (label)";

  $("hdStats").innerHTML = statCards([
    [hd.n_samples, "Records", hd.source, "blue"],
    [hd.n_attributes, "Attributes", `${hd.n_features} features + target`, "teal"],
    [hd.class_counts["1"], "With disease", `${pct(hd.class_counts["1"] / hd.n_samples)} of patients`, "disease"],
    [hd.class_counts["0"], "Without disease", `${pct(hd.class_counts["0"] / hd.n_samples)} of patients`, "healthy"],
  ]);

  $("hdBalance").innerHTML = balanceBar(hd.class_counts, hd.class_names) +
    `<p class="note">Mildly imbalanced, so every split is stratified and each model uses class weighting (balanced LR weights, XGBoost <code>scale_pos_weight</code>, weighted BCE for the MLP).</p>`;

  const groups = [
    ["Continuous", hd.groups.continuous, "StandardScaler"],
    ["Binary", hd.groups.binary, "passthrough (already 0/1)"],
    ["Nominal", hd.groups.nominal, "impute → one-hot"],
  ];
  $("hdTypes").innerHTML = `<thead><tr><th>Type</th><th class="left">Features</th><th class="left">Transform</th></tr></thead><tbody>` +
    groups.map(([t, fs, how]) => `<tr><td><b>${t}</b> (${fs.length})</td><td class="left">${fs.map((f) => esc(names[f])).join(", ")}</td><td class="left">${how}</td></tr>`).join("") + "</tbody>";

  const sel = $("hdFeature");
  sel.innerHTML = Object.entries(hd.features).map(([k, v]) => `<option value="${k}">${esc(v.label)}${v.unit && v.unit !== "code" ? ` (${v.unit})` : ""}</option>`).join("");
  sel.value = "thalach";
  const drawHist = () => {
    const f = hd.features[sel.value];
    let groups;
    if (f.type === "continuous") {
      groups = f.by_class["0"].map((_, i) => ({
        label: `${Math.round(f.edges[i])}`,
        values: { 0: f.by_class["0"][i], 1: f.by_class["1"][i] },
      }));
    } else {
      groups = f.categories.map((c, i) => ({
        label: f.values ? f.values[c] ?? c : `code ${c}`,
        values: { 0: f.by_class["0"][i], 1: f.by_class["1"][i] },
      }));
    }
    const title = f.type === "continuous" ? `${f.label}${f.unit ? ` (${f.unit}, bin start)` : ""}` : f.label;
    $("hdHist").innerHTML = groupedBars(groups, { xTitle: title }) +
      legend([{ color: CLASS_COLOR[0], label: "No Disease" }, { color: CLASS_COLOR[1], label: "Disease" }]) +
      (f.missing ? `<p class="note">${f.missing} value(s) missing (invalid code) — imputed inside each training fold.</p>` : "");
  };
  sel.onchange = drawHist;
  drawHist();

  $("hdCorr").innerHTML = heatmap(hd.correlation.labels, hd.correlation.matrix, names) +
    legend([{ color: divergingColor(-0.8), label: "negative" }, { color: divergingColor(0.8), label: "positive correlation" }]);

  $("hdEvidence").innerHTML = `<thead><tr><th class="left">Sign</th><th>CSV target=0</th><th>CSV target=1</th><th class="left">In disease</th></tr></thead><tbody>` +
    hd.label_evidence.map((e) => `<tr><td class="left">${esc(e.label)}</td><td><b>${e.csv_target_0}</b></td><td>${e.csv_target_1}</td><td class="left">${e.expected_in_disease}</td></tr>`).join("") + "</tbody>";
  $("hdInvalid").innerHTML = `<thead><tr><th class="left">Feature</th><th>Invalid code</th><th>Rows</th></tr></thead><tbody>` +
    Object.entries(hd.invalid_codes).map(([k, v]) => `<tr><td class="left">${esc(names[k])} (<code>${k}</code>)</td><td>${v.codes.join(", ")}</td><td>${v.rows}</td></tr>`).join("") + "</tbody>";

  $("bcStats").innerHTML = statCards([
    [bc.n_samples, "Samples", bc.source, "blue"],
    [bc.n_features, "Features", "all numeric", "teal"],
    [bc.class_counts["0"], "Benign", `${pct(bc.class_counts["0"] / bc.n_samples)} of samples`, "healthy"],
    [bc.class_counts["1"], "Malignant", `${pct(bc.class_counts["1"] / bc.n_samples)} of samples`, "disease"],
  ]);
  $("bcBalance").innerHTML = balanceBar(bc.class_counts, bc.class_names);

  document.querySelectorAll(".subtab").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".subtab").forEach((x) => x.classList.toggle("active", x === b));
    $("dsHeart").classList.toggle("hidden", b.dataset.ds !== "heart");
    $("dsBreast").classList.toggle("hidden", b.dataset.ds !== "breast");
  }));
}

function balanceBar(counts, names) {
  const n0 = counts["0"], n1 = counts["1"], t = n0 + n1;
  return `<div class="balance">
      <div style="width:${(n0 / t) * 100}%;background:${CLASS_COLOR[0]}">${n0}</div>
      <div style="width:${(n1 / t) * 100}%;background:${CLASS_COLOR[1]}">${n1}</div>
    </div>
    <div class="balance-legend"><span>${esc(names["0"])} · ${pct(n0 / t)}</span><span>${esc(names["1"])} · ${pct(n1 / t)}</span></div>`;
}

/* ═══════════════════ 3 · MODEL COMPARISON ═══════════════════ */
async function renderModels() {
  const { metrics, curves, model_order: order } = await get("/api/results");
  const M = metrics.models;

  const drawTable = (mode) => {
    const best = Object.fromEntries(METRICS.map(([k]) => [k, Math.max(...order.map((m) =>
      mode === "pooled" ? M[m].pooled[k] : M[m].summary[k].mean))]));
    $("mcTable").innerHTML = `<thead><tr><th>Model</th>${METRICS.map(([k, l]) =>
      `<th class="${k === "recall" ? "recall" : ""}">${l}${k === "recall" ? " ★" : ""}</th>`).join("")}</tr></thead><tbody>` +
      order.map((m) => `<tr><td><span class="model-name"><span class="swatch" style="background:${MODEL_COLOR[m]}"></span>${esc(M[m].name)}</span></td>` +
        METRICS.map(([k]) => {
          const v = mode === "pooled" ? M[m].pooled[k] : M[m].summary[k].mean;
          const f = k === "roc_auc" ? fix(v) : pct(v);
          const sd = mode === "folds" ? ` <span class="std">± ${k === "roc_auc" ? fix(M[m].summary[k].std) : pct(M[m].summary[k].std)}</span>` : "";
          return `<td class="${k === "recall" ? "recall " : ""}${v === best[k] ? "best" : ""}">${f}${sd}</td>`;
        }).join("") + "</tr>").join("") + "</tbody>";
    $("mcNote").innerHTML = mode === "pooled"
      ? `Pooled: each of the ${metrics.n_samples} patients predicted once, by the fold model that never saw them. ★ Recall = sensitivity to disease, the metric the synopsis weights most. Bold = best.`
      : `Mean ± standard deviation across the ${metrics.n_folds} test folds (~${Math.round(metrics.n_samples / metrics.n_folds)} patients each). With folds this small, one patient moves a fold's accuracy by ~1.6 points — differences inside one std are not meaningful.`;
  };
  document.querySelectorAll("#metricMode .toggle-btn").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#metricMode .toggle-btn").forEach((x) => x.classList.toggle("active", x === b));
    drawTable(b.dataset.mode);
  }));
  drawTable("pooled");

  $("mcRoc").innerHTML = lineChart(order.map((m) => ({
    x: M[m].roc.fpr, y: M[m].roc.tpr, color: MODEL_COLOR[m], name: M[m].name,
    tip: `${M[m].name} · AUC ${fix(M[m].pooled.roc_auc)}`,
  })), { square: true, diagonal: true, xLabel: "False positive rate (1 − specificity)", yLabel: "True positive rate (recall)", width: 460, height: 360 }) +
    legend(order.map((m) => ({ color: MODEL_COLOR[m], label: `${M[m].name} · AUC ${fix(M[m].pooled.roc_auc)}` })));

  $("mcConfusion").innerHTML = order.map((m) => {
    const c = M[m].pooled.confusion;
    const cellh = (cls, n, lbl) => `<div class="cm-cell ${cls}" data-tip="${lbl}: ${n}"><b>${n}</b><small>${lbl}</small></div>`;
    return `<div class="cm"><h4><span class="swatch" style="background:${MODEL_COLOR[m]}"></span>${esc(M[m].name)}</h4>
      <div class="cm-table">
        <span></span><span class="hdr">Pred. no</span><span class="hdr">Pred. disease</span>
        <span class="side">No disease</span>${cellh("good", c.tn, "true negative")}${cellh("fp", c.fp, "false alarm")}
        <span class="side">Disease</span>${cellh("fn", c.fn, "missed disease")}${cellh("good", c.tp, "caught")}
      </div></div>`;
  }).join("");

  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  $("mcLadder").innerHTML = order.map((m, i) => {
    const fi = M[m].fold_info, cx = fi.map((f) => f.complexity), p = M[m].pooled;
    let rows;
    if (m === "xgboost") {
      rows = [["Trees", range(cx.map((c) => c.n_trees))], ["Max depth", range(cx.map((c) => c.max_depth))],
        ["Total tree nodes", range(cx.map((c) => c.total_nodes))]];
    } else if (m === "mlp") {
      rows = [["Layers", cx[0].summary], ["Parameters", range(cx.map((c) => c.n_params))], ["Activation", "ReLU"]];
    } else {
      rows = [["Parameters", range(cx.map((c) => c.n_params))], ["Encoded features", range(cx.map((c) => c.n_features))], ["Form", "linear"]];
    }
    rows.push([m === "mlp" ? "Train time / fold" : "Search + train / fold", ms(mean(fi.map((f) => f.fit_seconds)) * 1000)]);
    return `<div class="rung" style="--c:${MODEL_COLOR[m]}">
      <div class="rung-step">Complexity ${i + 1} of ${order.length}</div>
      <h4>${esc(M[m].name)}</h4>
      <dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("")}</dl>
      <dl class="perf"><dt>Recall</dt><dd>${pct(p.recall)}</dd><dt>ROC-AUC</dt><dd>${fix(p.roc_auc)}</dd></dl>
    </div>`;
  }).join("");

  const fmtParams = (m, bp) => m === "logreg" ? `C = ${bp.C}`
    : m === "xgboost" ? `${bp.n_estimators} trees · depth ${bp.max_depth} · lr ${bp.learning_rate}`
    : `best epoch ${bp.best_epoch} / ${bp.epochs_run}`;
  $("mcFolds").innerHTML = `<thead><tr><th>Fold</th>${order.map((m) => `<th class="left">${esc(M[m].name)}</th>`).join("")}</tr></thead><tbody>` +
    M[order[0]].fold_info.map((_, k) => `<tr><td>${k + 1}</td>${order.map((m) =>
      `<td class="left">${esc(fmtParams(m, M[m].fold_info[k].best_params))}</td>`).join("")}</tr>`).join("") + "</tbody>";

  const fs = $("mcCurveFold");
  fs.innerHTML = curves.mlp.map((c) => `<option value="${c.fold}">Fold ${c.fold + 1}</option>`).join("");
  const drawCurve = () => {
    const c = curves.mlp[+fs.value];
    const ep = c.train_loss.map((_, i) => i + 1);
    $("mcCurve").innerHTML = lineChart([
      { x: ep, y: c.train_loss, color: "var(--chart-muted)", name: "training loss" },
      { x: ep, y: c.val_loss, color: MODEL_COLOR.mlp, name: "validation loss" },
    ], { xLabel: "epoch", yLabel: "weighted BCE loss", marker: { x: c.best_epoch + 1, label: `best epoch ${c.best_epoch + 1} (weights restored)` } }) +
      legend([{ color: "var(--chart-muted)", label: "training loss" }, { color: MODEL_COLOR.mlp, label: "validation loss" }]) +
      `<p class="note">Training stops ${c.val_loss.length - c.best_epoch - 1} epochs after the best validation loss; the weights from the best epoch are restored.</p>`;
  };
  fs.onchange = drawCurve;
  drawCurve();
}

/* ═══════════════════ 4 · ENCRYPTED INFERENCE ═══════════════════ */
async function renderFhe() {
  const [o, patients] = await Promise.all([get("/api/overview"), get("/api/patients")]);

  $("fxModel").innerHTML = o.model_order.map((k) => [k, o.models[k]]).map(([k, v]) =>
    `<option value="${k}">${esc(v.name)} — ${o.fhe_status[k].ready ? "encrypted ✓" : "plaintext only (FHE: Phase 2)"}</option>`).join("");
  $("fxPatient").innerHTML = patients.map((p) => `<option value="${p.id}">Patient #${p.id} · fold ${p.fold}</option>`).join("");

  const f = o.fhe, dis = o.fhe_disagreements;
  $("fxAggregate").innerHTML = `
    <div class="agg">
      <div class="agg-item tint-healthy"><b>${o.n_samples - f.n_disagree}/${o.n_samples}</b><span>same decision encrypted vs plaintext (${pct(f.agreement, 2)})</span></div>
      <div class="agg-item"><b>${fix(f.max_abs_prob_diff, 3)}</b><span>largest probability shift from 8-bit quantisation</span></div>
      <div class="agg-item tint-blue"><b>${ms(f.real_exec_ms_per_sample)}</b><span>mean genuine encrypted execution in the experiment</span></div>
    </div>
    ${dis.length ? `<p class="note">Disagreement${dis.length > 1 ? "s" : ""}: ${dis.map((d) =>
      `<a href="#" class="jump" data-id="${d.id}">patient #${d.id}</a> (plaintext ${fix(d.plain)} → encrypted ${fix(d.enc)})`).join(", ")}
      — a borderline probability where quantisation crosses the 0.5 threshold.</p>` : ""}`;
  document.querySelectorAll(".jump").forEach((a) => a.addEventListener("click", (e) => {
    e.preventDefault(); $("fxModel").value = "logreg"; $("fxPatient").value = a.dataset.id; loadPatient();
  }));

  const scenarios = await get("/api/scenarios");
  $("fxScenarios").innerHTML = scenarios.map((s) => `
    <button class="scenario ${s.tone}" data-key="${s.key}">
      <span class="scenario-title"><span class="dot"></span>${esc(s.title)}</span>
      <span class="scenario-p">p ≈ ${fix(s.p_disease, 2)}</span>
    </button>`).join("");
  document.querySelectorAll(".scenario").forEach((b) => b.addEventListener("click", async () => {
    const s = scenarios.find((x) => x.key === b.dataset.key);
    document.querySelectorAll(".scenario").forEach((x) => x.classList.toggle("active", x === b));
    $("fxModel").value = "logreg";     // scenarios are defined on the encrypted model
    $("fxPatient").value = s.patient_id;
    await loadPatient({ keepScenario: true });
    const [lo, hi] = s.band;
    $("fxScenarioNote").innerHTML = `<b>${esc(s.title)}</b> · ${esc(s.blurb)}. Patient #${s.patient_id} has
      out-of-fold p(disease) = ${fix(s.p_disease)}. Of the ${s.band_size} patients the model places in
      [${lo.toFixed(2)}, ${hi.toFixed(2)}${hi < 1 ? ")" : "]"}, <b>${pct(s.band_disease_rate, 0)}</b> actually have disease.`;
    $("fxScenarioNote").className = `scenario-note ${s.tone}`;
    runInference();
  }));

  $("fxPatient").onchange = () => loadPatient();
  $("fxModel").onchange = () => { clearScenario(); resetResult(); };
  $("fxRandom").onclick = () => {
    $("fxPatient").value = patients[Math.floor(Math.random() * patients.length)].id;
    loadPatient();
  };
  $("fxRun").onclick = runInference;
  $("fxPatient").value = patients[Math.floor(Math.random() * patients.length)].id;
  loadPatient();
}

function clearScenario() {
  document.querySelectorAll(".scenario").forEach((x) => x.classList.remove("active"));
  $("fxScenarioNote").className = "scenario-note hidden";
}

async function loadPatient({ keepScenario = false } = {}) {
  if (!keepScenario) clearScenario();
  resetResult();
  const p = await api(`/api/patient/${$("fxPatient").value}`);
  $("fxFoldNote").textContent = `Predicted by the fold-${p.fold} models, which never saw this patient during training.`;
  $("fxRecord").innerHTML = p.record.map((r) =>
    `<dt>${esc(r.label)}</dt><dd class="${r.value.startsWith("code") ? "code" : ""}">${esc(r.value)}</dd>`).join("");
}

function resetResult() {
  ["fxTEnc", "fxTSrv", "fxTDec"].forEach((id) => ($(id).textContent = "—"));
  $("fxCipher").classList.add("hidden");
  $("fxBanner").classList.add("hidden");
  for (const id of ["fxPlain", "fxEnc"]) {
    $(id).className = "result";
    $(id).querySelector(".result-label").textContent = "—";
    $(id).querySelector(".result-meta").textContent = "";
  }
}

async function runInference() {
  const btn = $("fxRun");
  btn.disabled = true; btn.textContent = "Running…";
  resetResult();
  document.querySelectorAll(".flow-step").forEach((s) => s.classList.add("running"));
  try {
    const r = await api(`/api/infer/${$("fxModel").value}`, { patient_id: +$("fxPatient").value });
    const p = r.plaintext, e = r.encrypted;
    setResult("fxPlain", p.label, `p(disease) = ${fix(p.prob)} · ${ms(p.latency_ms)}`);

    const banner = $("fxBanner");
    const truth = `Ground truth: <b>${r.truth.label}</b>`;
    if (e.available) {
      setResult("fxEnc", e.label, `p(disease) = ${fix(e.prob)} · ${ms(e.total_ms)} end-to-end`);
      const t = e.timings_ms;
      $("fxTEnc").textContent = ms(t.preprocess + t.encrypt_ms);
      $("fxTSrv").textContent = ms(t.server_ms);
      $("fxTDec").textContent = ms(t.decrypt_ms);
      $("fxCipher").innerHTML = `
        <div><span class="k">server receives</span> ${e.ciphertext_bytes.toLocaleString()} bytes of ciphertext · <span class="k">sha256</span> ${e.ciphertext_sha256}…</div>
        <div class="hex">${e.ciphertext_sample.match(/.{2}/g).join(" ")} …</div>
        <div><span class="k">server returns</span> ${e.result_ciphertext_bytes.toLocaleString()} bytes · fingerprint changes every run — encryption is randomised</div>`;
      $("fxCipher").classList.remove("hidden");
      const correct = (e.label === r.truth.label);
      banner.className = "banner " + (e.agree ? "agree" : "disagree");
      banner.innerHTML = e.agree
        ? `✓ Predictions agree — ${e.label}. ${truth} (${correct ? "correct" : "incorrect"}).<small>The server produced this without seeing any of the patient's values.</small>`
        : `⚠ Encrypted (${e.label}) and plaintext (${p.label}) disagree. ${truth}.<small>p(disease) is ${fix(p.prob)} in plaintext — close to 0.5, so 8-bit quantisation tips the decision.</small>`;
    } else {
      $("fxEnc").classList.add("off");
      $("fxEnc").querySelector(".result-label").textContent = "Phase 2";
      $("fxEnc").querySelector(".result-meta").textContent = "Not measured in current prototype";
      banner.className = "banner info";
      banner.innerHTML = `${esc(r.model_name)}: plaintext prediction ${p.label}. ${truth} (${p.label === r.truth.label ? "correct" : "incorrect"}).<small>${esc(e.reason)}</small>`;
    }
    banner.classList.remove("hidden");
  } catch (err) {
    $("fxBanner").className = "banner disagree";
    $("fxBanner").textContent = "Inference failed: " + err.message;
    $("fxBanner").classList.remove("hidden");
  } finally {
    btn.disabled = false; btn.textContent = "Run inference";
    document.querySelectorAll(".flow-step").forEach((s) => s.classList.remove("running"));
  }
}

function setResult(id, label, meta) {
  const el = $(id);
  el.className = "result " + (label === "Disease" ? "disease" : "healthy");
  el.querySelector(".result-label").textContent = label;
  el.querySelector(".result-meta").textContent = meta;
}

show("overview");
