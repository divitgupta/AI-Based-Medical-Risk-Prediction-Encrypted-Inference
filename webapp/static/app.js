/* HealthGuard dashboard (Phase 1 + Phase 2). Every number is fetched from the API, which
   serves what healthguard/experiment.py produced — nothing is hardcoded. */

const $ = (id) => document.getElementById(id);
// Colours are CSS variables, so switching theme recolours every chart
// without re-rendering. (SVG attributes cannot read var(), so charts use style.)
const MODEL_COLOR = { logreg: "var(--m-logreg)", xgboost: "var(--m-xgboost)", mlp: "var(--m-mlp)", elastic: "var(--accent)" };
const CLASS_COLOR = { 0: "var(--c-healthy)", 1: "var(--c-disease)" };
const CLASS_NAME = { 0: "No Disease", 1: "Disease" };
const METRICS = [
  ["accuracy", "Accuracy"], ["precision", "Precision"], ["recall", "Recall"],
  ["f1", "F1"], ["roc_auc", "ROC-AUC"],
];

const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
const fix = (v, d = 3) => Number(v).toFixed(d);
const ms = (v) => (v >= 1000 ? (v / 1000).toFixed(v >= 10000 ? 1 : 2) + " s" : v < 0.1 ? "< 0.1 ms" : v.toFixed(v < 10 ? 1 : 0) + " ms");
const bytes = (b) => b >= 1e9 ? (b / 1e9).toFixed(b >= 1e10 ? 0 : 2) + " GB" : b >= 1e6 ? (b / 1e6).toFixed(0) + " MB"
  : b >= 1e3 ? (b / 1e3).toFixed(1) + " kB" : b === 0 ? "none" : Math.round(b) + " B";
const qset = (st) => `${st.n_bits}-bit${st.rounding_threshold_bits ? `, rounding ${st.rounding_threshold_bits}` : ""}`;
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
const renderers = { overview: renderOverview, dataset: renderDataset, models: renderModels, fhe: renderFhe, phase2: renderPhase2, elastic: renderElastic };

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

/* Horizontal bars on a log scale: rows = [{label, value, color, text}] */
function logBars(rows, { width = 520, unit = "" } = {}) {
  const m = { l: 150, r: 90 }, rowH = 46, height = rows.length * rowH + 30;
  const vals = rows.map((r) => r.value).filter((v) => v > 0);
  const lo = Math.pow(10, Math.floor(Math.log10(Math.min(...vals)))), hi = Math.pow(10, Math.ceil(Math.log10(Math.max(...vals))));
  const x = (v) => m.l + (Math.log10(Math.max(v, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo) || 1) * (width - m.l - m.r);
  let body = "";
  for (let e = Math.log10(lo); e <= Math.log10(hi); e++) {
    const px = x(Math.pow(10, e));
    body += `<line x1="${px}" x2="${px}" y1="4" y2="${height - 24}" style="stroke:var(--chart-grid)"/>
      <text x="${px}" y="${height - 8}" text-anchor="middle">${esc(unit === "s" ? (Math.pow(10, e) >= 1 ? Math.pow(10, e) + " s" : Math.pow(10, e) * 1000 + " ms") : bytes(Math.pow(10, e)))}</text>`;
  }
  rows.forEach((r, i) => {
    const y = 8 + i * rowH;
    body += `<text x="${m.l - 10}" y="${y + 20}" text-anchor="end" style="font-weight:600">${esc(r.label)}</text>`;
    if (r.value > 0) {
      body += `<rect x="${m.l}" y="${y + 6}" width="${Math.max(2, x(r.value) - m.l)}" height="22" rx="4" style="fill:${r.color}" data-tip="${esc(r.label)}: ${esc(r.text)}"/>
        <text x="${x(r.value) + 8}" y="${y + 22}">${esc(r.text)}</text>`;
    } else {
      body += `<text x="${m.l + 4}" y="${y + 22}" style="fill:var(--text-2)">${esc(r.text)}</text>`;
    }
  });
  return `<svg viewBox="0 0 ${width} ${height}" role="img" style="max-width:${width * 1.2}px">${body}</svg>`;
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
    ["FHE", "LR · XGBoost · MLP via Concrete ML", "fhe"],
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

  $("ovFhe").innerHTML = `<table class="table compact"><thead><tr><th>Model</th><th>Same decision</th><th>Encrypted accuracy</th><th>Time / patient</th></tr></thead><tbody>` +
    keys.map((k) => {
      const f = o.fhe[k];
      return `<tr><td><span class="model-name"><span class="swatch" style="background:${MODEL_COLOR[k]}"></span>${esc(o.models[k].name)}</span></td>
        <td>${o.n_samples - f.n_disagree}/${o.n_samples}</td><td>${pct(f.pooled.accuracy)}</td><td>${ms(f.real_exec_ms_per_sample)}</td></tr>`;
    }).join("") + `</tbody></table>
    <p class="note">The server evaluates each model on ciphertext only — it never sees a feature value or holds the decryption key.
    Time is genuine encrypted execution on this laptop; cost grows steeply with model complexity (see the last tab).</p>`;
}

/* ═══════════════════ 2 · DATASET ═══════════════════ */
async function renderDataset() {
  const [hd, bp] = await Promise.all([get("/api/eda/heart"), get("/api/eda/base_paper")]);
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

  $("bpStats").innerHTML = statCards([
    [bp.n_rows.toLocaleString(), "Rows in the file", bp.source, "blue"],
    [bp.n_distinct, "Distinct patients", `${bp.n_duplicate_rows} rows are exact copies`, "amber"],
    [bp.n_shared_with_ours, "Shared with our data", `of our ${bp.n_ours_distinct} distinct Cleveland records`, "teal"],
    [bp.class_counts_distinct["1"], "Distinct with disease", `${bp.class_counts_distinct["0"]} without (same inverted labels)`, "disease"],
  ]);
  const copies = Object.entries(bp.copies_per_patient);
  const maxC = Math.max(...copies.map(([, c]) => c));
  $("bpCopies").innerHTML = `<div class="hbars">${copies.map(([n, c]) => `
    <div class="hbar-row"><span class="hbar-label">${n} cop${n === "1" ? "y" : "ies"}</span>
      <span class="hbar-track"><span class="hbar-fill" style="width:${(c / maxC) * 100}%"></span></span>
      <span class="hbar-val">${c} patients</span></div>`).join("")}</div>
    <p class="note">Each distinct patient record appears this many times in the base paper's file.</p>`;
  $("bpNote").innerHTML = `<p class="note">The base paper's file is not a larger dataset: it is <b>our ${bp.n_distinct} Cleveland
    patients repeated</b> (identical values and labels), ${bp.n_rows.toLocaleString()} rows in total.</p>
    <p class="note">If train and test are split by <i>row</i>, almost every test patient also sits in the training set,
    so the model is partly tested on patients it has already memorised. The Encryption Cost &amp; Base Paper tab
    measures how much that inflates accuracy.</p>`;

  document.querySelectorAll(".subtab").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".subtab").forEach((x) => x.classList.toggle("active", x === b));
    $("dsHeart").classList.toggle("hidden", b.dataset.ds !== "heart");
    $("dsBase").classList.toggle("hidden", b.dataset.ds !== "base");
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
let OV = null;   // overview payload, shared by the inference handlers

async function renderFhe() {
  const [o, patients] = await Promise.all([get("/api/overview"), get("/api/patients")]);
  OV = o;

  $("fxModel").innerHTML = o.model_order.map((k) =>
    `<option value="${k}">${esc(o.models[k].name)} — ${o.fhe_status[k].ready
      ? `encrypted ✓ (~${ms(o.fhe[k].real_exec_ms_per_sample)})` : "plaintext only"}</option>`).join("");
  $("fxPatient").innerHTML = patients.map((p) => `<option value="${p.id}">Patient #${p.id} · fold ${p.fold}</option>`).join("");

  const scenarios = await get("/api/scenarios");
  $("fxScenarios").innerHTML = scenarios.map((s) => `
    <button class="scenario ${s.tone}" data-key="${s.key}">
      <span class="scenario-title"><span class="dot"></span>${esc(s.title)}</span>
      <span class="scenario-p">p ≈ ${fix(s.p_disease, 2)}</span>
    </button>`).join("");
  document.querySelectorAll(".scenario").forEach((b) => b.addEventListener("click", async () => {
    const s = scenarios.find((x) => x.key === b.dataset.key);
    document.querySelectorAll(".scenario").forEach((x) => x.classList.toggle("active", x === b));
    $("fxModel").value = "logreg";     // scenarios are defined on the logistic regression
    modelChanged();
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
  $("fxModel").onchange = () => { clearScenario(); resetResult(); modelChanged(); };
  $("fxRandom").onclick = () => {
    $("fxPatient").value = patients[Math.floor(Math.random() * patients.length)].id;
    loadPatient();
  };
  $("fxRun").onclick = runInference;
  $("fxPatient").value = patients[Math.floor(Math.random() * patients.length)].id;
  modelChanged();
  loadPatient();
}

function modelChanged() {
  const o = OV, key = $("fxModel").value, f = o.fhe[key], dis = o.fhe_disagreements[key];
  const slow = f.real_exec_ms_per_sample > 2000;
  $("fxModelNote").classList.toggle("hidden", !slow);
  if (slow) {
    $("fxModelNote").innerHTML = `Genuine encrypted execution takes ~${ms(f.real_exec_ms_per_sample)} per patient for this model
      (${Math.round(f.cost_mean.pbs_count).toLocaleString()} bootstraps). The first run on a fold also loads
      ~${bytes(f.cost_mean.evaluation_key_bytes)} of evaluation keys.`;
  }
  $("fxAggModel").textContent = "· " + o.models[key].name;
  $("fxAggregate").innerHTML = `
    <div class="agg">
      <div class="agg-item tint-healthy"><b>${o.n_samples - f.n_disagree}/${o.n_samples}</b><span>same decision encrypted vs plaintext (${pct(f.agreement, 2)})</span></div>
      <div class="agg-item"><b>${fix(f.max_abs_prob_diff, 3)}</b><span>largest probability shift from ${qset(f.settings)} quantisation</span></div>
      <div class="agg-item tint-blue"><b>${ms(f.real_exec_ms_per_sample)}</b><span>mean genuine encrypted execution in the experiment</span></div>
    </div>
    ${dis.length ? `<p class="note">Disagreement${dis.length > 1 ? "s" : ""}: ${dis.map((d) =>
      `<a href="#" class="jump" data-id="${d.id}">patient #${d.id}</a> (plaintext ${fix(d.plain)} → encrypted ${fix(d.enc)})`).join(", ")}
      — borderline probabilities where quantisation crosses the 0.5 threshold.</p>` : `<p class="note">No disagreements: every encrypted decision matches plaintext.</p>`}`;
  document.querySelectorAll(".jump").forEach((a) => a.addEventListener("click", (e) => {
    e.preventDefault(); $("fxPatient").value = a.dataset.id; loadPatient();
  }));
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
  const btn = $("fxRun"), key = $("fxModel").value;
  const expected = OV ? OV.fhe[key].real_exec_ms_per_sample : 0;
  btn.disabled = true;
  btn.textContent = expected > 2000 ? `Running encrypted… (~${ms(expected)})` : "Running…";
  resetResult();
  document.querySelectorAll(".flow-step").forEach((s) => s.classList.add("running"));
  try {
    const r = await api(`/api/infer/${key}`, { patient_id: +$("fxPatient").value });
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
        <div><span class="k">server returns</span> ${e.result_ciphertext_bytes.toLocaleString()} bytes · fingerprint changes every run — encryption is randomised</div>
        ${e.setup_ms > 0 ? `<div><span class="k">one-off</span> ${ms(e.setup_ms)} to load this fold's circuit and generate its keys (not counted above)</div>` : ""}`;
      $("fxCipher").classList.remove("hidden");
      const correct = (e.label === r.truth.label);
      const q = OV ? qset(OV.fhe[key].settings) : "low-bit";
      banner.className = "banner " + (e.agree ? "agree" : "disagree");
      banner.innerHTML = e.agree
        ? `✓ Predictions agree — ${e.label}. ${truth} (${correct ? "correct" : "incorrect"}).<small>The server produced this without seeing any of the patient's values.</small>`
        : `⚠ Encrypted (${e.label}) and plaintext (${p.label}) disagree. ${truth}.<small>p(disease) is ${fix(p.prob)} in plaintext — close to 0.5, so ${q} quantisation tips the decision.</small>`;
    } else {
      $("fxEnc").classList.add("off");
      $("fxEnc").querySelector(".result-label").textContent = "Unavailable";
      $("fxEnc").querySelector(".result-meta").textContent = e.reason;
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

/* ═══════════════════ 5 · PHASE 2 ═══════════════════ */
async function renderPhase2() {
  const d = await get("/api/phase2");
  const order = d.model_order, n = 303;
  const name = (k) => `<span class="model-name"><span class="swatch" style="background:${MODEL_COLOR[k]}"></span>${esc(d.models[k].name)}</span>`;

  $("p2Accuracy").innerHTML = `<thead><tr><th>Model</th><th>Quantisation</th><th>Plaintext acc.</th><th>Encrypted acc.</th>
    <th class="recall">Encrypted recall ★</th><th>Encrypted AUC</th><th>Same decision</th><th>Max prob. shift</th></tr></thead><tbody>` +
    order.map((k) => {
      const f = d.fhe[k], p = d.models[k].pooled;
      return `<tr><td>${name(k)}</td><td>${qset(f.settings)}</td><td>${pct(p.accuracy)}</td><td><b>${pct(f.pooled.accuracy)}</b></td>
        <td class="recall">${pct(f.pooled.recall)}</td><td>${fix(f.pooled.roc_auc)}</td>
        <td>${n - f.n_disagree}/${n}</td><td>${fix(f.max_abs_prob_diff)}</td></tr>`;
    }).join("") + "</tbody>";
  $("p2AccuracyNote").textContent = "Encrypted predictions for all 303 patients come from FHE simulation (the circuit's exact integer arithmetic); " +
    "genuine encryption was run on sample patients in every fold and matched the simulation. A small accuracy change in either direction " +
    "comes from a few borderline patients crossing 0.5, not from encryption helping or hurting the model.";

  const cx = (k) => {
    const c = d.models[k].complexity;
    return k === "xgboost" ? `${range(c.map((x) => x.n_trees))} trees` : `${range(c.map((x) => x.n_params))} params`;
  };
  $("p2Cost").innerHTML = `<thead><tr><th>Model</th><th>Size</th><th>Bootstraps (PBS) / patient</th><th>Max bit-width</th>
    <th>Evaluation keys</th><th>Ciphertext in → out</th><th>Compile</th><th>Key generation</th><th>Encrypted time / patient</th><th>Peak extra memory</th></tr></thead><tbody>` +
    order.map((k) => {
      const f = d.fhe[k], c = f.cost_mean;
      return `<tr><td>${name(k)}</td><td>${esc(cx(k))}</td><td>${Math.round(c.pbs_count).toLocaleString()}</td><td>${f.cost_range.max_bit_width[1]}</td>
        <td>${bytes(c.evaluation_key_bytes)}</td><td>${bytes(c.input_ciphertext_bytes)} → ${bytes(c.output_ciphertext_bytes)}</td>
        <td>${ms(f.compile_seconds_mean * 1000)}</td><td>${ms(f.keygen_seconds_mean * 1000)}</td>
        <td><b>${ms(f.real_exec_ms_per_sample)}</b></td><td>${bytes(f.peak_extra_memory_mb_max * 1e6)}</td></tr>`;
    }).join("") + "</tbody>";

  $("p2Time").innerHTML = logBars(order.map((k) => ({
    label: d.models[k].name, value: d.fhe[k].real_exec_ms_per_sample / 1000, color: MODEL_COLOR[k],
    text: ms(d.fhe[k].real_exec_ms_per_sample),
  })), { unit: "s" }) + `<p class="note">Genuine encrypted execution per patient on this laptop (CPU).</p>`;
  $("p2Keys").innerHTML = logBars(order.map((k) => ({
    label: d.models[k].name, value: d.fhe[k].cost_mean.evaluation_key_bytes, color: MODEL_COLOR[k],
    text: d.fhe[k].cost_mean.evaluation_key_bytes ? bytes(d.fhe[k].cost_mean.evaluation_key_bytes) : "none (no bootstrapping)",
  }))) + `<p class="note">Logistic regression needs no bootstrapping: only its weighted sum runs encrypted, and the client applies the sigmoid after decrypting.</p>`;

  const q = d.study.quantisation;
  $("p2Quant").innerHTML = `<thead><tr><th class="left">Model</th><th>Setting</th><th>Same decision</th><th>Max prob. shift</th><th>PBS</th><th>Max bits</th><th>Eval. keys</th></tr></thead><tbody>` +
    Object.entries(q).map(([k, rows]) => rows.map((r) => `<tr class="${r.chosen ? "chosen" : ""}">
      <td class="left">${esc(d.models[k].name)}</td><td>${qset(r.setting)}${r.chosen ? " ✓" : ""}</td><td>${r.n_agree}/${n}</td>
      <td>${fix(r.max_abs_prob_diff)}</td><td>${Math.round(r.pbs_mean).toLocaleString()}</td><td>${r.max_bit_width}</td>
      <td>${bytes(r.evaluation_key_bytes_mean)}</td></tr>`).join("")).join("") + "</tbody>";

  const a = d.study.activation;
  $("p2Act").innerHTML = `<thead><tr><th class="left">Activation</th><th>Plaintext acc.</th><th class="recall">Recall ★</th><th>AUC</th><th>Encrypted acc.</th><th>PBS</th></tr></thead><tbody>` +
    [["ReLU (used)", a.relu], ["x² (CryptoNets-style)", a.square]].map(([l, r]) => `<tr><td class="left">${l}</td>
      <td>${pct(r.pooled.accuracy)}</td><td class="recall">${pct(r.pooled.recall)}</td><td>${fix(r.pooled.roc_auc)}</td>
      <td>${pct(r.encrypted.accuracy)}</td><td>${Math.round(r.pbs_mean)}</td></tr>`).join("") + "</tbody>";
  $("p2ActNote").textContent = "CKKS-style schemes can only add and multiply, so they need a polynomial such as x² in place of ReLU. " +
    "TFHE evaluates ReLU exactly as a table lookup, so the x² swap is not required here, and it does not make the circuit cheaper: " +
    "both activations cost one bootstrap per hidden unit.";

  const b = d.base_paper;
  if (!b) { $("p2Base").innerHTML = ""; $("p2BaseNote").textContent = "Run scripts/run_experiments.sh to produce results/base_paper.json."; return; }
  const rep = b.base_paper_reported;
  const row = (label, lr, enc, xgb, mlp, leak, cls = "") => `<tr class="${cls}"><td class="left">${label}</td><td>${lr}</td><td>${enc}</td><td>${xgb}</td><td>${mlp}</td><td>${leak}</td></tr>`;
  const ours = (r) => [pct(r.models.logreg.accuracy), pct(r.fhe_logreg.pooled.accuracy), pct(r.models.xgboost.accuracy), pct(r.models.mlp.accuracy),
    pct(r.test_rows_with_copy_in_training_pct, 0)];
  $("p2Base").innerHTML = `<thead><tr><th class="left">Evaluation (accuracy)</th><th>Logistic Reg.</th><th>Encrypted LR</th><th>XGBoost</th><th>MLP</th>
    <th>Test rows also in training</th></tr></thead><tbody>` +
    row(`Base paper, as reported <span class="muted">· ${esc(rep.scheme)}, split ${esc(rep.split)}</span>`, pct(rep.lr_plaintext_accuracy, 0),
      pct(rep.lr_encrypted_accuracy.poly_modulus_8192, 0), "—", "—", "?") +
    row("Ours · split by <b>row</b> (copies leak)", ...ours(b.row_split), "warn-row") +
    row("Ours · split by <b>patient</b> (no leakage)", ...ours(b.patient_split), "chosen") +
    "</tbody>";
  $("p2BaseNote").innerHTML = `On the base paper's own file, our logistic regression matches its reported accuracy, and our encrypted version
    matches plaintext more closely (${pct(b.patient_split.fhe_logreg.agreement, 1)} same decisions). With a row split,
    ${pct(b.row_split.test_rows_with_copy_in_training_pct, 0)} of test rows have an identical copy in training, so flexible models look much better
    than they are: XGBoost and the MLP gain ${(100 * (b.row_split.models.xgboost.accuracy - b.patient_split.models.xgboost.accuracy)).toFixed(0)} and
    ${(100 * (b.row_split.models.mlp.accuracy - b.patient_split.models.mlp.accuracy)).toFixed(0)} points. Splitting by patient removes the leak,
    and the result agrees with our main 303-patient study: added complexity does not help.`;
}

/* ═══════════════════ 6 · ELASTIC LEARNER ═══════════════════ */
async function renderElastic() {
  const [e, res] = await Promise.all([get("/api/elastic"), get("/api/results")]);
  const order = e.order, R = e.rows;
  const name = (k) => `<span class="model-name"><span class="swatch" style="background:${MODEL_COLOR[k]}"></span>${esc(R[k].name)}</span>`;

  $("elHow").innerHTML = [
    ["Base predictions", "LR, XGBoost and MLP each give p(disease)"],
    ["No leakage", "for training patients, from models that never saw them (inner 5-fold)"],
    ["Elastic-net learner", "logistic regression on the 3 logits · L1 can drop a model, L2 keeps weights stable"],
    ["Tuned per fold", `C and L1 ratio chosen by inner CV on F2 (${e.elastic.grid.C.length * e.elastic.grid.l1_ratio.length} combinations)`],
    ["Final answer", "one combined probability per patient"],
  ].map(([t, s]) => `<li><strong>${t}</strong><span>${s}</span></li>`).join("");

  const rec = e.recommendation, rk = rec.model;
  $("elRecommend").innerHTML = `<p class="eyebrow ok"><span class="dot"></span>Recommended model</p>
    <div class="rec-name">${name(rk)}</div>
    <div class="agg">
      <div class="agg-item tint-healthy"><b>${fix(R[rk].pooled.f2)}</b><span>F2 (recall-weighted)</span></div>
      <div class="agg-item"><b>${pct(R[rk].pooled.recall)}</b><span>recall</span></div>
      <div class="agg-item tint-blue"><b>${ms(R[rk].encrypted_ms)}</b><span>per patient, encrypted</span></div>
    </div>
    <p class="note"><b>Rule:</b> ${esc(rec.rule)}</p>
    <p class="note">Tied on F2: ${rec.tied_on_f2.map((k) => esc(R[k].name)).join(", ")}.
    ${rk !== "elastic" && rec.tied_on_f2.includes("elastic") ? "Combining the models does not beat the best single one, and it would need every encrypted circuit." : ""}</p>`;

  const cols = [["accuracy", "Accuracy", pct], ["precision", "Precision", pct], ["recall", "Recall ★", pct],
    ["f1", "F1", pct], ["f2", "F2 ★", fix], ["roc_auc", "ROC-AUC", fix]];
  const best = Object.fromEntries(cols.map(([k]) => [k, Math.max(...order.map((m) => R[m].pooled[k]))]));
  const fastest = Math.min(...order.map((m) => R[m].encrypted_ms));
  $("elBoard").innerHTML = `<thead><tr><th>Model</th>${cols.map(([, l]) => `<th>${l}</th>`).join("")}<th>Encrypted time</th></tr></thead><tbody>` +
    order.map((m) => `<tr class="${m === "elastic" ? "chosen" : ""}"><td>${name(m)}${m === rk ? " 👑" : ""}</td>` +
      cols.map(([k, , f]) => `<td class="${R[m].pooled[k] === best[k] ? "best" : ""}">${f(R[m].pooled[k])}</td>`).join("") +
      `<td class="${R[m].encrypted_ms === fastest ? "best" : ""}">${ms(R[m].encrypted_ms)}</td></tr>`).join("") + "</tbody>";

  $("elBest").innerHTML = Object.values(e.best).map((b) => `
    <div class="best-card" style="--c:${MODEL_COLOR[b.model]}">
      <span class="best-crit">${esc(b.label)}</span>
      <span class="best-model">${esc(R[b.model].name)}</span>
      <span class="best-val">${b.label.startsWith("Encrypted") ? ms(b.value) : b.label.startsWith("ROC") || b.label.startsWith("F2") ? fix(b.value) : pct(b.value)}
        <small>${b.direction === "lower" ? "lowest" : "highest"}</small></span>
    </div>`).join("");

  // Weight chart: one group per fold, one bar per base model (can be negative).
  const base = order.filter((m) => m !== "elastic"), F = e.elastic.folds;
  const w = 520, h = 260, m = { t: 14, r: 10, b: 36, l: 44 };
  const all = F.flatMap((f) => base.map((k) => f.weights[k]));
  const top = Math.max(0.5, ...all), bot = Math.min(0, ...all);
  const y = scale(bot, top, h - m.b, m.t), gw = (w - m.l - m.r) / F.length, bw = Math.min(22, gw / 4.5);
  let svg = "";
  [bot, 0, top / 2, top].forEach((v) => (svg += `<line x1="${m.l}" x2="${w - m.r}" y1="${y(v)}" y2="${y(v)}" style="stroke:${v === 0 ? "var(--chart-axis)" : "var(--chart-grid)"}"/>
    <text x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(2)}</text>`));
  F.forEach((f, i) => {
    const cx = m.l + gw * i + gw / 2;
    base.forEach((k, j) => {
      const v = f.weights[k], bx = cx + (j - 1.5) * bw + 1;
      svg += `<rect x="${bx}" y="${Math.min(y(v), y(0))}" width="${bw - 2}" height="${Math.max(1, Math.abs(y(v) - y(0)))}" rx="3"
        style="fill:${MODEL_COLOR[k]}" data-tip="Fold ${i + 1} · ${R[k].name}: ${v.toFixed(3)}"/>`;
    });
    svg += `<text x="${cx}" y="${h - m.b + 16}" text-anchor="middle">fold ${i + 1}</text>`;
  });
  $("elWeights").innerHTML = `<svg viewBox="0 0 ${w} ${h}" role="img" style="max-width:${w * 1.2}px">${svg}</svg>` +
    legend(base.map((k) => ({ color: MODEL_COLOR[k], label: R[k].name })));
  const dropped = F.flatMap((f) => f.dropped.map((k) => `${R[k].name} (fold ${f.fold + 1})`));
  $("elWeightsNote").textContent = `A larger weight means the learner trusts that model's opinion more. ` +
    (dropped.length ? `Dropped by the L1 penalty: ${dropped.join(", ")}. ` : "No model was dropped outright in any fold. ") +
    `The weights change a lot from fold to fold, another sign that no model is reliably better than the others on 303 patients.`;

  const M = res.metrics.models;
  $("elRoc").innerHTML = lineChart([
    ...base.map((k) => ({ x: M[k].roc.fpr, y: M[k].roc.tpr, color: MODEL_COLOR[k], name: R[k].name, width: 1.6, dash: "4 3" })),
    { x: e.elastic.roc.fpr, y: e.elastic.roc.tpr, color: MODEL_COLOR.elastic, name: "Elastic Learner", width: 2.8 },
  ], { square: true, diagonal: true, xLabel: "False positive rate", yLabel: "True positive rate (recall)", width: 460, height: 340 }) +
    legend(order.map((k) => ({ color: MODEL_COLOR[k], label: `${R[k].name} · AUC ${fix(R[k].pooled.roc_auc)}` })));
}

show("overview");
