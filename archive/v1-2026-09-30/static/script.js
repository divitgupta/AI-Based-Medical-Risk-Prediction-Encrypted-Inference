document.addEventListener('DOMContentLoaded', () => {
    const $ = id => document.getElementById(id);

    const tabs = document.querySelectorAll('.tab');
    const viewBtns = document.querySelectorAll('.view-btn');
    const btnLoadPatient = $('btnLoadPatient');
    const btnPredict = $('btnPredict');
    const patientDataContainer = $('patientData');
    const trueLabelBox = $('trueLabelBox');
    const trueLabelValue = $('trueLabelValue');

    let currentDataset = 'breast_cancer';
    let currentView = 'data';
    let currentFeatures = null;
    let currentTruth = null;
    let allMetrics = {};
    let status = null;

    const pct = v => (v * 100).toFixed(2) + '%';
    const ms = v => v >= 1000 ? (v / 1000).toFixed(2) + ' s' : v.toFixed(1) + ' ms';
    const bytes = b => b >= 1024 ? (b / 1024).toFixed(1) + ' KB' : b + ' B';
    const esc = s => String(s).replace(/[&<>"]/g, c =>
        ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

    // ── Boot ─────────────────────────────────────────────────────────────
    Promise.all([
        fetch('/api/metrics').then(r => r.json()),
        fetch('/api/status').then(r => r.json()),
    ]).then(([metrics, st]) => {
        allMetrics = metrics;
        status = st;
        renderStatus();
        renderAll();
    }).catch(err => {
        console.error(err);
        $('fheStatusText').innerText = 'Backend unreachable';
    });

    function renderAll() {
        renderEDA();
        renderModels();
        renderEvaluation();
        renderInferenceMetrics();
        renderBenchmarks();
    }

    function renderStatus() {
        const box = $('fheStatus');
        const ok = status.fhe_available[currentDataset];
        box.classList.toggle('ok', ok);
        box.classList.toggle('warn', !ok);
        $('fheStatusText').innerText = ok
            ? 'FHE circuit live — inference runs on real ciphertext'
            : 'FHE unavailable: ' + (status.fhe_error || 'circuit not compiled') + ' — plaintext only';
        btnPredict.querySelector('.btn-text').innerText = ok
            ? 'Run Plaintext & Encrypted Inference'
            : 'Run Plaintext Inference';
    }

    // ── View switching ───────────────────────────────────────────────────
    viewBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            viewBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentView = btn.dataset.view;
            document.querySelectorAll('.view').forEach(v => v.classList.add('hidden'));
            $('view-' + currentView).classList.remove('hidden');
            $('fheStatus').classList.toggle('hidden', currentView !== 'inference');
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    });
    $('fheStatus').classList.add('hidden');   // only shown on the inference view

    // ── Dataset switching ────────────────────────────────────────────────
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentDataset = tab.dataset.dataset;
            renderStatus();
            renderAll();
            resetPanel();
        });
    });

    // ═════════════════ VIEW 1 — DATA & EDA ═══════════════════════════════
    function renderEDA() {
        fetch(`/api/eda/${currentDataset}`).then(r => r.json()).then(d => {
            $('edaTitle').innerText = d.title;
            $('edaSource').innerText = d.source;

            $('edaStats').innerHTML = [
                ['Samples', d.n_samples],
                ['Raw features', d.n_features],
                ['Rows dropped', d.n_missing],
                ['Split', d.split.replace('Stratified ', '').replace(' train-test', '')],
            ].map(([k, v]) => `
                <div class="stat">
                    <span class="stat-value">${esc(v)}</span>
                    <span class="stat-label">${esc(k)}</span>
                </div>`).join('');

            const total = d.class_balance.reduce((a, c) => a + c.count, 0);
            $('classBalance').innerHTML = d.class_balance.map((c, i) => {
                const share = c.count / total;
                return `<div class="cb-row">
                    <span class="cb-label">${esc(c.label)}</span>
                    <div class="cb-track"><div class="cb-fill c${i}" style="width:${share * 100}%"></div></div>
                    <span class="cb-count">${c.count} · ${(share * 100).toFixed(1)}%</span>
                </div>`;
            }).join('');

            const maj = Math.max(...d.class_balance.map(c => c.count)) / total;
            $('balanceNote').innerText = maj > 0.6
                ? `Moderately imbalanced (majority class ${(maj * 100).toFixed(0)}%) — handled with class weighting during training.`
                : `Reasonably balanced (majority class ${(maj * 100).toFixed(0)}%), so accuracy is a meaningful metric here.`;

            $('prepGrid').innerHTML = [
                ['Feature types', d.feature_types],
                ['Pipeline', d.preprocessing],
                ['Missing values', d.n_missing === 0 ? 'None' : `${d.n_missing} rows dropped`],
                ['Train/test split', d.split],
            ].map(([k, v]) => `<div class="kv"><span class="k">${esc(k)}</span><span class="v small">${esc(v)}</span></div>`).join('');

            $('edaPlots').innerHTML = d.plots.map(p => `
                <figure class="plot-card" data-src="/plots/${p.file}" data-caption="${esc(p.caption)}">
                    <img src="/plots/${p.file}" alt="${esc(p.caption)}" loading="lazy">
                    <figcaption>${esc(p.caption)}</figcaption>
                </figure>`).join('');
            bindLightbox('#edaPlots .plot-card');
        });
    }

    // ═════════════════ VIEW 2 — MODELS ═══════════════════════════════════
    function renderModels() {
        fetch(`/api/models/${currentDataset}`).then(r => r.json()).then(d => {
            const p = d.protocol;
            $('protoScoring').innerText = 'Optimised for ' + p.scoring;
            $('protocolGrid').innerHTML = [
                ['Cross-validation', p.cv],
                ['Selection metric', p.scoring],
                ['Class imbalance', p.class_imbalance],
                ['Random seed', p.seed],
            ].map(([k, v]) => `<div class="kv"><span class="k">${esc(k)}</span><span class="v small">${esc(v)}</span></div>`).join('');
            $('protoNote').innerText = p.scoring_rationale;

            $('modelCards').innerHTML = d.models.map(m => {
                const selected = Object.entries(m.selected).map(([k, v]) =>
                    `<div class="hp"><span class="hp-k">${esc(k)}</span><span class="hp-v">${esc(v)}</span></div>`).join('');

                const searched = Object.keys(m.searched).length
                    ? `<div class="searched"><span class="k">Grid searched over</span>` +
                      Object.entries(m.searched).map(([k, v]) =>
                        `<code>${esc(k)} ∈ {${v.map(esc).join(', ')}}</code>`).join(' ') + `</div>`
                    : '';

                let weights = '';
                if (m.weights && m.weights.length) {
                    const max = Math.max(...m.weights.map(w => Math.abs(w.value)));
                    const signed = m.weights.some(w => w.value < 0);
                    weights = `<h4 class="sub-head">${esc(m.weights_label)}</h4>
                        <div class="weight-list">` + m.weights.map(w => {
                            const frac = Math.abs(w.value) / max * 100;
                            const neg = w.value < 0;
                            return `<div class="weight-row">
                                <span class="w-name">${esc(w.feature)}</span>
                                <div class="w-track ${signed ? 'signed' : ''}">
                                    <div class="w-fill ${neg ? 'neg' : 'pos'}" style="width:${frac / (signed ? 2 : 1)}%"></div>
                                </div>
                                <span class="w-val">${w.value.toFixed(3)}</span>
                            </div>`;
                        }).join('') + `</div>`;
                }

                return `<div class="panel glass model-card">
                    <div class="panel-header">
                        <h2>${esc(m.name)}</h2>
                        <span class="badge">${esc(m.family)}</span>
                    </div>
                    <div class="hp-grid">${selected}</div>
                    ${searched}
                    <p class="note">${esc(m.notes)}</p>
                    ${weights}
                </div>`;
            }).join('');
        });
    }

    // ═════════════════ VIEW 3 — EVALUATION ═══════════════════════════════
    function renderEvaluation() {
        // Grouped bar chart across the three plaintext models
        const block = allMetrics[currentDataset] || {};
        const models = ['LogisticRegression', 'XGBoost', 'MLP'].filter(m => block[m]);
        const metrics = [
            ['accuracy', 'Accuracy'], ['precision', 'Precision'],
            ['recall', 'Recall'], ['f1', 'F1'], ['roc_auc', 'ROC-AUC'],
        ];

        $('metricChart').innerHTML = `
            <div class="chart-legend">` + models.map((m, i) =>
                `<span class="legend-item"><i class="swatch s${i}"></i>${esc(m)}</span>`).join('') + `</div>
            <div class="chart-groups">` + metrics.map(([key, label]) => `
                <div class="chart-group">
                    <div class="bars">` + models.map((m, i) => {
                        const v = block[m][key];
                        return `<div class="bar-wrap" title="${esc(m)}: ${pct(v)}">
                            <span class="bar-val">${(v * 100).toFixed(1)}</span>
                            <div class="bar s${i}" style="height:${Math.max(0, (v - 0.5) / 0.5 * 100)}%"></div>
                        </div>`;
                    }).join('') + `</div>
                    <span class="group-label">${esc(label)}</span>
                </div>`).join('') + `</div>
            <p class="note">Bars are scaled from 0.50 to 1.00 so the differences between
            models are visible. All values are on the held-out test set.</p>`;

        // Confusion matrices + ROC curves
        fetch(`/api/evaluation/${currentDataset}`).then(r => r.json()).then(d => {
            $('evalCount').innerText = `${d.variants.length} model variants`;
            $('evalGrid').innerHTML = d.variants.map(v => `
                <div class="eval-card ${v.fhe ? 'is-fhe' : ''}">
                    <h4>${v.fhe ? '🔐 ' : ''}${esc(v.label)}</h4>
                    <div class="eval-pair">
                        <figure class="plot-card" data-src="/plots/${v.cm}" data-caption="${esc(v.label)} — confusion matrix">
                            <img src="/plots/${v.cm}" alt="confusion matrix" loading="lazy">
                        </figure>
                        <figure class="plot-card" data-src="/plots/${v.roc}" data-caption="${esc(v.label)} — ROC curve">
                            <img src="/plots/${v.roc}" alt="ROC curve" loading="lazy">
                        </figure>
                    </div>
                </div>`).join('');
            bindLightbox('#evalGrid .plot-card');
        });
    }

    function renderBenchmarks() {
        const body = $('benchBody');
        const names = { breast_cancer: 'Breast Cancer', heart_disease: 'Heart Disease' };
        let html = '';
        for (const ds of ['breast_cancer', 'heart_disease']) {
            const block = allMetrics[ds] || {};
            for (const model of ['LogisticRegression', 'XGBoost', 'MLP']) {
                const m = block[model];
                if (!m) continue;
                html += `<tr class="${ds === currentDataset ? 'hl' : ''}">
                    <td>${names[ds]}</td><td>${model}</td>
                    <td>${pct(m.accuracy)}</td><td>${pct(m.precision)}</td>
                    <td>${pct(m.recall)}</td><td>${pct(m.f1)}</td>
                    <td>${m.roc_auc.toFixed(4)}</td></tr>`;
            }
            for (const [mtype, data] of Object.entries(block.FHE || {})) {
                if (!data || !data.encrypted) continue;
                const m = data.encrypted, t = data.timing || {};
                const lat = t.real_fhe_per_sample_ms
                    ? ` · ${ms(t.real_fhe_per_sample_ms)}/patient encrypted` : '';
                html += `<tr class="fhe-row ${ds === currentDataset ? 'hl' : ''}">
                    <td>${names[ds]}</td><td>🔐 ${mtype} (FHE)${lat}</td>
                    <td>${pct(m.accuracy)}</td><td>${pct(m.precision)}</td>
                    <td>${pct(m.recall)}</td><td>${pct(m.f1)}</td>
                    <td>${m.roc_auc != null ? m.roc_auc.toFixed(4) : '--'}</td></tr>`;
            }
        }
        body.innerHTML = html || '<tr><td colspan="7" class="muted">No metrics found — run main.py first.</td></tr>';
    }

    // ═════════════════ VIEW 4 — INFERENCE ════════════════════════════════
    function renderInferenceMetrics() {
        const ds = currentDataset;
        const plain = (allMetrics[ds] || {}).LogisticRegression || null;
        const meta = (status.fhe_meta || {})[ds] || null;
        const enc = meta ? meta.metrics : null;

        [['mAccP', 'mAccE', 'accuracy', pct],
         ['mRecP', 'mRecE', 'recall', pct],
         ['mPrcP', 'mPrcE', 'precision', pct],
         ['mF1P', 'mF1E', 'f1', pct],
         ['mAucP', 'mAucE', 'roc_auc', v => v.toFixed(4)],
        ].forEach(([pId, eId, key, fmt]) => {
            $(pId).innerText = plain && plain[key] != null ? fmt(plain[key]) : '--';
            $(eId).innerText = enc && enc[key] != null ? fmt(enc[key]) : '--';
        });

        $('cBits').innerText = meta ? meta.n_bits + '-bit integers' : '--';
        $('cFeat').innerText = meta ? meta.n_features : '--';
        $('cCompile').innerText = meta ? meta.compile_time_s.toFixed(2) + ' s' : '--';
        const dsInfo = (status.datasets || {})[ds];
        $('cTest').innerText = dsInfo ? dsInfo.n_test + ' held-out patients' : '--';
    }

    function resetPanel() {
        currentFeatures = null;
        currentTruth = null;
        patientDataContainer.innerHTML = `
            <div class="empty-state">
                <p>No patient record loaded.</p>
                <p class="sub">Load a random held-out record from the test set to begin.</p>
            </div>`;
        trueLabelBox.classList.add('hidden');
        ['pipeline', 'ciphertextBox', 'resultsRow', 'verdict', 'explainBox']
            .forEach(id => $(id).classList.add('hidden'));
        btnPredict.disabled = true;
    }

    btnLoadPatient.addEventListener('click', async () => {
        btnLoadPatient.disabled = true;
        ['pipeline', 'ciphertextBox', 'resultsRow', 'verdict', 'explainBox']
            .forEach(id => $(id).classList.add('hidden'));
        patientDataContainer.innerHTML = '<div class="empty-state"><div class="spinner"></div></div>';

        try {
            const res = await fetch(`/api/random_patient/${currentDataset}`);
            const data = await res.json();
            if (data.error) throw new Error(data.error);

            currentFeatures = data.features;
            currentTruth = data;

            let html = '<div class="feature-grid">';
            for (const [key, val] of Object.entries(currentFeatures)) {
                const shown = typeof val === 'number'
                    ? (Number.isInteger(val) ? val : val.toFixed(3)) : val;
                html += `<div class="feature-item">
                    <span class="feature-name">${esc(key.replace(/_/g, ' '))}</span>
                    <span class="feature-val">${esc(shown)}</span></div>`;
            }
            patientDataContainer.innerHTML = html + '</div>';

            trueLabelValue.innerText = data.true_label_text;
            trueLabelValue.style.color = data.true_label_alarming ? 'var(--danger)' : 'var(--success)';
            trueLabelBox.classList.remove('hidden');
            btnPredict.disabled = false;
        } catch (err) {
            console.error(err);
            patientDataContainer.innerHTML =
                '<div class="empty-state"><p style="color:var(--danger)">Could not load a patient record.</p></div>';
        } finally {
            btnLoadPatient.disabled = false;
        }
    });

    btnPredict.addEventListener('click', async () => {
        if (!currentFeatures) return;

        const fheOn = status.fhe_available[currentDataset];
        btnPredict.disabled = true;
        btnPredict.querySelector('.btn-text').classList.add('hidden');
        btnPredict.querySelector('.spinner').classList.remove('hidden');
        ['resultsRow', 'verdict', 'ciphertextBox', 'explainBox']
            .forEach(id => $(id).classList.add('hidden'));

        if (fheOn) {
            $('pipeline').classList.remove('hidden');
            ['tEncrypt', 'tServer', 'tDecrypt'].forEach(id => $(id).innerText = '…');
            ['stEncrypt', 'stServer', 'stDecrypt'].forEach(id => $(id).classList.add('running'));
        }

        try {
            const res = await fetch(`/api/predict/${currentDataset}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ features: currentFeatures, mode: fheOn ? 'both' : 'plaintext' }),
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            renderPrediction(data);
        } catch (err) {
            console.error(err);
            $('verdict').classList.remove('hidden');
            $('verdict').className = 'verdict mismatch';
            $('verdict').innerText = 'Inference failed: ' + err.message;
        } finally {
            btnPredict.disabled = false;
            btnPredict.querySelector('.btn-text').classList.remove('hidden');
            btnPredict.querySelector('.spinner').classList.add('hidden');
            ['stEncrypt', 'stServer', 'stDecrypt'].forEach(id => $(id).classList.remove('running'));
        }
    });

    function renderPrediction(data) {
        const p = data.plaintext;
        const e = data.encrypted;

        $('boxPlain').classList.toggle('danger', !!(p && p.alarming));
        $('predPlain').innerText = p ? p.prediction_text : '--';
        $('confPlain').innerText = p ? pct(p.confidence) + ' confidence' : '--';
        $('latPlain').innerText = p ? ms(p.latency_ms) + (p.source ? ' · ' + p.source : '') : '--';

        const boxE = $('boxEnc');
        if (e && !e.error) {
            boxE.classList.remove('unavailable');
            boxE.classList.toggle('danger', !!e.alarming);
            $('predEnc').innerText = e.prediction_text;
            $('confEnc').innerText = pct(e.confidence) + ' confidence';
            $('latEnc').innerText = ms(e.timings_ms.total) + ' end-to-end';
            $('tEncrypt').innerText = ms(e.timings_ms.encrypt);
            $('tServer').innerText = ms(e.timings_ms.server_compute);
            $('tDecrypt').innerText = ms(e.timings_ms.decrypt);
            $('ctSize').innerText = bytes(e.ciphertext_bytes) + ' ciphertext';
            $('ctPreview').innerText = e.ciphertext_preview.replace(/(.{2})/g, '$1 ').trim() + ' …';
            $('ciphertextBox').classList.remove('hidden');
        } else {
            boxE.classList.add('unavailable');
            $('predEnc').innerText = '—';
            $('confEnc').innerText = 'unavailable';
            $('latEnc').innerText = e && e.error ? e.error : 'FHE circuit not compiled';
            $('pipeline').classList.add('hidden');
        }

        $('resultsRow').classList.remove('hidden');

        const v = $('verdict');
        v.classList.remove('hidden');
        if (e && !e.error) {
            const agree = e.prediction_raw === p.prediction_raw;
            const correct = currentTruth && e.prediction_raw === currentTruth.true_label_raw;
            v.className = 'verdict ' + (agree ? 'match' : 'mismatch');
            v.innerHTML = agree
                ? `✓ Encrypted and plaintext predictions agree — <strong>${esc(e.prediction_text)}</strong>` +
                  (currentTruth ? ` · ground truth <strong>${esc(currentTruth.true_label_text)}</strong> (${correct ? 'correct' : 'incorrect'})` : '') +
                  `<br><span class="sub">The server produced this answer without ever seeing a single feature value.` +
                  (data.baseline ? ` scikit-learn baseline for comparison: <strong>${esc(data.baseline.prediction_text)}</strong> (${pct(data.baseline.confidence)}).` : '') +
                  `</span>`
                : `⚠ Encrypted (<strong>${esc(e.prediction_text)}</strong>) and plaintext (<strong>${esc(p.prediction_text)}</strong>) disagree — quantisation artefact.`;
        } else {
            const correct = currentTruth && p.prediction_raw === currentTruth.true_label_raw;
            v.className = 'verdict match';
            v.innerHTML = `Plaintext prediction <strong>${esc(p.prediction_text)}</strong>` +
                (currentTruth ? ` · ground truth <strong>${esc(currentTruth.true_label_text)}</strong> (${correct ? 'correct' : 'incorrect'})` : '');
        }

        renderExplanation(data.explanation);
    }

    function renderExplanation(ex) {
        if (!ex || !ex.top_features || !ex.top_features.length) return;

        const lo = ex.log_odds;
        $('explainLogOdds').innerHTML =
            `log-odds <strong>${lo.toFixed(3)}</strong> → ` +
            `${lo > 0 ? esc(ex.positive_class) : esc(ex.negative_class)}`;

        const max = Math.max(...ex.top_features.map(f => Math.abs(f.contribution)));
        $('explainBars').innerHTML = ex.top_features.map(f => {
            const frac = Math.abs(f.contribution) / max * 50;   // half-width each side
            const neg = f.contribution < 0;
            return `<div class="ex-row">
                <span class="ex-name">${esc(f.feature)}</span>
                <div class="ex-track">
                    <div class="ex-axis"></div>
                    <div class="ex-fill ${neg ? 'neg' : 'pos'}"
                         style="width:${frac}%; ${neg ? 'right:50%' : 'left:50%'}"></div>
                </div>
                <span class="ex-val ${neg ? 'neg' : 'pos'}">${f.contribution > 0 ? '+' : ''}${f.contribution.toFixed(3)}</span>
            </div>`;
        }).join('') + `
            <div class="ex-legend">
                <span class="neg">← ${esc(ex.negative_class)}</span>
                <span class="pos">${esc(ex.positive_class)} →</span>
            </div>`;

        $('explainBox').classList.remove('hidden');
    }

    // ── Lightbox ─────────────────────────────────────────────────────────
    function bindLightbox(selector) {
        document.querySelectorAll(selector).forEach(card => {
            card.addEventListener('click', () => {
                $('lightboxImg').src = card.dataset.src;
                $('lightboxCaption').innerText = card.dataset.caption || '';
                $('lightbox').classList.remove('hidden');
            });
        });
    }
    $('lightbox').addEventListener('click', () => $('lightbox').classList.add('hidden'));
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') $('lightbox').classList.add('hidden');
    });
});
