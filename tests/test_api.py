"""
API tests. They load the real artifacts in results/, so run the experiment
first (scripts/run_experiments.sh).
"""

import json

import pytest

from healthguard import config

pytestmark = pytest.mark.skipif(
    not (config.RESULTS / "metrics.json").exists(),
    reason="run scripts/run_experiments.sh first",
)


@pytest.fixture(scope="module")
def client():
    from webapp import server
    server.load_state()
    return server.app.test_client()


@pytest.fixture(scope="module")
def predictions():
    return json.loads((config.RESULTS / "predictions.json").read_text())


def test_live_plaintext_matches_stored_out_of_fold_prediction(client, predictions):
    for i in (0, 85, 150, 302):
        for key in config.MODEL_KEYS:
            r = client.post(f"/api/infer/{key}?encrypted=0", json={"patient_id": i}).get_json()
            assert r["plaintext"]["prob"] == pytest.approx(predictions["prob"][key][i], abs=1e-5)
            assert r["fold"] == predictions["fold"][i] + 1


def test_encrypted_logreg_reports_agreement(client, predictions):
    r = client.post("/api/infer/logreg", json={"patient_id": 150}).get_json()
    enc = r["encrypted"]
    assert enc["available"] is True
    assert isinstance(enc["agree"], bool)
    assert enc["prob"] == pytest.approx(predictions["prob_fhe"]["logreg"][150], abs=1e-3)
    assert enc["ciphertext_bytes"] > 0 and set(enc["timings_ms"]) >= {"encrypt_ms", "server_ms", "decrypt_ms"}


def test_encrypted_mlp_matches_its_stored_encrypted_prediction(client, predictions):
    # One genuine encrypted MLP run (~5 s). XGBoost is the same code path but
    # takes ~30-60 s per patient, so it is left to the experiment.
    enc = client.post("/api/infer/mlp", json={"patient_id": 3}).get_json()["encrypted"]
    assert enc["available"] is True
    assert enc["prob"] == pytest.approx(predictions["prob_fhe"]["mlp"][3], abs=1e-3)


def test_phase2_reports_cost_for_every_model(client):
    d = client.get("/api/phase2").get_json()
    for key in config.MODEL_KEYS:
        f = d["fhe"][key]
        assert 0.9 <= f["agreement"] <= 1
        assert f["real_exec_ms_per_sample"] > 0
    # Complexity costs more under encryption: LR needs no bootstraps at all.
    pbs = {k: d["fhe"][k]["cost_mean"]["pbs_count"] for k in config.MODEL_KEYS}
    assert pbs["logreg"] == 0 < pbs["mlp"] < pbs["xgboost"]


@pytest.mark.parametrize("body", [{"patient_id": 999}, {"patient_id": -1}, {"patient_id": "x"}, {}])
def test_bad_patient_id_is_rejected(client, body):
    assert client.post("/api/infer/logreg", json=body).status_code == 400


def test_unknown_model_is_rejected(client):
    assert client.post("/api/infer/svm", json={"patient_id": 1}).status_code == 400


def test_patient_list_does_not_leak_labels(client):
    patients = client.get("/api/patients").get_json()
    assert len(patients) == 303
    assert set(patients[0]) == {"id", "fold"}


def test_every_model_has_its_own_scenarios(client, predictions):
    scenarios = client.get("/api/scenarios").get_json()
    assert list(scenarios) == [*config.MODEL_KEYS, "elastic"]
    for model, items in scenarios.items():
        assert [s["key"] for s in items] == ["no_disease", "probable", "disease"], model
        plain = predictions["prob_elastic" if model == "elastic" else "prob"]
        plain = plain if model == "elastic" else plain[model]
        enc = predictions["prob_elastic_fhe" if model == "elastic" else "prob_fhe"]
        enc = enc if model == "elastic" else enc[model]
        for s in items:
            i = s["patient_id"]
            lo, hi = s["band"]
            # picked by this model's own out-of-fold probability, inside the band
            assert lo <= plain[i] <= hi and s["p_disease"] == pytest.approx(plain[i])
            # the encrypted model must reach the same decision for a demo preset
            assert (enc[i] >= 0.5) == (plain[i] >= 0.5)
            assert 0 <= s["band_disease_rate"] <= 1 and s["band_size"] > 0


def test_elastic_learner_leaderboard(client):
    e = client.get("/api/elastic").get_json()
    assert e["order"] == [*config.MODEL_KEYS, "elastic"]
    assert e["recommendation"]["model"] in e["order"]
    assert e["recommendation"]["model"] in e["recommendation"]["tied_on_f2"]
    for b in e["best"].values():
        assert b["model"] in e["order"]
    assert len(e["elastic"]["folds"]) == config.N_FOLDS


def test_elastic_learner_runs_on_a_patient(client, predictions):
    for i in (0, 85, 302):
        r = client.post("/api/infer/elastic?encrypted=0", json={"patient_id": i}).get_json()
        assert r["plaintext"]["prob"] == pytest.approx(predictions["prob_elastic"][i], abs=1e-5)
        assert [b["model"] for b in r["elastic"]["base"]] == config.MODEL_KEYS


# ── Dataset switch: models retrained on the base paper's 1,025-row file ──

def _needs(ds):
    return pytest.mark.skipif(not (config.DATASETS[ds]["results"] / "metrics.json").exists(),
                              reason=f"run python -m healthguard.experiment --dataset {ds} first")


def test_datasets_endpoint_lists_trained_datasets(client):
    d = client.get("/api/datasets").get_json()
    assert d["default"] == "cleveland"
    keys = [x["key"] for x in d["datasets"]]
    assert keys[0] == "cleveland" and set(keys) <= set(config.DATASETS)


def test_unknown_dataset_is_rejected(client):
    assert client.get("/api/overview?ds=nope").status_code == 404


def test_cleveland_is_untouched_by_the_switch(client):
    cl = client.get("/api/overview").get_json()
    assert cl["n_samples"] == 303
    assert cl["models"]["logreg"]["pooled"]["accuracy"] == pytest.approx(0.8515, abs=1e-3)


@pytest.mark.parametrize("ds", ["base1025_patient", "base1025_row"])
def test_retrained_base_paper_models_are_served(client, ds):
    if not (config.DATASETS[ds]["results"] / "metrics.json").exists():
        pytest.skip(f"run python -m healthguard.experiment --dataset {ds} first")
    preds = json.loads((config.DATASETS[ds]["results"] / "predictions.json").read_text())
    o = client.get(f"/api/overview?ds={ds}").get_json()
    assert o["n_samples"] == 1025 and o["dataset"]["base"] is True
    assert o["dataset"]["grouped"] is (ds == "base1025_patient")
    for i in (0, 500, 1024):
        for key in config.MODEL_KEYS:
            r = client.post(f"/api/infer/{key}?ds={ds}&encrypted=0", json={"patient_id": i}).get_json()
            assert r["plaintext"]["prob"] == pytest.approx(preds["prob"][key][i], abs=1e-5)
    enc = client.post(f"/api/infer/logreg?ds={ds}", json={"patient_id": 700}).get_json()["encrypted"]
    assert enc["available"] is True
    assert enc["prob"] == pytest.approx(preds["prob_fhe"]["logreg"][700], abs=1e-3)


@_needs("base1025_patient")
def test_patient_split_never_tests_a_model_on_a_patient_it_trained_on(client):
    patients = client.get("/api/patients?ds=base1025_patient").get_json()
    folds_of = {}
    for p in patients:
        folds_of.setdefault(p["patient"], set()).add(p["fold"])
    assert all(len(f) == 1 for f in folds_of.values())
    assert client.get("/api/patient/700?ds=base1025_patient").get_json()["copies_in_training"] == 0


@_needs("base1025_row")
def test_row_split_leaks_copies_into_training(client):
    o = client.get("/api/overview?ds=base1025_row").get_json()
    assert o["dataset"]["leak_pct"] > 0.9


def test_base_paper_table_has_our_unchanged_models_row(client, predictions):
    e = client.get("/api/phase2").get_json()["our_models_on_base_file"]
    assert e["n_rows"] == 1025
    assert 0.8 < e["models"]["logreg"]["accuracy"] < 0.9 and e["encrypted_logreg"] is not None
