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


def test_scenarios_are_real_patients_in_their_band(client, predictions):
    scenarios = client.get("/api/scenarios").get_json()
    assert [s["key"] for s in scenarios] == ["no_disease", "probable", "disease"]
    for s in scenarios:
        i = s["patient_id"]
        p = predictions["prob"]["logreg"][i]
        lo, hi = s["band"]
        assert lo <= p <= hi and s["p_disease"] == pytest.approx(p)
        # the encrypted model must reach the same decision for a demo preset
        assert (predictions["prob_fhe"]["logreg"][i] >= 0.5) == (p >= 0.5)
        assert 0 <= s["band_disease_rate"] <= 1 and s["band_size"] > 0


def test_elastic_learner_leaderboard(client):
    e = client.get("/api/elastic").get_json()
    assert e["order"] == [*config.MODEL_KEYS, "elastic"]
    assert e["recommendation"]["model"] in e["order"]
    assert e["recommendation"]["model"] in e["recommendation"]["tied_on_f2"]
    for b in e["best"].values():
        assert b["model"] in e["order"]
    assert len(e["elastic"]["folds"]) == config.N_FOLDS
