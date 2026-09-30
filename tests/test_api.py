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
            r = client.post(f"/api/infer/{key}", json={"patient_id": i}).get_json()
            assert r["plaintext"]["prob"] == pytest.approx(predictions["prob"][key][i], abs=1e-5)
            assert r["fold"] == predictions["fold"][i] + 1


def test_encrypted_logreg_reports_agreement(client, predictions):
    r = client.post("/api/infer/logreg", json={"patient_id": 150}).get_json()
    enc = r["encrypted"]
    assert enc["available"] is True
    assert isinstance(enc["agree"], bool)
    assert enc["prob"] == pytest.approx(predictions["prob_fhe"]["logreg"][150], abs=1e-3)
    assert enc["ciphertext_bytes"] > 0 and set(enc["timings_ms"]) >= {"encrypt_ms", "server_ms", "decrypt_ms"}


def test_models_without_fhe_say_so_instead_of_inventing_numbers(client):
    enc = client.post("/api/infer/mlp", json={"patient_id": 3}).get_json()["encrypted"]
    assert enc["available"] is False and "Phase 2" in enc["reason"]
    assert "prob" not in enc


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
