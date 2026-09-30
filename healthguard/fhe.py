"""
Encrypted inference for the trained logistic regression (synopsis §8).

Build time (experiment stage):
    the exact trained scikit-learn model → Concrete ML quantised equivalent
    → compiled FHE circuit → saved deployment bundle.

Run time (web app):
    client: saved preprocessor → quantise → encrypt
    server: evaluate the circuit on ciphertext only
    client: decrypt → probability

The server object never receives a plaintext feature or the secret key.

NOTE: Concrete ML shells out to the system linker when compiling, and that
breaks on paths containing spaces. Run with ~/aimlel-venv/bin/python (see
scripts/setup_fhe_env.sh).
"""

import shutil
import time

import numpy as np
from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression

from . import config

# concrete.ml.deployment is imported lazily, inside the functions below, on
# purpose: importing it *before* the first circuit is compiled in a process
# makes the next compile crash inside LLVM (segfault, exit 139) with
# concrete-ml 1.9. Once one circuit has compiled, importing it is safe. The
# web app never compiles, so it can import it directly.


def compile_logreg(pipeline, X_train_raw):
    """Quantise + compile the trained LR of one fold. Returns (model, seconds)."""
    X_train = pipeline.named_steps["prep"].transform(X_train_raw)
    fhe_model = ConcreteLogisticRegression.from_sklearn_model(
        pipeline.named_steps["model"], X_train, n_bits=config.FHE_N_BITS,
    )
    t0 = time.perf_counter()
    fhe_model.compile(X_train)
    return fhe_model, time.perf_counter() - t0


def evaluate_fold(fhe_model, pipeline, X_test_raw, n_real):
    """
    Encrypted probabilities for every test row of the fold (FHE simulation:
    the circuit's exact integer arithmetic, run without the cryptography so
    all rows are fast), plus genuine encrypted execution timed on a few rows.
    """
    X_test = pipeline.named_steps["prep"].transform(X_test_raw)
    sim_prob = fhe_model.predict_proba(X_test, fhe="simulate")[:, 1]

    # Key generation and the first encrypted call are one-off costs; time them
    # separately so the per-patient figure is the steady-state cost, which is
    # what the web app shows live.
    t0 = time.perf_counter()
    fhe_model.fhe_circuit.keygen()
    keygen_s = time.perf_counter() - t0
    fhe_model.predict_proba(X_test[:1], fhe="execute")

    rows = X_test[:n_real]
    t0 = time.perf_counter()
    real_prob = fhe_model.predict_proba(rows, fhe="execute")[:, 1]
    per_sample_ms = (time.perf_counter() - t0) / max(len(rows), 1) * 1000
    real_agree = float(np.mean((real_prob >= 0.5) == (sim_prob[:n_real] >= 0.5)))
    return sim_prob, {"real_exec_ms_per_sample": per_sample_ms,
                      "real_exec_rows": int(len(rows)),
                      "real_vs_sim_agreement": real_agree,
                      "keygen_seconds": keygen_s}


def save_bundle(fhe_model, path):
    """Write the server-side deployment bundle (circuit + client specs)."""
    from concrete.ml.deployment import FHEModelDev

    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    FHEModelDev(path_dir=str(path), model=fhe_model).save()


class EncryptedModel:
    """
    A client/server pair for one saved bundle. In a real deployment these two
    halves live on different machines; here they share a process, but the
    server half is only ever handed serialized ciphertext and the public
    evaluation key.
    """

    def __init__(self, bundle_dir, key_dir):
        from concrete.ml.deployment import FHEModelClient, FHEModelServer

        self.client = FHEModelClient(path_dir=str(bundle_dir), key_dir=str(key_dir))
        self.evaluation_keys = self.client.get_serialized_evaluation_keys()
        self.server = FHEModelServer(path_dir=str(bundle_dir))
        self.server.load()

    def predict(self, x_preprocessed):
        """Returns (probability of disease, timings in ms, ciphertext bytes)."""
        t0 = time.perf_counter()
        ciphertext = self.client.quantize_encrypt_serialize(x_preprocessed)
        t1 = time.perf_counter()
        encrypted_result = self.server.run(ciphertext, self.evaluation_keys)
        t2 = time.perf_counter()
        probs = np.asarray(self.client.deserialize_decrypt_dequantize(encrypted_result))
        t3 = time.perf_counter()

        prob = float(probs.reshape(-1, 2)[0, 1]) if probs.size >= 2 else float(probs.ravel()[0])
        timings = {"encrypt_ms": (t1 - t0) * 1000,
                   "server_ms": (t2 - t1) * 1000,
                   "decrypt_ms": (t3 - t2) * 1000}
        return prob, timings, ciphertext, encrypted_result
