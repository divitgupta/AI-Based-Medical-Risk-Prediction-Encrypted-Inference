"""
Encrypted inference for all three trained models (synopsis §8).

Build time (experiment stage):
    the exact trained model → Concrete ML quantised equivalent
    → compiled FHE circuit → saved deployment bundle.

    logistic regression → concrete.ml.sklearn.LogisticRegression
    XGBoost             → concrete.ml.sklearn.XGBClassifier
    MLP (PyTorch)       → concrete.ml.torch.compile_torch_model

Run time (web app):
    client: saved preprocessor → quantise → encrypt
    server: evaluate the circuit on ciphertext only
    client: decrypt → probability

The server object never receives a plaintext feature or the secret key.

Concrete ML uses TFHE, which evaluates non-linear functions (ReLU, sigmoid,
tree comparisons) as table lookups inside programmable bootstrapping (PBS).
So, unlike CKKS, no polynomial activation is required; the number of PBS a
circuit needs is its main cost driver, and it is reported per model.

NOTE: Concrete ML shells out to the system linker when compiling, and that
breaks on paths containing spaces. Run with ~/aimlel-venv/bin/python (see
scripts/setup_fhe_env.sh).
"""

import shutil
import threading
import time

import numpy as np
import psutil
from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression
from concrete.ml.sklearn import XGBClassifier as ConcreteXGBClassifier
from concrete.ml.torch.compile import compile_torch_model

from . import config

# concrete.ml.deployment is imported lazily, inside the functions below, on
# purpose: importing it *before* the first circuit is compiled in a process
# makes the next compile crash inside LLVM (segfault, exit 139) with
# concrete-ml 1.9. Once one circuit has compiled, importing it is safe. The
# web app never compiles, so it can import it directly.

# What the decrypted output of each model is: class probabilities, or the
# MLP's single logit (its sigmoid is applied by the client after decryption).
OUTPUT_KIND = {"logreg": "proba", "xgboost": "proba", "mlp": "logit"}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


class TorchFHE:
    """Gives a compiled PyTorch module the same interface as the sklearn ones."""

    def __init__(self, quantized_module):
        self.module = quantized_module

    @property
    def fhe_circuit(self):
        return self.module.fhe_circuit

    def predict_proba(self, X, fhe="disable"):
        p = _sigmoid(self.module.forward(np.asarray(X, dtype=np.float32), fhe=fhe).ravel())
        return np.column_stack([1 - p, p])


def compile_model(key, pipeline, X_train_raw, settings=None):
    """Quantise + compile the trained model of one fold. Returns (model, seconds)."""
    settings = settings or config.FHE_SETTINGS[key]
    X_train = pipeline.named_steps["prep"].transform(X_train_raw)
    model = pipeline.named_steps["model"]
    t0 = time.perf_counter()
    if key == "logreg":
        fhe_model = ConcreteLogisticRegression.from_sklearn_model(model, X_train, **settings)
        fhe_model.compile(X_train)
    elif key == "xgboost":
        fhe_model = ConcreteXGBClassifier.from_sklearn_model(model, X_train, **settings)
        fhe_model.compile(X_train)
    elif key == "mlp":
        module = compile_torch_model(model.net_.eval(), X_train.astype(np.float32), **settings)
        fhe_model = TorchFHE(module)
    else:
        raise ValueError(key)
    return fhe_model, time.perf_counter() - t0


def circuit_cost(fhe_model):
    """Static cost of a compiled circuit, from the compiler's own statistics."""
    circuit = fhe_model.fhe_circuit
    st = circuit.statistics
    return {
        "pbs_count": int(st["programmable_bootstrap_count"]),
        "max_bit_width": int(circuit.graph.maximum_integer_bit_width()),
        "bootstrap_key_bytes": int(st["size_of_bootstrap_keys"]),
        "keyswitch_key_bytes": int(st["size_of_keyswitch_keys"]),
        "evaluation_key_bytes": int(st["size_of_bootstrap_keys"] + st["size_of_keyswitch_keys"]),
        "secret_key_bytes": int(st["size_of_secret_keys"]),
        "input_ciphertext_bytes": int(st["size_of_inputs"]),
        "output_ciphertext_bytes": int(st["size_of_outputs"]),
    }


class _PeakRSS:
    """Samples this process's resident memory in a background thread."""

    def __enter__(self):
        self._proc = psutil.Process()
        self.baseline = self.peak = self._proc.memory_info().rss
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.wait(0.02):
            self.peak = max(self.peak, self._proc.memory_info().rss)

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self._proc.memory_info().rss)

    @property
    def extra_mb(self):
        return (self.peak - self.baseline) / 1e6


def evaluate_fold(fhe_model, X_test, n_real, warmup):
    """
    Encrypted probabilities for every (preprocessed) test row of the fold via
    FHE simulation — the circuit's exact integer arithmetic, run without the
    cryptography so all rows are fast — plus genuine encrypted execution timed
    on `n_real` rows, with the memory it needed.
    """
    sim_prob = fhe_model.predict_proba(X_test, fhe="simulate")[:, 1]

    with _PeakRSS() as mem:
        # Key generation and the first encrypted call are one-off costs; time
        # them separately so the per-patient figure is the steady-state cost.
        t0 = time.perf_counter()
        fhe_model.fhe_circuit.keygen()
        keygen_s = time.perf_counter() - t0
        if warmup:
            fhe_model.predict_proba(X_test[:1], fhe="execute")

        rows = X_test[:n_real]
        t0 = time.perf_counter()
        real_prob = fhe_model.predict_proba(rows, fhe="execute")[:, 1]
        per_sample_ms = (time.perf_counter() - t0) / max(len(rows), 1) * 1000

    real_agree = float(np.mean((real_prob >= 0.5) == (sim_prob[:n_real] >= 0.5)))
    return sim_prob, {"real_exec_ms_per_sample": per_sample_ms,
                      "real_exec_rows": int(len(rows)),
                      "real_vs_sim_agreement": real_agree,
                      "keygen_seconds": keygen_s,
                      "peak_extra_memory_mb": mem.extra_mb}


def save_bundle(fhe_model, path):
    """Write the server-side deployment bundle (circuit + client specs)."""
    from concrete.ml.deployment import FHEModelDev

    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = fhe_model.module if isinstance(fhe_model, TorchFHE) else fhe_model
    FHEModelDev(path_dir=str(path), model=model).save()


class EncryptedModel:
    """
    A client/server pair for one saved bundle. In a real deployment these two
    halves live on different machines; here they share a process, but the
    server half is only ever handed serialized ciphertext and the public
    evaluation key. Keys are generated in memory when the pair is created.
    """

    def __init__(self, bundle_dir, output_kind):
        from concrete.ml.deployment import FHEModelClient, FHEModelServer

        self.output_kind = output_kind
        t0 = time.perf_counter()
        self.client = FHEModelClient(path_dir=str(bundle_dir))
        self.evaluation_keys = self.client.get_serialized_evaluation_keys()
        self.keygen_seconds = time.perf_counter() - t0
        self.server = FHEModelServer(path_dir=str(bundle_dir))
        self.server.load()

    def predict(self, x_preprocessed):
        """Returns (probability of disease, timings in ms, ciphertext, result ciphertext)."""
        x = np.asarray(x_preprocessed, dtype=np.float32 if self.output_kind == "logit" else float)
        t0 = time.perf_counter()
        ciphertext = self.client.quantize_encrypt_serialize(x)
        t1 = time.perf_counter()
        encrypted_result = self.server.run(ciphertext, self.evaluation_keys)
        t2 = time.perf_counter()
        out = np.asarray(self.client.deserialize_decrypt_dequantize(encrypted_result), dtype=float)
        t3 = time.perf_counter()

        if self.output_kind == "logit":
            prob = float(_sigmoid(out.ravel()[0]))
        else:
            prob = float(out.reshape(-1, 2)[0, 1]) if out.size >= 2 else float(out.ravel()[0])
        timings = {"encrypt_ms": (t1 - t0) * 1000,
                   "server_ms": (t2 - t1) * 1000,
                   "decrypt_ms": (t3 - t2) * 1000}
        return prob, timings, ciphertext, encrypted_result
