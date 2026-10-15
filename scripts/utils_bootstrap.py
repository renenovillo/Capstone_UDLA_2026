# -*- coding: utf-8 -*-
"""
Bootstrap percentil para IC 95%:
2000 remuestreos de preguntas, random_state=42, IC percentil 2.5-97.5.
Para diferencias entre modelos: bootstrap PAREADO (mismos indices
remuestreados para ambos modelos en cada replica).
"""
import numpy as np
from sklearn.metrics import roc_auc_score

N_BOOT = 2000
RANDOM_STATE = 42


def _resample_valido(y, rng, n, metric_fn, max_intentos=50):
    """Redibuja si el remuestreo queda con una sola clase (metric_fn fallaria)."""
    for _ in range(max_intentos):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) > 1:
            return idx
    return idx  # ultimo intento, se deja pasar (caso extremo)


def bootstrap_ci_metrica(y, p, metric_fn, n_boot=N_BOOT, random_state=RANDOM_STATE):
    """Punto estimado + IC 95% percentil para una metrica univariada metric_fn(y,p)."""
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    rng = np.random.default_rng(random_state)
    punto = metric_fn(y, p)
    estimaciones = np.empty(n_boot)
    for b in range(n_boot):
        idx = _resample_valido(y, rng, n, metric_fn)
        estimaciones[b] = metric_fn(y[idx], p[idx])
    lo, hi = np.percentile(estimaciones, [2.5, 97.5])
    return punto, lo, hi


def bootstrap_ci_diferencia(y, p_a, p_b, metric_fn=roc_auc_score, n_boot=N_BOOT,
                             random_state=RANDOM_STATE):
    """Bootstrap pareado de la diferencia metric_fn(y,p_a) - metric_fn(y,p_b)."""
    y = np.asarray(y)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)
    n = len(y)
    rng = np.random.default_rng(random_state)
    punto = metric_fn(y, p_a) - metric_fn(y, p_b)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = _resample_valido(y, rng, n, metric_fn)
        diffs[b] = metric_fn(y[idx], p_a[idx]) - metric_fn(y[idx], p_b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return punto, lo, hi
