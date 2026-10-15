# -*- coding: utf-8 -*-
"""
Calibracion: ECE con 10 intervalos de igual
anchura en [0, 1] y datos del diagrama de fiabilidad.
p = probabilidad estimada de y = 1 (respuesta correcta).
"""
import numpy as np
import pandas as pd

N_INTERVALOS = 10


def bins_fiabilidad(y, p, n_intervalos=N_INTERVALOS):
    """Un registro por intervalo: n, p media, frecuencia observada de y=1."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    idx = np.clip(np.floor(p * n_intervalos).astype(int), 0, n_intervalos - 1)
    filas = []
    for b in range(n_intervalos):
        m = idx == b
        n = int(m.sum())
        filas.append({
            "intervalo": b, "lo": b / n_intervalos, "hi": (b + 1) / n_intervalos, "n": n,
            "p_media": float(p[m].mean()) if n else np.nan,
            "frecuencia_observada": float(y[m].mean()) if n else np.nan,
        })
    return pd.DataFrame(filas)


def ece(y, p, n_intervalos=N_INTERVALOS):
    """ECE = sum_b (n_b / n) * |frecuencia observada_b - p media_b| (intervalos vacios omitidos)."""
    t = bins_fiabilidad(y, p, n_intervalos)
    t = t[t["n"] > 0]
    return float(np.sum(t["n"] / t["n"].sum() * np.abs(t["frecuencia_observada"] - t["p_media"])))
