# -*- coding: utf-8 -*-
"""
Fase 4.2 - Analisis (b): error silencioso.

Subconjunto desenlace in {correcta, silencioso} (excluye sintacticos y
timeouts). AUROC del modelo operativo en ese subconjunto, IC 95%.
Criterio: limite inferior del IC > 0.5.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from utils_bootstrap import bootstrap_ci_metrica

import os
RAIZ = Path(__file__).resolve().parents[1]  # raiz del repositorio
EXTERNO = Path(os.environ.get("TESIS_EXTERNO", RAIZ / "externo"))  # Spider, MultiSpider y modelo (descargados aparte)
BASE = RAIZ
RESULTADOS = BASE / "resultados"
TABLAS = RESULTADOS / "tablas"
LOGS = BASE / "logs"


def main():
    lineas = []

    def log(s=""):
        print(s, flush=True)
        lineas.append(s)

    log("=" * 70)
    log("Fase 4.2 - Analisis (b): error silencioso")
    log("=" * 70)

    modelo_operativo = (LOGS / "modelo_operativo.txt").read_text(encoding="utf-8").strip()
    log(f"Modelo operativo: {modelo_operativo.upper()}")

    oof = pd.read_csv(RESULTADOS / "oof_predicciones.csv", encoding="utf-8")
    subset = oof[oof["desenlace"].isin(["correcta", "silencioso"])].copy()
    n_correcta = (subset["desenlace"] == "correcta").sum()
    n_silencioso = (subset["desenlace"] == "silencioso").sum()
    log(f"n subconjunto = {len(subset)} (correcta={n_correcta}, silencioso={n_silencioso})")

    col = f"p_cal_{modelo_operativo}"
    punto, lo, hi = bootstrap_ci_metrica(subset["y"].values, subset[col].values, roc_auc_score)
    cumple = lo > 0.5
    log(f"\nAUROC {modelo_operativo.upper()} en subconjunto: {punto:.4f}  IC95% [{lo:.4f}, {hi:.4f}]")
    log(f"Criterio (limite inferior > 0.5): {'CUMPLE' if cumple else 'NO CUMPLE'}")

    tabla = pd.DataFrame([{
        "modelo": modelo_operativo, "n": len(subset), "n_correcta": n_correcta,
        "n_silencioso": n_silencioso, "auroc": punto, "ic_lo": lo, "ic_hi": hi,
        "cumple_criterio": cumple,
    }])
    tabla.to_csv(TABLAS / "tabla_b_silencioso.csv", index=False, encoding="utf-8")
    log(f"\nGuardado: {TABLAS / 'tabla_b_silencioso.csv'}")

    with open(LOGS / "analisis_b.log", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    return 0


if __name__ == "__main__":
    sys.exit(main())
