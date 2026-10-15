# -*- coding: utf-8 -*-
"""
Sensibilidad por bases de datos.

Las preguntas de una misma base no son independientes (20 bases). Aqui se remuestrean BASES COMPLETAS
(bootstrap por conglomerados, 2 000 remuestreos, random_state = 42, IC percentil 2,5-97,5) para las
diferencias pareadas de AUROC agrupado, y se registran: acierto por base, AUROC de GB por pliegue y
AUROC de GB para correcta frente a sintactico / silencioso.
Salida: resultados/tablas/tabla_sensibilidad_bases.csv y logs/sensibilidad_bases.log
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

D = Path(__file__).resolve().parents[1]
N_BOOT, SEED, MIN_N_BASE = 2000, 42, 30


def main():
    lineas = []

    def log(s=""):
        print(s, flush=True)
        lineas.append(s)

    oof = pd.read_csv(D / "resultados" / "oof_predicciones.csv")
    meta = pd.read_csv(D / "data" / "meta_es.csv").set_index("idx")
    oof["db_id"] = meta.loc[oof["idx"].values, "db_id"].values
    y = oof["y"].values
    oof["score_dif"] = oof["dificultad"].map({"easy": 3, "medium": 2, "hard": 1, "extra hard": 0}).astype(float)
    filas = []

    # ---- bootstrap por bases ----
    bases = oof["db_id"].unique()
    idx_por_base = {b: np.where(oof["db_id"].values == b)[0] for b in bases}
    rng = np.random.default_rng(SEED)
    comparaciones = {
        "gb_menos_lr": ("p_cal_gb", "p_cal_lr"),
        "gb_menos_longitud": ("p_cal_gb", "p_len"),
        "lr_menos_longitud": ("p_cal_lr", "p_len"),
        "gb_menos_confianza": ("p_cal_gb", "p_conf"),
        "gb_menos_dificultad": ("p_cal_gb", "score_dif"),
        "lr_menos_dificultad": ("p_cal_lr", "score_dif"),
    }
    difs = {k: [] for k in comparaciones}
    for _ in range(N_BOOT):
        for _intento in range(50):
            elegidas = rng.choice(bases, size=len(bases), replace=True)
            idx = np.concatenate([idx_por_base[b] for b in elegidas])
            if len(np.unique(y[idx])) > 1:
                break
        for k, (a, b) in comparaciones.items():
            difs[k].append(roc_auc_score(y[idx], oof[a].values[idx]) - roc_auc_score(y[idx], oof[b].values[idx]))
    log(f"Bootstrap por bases completas ({len(bases)} bases, {N_BOOT} remuestreos, semilla {SEED}):")
    for k, (a, b) in comparaciones.items():
        punto = roc_auc_score(y, oof[a].values) - roc_auc_score(y, oof[b].values)
        lo, hi = np.percentile(difs[k], [2.5, 97.5])
        log(f"  {k}: {punto:+.4f}  IC95% por bases [{lo:+.4f}, {hi:+.4f}]")
        filas.append({"metrica": "diff_AUROC_bootstrap_por_bases", "detalle": k, "valor": punto, "ic_lo": lo, "ic_hi": hi})

    # ---- acierto por base ----
    t = oof.groupby("db_id")["y"].agg(["mean", "size"])
    grandes = t[t["size"] >= MIN_N_BASE]
    log(f"\nAcierto por base: {len(t)} bases; con n >= {MIN_N_BASE} ({len(grandes)} bases): "
        f"min {100*grandes['mean'].min():.1f} % ({grandes['mean'].idxmin()}), max {100*grandes['mean'].max():.1f} % ({grandes['mean'].idxmax()}); "
        f"todas las bases: min {100*t['mean'].min():.1f} % ({t['mean'].idxmin()}, n = {int(t.loc[t['mean'].idxmin(), 'size'])}), max {100*t['mean'].max():.1f} %")
    filas.append({"metrica": "acierto_por_base_n_ge_30", "detalle": "min", "valor": grandes["mean"].min(), "ic_lo": None, "ic_hi": None})
    filas.append({"metrica": "acierto_por_base_n_ge_30", "detalle": "max", "valor": grandes["mean"].max(), "ic_lo": None, "ic_hi": None})
    filas.append({"metrica": "acierto_por_base_todas", "detalle": "min", "valor": t["mean"].min(), "ic_lo": None, "ic_hi": None})
    filas.append({"metrica": "acierto_por_base_todas", "detalle": "max", "valor": t["mean"].max(), "ic_lo": None, "ic_hi": None})

    # ---- AUROC de GB por pliegue ----
    por_fold = {int(f): roc_auc_score(g["y"], g["p_cal_gb"]) for f, g in oof.groupby("fold")}
    log("\nAUROC de GB por pliegue externo: " + ", ".join(f"{f}={v:.3f}" for f, v in por_fold.items()))
    filas.append({"metrica": "AUROC_gb_por_pliegue", "detalle": "min", "valor": min(por_fold.values()), "ic_lo": None, "ic_hi": None})
    filas.append({"metrica": "AUROC_gb_por_pliegue", "detalle": "max", "valor": max(por_fold.values()), "ic_lo": None, "ic_hi": None})

    # ---- separabilidad por tipo de error (GB) ----
    log("\nAUROC de GB (p_cal) para correcta frente a cada tipo de error:")
    for tipo in ["sintactico", "silencioso"]:
        s = oof[oof["desenlace"].isin(["correcta", tipo])]
        auc = roc_auc_score(s["y"], s["p_cal_gb"])
        log(f"  correcta vs {tipo}: n = {len(s)}  AUROC = {auc:.4f}")
        filas.append({"metrica": "AUROC_gb_correcta_vs", "detalle": tipo, "valor": auc, "ic_lo": None, "ic_hi": None})

    # ---- efecto del re-etiquetado v2 sobre la confianza (sin re-entrenar nada) ----
    # meta_es.csv (v2) conserva la etiqueta anterior en desenlace_v1; y_v1 = 1 si desenlace_v1 == 'correcta'
    v = meta[meta["gold_status"] == "ok"]
    y_v1 = (v["desenlace_v1"] == "correcta").astype(int)
    a1 = roc_auc_score(y_v1, v["conf_logprob_mean"])
    a2 = roc_auc_score(v["y"].astype(int), v["conf_logprob_mean"])
    cambian = int((y_v1 != v["y"].astype(int)).sum())
    log(f"\nConfianza cruda (conf_logprob_mean), AUROC con etiquetas v1 = {a1:.4f} y con v2 = {a2:.4f} "
        f"({cambian} etiquetas distintas; no se re-entrena nada)")
    filas.append({"metrica": "AUROC_confianza_cruda", "detalle": "etiquetas_v1", "valor": a1, "ic_lo": None, "ic_hi": None})
    filas.append({"metrica": "AUROC_confianza_cruda", "detalle": "etiquetas_v2", "valor": a2, "ic_lo": None, "ic_hi": None})

    pd.DataFrame(filas).to_csv(D / "resultados" / "tablas" / "tabla_sensibilidad_bases.csv", index=False, encoding="utf-8")
    (D / "logs" / "sensibilidad_bases.log").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
