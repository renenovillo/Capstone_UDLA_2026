# -*- coding: utf-8 -*-
"""
Fase 4.4 - Analisis (d): ruido asimetrico en los reportes.

Niveles rho in {0.10, 0.20, 0.30}. Por pliegue externo, nivel y replica (20):
en el entrenamiento, fraccion rho de las observaciones y=0 se cambian a y=1
(SOLO en esa direccion). Se entrena el modelo operativo con la capa e
hiperparametros ya elegidos (sin repetir busqueda) y se evalua en el
pliegue de prueba con las ETIQUETAS VERDADERAS.
"""
import ast
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning, message=".*'penalty' was deprecated.*")

import os
RAIZ = Path(__file__).resolve().parents[1]  # raiz del repositorio
EXTERNO = Path(os.environ.get("TESIS_EXTERNO", RAIZ / "externo"))  # Spider, MultiSpider y modelo (descargados aparte)
BASE = RAIZ
DATA = BASE / "data"
RESULTADOS = BASE / "resultados"
TABLAS = RESULTADOS / "tablas"
LOGS = BASE / "logs"

RANDOM_STATE = 42
N_REPLICAS = 20
NIVELES_RHO = [0.10, 0.20, 0.30]


def entrenar_predecir(modelo_operativo, cfg_fold, X_train, y_train, X_test):
    if modelo_operativo == "lr":
        clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, C=cfg_fold["C_lr"])
    elif modelo_operativo == "mlp":
        hidden = ast.literal_eval(str(cfg_fold["hidden_mlp"]))
        clf = MLPClassifier(hidden_layer_sizes=hidden, alpha=cfg_fold["alpha_mlp"],
                             activation="relu", solver="adam", early_stopping=True,
                             validation_fraction=0.1, max_iter=500, random_state=RANDOM_STATE)
    elif modelo_operativo == "gb":
        clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                              max_depth=int(cfg_fold["max_depth_gb"]),
                                              early_stopping=True, validation_fraction=0.1,
                                              random_state=RANDOM_STATE)
    else:
        raise ValueError(modelo_operativo)

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)
    clf.fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]


def main():
    lineas = []

    def log(s=""):
        print(s, flush=True)
        lineas.append(s)

    log("=" * 70)
    log("Fase 4.4 - Analisis (d): ruido asimetrico en los reportes")
    log("=" * 70)

    modelo_operativo = (LOGS / "modelo_operativo.txt").read_text(encoding="utf-8").strip()
    log(f"Modelo operativo: {modelo_operativo.upper()}")

    tabla_a = pd.read_csv(TABLAS / "tabla_a_modelos.csv", encoding="utf-8")
    auroc_longitud = tabla_a[(tabla_a["metrica"] == "AUROC") & (tabla_a["modelo"] == "longitud")]["punto"].iloc[0]
    log(f"AUROC de referencia (longitud): {auroc_longitud:.4f}")

    acts_es = np.load(DATA / "acts_es.npy")
    meta_es = pd.read_csv(DATA / "meta_es.csv", encoding="utf-8")
    folds = pd.read_csv(DATA / "folds.csv", encoding="utf-8")
    sel = pd.read_csv(LOGS / "seleccion_por_pliegue.csv", encoding="utf-8")
    sel_por_fold = {int(r["fold"]): r for _, r in sel.iterrows()}

    meta_idx = meta_es.set_index("idx")
    idx_validos = folds["idx"].values
    y_validos = meta_idx.loc[idx_validos, "y"].astype(int).values
    fold_externo = folds["fold_externo"].values
    acts_validos = acts_es[idx_validos]

    filas = []
    for rho in NIVELES_RHO:
        log(f"\nNivel de ruido rho = {rho}")
        aucs_replica = []
        for replica in range(N_REPLICAS):
            y_pool, p_pool = [], []
            for fold_ext in range(5):
                cfg = sel_por_fold[fold_ext]
                capa = int(cfg["capa_lr"])
                idx_train = np.where(fold_externo != fold_ext)[0]
                idx_test = np.where(fold_externo == fold_ext)[0]

                y_train_true = y_validos[idx_train]
                rng = np.random.default_rng(RANDOM_STATE + replica * 1000 + fold_ext)
                y_train_noisy = y_train_true.copy()
                idx_neg = np.where(y_train_true == 0)[0]
                n_flip = int(round(rho * len(idx_neg)))
                if n_flip > 0:
                    flip_pos = rng.choice(idx_neg, size=n_flip, replace=False)
                    y_train_noisy[flip_pos] = 1

                X_train = acts_validos[idx_train, capa, :].astype(np.float32)
                X_test = acts_validos[idx_test, capa, :].astype(np.float32)
                p_test = entrenar_predecir(modelo_operativo, cfg, X_train, y_train_noisy, X_test)

                y_pool.extend(y_validos[idx_test].tolist())  # etiquetas VERDADERAS
                p_pool.extend(p_test.tolist())

            aucs_replica.append(roc_auc_score(y_pool, p_pool))

        media = float(np.mean(aucs_replica))
        p2_5, p97_5 = np.percentile(aucs_replica, [2.5, 97.5])
        supera_en_010 = (media - auroc_longitud) >= 0.10
        supera_absoluto = media > auroc_longitud
        log(f"  AUROC medio (20 replicas): {media:.4f}  percentil [2.5,97.5] = [{p2_5:.4f}, {p97_5:.4f}]")
        log(f"  Supera a longitud en >= 0.10: {'si' if supera_en_010 else 'no'} "
            f"(diferencia = {media-auroc_longitud:+.4f})")
        log(f"  Supera a longitud en absoluto: {'si' if supera_absoluto else 'no'}")
        filas.append({"rho": rho, "auroc_medio": media, "p2_5": p2_5, "p97_5": p97_5,
                      "supera_longitud_en_010": supera_en_010, "supera_longitud_absoluto": supera_absoluto})

    df = pd.DataFrame(filas)
    df.to_csv(TABLAS / "tabla_d_ruido.csv", index=False, encoding="utf-8")
    log(f"\nGuardado: {TABLAS / 'tabla_d_ruido.csv'}")

    with open(LOGS / "analisis_d.log", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    return 0


if __name__ == "__main__":
    sys.exit(main())
