# -*- coding: utf-8 -*-
"""
Fase 4.3 - Analisis (c): curva de aprendizaje.

Para el modelo operativo, con la capa e hiperparametros YA ELEGIDOS por
pliegue externo (logs/seleccion_por_pliegue.csv, no se repite la busqueda):
tamanos de entrenamiento n in {100, 200, 400, todo}; 20 submuestras
estratificadas por y del entrenamiento externo por tamano y pliegue;
AUROC OOF agregado (pool de las 5 pruebas externas) por replica; media y
percentiles 2.5-97.5 entre las 20 replicas.
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
from sklearn.model_selection import train_test_split
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
TAMANOS = [100, 200, 400, "todo"]


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
    log("Fase 4.3 - Analisis (c): curva de aprendizaje")
    log("=" * 70)

    modelo_operativo = (LOGS / "modelo_operativo.txt").read_text(encoding="utf-8").strip()
    log(f"Modelo operativo: {modelo_operativo.upper()}")

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
    for tamano in TAMANOS:
        log(f"\nTamano n = {tamano}")
        aucs_replica = []
        marcado_todo = False
        for replica in range(N_REPLICAS):
            y_pool, p_pool = [], []
            for fold_ext in range(5):
                cfg = sel_por_fold[fold_ext]
                capa = int(cfg["capa_lr"])
                idx_train = np.where(fold_externo != fold_ext)[0]
                idx_test = np.where(fold_externo == fold_ext)[0]
                y_train_full = y_validos[idx_train]
                n_disponible = len(idx_train)

                if tamano == "todo" or tamano >= n_disponible:
                    sub_idx = idx_train
                    if tamano != "todo" and tamano >= n_disponible:
                        marcado_todo = True
                else:
                    sub_idx, _ = train_test_split(
                        idx_train, train_size=tamano, stratify=y_train_full,
                        random_state=RANDOM_STATE + replica,
                    )

                X_train = acts_validos[sub_idx, capa, :].astype(np.float32)
                y_train = y_validos[sub_idx]
                X_test = acts_validos[idx_test, capa, :].astype(np.float32)
                p_test = entrenar_predecir(modelo_operativo, cfg, X_train, y_train, X_test)

                y_pool.extend(y_validos[idx_test].tolist())
                p_pool.extend(p_test.tolist())

            aucs_replica.append(roc_auc_score(y_pool, p_pool))

        media = float(np.mean(aucs_replica))
        p2_5, p97_5 = np.percentile(aucs_replica, [2.5, 97.5])
        log(f"  AUROC medio (20 replicas): {media:.4f}  percentil [2.5,97.5] = [{p2_5:.4f}, {p97_5:.4f}]"
            f"{'  (n solicitado > disponible en algun pliegue -> se uso todo)' if marcado_todo else ''}")
        filas.append({"n": tamano, "auroc_medio": media, "p2_5": p2_5, "p97_5": p97_5,
                      "marcado_uso_todo": marcado_todo})

    df = pd.DataFrame(filas)

    # Criterio: menor tamano cuyo intervalo se solapa con el del modelo completo (todo)
    ref = df[df["n"] == "todo"].iloc[0]
    log("\n" + "-" * 70)
    log("Criterio: menor tamano cuyo IC se solapa con el de 'todo'")
    log("-" * 70)
    menor_tamano_suficiente = None
    for _, fila in df[df["n"] != "todo"].iterrows():
        solapa = (fila["p2_5"] <= ref["p97_5"]) and (fila["p97_5"] >= ref["p2_5"])
        log(f"  n={fila['n']}: IC=[{fila['p2_5']:.4f},{fila['p97_5']:.4f}] vs "
            f"todo=[{ref['p2_5']:.4f},{ref['p97_5']:.4f}] -> {'SOLAPA' if solapa else 'no solapa'}")
        if solapa and menor_tamano_suficiente is None:
            menor_tamano_suficiente = fila["n"]

    if menor_tamano_suficiente is not None:
        pct_desempeno = 100 * df.loc[df["n"] == menor_tamano_suficiente, "auroc_medio"].iloc[0] / ref["auroc_medio"]
        log(f"\nCon ~{menor_tamano_suficiente} respuestas reportadas, el detector alcanza "
            f"~{pct_desempeno:.0f}% del AUROC del modelo completo (n=todo).")
    else:
        log("\nNingun tamano reducido solapa con 'todo': se necesitan las {} preguntas completas.".format(
            len(y_validos)))

    df.to_csv(TABLAS / "tabla_c_aprendizaje.csv", index=False, encoding="utf-8")
    log(f"\nGuardado: {TABLAS / 'tabla_c_aprendizaje.csv'}")

    with open(LOGS / "analisis_c.log", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    return 0


if __name__ == "__main__":
    sys.exit(main())
