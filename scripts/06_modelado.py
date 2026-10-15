# -*- coding: utf-8 -*-
"""
Fase 3.2 - Modelado. Corre en CPU.

Para cada uno de los 5 pliegues externos (StratifiedGroupKFold, db_id como
grupo, guardados en data/folds.csv):
  1. Validacion interna (4 pliegues, mismos criterios) para elegir capa y
     C de la regresion logistica L2 (rejilla C x capa) por AUROC OOF interno.
  2. MLP y Gradient Boosting usan la MISMA capa elegida por LR; su propia
     rejilla de hiperparametros se elige tambien por AUROC OOF interno.
  3. Se reentrena cada modelo con todo el entrenamiento externo y se predice
     el pliegue de prueba externo (p_raw).
  4. Calibracion de Platt: puntuaciones OOF internas del modelo con la
     configuracion elegida -> ajustar LogisticRegression(C=1e6) univariada
     -> aplicar al pliegue de prueba externo (p_cal).
  5. Referencias (longitud, confianza): LR univariada C=1, sin busqueda de
     hiperparametros ni calibracion, mismos pliegues externos.

Salidas:
  resultados/oof_predicciones.csv
  logs/seleccion_por_pliegue.csv    (capa, C, cfg MLP, cfg GB por pliegue)
  resultados/tablas/perfil_capas.csv

Uso:
  python 06_modelado.py            # corrida completa (5 pliegues)
  python 06_modelado.py --prueba   # solo pliegue externo 0, rejilla reducida
"""
import argparse
import sys
import time
import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message=".*'penalty' was deprecated.*",
)
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import os
RAIZ = Path(__file__).resolve().parents[1]  # raiz del repositorio
EXTERNO = Path(os.environ.get("TESIS_EXTERNO", RAIZ / "externo"))  # Spider, MultiSpider y modelo (descargados aparte)
BASE = RAIZ
DATA = BASE / "data"
RESULTADOS = BASE / "resultados"
TABLAS = RESULTADOS / "tablas"
LOGS = BASE / "logs"
RESULTADOS.mkdir(parents=True, exist_ok=True)
TABLAS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS_EXT = 5
N_SPLITS_INT = 4
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
CAPAS_GRID = list(range(1, 37))  # capa 0 omitida (identica entre preguntas)
MLP_GRID = [(h, a) for h in [(32,), (128,)] for a in [1e-3, 1e-2, 1e-1]]
GB_GRID = [2, 3]


def log_a(lineas, s=""):
    print(s, flush=True)
    lineas.append(s)


def puntaje_logit(modelo, X):
    """decision_function si existe; si no, logit(predict_proba) recortado."""
    if hasattr(modelo, "decision_function"):
        return modelo.decision_function(X)
    p = np.clip(modelo.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ajustar_lr(X, y, C):
    return LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, C=C).fit(X, y)


def evaluar_capa_lr(layer, acts_train_ext, y_train_ext, internal_splits, c_grid):
    """Para una capa: escala por pliegue interno una vez, evalua todos los C
    de la rejilla, devuelve {C: auc_oof_interno} y los datos escalados
    (para reutilizar si esta capa resulta ser la elegida)."""
    X_l = acts_train_ext[:, layer, :].astype(np.float32)
    datos_internos = []
    for tr_pos, val_pos in internal_splits:
        scaler = StandardScaler().fit(X_l[tr_pos])
        Xtr = scaler.transform(X_l[tr_pos])
        Xval = scaler.transform(X_l[val_pos])
        datos_internos.append((Xtr, y_train_ext[tr_pos], Xval, val_pos))

    resultados_c = {}
    for C in c_grid:
        oof = np.zeros(len(y_train_ext))
        for Xtr, ytr, Xval, val_pos in datos_internos:
            clf = ajustar_lr(Xtr, ytr, C)
            oof[val_pos] = clf.predict_proba(Xval)[:, 1]
        resultados_c[C] = roc_auc_score(y_train_ext, oof)
    return layer, resultados_c


def procesar_pliegue_externo(fold_ext, acts_validos, y_validos, db_id_validos,
                              prompt_tokens, conf_logprob, qids, idxs,
                              capas_grid, c_grid, n_jobs, log_fn):
    idx_train = np.where(_fold_externo != fold_ext)[0]
    idx_test = np.where(_fold_externo == fold_ext)[0]

    y_train = y_validos[idx_train]
    y_test = y_validos[idx_test]
    grupos_train = db_id_validos[idx_train]
    acts_train_ext = acts_validos[idx_train]
    acts_test_ext = acts_validos[idx_test]

    log_fn(f"\n=== Pliegue externo {fold_ext} === (train={len(idx_train)}, test={len(idx_test)})")

    skf_int = StratifiedGroupKFold(n_splits=N_SPLITS_INT, shuffle=True, random_state=RANDOM_STATE)
    internal_splits = list(skf_int.split(np.zeros(len(idx_train)), y_train, groups=grupos_train))

    t0 = time.time()
    resultados_por_capa = Parallel(n_jobs=n_jobs)(
        delayed(evaluar_capa_lr)(layer, acts_train_ext, y_train, internal_splits, c_grid)
        for layer in capas_grid
    )
    log_fn(f"  Grilla LR (capa x C) evaluada en {time.time()-t0:.1f}s "
           f"({len(capas_grid)} capas x {len(c_grid)} C = {len(capas_grid)*len(c_grid)} combinaciones)")

    mejor_auc, mejor_layer, mejor_C = -1, None, None
    perfil_capas_pliegue = {}
    for layer, resultados_c in resultados_por_capa:
        mejor_c_capa = max(resultados_c, key=resultados_c.get)
        perfil_capas_pliegue[layer] = resultados_c  # se usa en 7.9 con C fijo, no aqui
        if resultados_c[mejor_c_capa] > mejor_auc:
            mejor_auc, mejor_layer, mejor_C = resultados_c[mejor_c_capa], layer, mejor_c_capa

    log_fn(f"  LR elegido: capa={mejor_layer}, C={mejor_C}, AUROC OOF interno={mejor_auc:.4f}")

    X_best_train = acts_train_ext[:, mejor_layer, :].astype(np.float32)
    X_best_test = acts_test_ext[:, mejor_layer, :].astype(np.float32)

    # ---- LR final: reentrenar con todo el train externo, predecir test ----
    scaler_lr = StandardScaler().fit(X_best_train)
    Xtr_lr = scaler_lr.transform(X_best_train)
    Xtest_lr = scaler_lr.transform(X_best_test)
    clf_lr = ajustar_lr(Xtr_lr, y_train, mejor_C)
    p_raw_lr_test = clf_lr.predict_proba(Xtest_lr)[:, 1]
    score_lr_test = puntaje_logit(clf_lr, Xtest_lr)

    # OOF interno con la config elegida (para Platt)
    oof_logit_lr = np.zeros(len(y_train))
    for tr_pos, val_pos in internal_splits:
        sc = StandardScaler().fit(X_best_train[tr_pos])
        Xtr = sc.transform(X_best_train[tr_pos])
        Xval = sc.transform(X_best_train[val_pos])
        clf = ajustar_lr(Xtr, y_train[tr_pos], mejor_C)
        oof_logit_lr[val_pos] = puntaje_logit(clf, Xval)
    platt_lr = LogisticRegression(C=1e6, solver="lbfgs").fit(oof_logit_lr.reshape(-1, 1), y_train)
    p_cal_lr_test = platt_lr.predict_proba(score_lr_test.reshape(-1, 1))[:, 1]

    # ---- MLP: misma capa, rejilla propia por AUROC OOF interno ----
    t0 = time.time()
    mejor_auc_mlp, mejor_cfg_mlp = -1, None
    for hidden, alpha in MLP_GRID:
        oof = np.zeros(len(y_train))
        for tr_pos, val_pos in internal_splits:
            sc = StandardScaler().fit(X_best_train[tr_pos])
            Xtr = sc.transform(X_best_train[tr_pos])
            Xval = sc.transform(X_best_train[val_pos])
            clf = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, activation="relu",
                                 solver="adam", early_stopping=True, validation_fraction=0.1,
                                 max_iter=500, random_state=RANDOM_STATE).fit(Xtr, y_train[tr_pos])
            oof[val_pos] = clf.predict_proba(Xval)[:, 1]
        auc = roc_auc_score(y_train, oof)
        if auc > mejor_auc_mlp:
            mejor_auc_mlp, mejor_cfg_mlp = auc, (hidden, alpha)
    log_fn(f"  MLP elegido: hidden={mejor_cfg_mlp[0]}, alpha={mejor_cfg_mlp[1]}, "
           f"AUROC OOF interno={mejor_auc_mlp:.4f} ({time.time()-t0:.1f}s)")

    scaler_mlp = StandardScaler().fit(X_best_train)
    Xtr_mlp = scaler_mlp.transform(X_best_train)
    Xtest_mlp = scaler_mlp.transform(X_best_test)
    clf_mlp = MLPClassifier(hidden_layer_sizes=mejor_cfg_mlp[0], alpha=mejor_cfg_mlp[1],
                             activation="relu", solver="adam", early_stopping=True,
                             validation_fraction=0.1, max_iter=500,
                             random_state=RANDOM_STATE).fit(Xtr_mlp, y_train)
    p_raw_mlp_test = clf_mlp.predict_proba(Xtest_mlp)[:, 1]
    score_mlp_test = puntaje_logit(clf_mlp, Xtest_mlp)

    oof_logit_mlp = np.zeros(len(y_train))
    for tr_pos, val_pos in internal_splits:
        sc = StandardScaler().fit(X_best_train[tr_pos])
        Xtr = sc.transform(X_best_train[tr_pos])
        Xval = sc.transform(X_best_train[val_pos])
        clf = MLPClassifier(hidden_layer_sizes=mejor_cfg_mlp[0], alpha=mejor_cfg_mlp[1],
                             activation="relu", solver="adam", early_stopping=True,
                             validation_fraction=0.1, max_iter=500,
                             random_state=RANDOM_STATE).fit(Xtr, y_train[tr_pos])
        oof_logit_mlp[val_pos] = puntaje_logit(clf, Xval)
    platt_mlp = LogisticRegression(C=1e6, solver="lbfgs").fit(oof_logit_mlp.reshape(-1, 1), y_train)
    p_cal_mlp_test = platt_mlp.predict_proba(score_mlp_test.reshape(-1, 1))[:, 1]

    # ---- Gradient Boosting: misma capa, rejilla propia ----
    t0 = time.time()
    mejor_auc_gb, mejor_depth = -1, None
    for depth in GB_GRID:
        oof = np.zeros(len(y_train))
        for tr_pos, val_pos in internal_splits:
            sc = StandardScaler().fit(X_best_train[tr_pos])
            Xtr = sc.transform(X_best_train[tr_pos])
            Xval = sc.transform(X_best_train[val_pos])
            clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=depth,
                                                  early_stopping=True, validation_fraction=0.1,
                                                  random_state=RANDOM_STATE).fit(Xtr, y_train[tr_pos])
            oof[val_pos] = clf.predict_proba(Xval)[:, 1]
        auc = roc_auc_score(y_train, oof)
        if auc > mejor_auc_gb:
            mejor_auc_gb, mejor_depth = auc, depth
    log_fn(f"  GB elegido: max_depth={mejor_depth}, AUROC OOF interno={mejor_auc_gb:.4f} "
           f"({time.time()-t0:.1f}s)")

    scaler_gb = StandardScaler().fit(X_best_train)
    Xtr_gb = scaler_gb.transform(X_best_train)
    Xtest_gb = scaler_gb.transform(X_best_test)
    clf_gb = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=mejor_depth,
                                             early_stopping=True, validation_fraction=0.1,
                                             random_state=RANDOM_STATE).fit(Xtr_gb, y_train)
    p_raw_gb_test = clf_gb.predict_proba(Xtest_gb)[:, 1]
    score_gb_test = puntaje_logit(clf_gb, Xtest_gb)

    oof_logit_gb = np.zeros(len(y_train))
    for tr_pos, val_pos in internal_splits:
        sc = StandardScaler().fit(X_best_train[tr_pos])
        Xtr = sc.transform(X_best_train[tr_pos])
        Xval = sc.transform(X_best_train[val_pos])
        clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=mejor_depth,
                                              early_stopping=True, validation_fraction=0.1,
                                              random_state=RANDOM_STATE).fit(Xtr, y_train[tr_pos])
        oof_logit_gb[val_pos] = puntaje_logit(clf, Xval)
    platt_gb = LogisticRegression(C=1e6, solver="lbfgs").fit(oof_logit_gb.reshape(-1, 1), y_train)
    p_cal_gb_test = platt_gb.predict_proba(score_gb_test.reshape(-1, 1))[:, 1]

    # ---- Referencias: longitud y confianza, univariadas C=1, sin busqueda ni calibracion ----
    len_train = prompt_tokens[idx_train].reshape(-1, 1).astype(np.float32)
    len_test = prompt_tokens[idx_test].reshape(-1, 1).astype(np.float32)
    sc_len = StandardScaler().fit(len_train)
    clf_len = LogisticRegression(C=1, solver="lbfgs", max_iter=5000).fit(sc_len.transform(len_train), y_train)
    p_len_test = clf_len.predict_proba(sc_len.transform(len_test))[:, 1]

    conf_train = conf_logprob[idx_train].reshape(-1, 1).astype(np.float32)
    conf_test = conf_logprob[idx_test].reshape(-1, 1).astype(np.float32)
    sc_conf = StandardScaler().fit(conf_train)
    clf_conf = LogisticRegression(C=1, solver="lbfgs", max_iter=5000).fit(sc_conf.transform(conf_train), y_train)
    p_conf_test = clf_conf.predict_proba(sc_conf.transform(conf_test))[:, 1]

    filas = []
    for j, pos in enumerate(idx_test):
        filas.append({
            "qid": qids[pos], "idx": idxs[pos], "fold": fold_ext, "y": int(y_validos[pos]),
            "p_raw_lr": p_raw_lr_test[j], "p_cal_lr": p_cal_lr_test[j],
            "p_raw_mlp": p_raw_mlp_test[j], "p_cal_mlp": p_cal_mlp_test[j],
            "p_raw_gb": p_raw_gb_test[j], "p_cal_gb": p_cal_gb_test[j],
            "p_len": p_len_test[j], "p_conf": p_conf_test[j],
            "capa_lr": mejor_layer, "C_lr": mejor_C,
            "cfg_mlp": f"{mejor_cfg_mlp[0]}|alpha={mejor_cfg_mlp[1]}",
            "cfg_gb": f"max_depth={mejor_depth}",
        })

    seleccion = {
        "fold": fold_ext, "capa_lr": mejor_layer, "C_lr": mejor_C,
        "auroc_oof_interno_lr": mejor_auc, "hidden_mlp": mejor_cfg_mlp[0],
        "alpha_mlp": mejor_cfg_mlp[1], "auroc_oof_interno_mlp": mejor_auc_mlp,
        "max_depth_gb": mejor_depth, "auroc_oof_interno_gb": mejor_auc_gb,
    }
    return filas, seleccion, resultados_por_capa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prueba", action="store_true",
                     help="Solo pliegue externo 0, rejilla reducida (sanity check).")
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    lineas = []

    def log(s=""):
        log_a(lineas, s)

    log("=" * 70)
    log(f"Fase 3.2 - Modelado {'(PRUEBA)' if args.prueba else '(COMPLETA)'} - "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    acts_es = np.load(DATA / "acts_es.npy")
    meta_es = pd.read_csv(DATA / "meta_es.csv", encoding="utf-8")
    folds = pd.read_csv(DATA / "folds.csv", encoding="utf-8")

    meta_idx = meta_es.set_index("idx")
    idx_validos = folds["idx"].values
    global _fold_externo
    _fold_externo = folds["fold_externo"].values

    y_validos = meta_idx.loc[idx_validos, "y"].astype(int).values
    db_id_validos = folds["db_id"].values
    qids = folds["qid"].values
    prompt_tokens = meta_idx.loc[idx_validos, "prompt_tokens"].astype(float).values
    conf_logprob = meta_idx.loc[idx_validos, "conf_logprob_mean"].astype(float).values
    acts_validos = acts_es[idx_validos]

    log(f"n valido = {len(idx_validos)} | acts_validos.shape = {acts_validos.shape}")

    capas_grid = CAPAS_GRID if not args.prueba else [10, 20, 27]
    c_grid = C_GRID if not args.prueba else [1e-2, 1]
    folds_a_correr = [0] if args.prueba else list(range(N_SPLITS_EXT))

    todas_filas = []
    todas_selecciones = []
    perfil_capas_todos = {}  # {fold: {layer: {C: auc}}}

    t_inicio = time.time()
    for fold_ext in folds_a_correr:
        filas, seleccion, resultados_por_capa = procesar_pliegue_externo(
            fold_ext, acts_validos, y_validos, db_id_validos, prompt_tokens,
            conf_logprob, qids, idx_validos, capas_grid, c_grid, args.n_jobs, log,
        )
        todas_filas.extend(filas)
        todas_selecciones.append(seleccion)
        perfil_capas_todos[fold_ext] = dict(resultados_por_capa)

    log(f"\nTiempo total modelado: {(time.time()-t_inicio)/60:.1f} min")

    oof = pd.DataFrame(todas_filas)
    sel = pd.DataFrame(todas_selecciones)

    if args.prueba:
        oof.to_csv(RESULTADOS / "oof_predicciones_prueba.csv", index=False, encoding="utf-8")
        sel.to_csv(LOGS / "seleccion_por_pliegue_prueba.csv", index=False, encoding="utf-8")
        log(f"\n[PRUEBA] Guardado resultados/oof_predicciones_prueba.csv y "
            f"logs/seleccion_por_pliegue_prueba.csv")
        with open(LOGS / "modelado_prueba.log", "w", encoding="utf-8") as f:
            f.write("\n".join(lineas))
        return 0

    # ---- 7.9: perfil por capas, descriptivo, C fijo = mas frecuente elegido en 7.3 ----
    log("\n" + "=" * 70)
    log("Perfil por capas (7.9) - descriptivo, C fijo = mas frecuente")
    log("=" * 70)
    c_mas_frecuente = Counter(sel["C_lr"]).most_common(1)[0][0]
    log(f"C mas frecuente entre los 5 pliegues: {c_mas_frecuente}")

    filas_perfil = []
    for layer in CAPAS_GRID:
        oof_layer = np.zeros(len(y_validos))
        for fold_ext in range(N_SPLITS_EXT):
            idx_train = np.where(_fold_externo != fold_ext)[0]
            idx_test = np.where(_fold_externo == fold_ext)[0]
            X_l = acts_validos[:, layer, :].astype(np.float32)
            sc = StandardScaler().fit(X_l[idx_train])
            Xtr = sc.transform(X_l[idx_train])
            Xtest = sc.transform(X_l[idx_test])
            clf = ajustar_lr(Xtr, y_validos[idx_train], c_mas_frecuente)
            oof_layer[idx_test] = clf.predict_proba(Xtest)[:, 1]
        auc_layer = roc_auc_score(y_validos, oof_layer)
        filas_perfil.append({"capa": layer, "auroc_oof": auc_layer})
    df_perfil = pd.DataFrame(filas_perfil)
    df_perfil.to_csv(TABLAS / "perfil_capas.csv", index=False, encoding="utf-8")
    log(f"Guardado: {TABLAS / 'perfil_capas.csv'}")
    log(str(df_perfil))

    oof = oof.merge(meta_es[["idx", "desenlace", "dificultad"]], on="idx", how="left")

    oof.to_csv(RESULTADOS / "oof_predicciones.csv", index=False, encoding="utf-8")
    sel.to_csv(LOGS / "seleccion_por_pliegue.csv", index=False, encoding="utf-8")
    log(f"\nGuardado: {RESULTADOS / 'oof_predicciones.csv'} ({len(oof)} filas)")
    log(f"Guardado: {LOGS / 'seleccion_por_pliegue.csv'}")

    for col in ["p_raw_lr", "p_cal_lr", "p_raw_mlp", "p_cal_mlp", "p_raw_gb", "p_cal_gb",
                "p_len", "p_conf"]:
        n_nan = oof[col].isna().sum()
        if n_nan:
            log(f"AVISO: columna {col} tiene {n_nan} NaN")
    log(f"NaN totales en oof_predicciones: {oof.isna().sum().sum()} (esperado 0)")

    with open(LOGS / "modelado.log", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    return 0


if __name__ == "__main__":
    sys.exit(main())
