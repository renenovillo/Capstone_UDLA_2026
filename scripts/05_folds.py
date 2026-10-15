# -*- coding: utf-8 -*-
"""
Fase 3.1 - Pliegues externos e internos.

Externo: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
  con y como estratificacion y db_id como grupo.
Guarda data/folds.csv con columnas: qid, idx, fold_externo.
Verifica que ningun db_id aparezca en train y test del mismo pliegue externo.

Excluye previamente las filas con gold_status == "error".
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import os
RAIZ = Path(__file__).resolve().parents[1]  # raiz del repositorio
EXTERNO = Path(os.environ.get("TESIS_EXTERNO", RAIZ / "externo"))  # Spider, MultiSpider y modelo (descargados aparte)
BASE = RAIZ
DATA = BASE / "data"
LOGS = BASE / "logs"
RANDOM_STATE = 42
N_SPLITS_EXT = 5


def main():
    lineas = []

    def log(s=""):
        print(s, flush=True)
        lineas.append(s)

    log("=" * 70)
    log("Fase 3.1 - Pliegues externos")
    log("=" * 70)

    meta = pd.read_csv(DATA / "meta_es.csv", encoding="utf-8")
    log(f"Filas totales en meta_es: {len(meta)}")

    meta_validas = meta[meta["gold_status"] != "error"].reset_index(drop=True)
    log(f"Filas validas (gold_status != 'error'): {len(meta_validas)}")

    X_dummy = np.zeros(len(meta_validas))
    y = meta_validas["y"].astype(int).values
    grupos = meta_validas["db_id"].values

    skf = StratifiedGroupKFold(n_splits=N_SPLITS_EXT, shuffle=True, random_state=RANDOM_STATE)

    fold_asignado = np.full(len(meta_validas), -1, dtype=int)
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X_dummy, y, groups=grupos)):
        fold_asignado[test_idx] = fold_id

    assert (fold_asignado >= 0).all(), "Algun registro no quedo asignado a un pliegue"

    meta_validas["fold_externo"] = fold_asignado

    log("\nVerificacion: db_id compartido entre train y test del mismo pliegue")
    ok = True
    for fold_id in range(N_SPLITS_EXT):
        test_dbs = set(meta_validas.loc[meta_validas["fold_externo"] == fold_id, "db_id"])
        train_dbs = set(meta_validas.loc[meta_validas["fold_externo"] != fold_id, "db_id"])
        solapan = test_dbs & train_dbs
        if solapan:
            ok = False
            log(f"  FALLA pliegue {fold_id}: db_id compartidos = {solapan}")
        else:
            log(f"  Pliegue {fold_id}: OK, sin solapamiento de db_id (n_test={len(test_dbs)} bases)")

    log(f"\nDistribucion de tamanos por pliegue:")
    log(str(meta_validas["fold_externo"].value_counts().sort_index()))
    log(f"\nTasa de acierto (y=1) por pliegue:")
    log(str(meta_validas.groupby("fold_externo")["y"].mean()))

    out = meta_validas[["qid", "idx", "db_id", "fold_externo"]]
    out.to_csv(DATA / "folds.csv", index=False, encoding="utf-8")
    log(f"\nGuardado: {DATA / 'folds.csv'}")

    with open(LOGS / "folds.log", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
