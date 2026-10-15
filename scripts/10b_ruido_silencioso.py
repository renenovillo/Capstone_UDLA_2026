# -*- coding: utf-8 -*-
"""
Variante del analisis (d): ruido DEPENDIENTE DE LA INSTANCIA.
Solo se reetiquetan como correctos errores SILENCIOSOS (el usuario siempre
advierte un error sintactico: la consulta no corre). rho se define igual
que en 10_analisis_d.py (fraccion de TODOS los errores del entrenamiento)
para que ambas curvas sean comparables; se extiende hasta 0,50.
Reutiliza entrenar_predecir() de 10_analisis_d.py sin cambios.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import os
RAIZ = Path(__file__).resolve().parents[1]  # raiz del repositorio
EXTERNO = Path(os.environ.get("TESIS_EXTERNO", RAIZ / "externo"))  # Spider, MultiSpider y modelo (descargados aparte)
BASE = RAIZ
SCRIPTS = BASE / "scripts"
DATA = BASE / "data"
TABLAS = BASE / "resultados" / "tablas"
LOGS = BASE / "logs"

spec = importlib.util.spec_from_file_location("d", SCRIPTS / "10_analisis_d.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)

RANDOM_STATE, N_REPLICAS = 42, 20
NIVELES_RHO = [0.10, 0.20, 0.30, 0.40, 0.50]
META = DATA / "meta_es.csv"


def main():
    modelo = (LOGS / "modelo_operativo.txt").read_text(encoding="utf-8").strip()
    acts = np.load(DATA / "acts_es.npy")
    meta = pd.read_csv(META, encoding="utf-8").set_index("idx")
    folds = pd.read_csv(DATA / "folds.csv", encoding="utf-8")
    sel = {int(r["fold"]): r for _, r in pd.read_csv(LOGS / "seleccion_por_pliegue.csv").iterrows()}

    idx = folds["idx"].values
    y = meta.loc[idx, "y"].astype(int).values
    silencioso = (meta.loc[idx, "desenlace"] == "silencioso").values
    fold = folds["fold_externo"].values
    acts = acts[idx]

    filas = []
    for rho in NIVELES_RHO:
        aucs = []
        for rep in range(N_REPLICAS):
            yp, pp = [], []
            for f in range(5):
                cfg = sel[f]
                capa = int(cfg["capa_lr"])
                tr = np.where(fold != f)[0]
                te = np.where(fold == f)[0]
                y_tr = y[tr].copy()
                n_flip = int(round(rho * np.sum(y_tr == 0)))
                cand = np.where(silencioso[tr])[0]
                if n_flip > len(cand):
                    raise ValueError(f"rho={rho}: se piden {n_flip} volteos y hay {len(cand)} silenciosos")
                rng = np.random.default_rng(RANDOM_STATE + rep * 1000 + f)
                y_tr[rng.choice(cand, size=n_flip, replace=False)] = 1
                p = d.entrenar_predecir(modelo, cfg, acts[tr, capa].astype(np.float32), y_tr,
                                        acts[te, capa].astype(np.float32))
                yp.extend(y[te]); pp.extend(p)
            aucs.append(roc_auc_score(yp, pp))
        lo, hi = np.percentile(aucs, [2.5, 97.5])
        filas.append({"rho": rho, "auroc_medio": np.mean(aucs), "p2_5": lo, "p97_5": hi})
        print(filas[-1], flush=True)
    pd.DataFrame(filas).to_csv(TABLAS / "tabla_d2_ruido_silencioso.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
