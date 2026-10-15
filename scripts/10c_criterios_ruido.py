# -*- coding: utf-8 -*-
"""
Fase R6.5 - Criterios del analisis (d) para AMBAS variantes de ruido
(uniforme: tabla_d_ruido.csv; solo silenciosos: tabla_d2_ruido_silencioso.csv).

Para cada nivel rho, con el AUROC medio de las 20 replicas frente al AUROC de longitud (tabla_a):
  - supera_010: AUROC medio - AUROC longitud >= 0,10
  - supera_abs: AUROC medio > AUROC longitud
Se reporta el primer nivel en que deja de cumplirse cada criterio; si no ocurre
dentro del rango evaluado, se reporta asi.
"""
import sys
from pathlib import Path

import pandas as pd

import os
RAIZ = Path(__file__).resolve().parents[1]  # raiz del repositorio
EXTERNO = Path(os.environ.get("TESIS_EXTERNO", RAIZ / "externo"))  # Spider, MultiSpider y modelo (descargados aparte)
BASE = RAIZ
TABLAS = BASE / "resultados" / "tablas"
LOGS = BASE / "logs"


def main():
    lineas = []

    def log(s=""):
        print(s, flush=True)
        lineas.append(s)

    ta = pd.read_csv(TABLAS / "tabla_a_modelos.csv", encoding="utf-8")
    auc_long = float(ta[(ta["metrica"] == "AUROC") & (ta["modelo"] == "longitud")]["punto"].iloc[0])
    log(f"Criterios del analisis (d) - AUROC de referencia (longitud) = {auc_long:.4f}")

    filas = []
    for variante, archivo in [("uniforme", "tabla_d_ruido.csv"), ("solo_silenciosos", "tabla_d2_ruido_silencioso.csv")]:
        t = pd.read_csv(TABLAS / archivo, encoding="utf-8")
        log(f"\nVariante: {variante} (rho evaluados: {', '.join(f'{r:.2f}' for r in t['rho'])})")
        primero_010, primero_abs = None, None
        for _, r in t.iterrows():
            dif = r["auroc_medio"] - auc_long
            s010, sabs = dif >= 0.10, r["auroc_medio"] > auc_long
            log(f"  rho={r['rho']:.2f}: AUROC medio={r['auroc_medio']:.4f} "
                f"[{r['p2_5']:.4f}, {r['p97_5']:.4f}]  diferencia vs longitud={dif:+.4f}  "
                f">=0,10: {'si' if s010 else 'no'}  >0: {'si' if sabs else 'no'}")
            filas.append({"variante": variante, "rho": r["rho"], "auroc_medio": r["auroc_medio"],
                          "p2_5": r["p2_5"], "p97_5": r["p97_5"], "diferencia_vs_longitud": dif,
                          "supera_010": bool(s010), "supera_abs": bool(sabs)})
            if primero_010 is None and not s010:
                primero_010 = r["rho"]
            if primero_abs is None and not sabs:
                primero_abs = r["rho"]
        rango = f"[{t['rho'].min():.2f}; {t['rho'].max():.2f}]"
        log(f"  -> primer nivel en que deja de superar a longitud en >= 0,10: "
            f"{primero_010 if primero_010 is not None else 'no ocurre dentro del rango ' + rango}")
        log(f"  -> primer nivel en que deja de superar a longitud en absoluto: "
            f"{primero_abs if primero_abs is not None else 'no ocurre dentro del rango ' + rango}")

    pd.DataFrame(filas).to_csv(TABLAS / "tabla_d_criterios.csv", index=False, encoding="utf-8")
    (LOGS / "analisis_d_criterios.log").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
