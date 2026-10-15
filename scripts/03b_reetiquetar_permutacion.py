# -*- coding: utf-8 -*-
"""
Re-etiqueta los casos 'silencioso' con el verificador v2 (tolerancia a
permutacion de columnas). No toca activaciones ni generacion.
Salidas:
  data/meta_es_v2.csv                  (meta con etiquetas corregidas)
  logs/reetiquetado_permutacion.csv    (casos que cambian, para revision)
  logs/auditoria_silenciosos.csv       (50 silenciosos v2 al azar, para anotar a mano)
Solo pueden cambiar silencioso -> correcta; sintacticos y timeouts no se tocan.
"""
from pathlib import Path

import pandas as pd

from utils_verificador import ejecutar_con_timeout
from utils_verificador_v2 import comparar

import os
RAIZ = Path(__file__).resolve().parents[1]  # raiz del repositorio
EXTERNO = Path(os.environ.get("TESIS_EXTERNO", RAIZ / "externo"))  # Spider, MultiSpider y modelo (descargados aparte)
BASE = RAIZ
SPIDER_DB = EXTERNO / "spider_data" / "database"
DATA = BASE / "data"
LOGS = BASE / "logs"


def main():
    meta = pd.read_csv(DATA / "meta_es.csv", encoding="utf-8")
    meta["desenlace_v1"] = meta["desenlace"]
    cambios = []
    for i, r in meta[meta["desenlace"] == "silencioso"].iterrows():
        ruta = SPIDER_DB / r["db_id"] / f"{r['db_id']}.sqlite"
        fg, eg = ejecutar_con_timeout(r["gold_sql"], ruta)
        fp, ep = ejecutar_con_timeout(r["pred_sql"], ruta)
        if eg is None and ep is None and comparar(fg, fp, bool(r["gold_has_orderby"])):
            meta.at[i, "desenlace"] = "correcta"
            meta.at[i, "y"] = 1
            cambios.append({"qid": r["qid"], "dificultad": r["dificultad"],
                            "gold_sql": r["gold_sql"], "pred_sql": r["pred_sql"]})

    pd.DataFrame(cambios).to_csv(LOGS / "reetiquetado_permutacion.csv", index=False, encoding="utf-8")
    meta.to_csv(DATA / "meta_es_v2.csv", index=False, encoding="utf-8")

    validos = meta[meta["gold_status"] == "ok"]
    print(f"Casos silencioso -> correcta: {len(cambios)}")
    print(validos["desenlace"].value_counts())
    print(f"Tasa de acierto v2: {100 * validos['y'].mean():.1f}%")

    aud = validos[validos["desenlace"] == "silencioso"].sample(n=50, random_state=42)
    aud = aud[["qid", "db_id", "dificultad", "question_es", "gold_sql", "pred_sql"]].copy()
    # Categorias sugeridas para anotar: error_real / gold_defectuoso /
    # traduccion_ambigua / equivalente_semantico / ambiguo
    aud["categoria_autor"] = ""
    aud["nota"] = ""
    aud.to_csv(LOGS / "auditoria_silenciosos.csv", index=False, encoding="utf-8-sig")


def agregar_question_en():
    """Agrega la pregunta en ingles original al CSV de auditoria (paso manual opcional)."""
    import json

    dev = json.load(open(SPIDER_DB.parent / "dev.json", encoding="utf-8"))
    aud = pd.read_csv(LOGS / "auditoria_silenciosos.csv", encoding="utf-8-sig")
    aud["question_en"] = aud["qid"].map(lambda q: dev[int(q[3:])]["question"])
    aud.to_csv(LOGS / "auditoria_silenciosos.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
