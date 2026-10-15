# -*- coding: utf-8 -*-
"""
Fase 2.4 - Etiquetado por ejecucion.

Aplica utils_verificador.verificar_pregunta a cada registro guardado en
data/extraccion/meta_{inicio}_{fin}.jsonl (generados por 02_extraer_y_generar.py)
y escribe data/extraccion/meta_etq_{inicio}_{fin}.jsonl con los campos del
verificador anadidos.

Independiente de la generacion: se puede re-correr (p.ej. tras corregir un
bug del verificador) sin volver a extraer activaciones ni generar SQL.
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils_verificador import verificar_pregunta

import os
RAIZ = Path(__file__).resolve().parents[1]  # raiz del repositorio
EXTERNO = Path(os.environ.get("TESIS_EXTERNO", RAIZ / "externo"))  # Spider, MultiSpider y modelo (descargados aparte)
BASE = RAIZ
SPIDER_DB = EXTERNO / "spider_data" / "database"
DIR_EXTRACCION = BASE / "data" / "extraccion"
LOGS = BASE / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

PATRON_META = re.compile(r"^meta_(\d{4})_(\d{4})\.jsonl$")


def cargar_jsonl(ruta):
    out = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            out.append(json.loads(linea))
    return out


def main():
    lineas = []

    def log(s=""):
        print(s, flush=True)
        lineas.append(s)

    lotes = sorted(
        (m.group(1), m.group(2), DIR_EXTRACCION / m.group(0))
        for f in DIR_EXTRACCION.iterdir()
        if (m := PATRON_META.match(f.name))
    )
    if not lotes:
        log("No se encontraron lotes meta_*.jsonl en data/extraccion/. "
            "Corre primero 02_extraer_y_generar.py.")
        return 1

    log(f"Lotes a etiquetar: {len(lotes)}")
    total = 0
    t0 = time.time()

    for inicio_s, fin_s, ruta_meta in lotes:
        registros = cargar_jsonl(ruta_meta)
        ruta_salida = DIR_EXTRACCION / f"meta_etq_{inicio_s}_{fin_s}.jsonl"
        etiquetados = []
        for reg in registros:
            ruta_db = SPIDER_DB / reg["db_id"] / f"{reg['db_id']}.sqlite"
            resultado = verificar_pregunta(
                reg["gold_sql"], reg["pred_sql"], ruta_db, reg["gold_has_orderby"],
            )
            etiquetados.append({**reg, **resultado})
        with open(ruta_salida, "w", encoding="utf-8") as f:
            for e in etiquetados:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        total += len(etiquetados)
        log(f"  [{inicio_s}:{fin_s}] {len(etiquetados)} preguntas etiquetadas -> {ruta_salida.name}")

    log(f"\nTotal etiquetado: {total} preguntas en {time.time()-t0:.1f}s")

    with open(LOGS / "etiquetado.log", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    return 0


if __name__ == "__main__":
    sys.exit(main())
