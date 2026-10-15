# -*- coding: utf-8 -*-
"""
Fase 1 - Preparacion del corpus.

1. Carga dev_es.json (1034) y dev.json (1034).
2. Verifica para cada indice i: db_id y query identicos. Aborta si hay
   una sola discrepancia.
3. Construye el registro base por pregunta (qid, idx, db_id, question_es,
   gold_sql, dificultad, gold_has_orderby).
4. Guarda data/corpus_es.jsonl.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import hardness as hd

import os
RAIZ = Path(__file__).resolve().parents[1]  # raiz del repositorio
EXTERNO = Path(os.environ.get("TESIS_EXTERNO", RAIZ / "externo"))  # Spider, MultiSpider y modelo (descargados aparte)
BASE = RAIZ
SPIDER = EXTERNO / "spider_data"
MULTISPIDER_ES = EXTERNO / "multispider" / "dataset" / "multispider" / "with_english_value"
DATA = BASE / "data"
LOGS = BASE / "logs"
DATA.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)


def cargar(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def main():
    lineas = []

    def log(s=""):
        print(s)
        lineas.append(s)

    log("=" * 70)
    log("Fase 1 - Preparacion del corpus ES")
    log("=" * 70)

    dev_es = cargar(MULTISPIDER_ES / "dev_es.json")
    dev_en = cargar(SPIDER / "dev.json")

    log(f"dev_es.json: {len(dev_es)} registros")
    log(f"dev.json:    {len(dev_en)} registros")

    if len(dev_es) != len(dev_en):
        log("ABORTAR: longitudes distintas entre dev_es.json y dev.json.")
        return 1

    discrepancias = []
    for i, (es, en) in enumerate(zip(dev_es, dev_en)):
        if es["db_id"] != en["db_id"] or es["query"] != en["query"]:
            discrepancias.append(i)

    log(f"\nDiscrepancias de alineacion (db_id o query distintos): {len(discrepancias)}")
    if discrepancias:
        log(f"Indices con discrepancia (primeros 20): {discrepancias[:20]}")
        log("ABORTAR: se exige 0 discrepancias.")
        with open(LOGS / "verificacion_corpus.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lineas))
        return 1

    log("Alineacion verificada: 0 discrepancias (mismo db_id y mismo query en cada indice).")

    registros = []
    for i, (es, en) in enumerate(zip(dev_es, dev_en)):
        sql_ast = en["sql"]
        registros.append({
            "qid": f"es_{i:04d}",
            "idx": i,
            "db_id": en["db_id"],
            "question_es": es["question"],
            "gold_sql": en["query"],
            "dificultad": hd.eval_hardness(sql_ast),
            "gold_has_orderby": hd.has_orderby(sql_ast),
        })

    log(f"\nRegistros construidos: {len(registros)}")

    out_path = DATA / "corpus_es.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"Guardado: {out_path}")

    log("\n" + "=" * 70)
    log("Distribucion de dificultad (esperada: 24.0% / 43.1% / 16.8% / 16.1%)")
    log("=" * 70)
    orden = ["easy", "medium", "hard", "extra hard"]
    c = Counter(r["dificultad"] for r in registros)
    total = sum(c.values())
    for d in orden:
        pct = 100 * c.get(d, 0) / total
        log(f"  {d:12s}: {c.get(d, 0):4d} ({pct:.1f}%)")

    n_db = len(set(r["db_id"] for r in registros))
    log(f"\nBases distintas (db_id): {n_db} (esperado: 20)")

    n_orderby = sum(1 for r in registros if r["gold_has_orderby"])
    log(f"Con ORDER BY en el gold: {n_orderby} ({100*n_orderby/total:.1f}%) (esperado ~22.3%)")

    with open(LOGS / "verificacion_corpus.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))

    log(f"\nOK: n = {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
