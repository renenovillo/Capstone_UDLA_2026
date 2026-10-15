# -*- coding: utf-8 -*-
"""
Fase 2.5 - Consolidacion y verificacion.

Consolida los lotes data/extraccion/{acts,meta_etq}_{inicio}_{fin}.{npy,jsonl}
en:
  data/acts_es.npy       float16, [1034, 37, 2048]
  data/meta_es.parquet

Y ejecuta las 8 verificaciones obligatorias de la seccion 6.8, registrando
todo en logs/verificacion_extraccion.md.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import os
RAIZ = Path(__file__).resolve().parents[1]  # raiz del repositorio
EXTERNO = Path(os.environ.get("TESIS_EXTERNO", RAIZ / "externo"))  # Spider, MultiSpider y modelo (descargados aparte)
BASE = RAIZ
DIR_EXTRACCION = BASE / "data" / "extraccion"
DATA = BASE / "data"
LOGS = BASE / "logs"

PATRON_ACTS = re.compile(r"^acts_(\d{4})_(\d{4})\.npy$")
PATRON_META_ETQ = re.compile(r"^meta_etq_(\d{4})_(\d{4})\.jsonl$")

N_ESPERADO = 1034
N_CAPAS = 37
DIM = 2048


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

    log("=" * 70)
    log("Fase 2.5 - Consolidacion y verificacion")
    log("=" * 70)

    lotes_acts = sorted(
        (m.group(1), m.group(2)) for f in DIR_EXTRACCION.iterdir()
        if (m := PATRON_ACTS.match(f.name))
    )
    lotes_meta = sorted(
        (m.group(1), m.group(2)) for f in DIR_EXTRACCION.iterdir()
        if (m := PATRON_META_ETQ.match(f.name))
    )

    if lotes_acts != lotes_meta:
        log(f"ERROR: los lotes de acts ({len(lotes_acts)}) y meta_etq ({len(lotes_meta)}) no coinciden.")
        log(f"  acts: {lotes_acts}")
        log(f"  meta_etq: {lotes_meta}")
        return 1

    log(f"Lotes encontrados: {len(lotes_acts)}")

    todas_acts = []
    todos_meta = []
    for inicio_s, fin_s in lotes_acts:
        acts = np.load(DIR_EXTRACCION / f"acts_{inicio_s}_{fin_s}.npy")
        meta = cargar_jsonl(DIR_EXTRACCION / f"meta_etq_{inicio_s}_{fin_s}.jsonl")
        if acts.shape[0] != len(meta):
            log(f"ERROR: lote [{inicio_s}:{fin_s}] acts.shape[0]={acts.shape[0]} != len(meta)={len(meta)}")
            return 1
        todas_acts.append(acts)
        todos_meta.extend(meta)

    acts_es = np.concatenate(todas_acts, axis=0)
    meta_es = pd.DataFrame(todos_meta).sort_values("idx").reset_index(drop=True)
    # reordenar acts_es igual que meta_es tras el sort
    orden = pd.DataFrame(todos_meta)["idx"].argsort().values
    acts_es = acts_es[orden]

    log("\n" + "-" * 70)
    log("VERIFICACION 1: dimensiones y NaN/Inf")
    log("-" * 70)
    ok1 = True
    log(f"meta_es filas: {len(meta_es)} (esperado {N_ESPERADO})")
    ok1 &= len(meta_es) == N_ESPERADO
    log(f"acts_es.shape: {acts_es.shape} (esperado ({N_ESPERADO}, {N_CAPAS}, {DIM}))")
    ok1 &= acts_es.shape == (N_ESPERADO, N_CAPAS, DIM)
    tiene_nan = np.isnan(acts_es.astype(np.float32)).any()
    tiene_inf = np.isinf(acts_es.astype(np.float32)).any()
    log(f"NaN en acts_es: {tiene_nan} | Inf en acts_es: {tiene_inf}")
    ok1 &= (not tiene_nan) and (not tiene_inf)
    log(f"RESULTADO 1: {'OK' if ok1 else 'FALLA'}")

    log("\n" + "-" * 70)
    log("VERIFICACION 2: gold_status y gold_empty")
    log("-" * 70)
    n_gold_error = (meta_es["gold_status"] == "error").sum()
    n_gold_empty = meta_es["gold_empty"].fillna(False).astype(bool).sum()
    log(f"gold_status == 'error': {n_gold_error}")
    log(f"gold_empty == True: {n_gold_empty}")

    meta_validas = meta_es[meta_es["gold_status"] != "error"].copy()
    n_validas = len(meta_validas)
    log(f"Preguntas validas (gold_status == 'ok'): {n_validas}")

    log("\n" + "-" * 70)
    log("VERIFICACION 3: distribucion de desenlace y tasa de acierto")
    log("-" * 70)
    dist = meta_validas["desenlace"].value_counts()
    log(str(dist))
    tasa_acierto = meta_validas["y"].mean()
    log(f"Tasa de acierto (y=1): {tasa_acierto*100:.1f}%")
    ok3 = 0.20 <= tasa_acierto <= 0.80
    log(f"RESULTADO 3 (criterio 20%-80%): {'OK' if ok3 else 'FALLA -- DETENERSE Y REPORTAR'}")

    log("\n" + "-" * 70)
    log("VERIFICACION 4: conteo de errores silenciosos")
    log("-" * 70)
    n_silencioso = (meta_validas["desenlace"] == "silencioso").sum()
    log(f"Errores silenciosos: {n_silencioso}")
    if n_silencioso < 100:
        log("AVISO: n < 100 -- el analisis (b) se ejecuta igual pero se marca de baja potencia.")

    log("\n" + "-" * 70)
    log("VERIFICACION 5: tasa de acierto por nivel de dificultad")
    log("-" * 70)
    tabla_dif = meta_validas.groupby("dificultad")["y"].agg(["mean", "count"])
    log(str(tabla_dif))

    log("\n" + "-" * 70)
    log("VERIFICACION 6: sanidad de capa 0 (varianza entre preguntas ~ 0)")
    log("-" * 70)
    var_capa0 = acts_es[:, 0, :].astype(np.float32).var(axis=0).mean()
    log(f"Varianza media entre preguntas en capa 0: {var_capa0:.8f} (esperado ~0)")
    ok6 = var_capa0 < 1e-4
    log(f"RESULTADO 6: {'OK' if ok6 else 'REVISAR -- posible error de indexacion'}")

    log("\n" + "-" * 70)
    log("VERIFICACION 7: varianza > 0 en capas >= 1")
    log("-" * 70)
    varianzas = [acts_es[:, l, :].astype(np.float32).var(axis=0).mean() for l in range(1, N_CAPAS)]
    ok7 = all(v > 0 for v in varianzas)
    log(f"Min varianza entre capas 1..36: {min(varianzas):.8f}")
    log(f"RESULTADO 7: {'OK' if ok7 else 'FALLA -- alguna capa con varianza 0'}")

    log("\n" + "-" * 70)
    log("VERIFICACION 8: revision manual de 10 casos (3 correctas, 4 silenciosas, 3 sintacticas)")
    log("-" * 70)
    log("PENDIENTE DE REVISION MANUAL -- ver casos listados abajo.")
    for desenlace, n in [("correcta", 3), ("silencioso", 4), ("sintactico", 3)]:
        sub = meta_validas[meta_validas["desenlace"] == desenlace].head(n)
        for _, r in sub.iterrows():
            log(f"  [{desenlace}] qid={r['qid']} db_id={r['db_id']}")
            log(f"    pregunta: {r['question_es']}")
            log(f"    gold: {r['gold_sql']}")
            log(f"    pred: {r['pred_sql']}")
            if pd.notna(r.get("sqlite_error")):
                log(f"    error: {r['sqlite_error']}")

    DATA.mkdir(parents=True, exist_ok=True)
    np.save(DATA / "acts_es.npy", acts_es)
    # .csv en vez de .parquet: pyarrow no esta instalado en el entorno de ejecucion;
    meta_es.to_csv(DATA / "meta_es.csv", index=False, encoding="utf-8")
    log(f"\nGuardado: {DATA / 'acts_es.npy'}")
    log(f"Guardado: {DATA / 'meta_es.csv'}")

    with open(LOGS / "verificacion_extraccion.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    log(f"\nLog completo: {LOGS / 'verificacion_extraccion.md'}")

    return 0 if (ok1 and ok3 and ok6 and ok7) else 1


if __name__ == "__main__":
    sys.exit(main())
