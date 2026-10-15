# -*- coding: utf-8 -*-
"""
Fase 2 - Extraccion de activaciones y generacion de SQL.

Recorre las 1034 preguntas de data/corpus_es.jsonl con la plantilla de prompt
y la serializacion de esquema definidas en este script, con guardado
incremental por lotes de 50 y reanudacion.

Uso:
  python 02_extraer_y_generar.py             # corrida completa (con reanudacion)
  python 02_extraer_y_generar.py --prueba 10  # prueba corta obligatoria (10 preguntas)
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
RAIZ = Path(__file__).resolve().parents[1]  # raiz del repositorio
EXTERNO = Path(os.environ.get("TESIS_EXTERNO", RAIZ / "externo"))  # Spider, MultiSpider y modelo (descargados aparte)
BASE = RAIZ
RUTA_MODELO = EXTERNO / "qwen2.5-coder-3b"
RUTA_TABLES_ES = (EXTERNO / "multispider" / "dataset" / "multispider"
                  / "with_english_value" / "tables_es.json")
RUTA_CORPUS = BASE / "data" / "corpus_es.jsonl"
DIR_EXTRACCION = BASE / "data" / "extraccion"
LOGS = BASE / "logs"
DIR_EXTRACCION.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

MAX_NEW_TOKENS = 200  # maximo de tokens generados por consulta
TAMANO_LOTE = 50
N_CAPAS = 37
DIM = 2048

# ---- Plantilla de sistema/usuario ----
SISTEMA_ES = (
    "Eres un asistente que traduce preguntas en lenguaje natural a "
    "consultas SQL válidas para SQLite. Responde ÚNICAMENTE con la "
    "consulta SQL, sin explicaciones ni comentarios."
)


def cargar_json(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def cargar_jsonl(ruta):
    out = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            out.append(json.loads(linea))
    return out


def indexar_esquemas(lista_tables):
    return {t["db_id"]: t for t in lista_tables}


def construir_esquema_texto(schema):
    """Serializa el esquema de la base como sentencias CREATE TABLE."""
    tablas = schema["table_names_original"]
    columnas = schema["column_names_original"]
    tipos = schema["column_types"]
    pks = set(schema["primary_keys"])

    lineas = []
    for i, nombre_tabla in enumerate(tablas):
        cols_de_tabla = []
        for col_idx, (idx_tabla, nombre_col) in enumerate(columnas):
            if idx_tabla == i:
                marca_pk = " PRIMARY KEY" if col_idx in pks else ""
                cols_de_tabla.append(f"  {nombre_col} {tipos[col_idx]}{marca_pk}")
        lineas.append(f"CREATE TABLE {nombre_tabla} (\n" + ",\n".join(cols_de_tabla) + "\n)")

    fks_texto = []
    for col_a, col_b in schema["foreign_keys"]:
        nombre_a = columnas[col_a][1]
        nombre_b = columnas[col_b][1]
        fks_texto.append(f"-- FK: {nombre_a} -> {nombre_b}")

    return "\n\n".join(lineas) + ("\n\n" + "\n".join(fks_texto) if fks_texto else "")


def construir_prompt(pregunta, esquema_texto):
    """Construye los mensajes de sistema y usuario."""
    usuario = f"Esquema de la base de datos:\n{esquema_texto}\n\nPregunta: {pregunta}\n\nSQL:"
    return [
        {"role": "system", "content": SISTEMA_ES},
        {"role": "user", "content": usuario},
    ]


def limpiar_sql_generado(texto):
    """Quita los bloques de codigo markdown de la salida generada."""
    texto = texto.strip()
    if texto.startswith("```"):
        lineas = texto.split("\n")
        lineas = [l for l in lineas if not l.strip().startswith("```")]
        texto = "\n".join(lineas).strip()
    return texto


def primera_sentencia(texto):
    """Extraccion del SQL: primera sentencia hasta el primer ';' si existe."""
    if ";" in texto:
        return texto.split(";")[0].strip()
    return texto.strip()


def guardar_plantilla_y_hash(tokenizer, esquema_texto_ejemplo, pregunta_ejemplo):
    mensajes = construir_prompt(pregunta_ejemplo, esquema_texto_ejemplo)
    texto_plantilla = tokenizer.apply_chat_template(
        mensajes, add_generation_prompt=True, tokenize=False,
    )
    hash_sha256 = hashlib.sha256(texto_plantilla.encode("utf-8")).hexdigest()
    with open(LOGS / "prompt_template.txt", "w", encoding="utf-8") as f:
        f.write("SISTEMA:\n" + SISTEMA_ES + "\n\n")
        f.write("PLANTILLA DE MENSAJE USUARIO:\n")
        f.write("Esquema de la base de datos:\n{esquema_texto}\n\nPregunta: {pregunta}\n\nSQL:\n\n")
        f.write("EJEMPLO RENDERIZADO CON apply_chat_template:\n")
        f.write("-" * 70 + "\n")
        f.write(texto_plantilla)
        f.write("\n" + "-" * 70 + "\n")
        f.write(f"\nSHA-256 del ejemplo renderizado: {hash_sha256}\n")
    return hash_sha256


def lote_completo(inicio, fin):
    p_npy = DIR_EXTRACCION / f"acts_{inicio:04d}_{fin:04d}.npy"
    p_jsonl = DIR_EXTRACCION / f"meta_{inicio:04d}_{fin:04d}.jsonl"
    if not (p_npy.exists() and p_jsonl.exists()):
        return False
    try:
        acts = np.load(p_npy)
        n_meta = sum(1 for _ in open(p_jsonl, encoding="utf-8"))
        return acts.shape[0] == n_meta == (fin - inicio)
    except Exception:
        return False


def procesar_lote(model, tokenizer, esquemas, registros_lote, log_fn):
    acts_lote = np.zeros((len(registros_lote), N_CAPAS, DIM), dtype=np.float16)
    metas = []

    for j, reg in enumerate(registros_lote):
        t0 = time.time()
        esquema = esquemas.get(reg["db_id"])
        if esquema is None:
            raise RuntimeError(f"Sin esquema para db_id={reg['db_id']} (qid={reg['qid']})")

        esquema_texto = construir_esquema_texto(esquema)
        mensajes = construir_prompt(reg["question_es"], esquema_texto)

        inputs = tokenizer.apply_chat_template(
            mensajes, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to("cuda")
        prompt_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out_fwd = model(**inputs, output_hidden_states=True)
            capas = [h[0, -1, :].to(torch.float16).cpu().numpy() for h in out_fwd.hidden_states]
            del out_fwd
        acts_lote[j] = np.stack(capas, axis=0)
        del capas
        torch.cuda.empty_cache()

        with torch.no_grad():
            t_gen0 = time.time()
            salida = model.generate(
                **inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True, output_scores=True,
            )
            t_gen_s = time.time() - t_gen0

        secuencia = salida.sequences
        n_gen_tokens = secuencia.shape[1] - prompt_tokens

        try:
            puntajes_transicion = model.compute_transition_scores(
                secuencia, salida.scores, normalize_logits=True,
            )
            logprobs = puntajes_transicion[0]
            logprobs = logprobs[torch.isfinite(logprobs)]
            conf_logprob_mean = float(logprobs.mean().item()) if logprobs.numel() > 0 else float("nan")
        except Exception as e:
            conf_logprob_mean = float("nan")
            log_fn(f"  AVISO: fallo compute_transition_scores en {reg['qid']}: {e}")

        raw_output = tokenizer.decode(secuencia[0][prompt_tokens:], skip_special_tokens=True)
        sql_limpio = limpiar_sql_generado(raw_output)
        pred_sql = primera_sentencia(sql_limpio)

        del inputs, salida, secuencia
        torch.cuda.empty_cache()

        t_total = time.time() - t0
        metas.append({
            "qid": reg["qid"], "idx": reg["idx"], "db_id": reg["db_id"],
            "question_es": reg["question_es"], "gold_sql": reg["gold_sql"],
            "pred_sql": pred_sql, "raw_output": raw_output,
            "dificultad": reg["dificultad"], "gold_has_orderby": reg["gold_has_orderby"],
            "prompt_tokens": prompt_tokens, "n_gen_tokens": n_gen_tokens,
            "conf_logprob_mean": conf_logprob_mean,
            "t_gen_s": t_gen_s, "t_total_s": t_total,
        })
        vram = torch.cuda.memory_allocated() / 1e9
        log_fn(f"  [{reg['qid']}] {reg['db_id']} - {t_total:.2f}s (gen {t_gen_s:.2f}s) - "
               f"VRAM {vram:.2f} GB - prompt_tok={prompt_tokens} gen_tok={n_gen_tokens}")

    return acts_lote, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prueba", type=int, default=None,
                     help="Si se indica, corre solo las primeras N preguntas (prueba corta).")
    args = ap.parse_args()

    log_lines = []

    def log(s=""):
        print(s, flush=True)
        log_lines.append(s)

    def guardar_log():
        nombre = "extraccion_prueba.log" if args.prueba else "extraccion.log"
        with open(LOGS / nombre, "a", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
        log_lines.clear()

    log("=" * 70)
    log(f"Fase 2 - Extraccion y generacion - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Modo: {'PRUEBA CORTA n=' + str(args.prueba) if args.prueba else 'CORRIDA COMPLETA'}")
    log("=" * 70)

    registros = cargar_jsonl(RUTA_CORPUS)
    if args.prueba:
        registros = registros[:args.prueba]
    n_total = len(registros)
    log(f"Preguntas a procesar: {n_total}")

    esquemas = indexar_esquemas(cargar_json(RUTA_TABLES_ES))

    log("\nCargando tokenizer y modelo...")
    tokenizer = AutoTokenizer.from_pretrained(RUTA_MODELO)
    model = AutoModelForCausalLM.from_pretrained(
        RUTA_MODELO, dtype=torch.float16, device_map="cuda",
    )
    model.eval()
    vram_carga = torch.cuda.memory_allocated() / 1e9
    log(f"VRAM tras cargar el modelo: {vram_carga:.2f} GB")

    hash_plantilla = guardar_plantilla_y_hash(
        tokenizer, construir_esquema_texto(esquemas[registros[0]["db_id"]]),
        registros[0]["question_es"],
    )
    log(f"Plantilla guardada en logs/prompt_template.txt (SHA-256: {hash_plantilla})")

    t_inicio_total = time.time()
    tiempos_por_pregunta = []

    lotes = [(i, min(i + TAMANO_LOTE, n_total)) for i in range(0, n_total, TAMANO_LOTE)]
    for inicio, fin in lotes:
        if not args.prueba and lote_completo(inicio, fin):
            log(f"\nLote [{inicio:04d}:{fin:04d}] ya completo -- se salta (reanudacion).")
            continue

        log(f"\nLote [{inicio:04d}:{fin:04d}] ({fin - inicio} preguntas)...")
        t_lote0 = time.time()
        acts_lote, metas = procesar_lote(model, tokenizer, esquemas, registros[inicio:fin], log)
        t_lote = time.time() - t_lote0
        tiempos_por_pregunta.append(t_lote / (fin - inicio))

        prefijo = "prueba_" if args.prueba else ""
        p_npy = DIR_EXTRACCION / f"{prefijo}acts_{inicio:04d}_{fin:04d}.npy"
        p_jsonl = DIR_EXTRACCION / f"{prefijo}meta_{inicio:04d}_{fin:04d}.jsonl"
        np.save(p_npy, acts_lote)
        with open(p_jsonl, "w", encoding="utf-8") as f:
            for m in metas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        tiempo_medio = sum(tiempos_por_pregunta) / len(tiempos_por_pregunta)
        restantes = n_total - fin if not args.prueba else 0
        eta_s = tiempo_medio * restantes
        log(f"Lote guardado: {p_npy.name}, {p_jsonl.name} "
            f"(tiempo/pregunta lote={t_lote/(fin-inicio):.2f}s, "
            f"tiempo/pregunta medio={tiempo_medio:.2f}s, "
            f"ETA restante={eta_s/60:.1f} min)")
        guardar_log()

    t_total = time.time() - t_inicio_total
    log(f"\nTiempo total: {t_total/60:.2f} min para {n_total} preguntas "
        f"({t_total/n_total:.2f} s/pregunta)")
    if args.prueba:
        estim_1034 = (t_total / n_total) * 1034 / 60
        log(f"ESTIMACION para 1034 preguntas: {estim_1034:.1f} min (~{estim_1034/60:.2f} h)")
    guardar_log()
    return 0


if __name__ == "__main__":
    sys.exit(main())
