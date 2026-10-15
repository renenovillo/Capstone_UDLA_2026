# -*- coding: utf-8 -*-
"""
Fase 4.1 - Analisis (a): comparacion de modelos.

Usa las predicciones OOF agrupadas de resultados/oof_predicciones.csv.
Decide el MODELO OPERATIVO (usado en los analisis b-e) y lo registra en
logs/modelo_operativo.txt.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from utils_bootstrap import bootstrap_ci_diferencia, bootstrap_ci_metrica
from utils_calibracion import bins_fiabilidad, ece

import os
RAIZ = Path(__file__).resolve().parents[1]  # raiz del repositorio
EXTERNO = Path(os.environ.get("TESIS_EXTERNO", RAIZ / "externo"))  # Spider, MultiSpider y modelo (descargados aparte)
BASE = RAIZ
DATA = BASE / "data"
RESULTADOS = BASE / "resultados"
TABLAS = RESULTADOS / "tablas"
LOGS = BASE / "logs"
TABLAS.mkdir(parents=True, exist_ok=True)

MODELOS_PRINCIPALES = {
    "lr": ("p_raw_lr", "p_cal_lr"),
    "mlp": ("p_raw_mlp", "p_cal_mlp"),
    "gb": ("p_raw_gb", "p_cal_gb"),
}
REFERENCIAS = {"longitud": "p_len", "confianza": "p_conf"}


def main():
    lineas = []

    def log(s=""):
        print(s, flush=True)
        lineas.append(s)

    log("=" * 70)
    log("Fase 4.1 - Analisis (a): comparacion de modelos")
    log("=" * 70)

    oof = pd.read_csv(RESULTADOS / "oof_predicciones.csv", encoding="utf-8")
    meta = pd.read_csv(DATA / "meta_es.csv", encoding="utf-8")
    oof = oof.merge(meta[["idx", "gold_empty"]], on="idx", how="left")
    y = oof["y"].values
    log(f"n = {len(oof)} (OOF agrupadas de los 5 pliegues externos)")

    filas_tabla = []

    # ---- AUROC de los 3 modelos (p_cal) y las 2 referencias ----
    log("\n" + "-" * 70)
    log("AUROC (con p_cal para LR/MLP/GB; p_raw para referencias) + IC 95%")
    log("-" * 70)
    auroc_resultados = {}
    for nombre, (p_raw_col, p_cal_col) in MODELOS_PRINCIPALES.items():
        punto, lo, hi = bootstrap_ci_metrica(y, oof[p_cal_col].values, roc_auc_score)
        auroc_resultados[nombre] = punto
        log(f"  {nombre.upper():10s}: AUROC = {punto:.4f}  IC95% [{lo:.4f}, {hi:.4f}]")
        filas_tabla.append({"metrica": "AUROC", "modelo": nombre, "punto": punto,
                             "ic_lo": lo, "ic_hi": hi})
    for nombre, col in REFERENCIAS.items():
        punto, lo, hi = bootstrap_ci_metrica(y, oof[col].values, roc_auc_score)
        auroc_resultados[nombre] = punto
        log(f"  {nombre.upper():10s}: AUROC = {punto:.4f}  IC95% [{lo:.4f}, {hi:.4f}]")
        filas_tabla.append({"metrica": "AUROC", "modelo": nombre, "punto": punto,
                             "ic_lo": lo, "ic_hi": hi})

    # ---- Brier (con p_cal) de los 3 modelos + prevalencia de referencia ----
    log("\n" + "-" * 70)
    log("Brier (con p_cal) + IC 95%; referencia: Brier de la prevalencia")
    log("-" * 70)
    for nombre, (p_raw_col, p_cal_col) in MODELOS_PRINCIPALES.items():
        punto, lo, hi = bootstrap_ci_metrica(y, oof[p_cal_col].values, brier_score_loss)
        log(f"  {nombre.upper():10s}: Brier = {punto:.4f}  IC95% [{lo:.4f}, {hi:.4f}]")
        filas_tabla.append({"metrica": "Brier", "modelo": nombre, "punto": punto,
                             "ic_lo": lo, "ic_hi": hi})
    ybar = y.mean()
    brier_prevalencia = np.mean((ybar - y) ** 2)
    log(f"  PREVALENCIA: Brier = {brier_prevalencia:.4f} (referencia, sin IC)")
    filas_tabla.append({"metrica": "Brier", "modelo": "prevalencia", "punto": brier_prevalencia,
                         "ic_lo": None, "ic_hi": None})

    # ---- Diferencias pareadas ----
    log("\n" + "-" * 70)
    log("Diferencias pareadas (bootstrap pareado, mismos indices)")
    log("-" * 70)

    d_lr_len, lo, hi = bootstrap_ci_diferencia(y, oof["p_cal_lr"].values, oof["p_len"].values)
    exito_a = d_lr_len >= 0.10
    log(f"  LR - longitud: {d_lr_len:+.4f}  IC95% [{lo:+.4f}, {hi:+.4f}]  "
        f"(criterio >= 0.10: {'CUMPLE' if exito_a else 'NO CUMPLE'}; "
        f"limite inferior > 0: {'si' if lo > 0 else 'no'})")
    filas_tabla.append({"metrica": "diff_AUROC", "modelo": "lr_menos_longitud", "punto": d_lr_len,
                         "ic_lo": lo, "ic_hi": hi})

    d_mlp_lr, lo_m, hi_m = bootstrap_ci_diferencia(y, oof["p_cal_mlp"].values, oof["p_cal_lr"].values)
    mlp_supera = lo_m > 0
    log(f"  MLP - LR: {d_mlp_lr:+.4f}  IC95% [{lo_m:+.4f}, {hi_m:+.4f}]  "
        f"({'MLP supera a LR (IC > 0)' if mlp_supera else 'IC incluye 0 -> senal esencialmente lineal'})")
    filas_tabla.append({"metrica": "diff_AUROC", "modelo": "mlp_menos_lr", "punto": d_mlp_lr,
                         "ic_lo": lo_m, "ic_hi": hi_m})

    d_gb_lr, lo_g, hi_g = bootstrap_ci_diferencia(y, oof["p_cal_gb"].values, oof["p_cal_lr"].values)
    gb_supera = lo_g > 0
    log(f"  GB - LR: {d_gb_lr:+.4f}  IC95% [{lo_g:+.4f}, {hi_g:+.4f}]  "
        f"({'GB supera a LR (IC > 0)' if gb_supera else 'IC incluye 0 -> senal esencialmente lineal'})")
    filas_tabla.append({"metrica": "diff_AUROC", "modelo": "gb_menos_lr", "punto": d_gb_lr,
                         "ic_lo": lo_g, "ic_hi": hi_g})

    # ---- Modelo operativo ----
    candidatos = [("lr", auroc_resultados["lr"])]
    if mlp_supera:
        candidatos.append(("mlp", auroc_resultados["mlp"]))
    if gb_supera:
        candidatos.append(("gb", auroc_resultados["gb"]))
    modelo_operativo = max(candidatos, key=lambda t: t[1])[0]
    log(f"\nMODELO OPERATIVO: {modelo_operativo.upper()} "
        f"({'LR por defecto (parsimonia)' if modelo_operativo == 'lr' else 'supera a LR segun regla 7.7/8.1'})")

    p_cal_operativo = oof[f"p_cal_{modelo_operativo}"].values

    d_mejor_conf, lo_c, hi_c = bootstrap_ci_diferencia(y, p_cal_operativo, oof["p_conf"].values)
    log(f"  {modelo_operativo.upper()} - confianza: {d_mejor_conf:+.4f}  "
        f"IC95% [{lo_c:+.4f}, {hi_c:+.4f}] (comparacion, sin umbral)")
    filas_tabla.append({"metrica": "diff_AUROC", "modelo": f"{modelo_operativo}_menos_confianza",
                         "punto": d_mejor_conf, "ic_lo": lo_c, "ic_hi": hi_c})

    # ---- Sensibilidad: excluir gold_empty ----
    log("\n" + "-" * 70)
    log("Sensibilidad: AUROC del modelo operativo excluyendo gold_empty==True")
    log("-" * 70)
    sin_vacios = oof[oof["gold_empty"] == False]
    auroc_sin_vacios = roc_auc_score(sin_vacios["y"].values, sin_vacios[f"p_cal_{modelo_operativo}"].values)
    log(f"  AUROC {modelo_operativo.upper()} (n={len(oof)}, con gold_empty): {auroc_resultados[modelo_operativo]:.4f}")
    log(f"  AUROC {modelo_operativo.upper()} (n={len(sin_vacios)}, sin gold_empty): {auroc_sin_vacios:.4f}")
    filas_tabla.append({"metrica": "AUROC_sensibilidad_sin_gold_empty", "modelo": modelo_operativo,
                         "punto": auroc_sin_vacios, "ic_lo": None, "ic_hi": None})

    # ---- Desagregacion por dificultad ----
    log("\n" + "-" * 70)
    log(f"Desagregacion (descriptiva): AUROC de {modelo_operativo.upper()} por dificultad")
    log("-" * 70)
    for dif, grupo in oof.groupby("dificultad"):
        n_pos = (grupo["y"] == 1).sum()
        n_neg = (grupo["y"] == 0).sum()
        if n_pos >= 20 and n_neg >= 20:
            auc_dif = roc_auc_score(grupo["y"].values, grupo[f"p_cal_{modelo_operativo}"].values)
            log(f"  {dif:12s} (n_pos={n_pos}, n_neg={n_neg}): AUROC = {auc_dif:.4f}")
            filas_tabla.append({"metrica": "AUROC_por_dificultad", "modelo": dif,
                                 "punto": auc_dif, "ic_lo": None, "ic_hi": None})
        else:
            log(f"  {dif:12s} (n_pos={n_pos}, n_neg={n_neg}): no estimable (< 20 casos de alguna clase)")
            filas_tabla.append({"metrica": "AUROC_por_dificultad", "modelo": dif,
                                 "punto": None, "ic_lo": None, "ic_hi": None})

    # =====================================================================
    # ---- Analisis adicionales ----
    # =====================================================================

    # ---- Referencia descriptiva de dificultad: nunca como predictor ----
    log("\n" + "-" * 70)
    log("R4.1 Referencia de dificultad (descriptiva; easy=3, medium=2, hard=1, extra hard=0)")
    log("-" * 70)
    puntaje_dif = oof["dificultad"].map({"easy": 3, "medium": 2, "hard": 1, "extra hard": 0})
    if puntaje_dif.isna().any():
        raise ValueError("Niveles de dificultad no reconocidos: "
                         f"{sorted(oof.loc[puntaje_dif.isna(), 'dificultad'].unique())}")
    puntaje_dif = puntaje_dif.astype(float).values

    punto, lo, hi = bootstrap_ci_metrica(y, puntaje_dif, roc_auc_score)
    log(f"  AUROC dificultad (ordinal): {punto:.4f}  IC95% [{lo:.4f}, {hi:.4f}]")
    filas_tabla.append({"metrica": "AUROC_dificultad_ordinal", "modelo": "dificultad",
                         "punto": punto, "ic_lo": lo, "ic_hi": hi})

    d, lo, hi = bootstrap_ci_diferencia(y, oof["p_cal_lr"].values, puntaje_dif)
    log(f"  LR - dificultad: {d:+.4f}  IC95% [{lo:+.4f}, {hi:+.4f}] (bootstrap pareado)")
    filas_tabla.append({"metrica": "diff_AUROC", "modelo": "lr_menos_dificultad",
                         "punto": d, "ic_lo": lo, "ic_hi": hi})

    def auroc_intra_nivel(score):
        """Promedio de los AUROC por nivel ponderado por n_pos*n_neg (= AUROC sobre pares de igual dificultad)."""
        num, den = 0.0, 0.0
        for _, g in oof.assign(_s=score).groupby("dificultad"):
            n_pos, n_neg = int((g["y"] == 1).sum()), int((g["y"] == 0).sum())
            if n_pos == 0 or n_neg == 0:
                continue
            w = n_pos * n_neg
            num += w * roc_auc_score(g["y"].values, g["_s"].values)
            den += w
        return num / den

    for nombre, col in [("lr", "p_cal_lr"), ("confianza", "p_conf")]:
        v = auroc_intra_nivel(oof[col].values)
        log(f"  AUROC intra-nivel de dificultad, {nombre} ({col}): {v:.4f} (descriptivo, sin IC)")
        filas_tabla.append({"metrica": "AUROC_intra_nivel_dificultad", "modelo": nombre,
                             "punto": v, "ic_lo": None, "ic_hi": None})

    # ---- Comparaciones para el modelo operativo (GB en v2) ----
    # GB - longitud es el criterio (a) aplicado al modelo operativo. GB - dificultad y
    # LR - confianza son complementos no preespecificados: se formulan para LR.
    log("\n" + "-" * 70)
    log("Comparaciones del modelo operativo (GB - longitud: criterio a; el resto, complementos)")
    log("-" * 70)
    for etiqueta, a, b, nombre_b in [
        ("gb_menos_longitud", "p_cal_gb", "p_len", "longitud"),
        ("gb_menos_dificultad", "p_cal_gb", None, "dificultad (ordinal)"),
        ("lr_menos_confianza", "p_cal_lr", "p_conf", "confianza"),
    ]:
        score_b = puntaje_dif if b is None else oof[b].values
        d, lo, hi = bootstrap_ci_diferencia(y, oof[a].values, score_b)
        log(f"  {etiqueta}: {d:+.4f}  IC95% [{lo:+.4f}, {hi:+.4f}] (bootstrap pareado)")
        filas_tabla.append({"metrica": "diff_AUROC" if etiqueta == "gb_menos_longitud" else "diff_AUROC_complemento",
                             "modelo": etiqueta, "punto": d, "ic_lo": lo, "ic_hi": hi})
    v = auroc_intra_nivel(oof["p_cal_gb"].values)
    log(f"  AUROC intra-nivel de dificultad, gb (p_cal_gb): {v:.4f} (descriptivo, sin IC)")
    filas_tabla.append({"metrica": "AUROC_intra_nivel_dificultad", "modelo": "gb",
                         "punto": v, "ic_lo": None, "ic_hi": None})

    # ---- Calibracion: ECE + diagrama de fiabilidad ----
    log("\n" + "-" * 70)
    log("R4.2 Calibracion de p_cal_lr: ECE (10 intervalos de igual anchura) y fiabilidad")
    log("-" * 70)
    bins = bins_fiabilidad(y, oof["p_cal_lr"].values)
    ece_lr = ece(y, oof["p_cal_lr"].values)
    log(bins.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log(f"  ECE (p_cal_lr) = {ece_lr:.4f}")
    filas_tabla.append({"metrica": "ECE_10_intervalos", "modelo": "lr",
                         "punto": ece_lr, "ic_lo": None, "ic_hi": None})
    bins.to_csv(TABLAS / "tabla_a_fiabilidad_bins.csv", index=False, encoding="utf-8")
    for nombre, col in [("mlp", "p_cal_mlp"), ("gb", "p_cal_gb")]:   # complemento: calibracion de los otros modelos
        e = ece(y, oof[col].values)
        log(f"  ECE ({col}) = {e:.4f}")
        filas_tabla.append({"metrica": "ECE_10_intervalos", "modelo": nombre,
                             "punto": e, "ic_lo": None, "ic_hi": None})

    spec = importlib.util.spec_from_file_location("figs12", Path(__file__).parent / "12_figuras_y_tablas.py")
    figs12 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figs12)
    figs12.fig_a_fiabilidad()

    # ---- R4.3 Coherencia de AUROC agrupado vs por pliegue ----
    log("\n" + "-" * 70)
    log("R4.3 Coherencia: AUROC agrupado vs. media por pliegue (p_raw_lr)")
    log("-" * 70)
    auroc_agrupado_raw = roc_auc_score(y, oof["p_raw_lr"].values)
    por_pliegue = {f: roc_auc_score(g["y"].values, g["p_raw_lr"].values) for f, g in oof.groupby("fold")}
    media_pliegues = float(np.mean(list(por_pliegue.values())))
    log(f"  AUROC agrupado con p_raw_lr: {auroc_agrupado_raw:.4f}")
    log(f"  AUROC agrupado con p_cal_lr: {auroc_resultados['lr']:.4f}  (el de la tabla principal)")
    log(f"  Media de AUROC por pliegue (p_raw_lr): {media_pliegues:.4f}  "
        f"(por pliegue: " + ", ".join(f"{f}={v:.3f}" for f, v in por_pliegue.items()) + ")")
    log("  Nota: dentro de cada pliegue Platt es monotono, asi que el AUROC por pliegue es el mismo con p_raw y p_cal;"
        " solo el AUROC agrupado difiere, porque Platt reordena las puntuaciones entre pliegues.")
    filas_tabla.append({"metrica": "AUROC_agrupado_p_raw", "modelo": "lr",
                         "punto": auroc_agrupado_raw, "ic_lo": None, "ic_hi": None})
    filas_tabla.append({"metrica": "AUROC_media_por_pliegue_p_raw", "modelo": "lr",
                         "punto": media_pliegues, "ic_lo": None, "ic_hi": None})

    df_tabla = pd.DataFrame(filas_tabla)
    df_tabla.to_csv(TABLAS / "tabla_a_modelos.csv", index=False, encoding="utf-8")
    log(f"\nGuardado: {TABLAS / 'tabla_a_modelos.csv'}")

    with open(LOGS / "modelo_operativo.txt", "w", encoding="utf-8") as f:
        f.write(f"{modelo_operativo}\n")
    log(f"Guardado: {LOGS / 'modelo_operativo.txt'} -> '{modelo_operativo}'")

    with open(LOGS / "analisis_a.log", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))

    return 0


if __name__ == "__main__":
    sys.exit(main())
