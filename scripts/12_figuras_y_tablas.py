# -*- coding: utf-8 -*-
"""
Fase 5.1 - Figuras y tablas finales.

Regenera las figuras de los analisis (a)-(d) con estilo uniforme: texto en
espanol, coma decimal, fuente legible, 300 dpi.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
from utils_calibracion import bins_fiabilidad, ece  # noqa: E402

import os
RAIZ = Path(__file__).resolve().parents[1]  # raiz del repositorio
EXTERNO = Path(os.environ.get("TESIS_EXTERNO", RAIZ / "externo"))  # Spider, MultiSpider y modelo (descargados aparte)
BASE = RAIZ
DATA = BASE / "data"
RESULTADOS = BASE / "resultados"
TABLAS = RESULTADOS / "tablas"
FIGURAS = RESULTADOS / "figuras"
LOGS = BASE / "logs"
FIGURAS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

COLORES = {"lr": "#1b6ca8", "mlp": "#e07b39", "gb": "#3e9b4f",
           "longitud": "#999999", "confianza": "#c44e52"}
NOMBRES = {"lr": "Regresión logística", "mlp": "MLP", "gb": "Gradient boosting",
           "longitud": "Longitud del prompt", "confianza": "Confianza del modelo"}


def coma(x, pos=None):
    return f"{x:.2f}".replace(".", ",")


def aplicar_coma(ax, eje="y"):
    formatter = mticker.FuncFormatter(coma)
    if eje in ("y", "ambos"):
        ax.yaxis.set_major_formatter(formatter)
    if eje in ("x", "ambos"):
        ax.xaxis.set_major_formatter(formatter)


def fig_a_roc():
    oof = pd.read_csv(RESULTADOS / "oof_predicciones.csv", encoding="utf-8")
    y = oof["y"].values
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for modelo, col in [("lr", "p_cal_lr"), ("mlp", "p_cal_mlp"), ("gb", "p_cal_gb"),
                        ("longitud", "p_len"), ("confianza", "p_conf")]:
        fpr, tpr, _ = roc_curve(y, oof[col].values)
        auc = roc_auc_score(y, oof[col].values)
        ax.plot(fpr, tpr, label=f"{NOMBRES[modelo]} (AUROC={auc:.3f})".replace(".", ","),
                color=COLORES[modelo], linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Azar (AUROC=0,500)")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC — modelos y referencias\n(OOF agrupadas, n=1 032)")
    ax.legend(loc="lower right", fontsize=9)
    aplicar_coma(ax, "ambos")
    fig.tight_layout()
    fig.savefig(FIGURAS / "fig_a_roc.png")
    plt.close(fig)
    print("Guardado: fig_a_roc.png")


def fig_a_fiabilidad():
    oof = pd.read_csv(RESULTADOS / "oof_predicciones.csv", encoding="utf-8")
    y = oof["y"].values

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Calibración perfecta")
    for modelo, col, desplaz in [("lr", "p_cal_lr", (6, -12)), ("gb", "p_cal_gb", (-34, 6))]:
        bins = bins_fiabilidad(y, oof[col].values)
        valor_ece = ece(y, oof[col].values)
        llena = bins[bins["n"] > 0]
        ax.plot(llena["p_media"], llena["frecuencia_observada"], marker="o", color=COLORES[modelo],
                linewidth=2, label=f"{NOMBRES[modelo]} (ECE={valor_ece:.3f})".replace(".", ","))
        for _, f in llena.iterrows():
            ax.annotate(f"n={int(f['n'])}", (f["p_media"], f["frecuencia_observada"]),
                        textcoords="offset points", xytext=desplaz, fontsize=7, color=COLORES[modelo])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Probabilidad predicha de respuesta correcta (media por intervalo)")
    ax.set_ylabel("Frecuencia observada de respuestas correctas")
    ax.set_title("Diagrama de fiabilidad (p calibrada con Platt)\n10 intervalos de igual anchura; n por intervalo anotado")
    ax.legend(loc="upper left", fontsize=9)
    aplicar_coma(ax, "ambos")
    fig.tight_layout()
    fig.savefig(FIGURAS / "fig_a_fiabilidad.png")
    plt.close(fig)
    print("Guardado: fig_a_fiabilidad.png")


def fig_a_perfil_capas():
    perfil = pd.read_csv(TABLAS / "perfil_capas.csv", encoding="utf-8")
    tabla_a = pd.read_csv(TABLAS / "tabla_a_modelos.csv", encoding="utf-8")
    sel = pd.read_csv(LOGS / "seleccion_por_pliegue.csv", encoding="utf-8")
    auroc_long = tabla_a[(tabla_a.metrica == "AUROC") & (tabla_a.modelo == "longitud")].punto.iloc[0]
    auroc_conf = tabla_a[(tabla_a.metrica == "AUROC") & (tabla_a.modelo == "confianza")].punto.iloc[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(perfil["capa"], perfil["auroc_oof"], color=COLORES["lr"], linewidth=2,
            marker="o", markersize=3, label="AUROC por capa (C fijo)")
    ax.axhline(auroc_long, color=COLORES["longitud"], linestyle="--", linewidth=1.5,
               label=f"Longitud (AUROC={auroc_long:.3f})".replace(".", ","))
    ax.axhline(auroc_conf, color=COLORES["confianza"], linestyle="--", linewidth=1.5,
               label=f"Confianza (AUROC={auroc_conf:.3f})".replace(".", ","))
    for capa in sorted(sel["capa_lr"].unique()):
        ax.axvline(capa, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Capa (índice, 0 = embedding)")
    ax.set_ylabel("AUROC (OOF, descriptivo)")
    ax.set_title("Perfil de AUROC por capa — líneas punteadas: capas elegidas por pliegue")
    ax.legend(loc="lower right", fontsize=9)
    aplicar_coma(ax, "y")
    fig.tight_layout()
    fig.savefig(FIGURAS / "fig_a_perfil_capas.png")
    plt.close(fig)
    print("Guardado: fig_a_perfil_capas.png")


def fig_c_aprendizaje():
    tabla = pd.read_csv(TABLAS / "tabla_c_aprendizaje.csv", encoding="utf-8")
    tabla_a = pd.read_csv(TABLAS / "tabla_a_modelos.csv", encoding="utf-8")
    auroc_long = tabla_a[(tabla_a.metrica == "AUROC") & (tabla_a.modelo == "longitud")].punto.iloc[0]

    folds = pd.read_csv(DATA / "folds.csv", encoding="utf-8")
    tam_train = len(folds) - folds["fold_externo"].value_counts()
    etiqueta_todo = f"Todo\n(n={tam_train.min()}-{tam_train.max()})"
    etiquetas = [str(n) if n != "todo" else etiqueta_todo for n in tabla["n"]]
    x = np.arange(len(tabla))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, tabla["auroc_medio"], marker="o", color=COLORES["lr"], linewidth=2,
            label="AUROC medio (20 réplicas)")
    ax.fill_between(x, tabla["p2_5"], tabla["p97_5"], color=COLORES["lr"], alpha=0.2,
                     label="Percentil [2,5 – 97,5]")
    ax.axhline(auroc_long, color=COLORES["longitud"], linestyle="--", linewidth=1.5,
               label=f"Longitud (AUROC={auroc_long:.3f})".replace(".", ","))
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas)
    ax.set_xlabel("Tamaño de entrenamiento (n de respuestas reportadas)")
    ax.set_ylabel("AUROC")
    ax.set_title("Curva de aprendizaje\nviabilidad del etiquetado por retroalimentación")
    ax.legend(loc="lower right", fontsize=9)
    aplicar_coma(ax, "y")
    fig.tight_layout()
    fig.savefig(FIGURAS / "fig_c_aprendizaje.png")
    plt.close(fig)
    print("Guardado: fig_c_aprendizaje.png")


def fig_d_ruido():
    uniforme = pd.read_csv(TABLAS / "tabla_d_ruido.csv", encoding="utf-8")
    silencioso = pd.read_csv(TABLAS / "tabla_d2_ruido_silencioso.csv", encoding="utf-8")
    tabla_a = pd.read_csv(TABLAS / "tabla_a_modelos.csv", encoding="utf-8")
    auroc_long = tabla_a[(tabla_a.metrica == "AUROC") & (tabla_a.modelo == "longitud")].punto.iloc[0]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for tabla, color, etiqueta in [
        (uniforme, "#3e9b4f", "Ruido uniforme (todos los errores)"),
        (silencioso, "#e07b39", "Ruido solo en errores silenciosos"),
    ]:
        x = tabla["rho"].values
        ax.plot(x, tabla["auroc_medio"], marker="o", color=color, linewidth=2, label=etiqueta)
        ax.fill_between(x, tabla["p2_5"], tabla["p97_5"], color=color, alpha=0.18)
    ax.axhline(auroc_long, color=COLORES["longitud"], linestyle="--", linewidth=1.5,
               label=f"Longitud (AUROC={auroc_long:.3f})".replace(".", ","))
    ax.axhline(auroc_long + 0.10, color=COLORES["longitud"], linestyle=":", linewidth=1.2,
               label="Longitud + 0,10 (criterio a)")
    ax.set_xlabel("Nivel de ruido ρ (fracción de los errores de entrenamiento reetiquetados como correctos)")
    ax.set_ylabel("AUROC (evaluado con etiquetas verdaderas)")
    ax.set_title("Robustez al ruido en los reportes de usuarios\n(bandas: percentil 2,5 – 97,5 de 20 réplicas)")
    ax.legend(loc="lower left", fontsize=9)
    aplicar_coma(ax, "ambos")
    fig.tight_layout()
    fig.savefig(FIGURAS / "fig_d_ruido.png")
    plt.close(fig)
    print("Guardado: fig_d_ruido.png")


def main():
    fig_a_roc()
    fig_a_fiabilidad()
    fig_a_perfil_capas()
    fig_c_aprendizaje()
    fig_d_ruido()
    return 0


if __name__ == "__main__":
    sys.exit(main())
