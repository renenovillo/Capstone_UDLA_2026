# -*- coding: utf-8 -*-
"""
Fase 0 - Verificacion de entorno.
Imprime y registra en logs/entorno.txt: versiones, CUDA, VRAM, rutas.
"""
import datetime
import sys
from pathlib import Path

import os
RAIZ = Path(__file__).resolve().parents[1]  # raiz del repositorio
EXTERNO = Path(os.environ.get("TESIS_EXTERNO", RAIZ / "externo"))  # Spider, MultiSpider y modelo (descargados aparte)
BASE = RAIZ
LOGS = BASE / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

RUTAS_ESPERADAS = {
    "SPIDER dev.json": EXTERNO / "spider_data" / "dev.json",
    "SPIDER tables.json": EXTERNO / "spider_data" / "tables.json",
    "SPIDER database/": EXTERNO / "spider_data" / "database",
    "MULTISPIDER dev_es.json": EXTERNO / "multispider" / "dataset"
        / "multispider" / "with_english_value" / "dev_es.json",
    "MULTISPIDER tables_es.json": EXTERNO / "multispider" / "dataset"
        / "multispider" / "with_english_value" / "tables_es.json",
    "Modelo Qwen2.5-Coder-3B": EXTERNO / "qwen2.5-coder-3b",
    "hardness.py": RAIZ / "scripts" / "hardness.py",
}


def main():
    lineas = []

    def log(s=""):
        print(s)
        lineas.append(s)

    log("=" * 70)
    log(f"Verificacion de entorno - {datetime.datetime.now().isoformat()}")
    log("=" * 70)

    log(f"\nPython: {sys.version}")
    log(f"Ejecutable: {sys.executable}")

    try:
        import torch
        log(f"\ntorch: {torch.__version__}")
        log(f"CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(f"Dispositivo: {torch.cuda.get_device_name(0)}")
            libre, total = torch.cuda.mem_get_info()
            log(f"VRAM libre: {libre/1e9:.2f} GB / total: {total/1e9:.2f} GB")
    except ImportError as e:
        log(f"ERROR importando torch: {e}")

    for nombre, mod in [("transformers", "transformers"), ("accelerate", "accelerate"),
                        ("sklearn", "sklearn"), ("numpy", "numpy"), ("pandas", "pandas")]:
        try:
            m = __import__(mod)
            log(f"{nombre}: {m.__version__}")
        except ImportError as e:
            log(f"ERROR importando {nombre}: {e}")

    log("\n" + "=" * 70)
    log("Verificacion de rutas")
    log("=" * 70)
    todas_ok = True
    for nombre, ruta in RUTAS_ESPERADAS.items():
        existe = ruta.exists()
        todas_ok = todas_ok and existe
        log(f"  [{'OK' if existe else 'FALTA'}] {nombre}: {ruta}")

    log("\n" + "=" * 70)
    log(f"RESULTADO: {'TODAS LAS RUTAS ENCONTRADAS' if todas_ok else 'FALTAN RUTAS -- DETENER'}")
    log("=" * 70)

    with open(LOGS / "entorno.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))

    return 0 if todas_ok else 1


if __name__ == "__main__":
    sys.exit(main())
