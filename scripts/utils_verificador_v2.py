# -*- coding: utf-8 -*-
"""
Verificador v2: identico a utils_verificador.py salvo la funcion comparar(),
que ahora tolera la PERMUTACION DE COLUMNAS (criterio del test-suite de
Spider, Zhong et al., 2020): dos resultados coinciden si existe un
reordenamiento de las columnas de la prediccion que los hace iguales.
Filas: se mantiene la regla de ORDER BY (lista si el gold ordena,
multiconjunto si no). Columnas extra o faltantes siguen siendo error.
"""
from collections import Counter
from itertools import product

from utils_verificador import (TIMEOUT_S, ejecutar_con_timeout,  # noqa: F401
                               normalizar_filas)

MAX_PERMUTACIONES = 20_000


def _coinciden_filas(g, p, gold_has_orderby):
    return g == p if gold_has_orderby else Counter(g) == Counter(p)


def comparar(filas_gold, filas_pred, gold_has_orderby):
    g = normalizar_filas(filas_gold)
    p = normalizar_filas(filas_pred)
    if _coinciden_filas(g, p, gold_has_orderby):
        return True
    if len(g) != len(p):
        return False
    if len(g) == 0:
        return True
    n_cols = len(g[0])
    if any(len(f) != n_cols for f in p):
        return False
    # Columnas candidatas: la columna j de la prediccion solo puede ocupar la
    # posicion i del gold si ambas tienen el mismo multiconjunto de valores.
    firma_g = [Counter(f[i] for f in g) for i in range(n_cols)]
    firma_p = [Counter(f[j] for f in p) for j in range(n_cols)]
    candidatas = [[j for j in range(n_cols) if firma_p[j] == firma_g[i]] for i in range(n_cols)]
    if any(len(c) == 0 for c in candidatas):
        return False
    intentos = 0
    for perm in product(*candidatas):
        if len(set(perm)) != n_cols:
            continue
        intentos += 1
        if intentos > MAX_PERMUTACIONES:
            return False
        p_perm = [tuple(f[j] for j in perm) for f in p]
        if _coinciden_filas(g, p_perm, gold_has_orderby):
            return True
    return False
