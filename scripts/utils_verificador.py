# -*- coding: utf-8 -*-
"""
Verificador de corrección por ejecución.

Ejecuta gold y prediccion contra la base .sqlite correspondiente, ABIERTA
SIEMPRE EN SOLO LECTURA (uri file:...?mode=ro), y compara resultados con
la regla de ORDER BY (lista ordenada si el gold usa ORDER BY; multiconjunto
si no).

Timeout de 30s por consulta implementado con
sqlite3.Connection.set_progress_handler (chequeo de reloj cada N
instrucciones de la VM de SQLite), en vez de signal.alarm (no disponible
en Windows) o un proceso hijo (overhead de crear 1034+ procesos).
"""
import sqlite3
import time

TIMEOUT_S = 30
N_OPS_PROGRESS = 1000  # granularidad del chequeo de reloj dentro de set_progress_handler


class ErrorTimeout(Exception):
    pass


# El modo solo lectura impide escribir en la base, pero no impide ATTACH, que puede crear
# archivos vacios en disco. Como se ejecuta SQL generado por el modelo, se bloquean
# ATTACH, DETACH y PRAGMA con un autorizador (ninguna consulta del corpus los usa).
_ACCIONES_BLOQUEADAS = {sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH, sqlite3.SQLITE_PRAGMA}


def _autorizador(accion, *_):
    return sqlite3.SQLITE_DENY if accion in _ACCIONES_BLOQUEADAS else sqlite3.SQLITE_OK


def _conectar_ro(ruta_db):
    uri = f"file:{ruta_db}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def ejecutar_con_timeout(sql, ruta_db, timeout_s=TIMEOUT_S):
    """Devuelve (filas, error). filas=None si hubo error o timeout.
    error in {None, 'timeout', '<mensaje de excepcion>'}."""
    t0 = time.time()

    def _handler():
        return 1 if (time.time() - t0) > timeout_s else 0

    con = _conectar_ro(ruta_db)
    con.set_progress_handler(_handler, N_OPS_PROGRESS)
    con.set_authorizer(_autorizador)
    try:
        cur = con.cursor()
        cur.execute(sql)
        filas = cur.fetchall()
        return filas, None
    except sqlite3.OperationalError as e:
        msg = str(e).lower()
        if "interrupted" in msg or (time.time() - t0) > timeout_s:
            return None, "timeout"
        return None, str(e)
    except Exception as e:
        if (time.time() - t0) > timeout_s:
            return None, "timeout"
        return None, str(e)
    finally:
        con.set_progress_handler(None, 0)
        con.close()


def normalizar_valor(v):
    """Enteros/flotantes al mismo tipo numerico, flotantes a 6 decimales.
    None se mantiene; cadenas sin modificar."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v  # bool es subtipo de int en Python; se deja tal cual (SQLite no distingue)
    if isinstance(v, int):
        return round(float(v), 6)
    if isinstance(v, float):
        return round(v, 6)
    return v


def normalizar_filas(filas):
    return [tuple(normalizar_valor(v) for v in fila) for fila in filas]


def comparar(filas_gold, filas_pred, gold_has_orderby):
    """True si coinciden segun la regla de ORDER BY."""
    g = normalizar_filas(filas_gold)
    p = normalizar_filas(filas_pred)
    if gold_has_orderby:
        return g == p
    from collections import Counter
    return Counter(g) == Counter(p)


def verificar_pregunta(gold_sql, pred_sql, ruta_db, gold_has_orderby):
    """Devuelve un dict con: gold_status, gold_empty, desenlace, y,
    sqlite_error, t_exec_s (tiempo total de ejecucion gold+pred)."""
    t0 = time.time()

    filas_gold, err_gold = ejecutar_con_timeout(gold_sql, ruta_db)
    if err_gold is not None:
        # El gold fallo (bug de dataset o esquema): la pregunta se excluye del analisis
        #. desenlace/y quedan None; el filtro aguas abajo usa gold_status.
        return {
            "gold_status": "error", "gold_empty": False,
            "desenlace": None, "y": None,
            "sqlite_error": f"gold: {err_gold}",
            "t_exec_s": time.time() - t0,
        }

    gold_empty = len(filas_gold) == 0

    filas_pred, err_pred = ejecutar_con_timeout(pred_sql, ruta_db)
    if err_pred == "timeout":
        return {
            "gold_status": "ok", "gold_empty": gold_empty,
            "desenlace": "timeout", "y": 0,
            "sqlite_error": None,
            "t_exec_s": time.time() - t0,
        }
    if err_pred is not None:
        return {
            "gold_status": "ok", "gold_empty": gold_empty,
            "desenlace": "sintactico", "y": 0,
            "sqlite_error": err_pred,
            "t_exec_s": time.time() - t0,
        }

    coincide = comparar(filas_gold, filas_pred, gold_has_orderby)
    return {
        "gold_status": "ok", "gold_empty": gold_empty,
        "desenlace": "correcta" if coincide else "silencioso",
        "y": 1 if coincide else 0,
        "sqlite_error": None,
        "t_exec_s": time.time() - t0,
    }
