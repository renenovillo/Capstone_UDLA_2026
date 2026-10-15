"""
Clasificador oficial de dificultad de Spider (easy/medium/hard/extra hard).

Este modulo es un PUERTO FIEL (no una reinterpretacion) de la logica de
`evaluation.py` del repositorio oficial de Spider
(https://github.com/taoyds/spider/blob/master/evaluation.py, descargado el
2026-09-14). No se reescribio la heuristica: se copiaron las funciones
`count_agg`, `count_component1`, `count_component2`, `count_others`,
`has_agg`, `get_nestedSQL` y `Evaluator.eval_hardness` tal como estan en el
script oficial, adaptando solo el nombre de la etiqueta "extra" -> "extra hard"
para que coincida con la nomenclatura de la literatura (Yu et al., 2018).

Se reutiliza sobre el campo `sql` que ya viene pre-parseado (formato AST de
`process_sql.py`) en los JSON de Spider y de MultiSpider ES, por lo que no
hace falta re-parsear las consultas ni tocar las bases .sqlite.

NOTA: dev_es.json (MultiSpider, with_english_value) conserva el SQL gold de dev.json
(Spider) en cada indice; la etiqueta de dificultad depende solo de ese SQL.
"""

AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')

HARDNESS = {
    "component1": ('where', 'group', 'order', 'limit', 'join', 'or', 'like'),
    "component2": ('except', 'union', 'intersect'),
}


def has_agg(unit):
    return unit[0] != AGG_OPS.index('none')


def count_agg(units):
    return len([unit for unit in units if has_agg(unit)])


def get_nestedSQL(sql):
    nested = []
    for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql['intersect'] is not None:
        nested.append(sql['intersect'])
    if sql['except'] is not None:
        nested.append(sql['except'])
    if sql['union'] is not None:
        nested.append(sql['union'])
    return nested


def count_component1(sql):
    count = 0
    if len(sql['where']) > 0:
        count += 1
    if len(sql['groupBy']) > 0:
        count += 1
    if len(sql['orderBy']) > 0:
        count += 1
    if sql['limit'] is not None:
        count += 1
    if len(sql['from']['table_units']) > 0:  # JOIN
        count += len(sql['from']['table_units']) - 1

    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    count += len([token for token in ao if token == 'or'])
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')])

    return count


def count_component2(sql):
    nested = get_nestedSQL(sql)
    return len(nested)


def count_others(sql):
    count = 0
    agg_count = count_agg(sql['select'][1])
    agg_count += count_agg(sql['where'][::2])
    agg_count += count_agg(sql['groupBy'])
    if len(sql['orderBy']) > 0:
        agg_count += count_agg([unit[1] for unit in sql['orderBy'][1] if unit[1]] +
                                [unit[2] for unit in sql['orderBy'][1] if unit[2]])
    agg_count += count_agg(sql['having'])
    if agg_count > 1:
        count += 1

    if len(sql['select'][1]) > 1:
        count += 1
    if len(sql['where']) > 1:
        count += 1
    if len(sql['groupBy']) > 1:
        count += 1

    return count


def eval_hardness(sql):
    """Devuelve 'easy' | 'medium' | 'hard' | 'extra hard', identico a la
    logica oficial de evaluation.py (con 'extra' renombrado a 'extra hard')."""
    count_comp1_ = count_component1(sql)
    count_comp2_ = count_component2(sql)
    count_others_ = count_others(sql)

    if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
        return "easy"
    elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
            (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
        return "medium"
    elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
            (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
            (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
        return "hard"
    else:
        return "extra hard"


# --- Metricas estructurales adicionales (no parte del clasificador oficial,
# pero derivadas del mismo AST `sql`, para el analisis de complejidad
# estructural pedido en el EDA: JOIN, subconsultas, GROUP BY, ORDER BY, agregaciones). ---

def n_joins(sql):
    tu = sql['from']['table_units']
    return max(len(tu) - 1, 0)


def has_subquery(sql):
    return count_component2(sql) > 0


def n_subqueries(sql):
    return count_component2(sql)


def has_groupby(sql):
    return len(sql['groupBy']) > 0


def has_orderby(sql):
    return len(sql['orderBy']) > 0


def has_where(sql):
    return len(sql['where']) > 0


def n_aggregations(sql):
    """Conteo total de usos de funciones de agregacion (COUNT/SUM/AVG/MIN/MAX)
    en SELECT, WHERE, GROUP BY, ORDER BY y HAVING."""
    total = count_agg(sql['select'][1])
    total += count_agg(sql['where'][::2])
    total += count_agg(sql['groupBy'])
    if len(sql['orderBy']) > 0:
        total += count_agg([unit[1] for unit in sql['orderBy'][1] if unit[1]] +
                            [unit[2] for unit in sql['orderBy'][1] if unit[2]])
    total += count_agg(sql['having'])
    return total


def has_set_op(sql):
    return sql['intersect'] is not None or sql['union'] is not None or sql['except'] is not None
