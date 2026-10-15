======================================================================
Fase 2.5 - Consolidacion y verificacion
======================================================================
Lotes encontrados: 21

----------------------------------------------------------------------
VERIFICACION 1: dimensiones y NaN/Inf
----------------------------------------------------------------------
meta_es filas: 1034 (esperado 1034)
acts_es.shape: (1034, 37, 2048) (esperado (1034, 37, 2048))
NaN en acts_es: False | Inf en acts_es: False
RESULTADO 1: OK

----------------------------------------------------------------------
VERIFICACION 2: gold_status y gold_empty
----------------------------------------------------------------------
gold_status == 'error': 2
gold_empty == True: 49
Preguntas validas (gold_status == 'ok'): 1032

----------------------------------------------------------------------
VERIFICACION 3: distribucion de desenlace y tasa de acierto
----------------------------------------------------------------------
desenlace
correcta      632
silencioso    261
sintactico    139
Name: count, dtype: int64
Tasa de acierto (y=1): 61.2%
RESULTADO 3 (criterio 20%-80%): OK

----------------------------------------------------------------------
VERIFICACION 4: conteo de errores silenciosos
----------------------------------------------------------------------
Errores silenciosos: 261

----------------------------------------------------------------------
VERIFICACION 5: tasa de acierto por nivel de dificultad
----------------------------------------------------------------------
                mean  count
dificultad                 
easy        0.830645    248
extra hard  0.403614    166
hard        0.545977    174
medium      0.594595    444

----------------------------------------------------------------------
VERIFICACION 6: sanidad de capa 0 (varianza entre preguntas ~ 0)
----------------------------------------------------------------------
Varianza media entre preguntas en capa 0: 0.00000000 (esperado ~0)
RESULTADO 6: OK

----------------------------------------------------------------------
VERIFICACION 7: varianza > 0 en capas >= 1
----------------------------------------------------------------------
Min varianza entre capas 1..36: 0.00032337
RESULTADO 7: OK

----------------------------------------------------------------------
VERIFICACION 8: revision manual de 10 casos (3 correctas, 4 silenciosas, 3 sintacticas)
----------------------------------------------------------------------
PENDIENTE DE REVISION MANUAL POR EL AUTOR -- ver casos listados abajo.
  [correcta] qid=es_0000 db_id=concert_singer
    pregunta: ¿Cuántos cantantes tenemos?
    gold: SELECT count(*) FROM singer
    pred: SELECT COUNT(*) FROM singer
  [correcta] qid=es_0001 db_id=concert_singer
    pregunta: ¿Cuál es el número total de cantantes?
    gold: SELECT count(*) FROM singer
    pred: SELECT COUNT(*) FROM singer
  [correcta] qid=es_0002 db_id=concert_singer
    pregunta: Muestre el nombre, el país y la edad de todos los cantantes ordenados por edad, desde el mayor al más joven.
    gold: SELECT name ,  country ,  age FROM singer ORDER BY age DESC
    pred: SELECT Name, Country, Age FROM singer ORDER BY Age DESC
  [silencioso] qid=es_0011 db_id=concert_singer
    pregunta: ¿Cuántos cantantes vienen de cada país?
    gold: SELECT country ,  count(*) FROM singer GROUP BY country
    pred: SELECT COUNT(Singer_ID), Country FROM singer GROUP BY Country
  [silencioso] qid=es_0016 db_id=concert_singer
    pregunta: ¿Cuál es la asistencia media y la capacidad máxima de todos los estadios?
    gold: select max(capacity), average from stadium
    pred: SELECT AVG(Average) AS Average_Attendance, MAX(Capacity) AS Maximum_Capacity FROM stadium
  [silencioso] qid=es_0023 db_id=concert_singer
    pregunta: Para cada estadio, ¿cuántos conciertos se celebran?
    gold: SELECT T2.name ,  count(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id GROUP BY T1.stadium_id
    pred: SELECT s.Stadium_ID, COUNT(c.concert_ID) AS Number_of_Concerts
FROM stadium s
JOIN concert c ON s.Stadium_ID = c.Stadium_ID
GROUP BY s.Stadium_ID
  [silencioso] qid=es_0030 db_id=concert_singer
    pregunta: Muestre los países de donde vienen los cantantes mayores de 40 años o menores de 30.
    gold: SELECT country FROM singer WHERE age  >  40 INTERSECT SELECT country FROM singer WHERE age  <  30
    pred: SELECT Country FROM singer WHERE Age > 40 OR Age < 30
  [sintactico] qid=es_0006 db_id=concert_singer
    pregunta: Muestra el nombre y el año de lanzamiento de la canción del cantante más joven.
    gold: SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1
    pred: SELECT T2.Name, T2.Song_release_year FROM singer AS T1 INNER JOIN concert AS T3 ON T1.Singer_ID = T3.Singer_ID INNER JOIN singer_in_concert AS T4 ON T1.Singer_ID = T4.Singer_ID INNER JOIN song AS T2 ON T1.Singer_ID = T2.Singer_ID ORDER BY T1.Age ASC LIMIT 1
    error: no such table: song
  [sintactico] qid=es_0007 db_id=concert_singer
    pregunta: ¿Cuáles son los nombres y años de lanzamiento de todas las canciones del cantante más joven?
    gold: SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1
    pred: SELECT T2.Name, T2.Song_release_year FROM singer AS T1 INNER JOIN singer_in_concert AS T3 ON T1.Singer_ID = T3.Singer_ID INNER JOIN concert AS T4 ON T3.concert_ID = T4.concert_ID INNER JOIN song AS T2 ON T4.concert_Name = T2.Song_Name ORDER BY T1.Age ASC LIMIT 1
    error: no such table: song
  [sintactico] qid=es_0012 db_id=concert_singer
    pregunta: Enumere todos los nombres de las canciones por cantantes que están por encima de la edad promedio.
    gold: SELECT song_name FROM singer WHERE age  >  (SELECT avg(age) FROM singer)
    pred: SELECT T2.Song_Name FROM singer AS T1 INNER JOIN singer_in_concert AS T3 ON T1.Singer_ID = T3.Singer_ID INNER JOIN concert AS T4 ON T3.concert_ID = T4.concert_ID INNER JOIN stadium AS T5 ON T4.Stadium_ID = T5.Stadium_ID INNER JOIN singer AS T6 ON T3.Singer_ID = T6.Singer_ID WHERE T1.Age > (SELECT AVG(Age) FROM singer)
    error: no such column: T2.Song_Name

CONCLUSION VERIFICACION 8 (revision manual, asistida por un modelo de lenguaje, el 2026-09-27):
  - 3 correctas: SQL generado semanticamente equivalente al gold (diferencias
    solo de mayus/minuscula en nombres de columna, sin efecto en SQLite).
    Verificador acerto en las 3.
  - 4 silenciosas: en los 4 casos el SQL generado difiere genuinamente del
    gold (orden/seleccion de columnas distinta con contenido distinto, o
    semantica distinta, p.ej. es_0030 usa OR en vez de INTERSECT). El
    verificador acerto en marcarlas como error silencioso.
  - 3 sintacticas: errores reales de SQLite (tabla/columna inexistente).
    Verificador acerto en las 3.
  - No se encontro ningun error del verificador en esta muestra de 10 casos.
    No hace falta re-etiquetar.

Guardado: data\acts_es.npy
Guardado: data\meta_es.csv