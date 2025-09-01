# Proyecto1TC
Proyecto 1 — AFN/AFD desde Expresiones Regulares
Convierte una expresión regular infix a postfix (Shunting Yard), construye un AFN (Thompson), lo transforma a AFD (subconjuntos), minimiza el AFD (algoritmo de particiones/Moore) y simula AFN/AFD sobre una cadena w.
También genera imágenes .png (y sus .dot) del AFN, AFD y AFD minimizado.
⚠️ El archivo principal se llama main-1.py.
Requisitos
Python 3.9+
Graphviz (sistema) para diagramas con layout dot
macOS: brew install graphviz
Windows (PowerShell admin): choco install graphviz
Ubuntu/Debian: sudo apt-get install graphviz
Paquetes Python: rich, networkx, matplotlib, graphviz
Instálalos con:
pip install -r requirements.txt
Verifica Graphviz:
dot -V
# debe imprimir: dot - graphviz version X.YY
Instalación rápida
macOS / Linux
cd /ruta/al/repositorio
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# (si falta) brew install graphviz     # macOS
# (si falta) sudo apt-get install graphviz  # Ubuntu/Debian
Windows (PowerShell)
cd C:\ruta\al\repositorio
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# (si falta) choco install graphviz
Uso
En macOS/Linux usa python3 si python no existe. Dentro del venv, python ya apunta a Python 3.
1) Una sola expresión
python main-1.py -r "(a|b)*abb(a|b)*" -w "babbaaaaa" -o out
# Imágenes en: out/single/afn.png, afd.png, afd_min.png (+ .dot)
2) Archivo con varias expresiones
python main-1.py -f expresiones.txt -w "babbaaaaa" -o out_lote
# Imágenes por línea: out_lote/linea_1/, out_lote/linea_2/, ...
Opciones útiles
--steps → imprime los pasos del Shunting Yard (pila/salida).
--epsilon E → define el símbolo de épsilon (por defecto ε).
Ejemplo:
python main-1.py -r "(E|a)*" -w "" --epsilon E -o out
Sintaxis soportada y no soportada
Soportado
Literales de 1 carácter (Unicode simple)
Agrupación: ( )
Unión: |
Concatenación: implícita (el programa inserta · internamente)
Kleene: *
Azúcares:
+ ⇒ X X * ·
? ⇒ X ε |
Épsilon: ε (o lo que definas con --epsilon)
No soportado
Clases [ ... ], rangos [a-z]
Comodín .
Anclas ^ / $
Cuantificadores {m,n}
Grupos especiales (?:...)
¿Qué hace internamente?
Infix → Postfix (Shunting Yard + inserción de ·)
Desazucarado de + y ?
AFN (Thompson) con transiciones ε
AFD (subconjuntos)
Minimización (particiones/Moore)
Render con Graphviz/DOT (además guarda el .dot junto al .png)
Archivos de ejemplo
expresiones.txt — exp. de prueba básicas.
expresiones_avanzadas.txt — exp. más “retadoras”.
casos_extremos.txt — stress tests y casos límite (errores de sintaxis, no soportados, profundidad, etc.).
Ejemplo:
python main-1.py -f expresiones_avanzadas.txt -w "babbaaaaa" -o out_avz
python main-1.py -f casos_extremos.txt -w "" -o out_extremos
Salida generada
Por cada regex:
afn.png / afn.dot
afd.png / afd.dot
afd_min.png / afd_min.dot
En consola:
Postfix (simplificado) y alfabeto inferido.
Acepta o rechaza la palabra w en AFN/AFD/AFDmin.
Solución de problemas
zsh: command not found: python (macOS)
Usa python3 o activa el venv (source .venv/bin/activate).
can't open file 'main-1.py'
Asegúrate de estar en la carpeta correcta (ls/dir).
Graphviz no se usa / diagramas desordenados
Verifica dot -V. Si no aparece, instala Graphviz y asegúrate de que esté en PATH.
Se guardan los .dot; ábrelos con Graphviz si quieres ajustar el layout.
No se ve ε
Usa --epsilon e y reemplaza ε por e en tus expresiones.
Imágenes muy densas
Abre el .dot y vuelve a renderizar ajustando parámetros (p. ej. nodesep, ranksep).
