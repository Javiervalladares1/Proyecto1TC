# Proyecto 1 — AFN/AFD desde Expresiones Regulares

Convierte una regex **infix** a **postfix**, construye **AFN** (Thompson), **AFD** (subconjuntos), lo **minimiza** (particiones/Moore) y **simula** la cadena `w`.  
Genera imágenes **AFN**, **AFD** y **AFD minimizado** (también sus **.dot**).

> Archivo principal: **`main-1.py`**

---

## Requisitos

- **Python 3.9+**
- **Paquetes** (se instalan con `requirements.txt`):
  ```bash
  pip install -r requirements.txt
Graphviz (para diagramas con layout dot):
macOS: brew install graphviz
Windows (PowerShell admin): choco install graphviz
Ubuntu/Debian: sudo apt-get install graphviz
Verifica: dot -V
Si python no existe en tu sistema, usa python3.
Ejecución rápida
1) Una sola expresión
# macOS / Linux
python3 main-1.py -r "(a|b)*abb(a|b)*" -w "babbaaaaa" -o out

# Windows
python main-1.py -r "(a|b)*abb(a|b)*" -w "babbaaaaa" -o out
Salida (en out/single/):
afn.png / afn.dot, afd.png / afd.dot, afd_min.png / afd_min.dot
2) Archivo con varias expresiones (una por línea)
# macOS / Linux
python3 main-1.py -f expresiones.txt -w "babbaaaaa" -o out_lote

# Windows
python main-1.py -f expresiones.txt -w "babbaaaaa" -o out_lote
Salida: out_lote/linea_1/, out_lote/linea_2/, …
Opciones útiles
--steps → imprime los pasos del Shunting Yard.
--epsilon E → cambia el símbolo de épsilon (por defecto ε):
python3 main-1.py -r "(E|a)*" -w "" --epsilon E -o out
