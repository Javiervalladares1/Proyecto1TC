#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proyecto No. 1 — Construcción y simulación de AFN/AFD desde una expresión regular
Autor: (Completar con su nombre)
Descripción:
  - Convierte una regex r (infix) a postfix (Shunting Yard con concatenación explícita '·').
  - Desazucara operadores + y ?
  - Construye un AFN con Thompson y lo simula para una cadena w.
  - Construye un AFD por subconjuntos, lo minimiza y simula.
  - Dibuja imágenes de AFN, AFD y AFD minimizado.
Uso básico:
  python main.py -r "(a|b)*abb(a|b)*" -w "babbaaaaa" -o out
  python main.py -f expresiones.txt -w "babbaaaaa" -o out
Requiere:
  - Python 3.9+
  - Paquetes: graphviz (opcional), networkx, matplotlib, rich
  - (Opcional) Tener Graphviz instalado en el sistema para mejor renderizado.
"""
from __future__ import annotations

import argparse
import subprocess

import os
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional, Iterable

# ===== Pretty print opcional =====
try:
    from rich import print as rprint
except Exception:
    def rprint(*a, **k): print(*a, **k)

# ====== Tokenización y Shunting Yard (basado en tus Labs 3/4, simplificado para este proyecto) ======

PREC = {'|': 2, '·': 3, '?': 4, '*': 4, '+': 4, '^': 5, '(': 1}
RIGHT_ASSOC = {'^'}

def es_operador(tok: str) -> bool:
    return tok in PREC and tok != '('

def tokenizar(regex: str) -> List[str]:
    """Tokeniza respetando escapes con '\\' (e.g., '\\(' = literal '(')."""
    tokens: List[str] = []
    i = 0
    while i < len(regex):
        c = regex[i]
        if c == '\\' and i + 1 < len(regex):
            tokens.append(regex[i:i+2])
            i += 2
        else:
            if not c.isspace():
                tokens.append(c)
            i += 1
    return tokens

def necesita_concat(c1: str, c2: str) -> bool:
    # evita concat antes de ')', después de '(' y antes de operadores binarios/postfijos
    if c1 == '(' or c2 == ')':
        return False
    if c2 == '|':
        return False
    if c2 in {'?', '*', '+'}:
        return False
    if es_operador(c1) and PREC[c1] != 4:  # 4 = ?,*,+ (postfijos)
        return False
    return True

def format_regex(regex: str) -> List[str]:
    """Inserta el operador de concatenación '·' donde corresponda."""
    toks = tokenizar(regex)
    if not toks:
        return []
    res: List[str] = []
    for i, t1 in enumerate(toks[:-1]):
        t2 = toks[i+1]
        res.append(t1)
        c1 = t1[-1]
        c2 = t2[0]
        if necesita_concat(c1, c2):
            res.append('·')
    res.append(toks[-1])
    return res

@dataclass
class PasoSY:
    paso: int
    token: str
    accion: str
    pila: str
    salida: str

@dataclass
class ResultadoSY:
    postfix: str
    postfix_tokens: List[str]
    error_msg: Optional[str]
    pasos: List[PasoSY]

def infix_to_postfix(regex: str) -> ResultadoSY:
    tokens = format_regex(regex)
    salida: List[str] = []
    pila: List[str] = []
    pasos: List[PasoSY] = []
    step = 1
    error: Optional[str] = None

    def push(tok: str):
        nonlocal step
        pila.append(tok)
        pasos.append(PasoSY(step, tok, 'push', ''.join(pila), ''.join(salida))); step += 1

    def pop_to_output():
        nonlocal step
        tok = pila.pop()
        salida.append(tok)
        pasos.append(PasoSY(step, tok, 'pop→out', ''.join(pila), ''.join(salida))); step += 1

    for tok in tokens:
        if error: break
        if tok == '(':
            push(tok)
        elif tok == ')':
            if not pila:
                error = "Paréntesis de cierre sin apertura."
                pasos.append(PasoSY(step, tok, 'error', ''.join(pila), ''.join(salida))); step += 1
                break
            while pila and pila[-1] != '(':
                pop_to_output()
            if not pila or pila[-1] != '(':
                error = "Paréntesis de cierre sin apertura."
                pasos.append(PasoSY(step, tok, 'error', ''.join(pila), ''.join(salida))); step += 1
                break
            pila.pop()
            pasos.append(PasoSY(step, ')', 'pop (', ''.join(pila), ''.join(salida))); step += 1
        elif es_operador(tok):
            while (pila and es_operador(pila[-1]) and
                   ((PREC[pila[-1]] > PREC[tok]) or
                    (PREC[pila[-1]] == PREC[tok] and tok not in RIGHT_ASSOC))):
                pop_to_output()
            push(tok)
        else:
            salida.append(tok)
            pasos.append(PasoSY(step, tok, 'output', ''.join(pila), ''.join(salida))); step += 1

    if not error:
        while pila and pila[-1] != '(':
            pop_to_output()
        if pila and pila[-1] == '(':
            error = "Paréntesis de apertura sin cierre."
            pasos.append(PasoSY(step, '', 'error', ''.join(pila), ''.join(salida))); step += 1

    postfix_tokens = salida[:]
    postfix_str = ''.join(salida)
    return ResultadoSY(postfix=postfix_str, postfix_tokens=postfix_tokens, error_msg=error, pasos=pasos)

def desugar_postfix_extensions(postfix_tokens: List[str]) -> List[str]:
    """
    Reescribe + y ? sobre la secuencia postfix (no-variadic) en términos de *, ε y |.
    X+ -> X X * ·
    X? -> X ε |
    """
    stack: List[List[str]] = []
    for tok in postfix_tokens:
        if tok == '·':
            if len(stack) < 2: raise ValueError("Postfix inválido: faltan operandos para '·'.")
            b = stack.pop(); a = stack.pop()
            stack.append(a + b + ['·'])
        elif tok == '|':
            if len(stack) < 2: raise ValueError("Postfix inválido: faltan operandos para '|'.")
            b = stack.pop(); a = stack.pop()
            stack.append(a + b + ['|'])
        elif tok == '*':
            if not stack: raise ValueError("Postfix inválido: falta operando para '*'.")
            a = stack.pop()
            stack.append(a + ['*'])
        elif tok == '+':
            if not stack: raise ValueError("Postfix inválido: falta operando para '+'.")
            a = stack.pop()
            stack.append(a + a + ['*', '·'])
        elif tok == '?':
            if not stack: raise ValueError("Postfix inválido: falta operando para '?'.")
            a = stack.pop()
            stack.append(a + ['ε', '|'])
        else:
            stack.append([tok])
    if len(stack) != 1:
        raise ValueError("Postfix inválido: sobran elementos tras la desazucaración.")
    return stack[0]

def tok_label(tok: str) -> str:
    """Devuelve el símbolo literal que representa un token (considerando escapes)."""
    if len(tok) == 2 and tok[0] == '\\':
        return tok[1]
    return tok

def build_alphabet_from_tokens(tokens: List[str], epsilon: str = 'ε') -> List[str]:
    ops = {'·','|','*'}
    alphabet = []
    for t in tokens:
        if t in ops or t == epsilon:
            continue
        sym = tok_label(t)
        if sym not in alphabet:
            alphabet.append(sym)
    return alphabet

# ====== NFA via Thompson ======

@dataclass
class NFAFragment:
    start: int
    accept: int

class NFABuilder:
    def __init__(self, epsilon: str = 'ε'):
        self.epsilon = epsilon
        self.next_id = 0
        self.trans: Dict[int, Dict[str, Set[int]]] = {}

    def new_state(self) -> int:
        s = self.next_id
        self.next_id += 1
        if s not in self.trans:
            self.trans[s] = {}
        return s

    def add_edge(self, u: int, symbol: str, v: int) -> None:
        self.trans.setdefault(u, {}).setdefault(symbol, set()).add(v)

    def literal(self, a: str) -> NFAFragment:
        s = self.new_state()
        t = self.new_state()
        self.add_edge(s, a, t)
        return NFAFragment(s, t)

    def epsilon_edge(self) -> NFAFragment:
        s = self.new_state()
        t = self.new_state()
        self.add_edge(s, self.epsilon, t)
        return NFAFragment(s, t)

    def concat(self, A: NFAFragment, B: NFAFragment) -> NFAFragment:
        self.add_edge(A.accept, self.epsilon, B.start)
        return NFAFragment(A.start, B.accept)

    def union(self, A: NFAFragment, B: NFAFragment) -> NFAFragment:
        s = self.new_state()
        t = self.new_state()
        self.add_edge(s, self.epsilon, A.start)
        self.add_edge(s, self.epsilon, B.start)
        self.add_edge(A.accept, self.epsilon, t)
        self.add_edge(B.accept, self.epsilon, t)
        return NFAFragment(s, t)

    def star(self, A: NFAFragment) -> NFAFragment:
        s = self.new_state()
        t = self.new_state()
        self.add_edge(s, self.epsilon, A.start)
        self.add_edge(s, self.epsilon, t)
        self.add_edge(A.accept, self.epsilon, A.start)
        self.add_edge(A.accept, self.epsilon, t)
        return NFAFragment(s, t)

@dataclass
class NFA:
    start: int
    accept: int
    trans: Dict[int, Dict[str, Set[int]]]
    alphabet: List[str]
    epsilon: str = 'ε'

def postfix_to_nfa(tokens: List[str], epsilon: str = 'ε') -> NFA:
    builder = NFABuilder(epsilon=epsilon)
    stack: List[NFAFragment] = []
    for tok in tokens:
        if tok == '·':
            b = stack.pop(); a = stack.pop()
            stack.append(builder.concat(a, b))
        elif tok == '|':
            b = stack.pop(); a = stack.pop()
            stack.append(builder.union(a, b))
        elif tok == '*':
            a = stack.pop()
            stack.append(builder.star(a))
        elif tok == epsilon:
            stack.append(builder.epsilon_edge())
        else:
            stack.append(builder.literal(tok_label(tok)))
    assert len(stack) == 1, "Error al construir AFN: pila inválida."
    frag = stack.pop()
    # calcular alfabeto (excluye epsilon y operadores)
    alpha = build_alphabet_from_tokens(tokens, epsilon=epsilon)
    return NFA(start=frag.start, accept=frag.accept, trans=builder.trans, alphabet=alpha, epsilon=epsilon)

# ====== Simulación de AFN ======

def eclosure(states: Set[int], trans: Dict[int, Dict[str, Set[int]]], epsilon: str) -> Set[int]:
    stack = list(states)
    closure = set(states)
    while stack:
        s = stack.pop()
        for t in trans.get(s, {}).get(epsilon, set()):
            if t not in closure:
                closure.add(t)
                stack.append(t)
    return closure

def move(states: Set[int], a: str, trans: Dict[int, Dict[str, Set[int]]]) -> Set[int]:
    out: Set[int] = set()
    for s in states:
        out |= trans.get(s, {}).get(a, set())
    return out

def simulate_nfa(nfa: NFA, w: str) -> bool:
    current = eclosure({nfa.start}, nfa.trans, nfa.epsilon)
    for ch in w:
        current = eclosure(move(current, ch, nfa.trans), nfa.trans, nfa.epsilon)
    return nfa.accept in current

# ====== Subconjuntos (AFN -> AFD) ======

@dataclass
class DFA:
    start: int
    accept_states: Set[int]
    trans: Dict[int, Dict[str, int]]
    alphabet: List[str]

def nfa_to_dfa(nfa: NFA) -> DFA:
    epsilon = nfa.epsilon
    start_set = frozenset(eclosure({nfa.start}, nfa.trans, epsilon))
    unmarked: List[frozenset[int]] = [start_set]
    dfa_states: List[frozenset[int]] = [start_set]
    index: Dict[frozenset[int], int] = {start_set: 0}
    trans: Dict[int, Dict[str, int]] = {}

    while unmarked:
        S = unmarked.pop()
        Sid = index[S]
        trans.setdefault(Sid, {})
        for a in nfa.alphabet:
            U = eclosure(move(set(S), a, nfa.trans), nfa.trans, epsilon)
            Ufs = frozenset(U)
            if not U:
                continue
            if Ufs not in index:
                index[Ufs] = len(dfa_states)
                dfa_states.append(Ufs)
                unmarked.append(Ufs)
            trans[Sid][a] = index[Ufs]

    accept_states = { index[S] for S in dfa_states if nfa.accept in S }
    return DFA(start=0, accept_states=accept_states, trans=trans, alphabet=list(nfa.alphabet))

# ====== Minimizacion (partitions) ======

def complete_dfa(dfa: DFA) -> DFA:
    """Asegura que todas las transiciones estén definidas, agregando un estado sumidero si es necesario."""
    trans = {s: dict(dfa.trans.get(s, {})) for s in dfa.trans.keys()}
    states = set(trans.keys())
    # incluir estados que solo aparecen como destino
    for m in dfa.trans.values():
        for t in m.values():
            states.add(t)
    sink_needed = False
    sink_id = max(states) + 1 if states else 1
    for s in list(states):
        trans.setdefault(s, {})
        for a in dfa.alphabet:
            if a not in trans[s]:
                trans[s][a] = sink_id
                sink_needed = True
    if sink_needed:
        trans[sink_id] = {a: sink_id for a in dfa.alphabet}
        states.add(sink_id)
    # recomputar aceptaciones (no cambia)
    accepts = set(dfa.accept_states)
    start = dfa.start
    return DFA(start=start, accept_states=accepts, trans=trans, alphabet=list(dfa.alphabet))

def partitions_minimize(dfa: DFA) -> DFA:
    """
    Minimización por ALGORTIMO DE PARTICIONES (Moore):
    1) Completa el AFD (agrega sumidero si falta) para que δ esté total.
    2) Partición inicial P = { F, Q\F } (sin bloques vacíos).
    3) Mientras haya divisiones: refina cada bloque agrupando por firma
       (para cada estado s: tuple( id_bloque( δ(s,a) ) para a en alfabeto )).
    4) Construye el AFD mínimo tomando 1 representante por bloque.
    """
    dfa = complete_dfa(dfa)
    alphabet = list(dfa.alphabet)

    # Conjunto de estados (asegurado por complete_dfa)
    states = set(dfa.trans.keys())
    for m in dfa.trans.values():
        states.update(m.values())

    F = set(dfa.accept_states)
    NF = states - F

    # Partición inicial (sin bloques vacíos)
    P: List[Set[int]] = [B for B in [F, NF] if B]

    changed = True
    while changed:
        changed = False

        # Mapa estado -> id de bloque actual
        block_id: Dict[int, int] = {}
        for i, block in enumerate(P):
            for s in block:
                block_id[s] = i

        newP: List[Set[int]] = []
        for block in P:
            # Agrupar el bloque por firma de transiciones
            groups: Dict[Tuple[int, ...], Set[int]] = {}
            for s in block:
                # Firma: a qué bloque va δ(s,a) para cada símbolo
                sig = tuple(block_id[dfa.trans[s][a]] for a in alphabet)
                groups.setdefault(sig, set()).add(s)

            # Si el bloque se divide en >1 grupos, hubo refinamiento
            if len(groups) > 1:
                changed = True
                newP.extend(groups.values())
            else:
                newP.append(block)

        P = newP

    # Construcción del AFD mínimo
    block_id: Dict[int, int] = {}
    for i, block in enumerate(P):
        for s in block:
            block_id[s] = i

    new_start = block_id[dfa.start]
    new_accepts = { block_id[s] for s in F }

    new_trans: Dict[int, Dict[str, int]] = {}
    for i, block in enumerate(P):
        rep = next(iter(block))              # representante del bloque
        new_trans[i] = {}
        for a in alphabet:
            t = dfa.trans[rep][a]
            new_trans[i][a] = block_id[t]

    return DFA(start=new_start, accept_states=new_accepts, trans=new_trans, alphabet=alphabet)


# ====== Simulación de AFD ======

def simulate_dfa(dfa: DFA, w: str) -> bool:
    s = dfa.start
    for ch in w:
        s = dfa.trans.get(s, {}).get(ch, None)
        if s is None:
            return False
    return s in dfa.accept_states

# ====== Graficación (Graphviz/DOT) ======

def _aggregate_edge_labels_nfa(nfa: NFA):
    labels: Dict[Tuple[int,int], Set[str]] = {}
    for u, m in nfa.trans.items():
        for a, dests in m.items():
            for v in dests:
                labels.setdefault((u, v), set()).add(a)
    # ordena con ε primero y luego alfabético
    def _key(s: str): return (s != 'ε', s)
    return {k: ",".join(sorted(v, key=_key)) for k, v in labels.items()}

def _aggregate_edge_labels_dfa(dfa: DFA):
    labels: Dict[Tuple[int,int], Set[str]] = {}
    for u, m in dfa.trans.items():
        for a, v in m.items():
            labels.setdefault((u, v), set()).add(a)
    return {k: ",".join(sorted(v)) for k, v in labels.items()}

def _render_dot(dot_src: str, out_png: str) -> bool:
    """
    Genera PNG desde DOT. Primero intenta con python-graphviz; si falla,
    invoca la CLI 'dot' (Graphviz del sistema). Siempre guarda el .dot
    junto al .png para poder revisarlo.
    """
    base = out_png.rsplit(".", 1)[0]
    dot_path = base + ".dot"
    # guardo el .dot
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot_src)

    # 1) binding de Python (graphviz.Source)
    try:
        import graphviz  # type: ignore
        g = graphviz.Source(dot_src, format="png")
        g.render(filename=base, cleanup=True)
        return True
    except Exception:
        pass

    # 2) CLI 'dot'
    try:
        subprocess.run(["dot", "-Tpng", dot_path, "-o", out_png], check=True)
        return True
    except Exception as e:
        rprint(f"[yellow]No se pudo usar Graphviz (dot): {e}[/yellow]")
        return False

def draw_nfa(nfa: NFA, path_png: str) -> None:
    edge_labels = _aggregate_edge_labels_nfa(nfa)

    # recolecta nodos (opc.)
    nodes = set(nfa.trans.keys())
    for m in nfa.trans.values():
        for dests in m.values():
            nodes |= dests

    lines = [
        "digraph NFA {",
        # layout elegante y claro
        '  rankdir=LR; layout=dot; splines=true; overlap=false; concentrate=true; '
        'outputorder=edgesfirst; nodesep=0.7; ranksep=1.1; margin=0.25;',
        '  labelloc="t"; label="AFN (Thompson)"; fontsize=20; fontname="Helvetica";',
        '  node [shape=circle, width=0.6, height=0.6, fontname="Helvetica"];',
        '  edge [fontname="Helvetica", fontsize=11, arrowsize=0.9, penwidth=1.2, labeldistance=1.6];',
        '  __start [shape=point, width=0.1, label=""];',
        f'  __start -> {nfa.start};',
        f'  {nfa.accept} [shape=doublecircle];'
    ]
    for (u, v), lab in edge_labels.items():
        safe = lab.replace('"', r'\"')
        lines.append(f'  {u} -> {v} [label="{safe}"];')
    lines.append("}")

    _render_dot("\n".join(lines), path_png)

def draw_dfa(dfa: DFA, path_png: str, title: str = "AFD") -> None:
    edge_labels = _aggregate_edge_labels_dfa(dfa)

    nodes = set(dfa.trans.keys())
    for m in dfa.trans.values():
        for t in m.values():
            nodes.add(t)

    lines = [
        "digraph DFA {",
        '  rankdir=LR; layout=dot; splines=true; overlap=false; concentrate=true; '
        'outputorder=edgesfirst; nodesep=0.7; ranksep=1.1; margin=0.25;',
        f'  labelloc="t"; label="{title}"; fontsize=20; fontname="Helvetica";',
        '  node [shape=circle, width=0.6, height=0.6, fontname="Helvetica"];',
        '  edge [fontname="Helvetica", fontsize=11, arrowsize=0.9, penwidth=1.2, labeldistance=1.6];',
        '  __start [shape=point, width=0.1, label=""];',
        f'  __start -> {dfa.start};'
    ]
    # doble círculo en aceptaciones
    for acc in sorted(dfa.accept_states):
        lines.append(f'  {acc} [shape=doublecircle];')
    for (u, v), lab in edge_labels.items():
        safe = lab.replace('"', r'\"')
        lines.append(f'  {u} -> {v} [label="{safe}"];')
    lines.append("}")

    _render_dot("\n".join(lines), path_png)

# ====== CLI ======

def process_regex(r: str, w: str, out_dir: str, epsilon: str = 'ε', show_steps: bool = False) -> Dict[str, str]:
    rprint(f"[bold cyan]Regex:[/bold cyan] {r}")
    res = infix_to_postfix(r)
    if res.error_msg:
        rprint(f"[red]Error Shunting Yard:[/red] {res.error_msg}")
        return {}
    try:
        simple = desugar_postfix_extensions(res.postfix_tokens)
    except Exception as e:
        rprint(f"[red]Error desazucarando:[/red] {e}")
        return {}
    rprint(f"Postfix simplificado: [green]{''.join(simple)}[/green]  Alfabeto: {build_alphabet_from_tokens(simple, epsilon)}")

    # AFN
    nfa = postfix_to_nfa(simple, epsilon=epsilon)
    ok_nfa = simulate_nfa(nfa, w)
    nfa_img = f"{out_dir}/afn.png"
    draw_nfa(nfa, nfa_img)
    rprint(f"AFN acepta '{w}'? -> [bold]{'sí' if ok_nfa else 'no'}[/bold]. Imagen: {nfa_img}")

    # AFD
    dfa = nfa_to_dfa(nfa)
    ok_dfa = simulate_dfa(dfa, w)
    dfa_img = f"{out_dir}/afd.png"
    draw_dfa(dfa, dfa_img, title="AFD (Subconjuntos)")
    rprint(f"AFD acepta '{w}'? -> [bold]{'sí' if ok_dfa else 'no'}[/bold]. Imagen: {dfa_img}")

    # Min
    mdfa = partitions_minimize(dfa)
    ok_mdfa = simulate_dfa(mdfa, w)
    mdfa_img = f"{out_dir}/afd_min.png"
    draw_dfa(mdfa, mdfa_img, title="AFD Minimizado (partitions)")
    rprint(f"AFD minimizado acepta '{w}'? -> [bold]{'sí' if ok_mdfa else 'no'}[/bold]. Imagen: {mdfa_img}")

    if show_steps:
        rprint("[bold]Pasos shunting yard:[/bold]")
        for p in res.pasos:
            rprint(f" {p.paso:>3}  tok='{p.token}'  {p.accion:>8}  pila='{p.pila}'  salida='{p.salida}'")

    return {"nfa_img": nfa_img, "dfa_img": dfa_img, "dfa_min_img": mdfa_img}

def main():
    ap = argparse.ArgumentParser(description="Proyecto 1: AFN/AFD desde regex")
    ap.add_argument("-r", "--regex", type=str, help="Expresión regular r (infix)")
    ap.add_argument("-w", "--word", type=str, required=True, help="Cadena w a evaluar")
    ap.add_argument("-f", "--file", type=str, help="Archivo con expresiones regulares (una por línea)")
    ap.add_argument("-o", "--outdir", type=str, default="out", help="Directorio de salida para imágenes")
    ap.add_argument("--epsilon", type=str, default="ε", help="Símbolo para epsilon (default: ε)")
    ap.add_argument("--steps", action="store_true", help="Mostrar pasos del shunting yard")
    args = ap.parse_args()

    if not args.regex and not args.file:
        ap.error("Debe proporcionar --regex o --file")

    # preparar salida
    out_base = args.outdir
    os.makedirs(out_base, exist_ok=True)

    if args.regex:
        out_dir = f"{out_base}/single"
        os.makedirs(out_dir, exist_ok=True)
        process_regex(args.regex.strip(), args.word, out_dir, epsilon=args.epsilon, show_steps=args.steps)

    if args.file:
        # procesa cada línea del archivo
        with open(args.file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                rx = (line or "").strip()
                if not rx:
                    continue
                subdir = f"{out_base}/linea_{i}"
                os.makedirs(subdir, exist_ok=True)
                rprint("\n" + "═" * 80)
                rprint(f"[yellow]Línea {i}:[/yellow] {rx}")
                process_regex(rx, args.word, subdir, epsilon=args.epsilon, show_steps=args.steps)

if __name__ == "__main__":
    main()
