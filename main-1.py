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
