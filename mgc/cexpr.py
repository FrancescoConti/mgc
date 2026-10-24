# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""Integer C expressions of `#define`s, evaluated at compile time.

The L1 check (`driver.check_l1`) needs the value of macros written as C expressions, e.g.
`define("(EMBED_SIZE % 256 == 0 && 2 * PRJ_TS * 256 <= PRJ_L1 ? 256 : 128)")` or
`#define L1_SIZE (0x000DFFFF)`. `object_macros` reads the object-like macros of a C file,
`evaluate` computes an expression over them with the C semantics of integer arithmetic
(`/` and `%` truncate toward zero, comparisons and logical operators give 0 or 1, `&&`,
`||` and `?:` only need the operand they select).

Example: `evaluate('A > 2 ? A * 3 : -1', {'A': '(4)'})` is 12; an undefined name, a cast or a
function-like macro makes the value None.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

_TOKEN = re.compile(r'\s*(?:(0[xX][0-9a-fA-F]+|\d+)[uUlL]*|([A-Za-z_]\w*)|(<<|>>|<=|>=|==|!=|&&|\|\||[-+*/%<>!~&|^?:()]))')

# binary operators by precedence, lowest first (the conditional operator is below them all)
_BINARY = [('||',), ('&&',), ('|',), ('^',), ('&',), ('==', '!='), ('<', '<=', '>', '>='), ('<<', '>>'),
           ('+', '-'), ('*', '/', '%')]


def object_macros(text: str) -> Dict[str, str]:
    """`NAME -> body` for each object-like `#define NAME body` of a C file (comments removed,
    line continuations joined); function-like macros (`#define F(x) ...`) are left out."""
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
    text = re.sub(r'\\\n', ' ', text)
    out = {}
    for m in re.finditer(r'^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)(?!\()[ \t]*(.*)$', text, re.M):
        out[m.group(1)] = m.group(2).split('//')[0].strip()
    return out


def _tokens(text: str) -> Optional[List[str]]:
    toks, pos, text = [], 0, text.rstrip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            return None
        num = m.group(1)
        toks.append(str(int(num, 8 if num[0] == '0' and num.isdigit() else 0)) if num else m.group(2) or m.group(3))
        pos = m.end()
    return toks


def _cdiv(a, b):
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


_OPS: Dict[str, Callable[[int, int], Optional[int]]] = {
    '|': lambda a, b: a | b, '^': lambda a, b: a ^ b, '&': lambda a, b: a & b,
    '==': lambda a, b: int(a == b), '!=': lambda a, b: int(a != b), '<': lambda a, b: int(a < b),
    '<=': lambda a, b: int(a <= b), '>': lambda a, b: int(a > b), '>=': lambda a, b: int(a >= b),
    '<<': lambda a, b: a << b if b >= 0 else None, '>>': lambda a, b: a >> b if b >= 0 else None,
    '+': lambda a, b: a + b, '-': lambda a, b: a - b, '*': lambda a, b: a * b,
    '/': lambda a, b: _cdiv(a, b) if b else None, '%': lambda a, b: a - b * _cdiv(a, b) if b else None,
}


class _Parser:
    """Recursive descent over the tokens of one expression; every rule returns the value, or
    None if it cannot be computed (the tokens are consumed either way)."""

    def __init__(self, toks, name_value):
        self.toks, self.i, self.name_value = toks, 0, name_value

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, tok=None):
        t = self.peek()
        if t is None or (tok is not None and t != tok):
            raise SyntaxError(tok or 'operand')
        self.i += 1
        return t

    def cond(self):
        c = self.binary(0)
        if self.peek() != '?':
            return c
        self.take('?')
        a = self.cond()
        self.take(':')
        b = self.cond()
        return None if c is None else a if c else b

    def binary(self, level):
        if level == len(_BINARY):
            return self.unary()
        a = self.binary(level + 1)
        while self.peek() in _BINARY[level]:
            op = self.take()
            b = self.binary(level + 1)
            if op == '&&':
                a = 0 if a == 0 or b == 0 else None if a is None or b is None else 1
            elif op == '||':
                a = 1 if (a is not None and a != 0) or (b is not None and b != 0) else \
                    None if a is None or b is None else 0
            else:
                a = None if a is None or b is None else _OPS[op](a, b)
        return a

    def unary(self):
        t = self.peek()
        if t in ('+', '-', '!', '~'):
            self.take()
            v = self.unary()
            return None if v is None else {'+': v, '-': -v, '!': int(not v), '~': ~v}[t]
        if t == '(':
            self.take()
            v = self.cond()
            self.take(')')
            return v
        t = self.take()
        if t.isdigit():
            return int(t)
        if t[0].isalpha() or t[0] == '_':
            if self.peek() == '(':  # function-like macro or function call
                raise SyntaxError(t)
            return self.name_value(t)
        raise SyntaxError(t)


def evaluate(text: str, macros: Dict[str, str], values: Optional[Dict[str, int]] = None) -> Optional[int]:
    """The value of the C expression `text`, a name being looked up in `values` first, then
    expanded from `macros` (recursively); None if it cannot be computed."""
    values = dict(values or {})
    busy = set()

    def name_value(name):
        if name in values:
            return values[name]
        if name not in macros or name in busy:
            return None
        busy.add(name)
        v = expr(macros[name])
        busy.discard(name)
        values[name] = v
        return v

    def expr(t):
        toks = _tokens(t)
        if not toks:
            return None
        p = _Parser(toks, name_value)
        try:
            v = p.cond()
        except SyntaxError:
            return None
        return v if p.peek() is None else None

    return expr(text)


def undefined(name: str, macros: Dict[str, str], values: Optional[Dict[str, int]] = None) -> set:
    """The names that `name` needs, through `macros`, and that are neither in `values` nor
    macros: `undefined('B', {'B': 'A + C', 'A': '1'})` is {'C'}."""
    values, seen, out = values or {}, set(), set()

    def walk(n):
        if n in values or n in seen:
            return
        seen.add(n)
        if n not in macros:
            out.add(n)
            return
        for t in _tokens(macros[n]) or []:
            if t[0].isalpha() or t[0] == '_':
                walk(t)

    walk(name)
    return out
