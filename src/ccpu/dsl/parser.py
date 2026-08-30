"""Small Pratt parser for the extensible ASL-Core surface language."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

ASL_VERSION = "asl-core-v0"
ASL_PROFILE = "asl-arith-v0"


class ASLParseError(ValueError):
    pass


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    offset: int


_TOKEN = re.compile(
    r"(?P<SPACE>\s+)"
    r"|(?P<STRING>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')"
    r"|(?P<NUMBER>(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+))"
    r"|(?P<VARIABLE>\?[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<IDENT>[A-Za-z_][A-Za-z0-9_-]*)"
    r"|(?P<OP>:-|<-|<=|>=|==|!=|[=+*/<>!?,;:\-])"
    r"|(?P<PUNCT>[().\[\]{}])"
)


def _tokenize(text: str) -> list[Token]:
    tokens = []
    offset = 0
    while offset < len(text):
        match = _TOKEN.match(text, offset)
        if match is None:
            raise ASLParseError(f"unexpected character at offset {offset}: {text[offset]!r}")
        kind = match.lastgroup or ""
        if kind != "SPACE":
            tokens.append(Token(kind, match.group(), offset))
        offset = match.end()
    tokens.append(Token("EOF", "", len(text)))
    return tokens


_PRECEDENCE = {
    "=": 1,
    "<-": 1,
    ":-": 1,
    "OR": 2,
    ";": 2,
    "AND": 3,
    ",": 3,
    "==": 4,
    "!=": 4,
    "<": 4,
    "<=": 4,
    ">": 4,
    ">=": 4,
    "+": 5,
    "-": 5,
    "*": 6,
    "/": 6,
}
_STATEMENT_OPERATORS = {"=", "<-", ":-"}


class _Parser:
    def __init__(self, text: str) -> None:
        self.tokens = _tokenize(text)
        self.index = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    def advance(self) -> Token:
        token = self.current
        self.index += 1
        return token

    def accept(self, value: str) -> bool:
        if self.current.value.upper() == value.upper():
            self.advance()
            return True
        return False

    def expect(self, value: str) -> Token:
        if not self.accept(value):
            raise ASLParseError(
                f"expected {value!r} at offset {self.current.offset}, got {self.current.value!r}"
            )
        return self.tokens[self.index - 1]

    def parse_statement(self) -> dict[str, Any]:
        if self.current.kind == "IDENT" and self.current.value.upper() == "RETURN":
            self.advance()
            statement = {"type": "return", "expression": self.parse_expression()}
        elif self.current.value == "?":
            self.advance()
            statement = {"type": "query", "expression": self.parse_expression()}
        else:
            expression = self.parse_expression()
            if (
                expression.get("type") == "binary"
                and expression.get("operator") in _STATEMENT_OPERATORS
            ):
                statement = {
                    "type": "statement",
                    "operator": expression["operator"],
                    "left": expression["left"],
                    "right": expression["right"],
                }
            else:
                statement = {"type": "expression_statement", "expression": expression}
        if self.current.kind != "EOF":
            raise ASLParseError(
                f"unexpected token at offset {self.current.offset}: {self.current.value!r}"
            )
        return statement

    def parse_expression(self, minimum: int = 0) -> dict[str, Any]:
        left = self.parse_prefix()
        while True:
            value = self.current.value.upper()
            if self.current.kind == "IDENT" and value not in {"AND", "OR"}:
                break
            precedence = _PRECEDENCE.get(value)
            if precedence is None or precedence < minimum:
                break
            operator = self.advance().value.upper()
            right = self.parse_expression(
                precedence + (0 if operator in _STATEMENT_OPERATORS else 1)
            )
            left = {"type": "binary", "operator": operator, "left": left, "right": right}
        return left

    def parse_prefix(self) -> dict[str, Any]:
        token = self.advance()
        upper = token.value.upper()
        if token.value in {"-", "!"} or upper == "NOT":
            return {"type": "unary", "operator": upper, "operand": self.parse_expression(7)}
        if token.kind == "NUMBER":
            cleaned = token.value.replace(",", "")
            value: Any = int(cleaned) if "." not in cleaned else cleaned
            return {"type": "number", "value": value}
        if token.kind == "STRING":
            return {"type": "string", "value": ast.literal_eval(token.value)}
        if token.kind == "VARIABLE":
            return {"type": "variable", "name": token.value[1:]}
        if token.kind == "IDENT":
            if upper in {"TRUE", "FALSE"}:
                return {"type": "boolean", "value": upper == "TRUE"}
            if upper == "NULL":
                return {"type": "null", "value": None}
            return self.parse_identifier(token.value)
        if token.value == "(":
            expression = self.parse_expression()
            self.expect(")")
            return expression
        if token.value == "[":
            return self.parse_list()
        if token.value == "{":
            return self.parse_record()
        raise ASLParseError(f"expected expression at offset {token.offset}, got {token.value!r}")

    def parse_identifier(self, name: str) -> dict[str, Any]:
        parts = [name]
        while self.accept("."):
            token = self.advance()
            if token.kind != "IDENT":
                raise ASLParseError(f"expected path component at offset {token.offset}")
            parts.append(token.value)
        node: dict[str, Any] = {
            "type": "identifier" if len(parts) == 1 else "path",
            "name": parts[0],
        }
        if len(parts) > 1:
            node = {"type": "path", "parts": parts}
        if not self.accept("("):
            return node
        arguments = []
        if not self.accept(")"):
            while True:
                if self.current.kind == "IDENT" and self.tokens[self.index + 1].value == "=":
                    argument_name = self.advance().value
                    self.advance()
                    arguments.append(
                        {
                            "type": "named_argument",
                            "name": argument_name,
                            "value": self.parse_expression(4),
                        }
                    )
                else:
                    arguments.append(self.parse_expression(4))
                if self.accept(")"):
                    break
                self.expect(",")
        return {"type": "call", "function": ".".join(parts), "arguments": arguments}

    def parse_list(self) -> dict[str, Any]:
        items = []
        if not self.accept("]"):
            while True:
                items.append(self.parse_expression(4))
                if self.accept("]"):
                    break
                self.expect(",")
        return {"type": "list", "items": items}

    def parse_record(self) -> dict[str, Any]:
        fields = []
        if not self.accept("}"):
            while True:
                key = self.advance()
                if key.kind not in {"IDENT", "STRING"}:
                    raise ASLParseError(f"invalid record key at offset {key.offset}")
                self.expect(":")
                fields.append(
                    {
                        "name": key.value.strip("\"'"),
                        "value": self.parse_expression(4),
                    }
                )
                if self.accept("}"):
                    break
                self.expect(",")
        return {"type": "record", "fields": fields}


def _scope_record(scope: dict[str, Any], statement: dict[str, Any], line: int) -> dict[str, Any]:
    return {"scope": dict(scope), "source_line": line, "statement": statement}


def parse_asl(
    text: str,
    *,
    effective_scope: dict[str, Any] | None = None,
    profile: str = ASL_PROFILE,
) -> dict[str, Any]:
    root = dict(
        effective_scope
        or {"id": "root", "parent": None, "kind": "workspace", "source": "runtime_default"}
    )
    root.setdefault("parent", None)
    root.setdefault("kind", "workspace")
    root.setdefault("source", "runtime_default")
    stack = [root]
    records = []
    scopes = [root]
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.upper().startswith("SCOPE "):
            scope_id = line[6:].strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", scope_id):
                raise ASLParseError(f"invalid scope id on line {line_number}: {scope_id!r}")
            parent = stack[-1]
            scope = {
                "id": scope_id,
                "parent": parent["id"],
                "kind": "explicit",
                "source": "model_asl",
                "path": [*parent.get("path", [parent["id"]]), scope_id],
            }
            records.append(
                _scope_record(parent, {"type": "scope_start", "id": scope_id}, line_number)
            )
            stack.append(scope)
            scopes.append(scope)
            continue
        if line.upper() == "END":
            if len(stack) == 1:
                raise ASLParseError(f"unmatched END on line {line_number}")
            records.append(_scope_record(stack[-1], {"type": "scope_end"}, line_number))
            stack.pop()
            continue
        records.append(_scope_record(stack[-1], _Parser(line).parse_statement(), line_number))
    if len(stack) != 1:
        raise ASLParseError(f"unclosed scope: {stack[-1]['id']}")
    return {
        "asl_version": ASL_VERSION,
        "profile": profile,
        "root_scope": root,
        "scopes": scopes,
        "records": records,
    }
