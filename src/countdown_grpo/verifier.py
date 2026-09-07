"""A small, auditable arithmetic-expression verifier.

Generated model text is never passed to ``eval``. Expressions are tokenized,
parsed with a recursive-descent parser, and evaluated with exact Fractions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import re


_TOKEN = re.compile(r"\s*(?:(\d+)|([()+*/-]))")


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    value: Fraction | None = None
    reason: str = ""


class _Parser:
    def __init__(self, expression: str, allowed_numbers: list[int]):
        self.tokens = self._tokenize(expression)
        self.position = 0
        self.used_numbers: list[int] = []
        self.allowed_numbers = Counter(allowed_numbers)

    @staticmethod
    def _tokenize(expression: str) -> list[str]:
        tokens: list[str] = []
        cursor = 0
        while cursor < len(expression):
            match = _TOKEN.match(expression, cursor)
            if not match:
                raise ValueError("expression contains unsupported text")
            tokens.append(match.group(1) or match.group(2))
            cursor = match.end()
        return tokens

    def _peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def _take(self, expected: str | None = None) -> str:
        token = self._peek()
        if token is None or (expected is not None and token != expected):
            raise ValueError(f"expected {expected or 'another token'}")
        self.position += 1
        return token

    def parse(self) -> Fraction:
        value = self._parse_sum()
        if self._peek() is not None:
            raise ValueError("trailing tokens")
        if Counter(self.used_numbers) != self.allowed_numbers:
            raise ValueError("each supplied number must be used exactly once")
        return value

    def _parse_sum(self) -> Fraction:
        value = self._parse_product()
        while self._peek() in {"+", "-"}:
            operator = self._take()
            rhs = self._parse_product()
            value = value + rhs if operator == "+" else value - rhs
        return value

    def _parse_product(self) -> Fraction:
        value = self._parse_atom()
        while self._peek() in {"*", "/"}:
            operator = self._take()
            rhs = self._parse_atom()
            if operator == "/":
                if rhs == 0:
                    raise ValueError("division by zero")
                value /= rhs
            else:
                value *= rhs
        return value

    def _parse_atom(self) -> Fraction:
        token = self._peek()
        if token == "(":
            self._take("(")
            value = self._parse_sum()
            self._take(")")
            return value
        if token == "-":
            self._take("-")
            return -self._parse_atom()
        if token is None or not token.isdigit():
            raise ValueError("expected a supplied number")
        number = int(self._take())
        if self.used_numbers.count(number) >= self.allowed_numbers[number]:
            raise ValueError("a supplied number was reused")
        self.used_numbers.append(number)
        return Fraction(number)


def extract_expression(completion: str) -> str:
    """Extract the preferred answer span from a model completion."""

    tagged = re.findall(r"<answer>\s*(.*?)\s*</answer>", completion, flags=re.I | re.S)
    if tagged:
        return tagged[-1].strip()
    lines = [line.strip() for line in completion.splitlines() if line.strip()]
    return lines[-1] if lines else completion.strip()


def verify_expression(expression: str, target: int, nums: list[int]) -> VerificationResult:
    """Return an exact, auditable verdict for one Countdown expression."""

    try:
        value = _Parser(expression, nums).parse()
    except (ValueError, ZeroDivisionError) as error:
        return VerificationResult(valid=False, reason=str(error))
    valid = value == Fraction(target)
    return VerificationResult(
        valid=valid,
        value=value,
        reason="" if valid else f"expression evaluates to {value}, not {target}",
    )
