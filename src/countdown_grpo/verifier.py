"""Safe, exact verification for the locked Countdown arithmetic contract.

Model output is parsed by this module; it is never evaluated as Python. The
parser accepts integer literals, four binary operators, and parentheses. It
uses :class:`fractions.Fraction` while rejecting a binary operation as soon as
it produces a non-integer intermediate value.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

FAILURE_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "ok",
        "no_answer",
        "malformed_answer",
        "syntax_error",
        "unsupported_syntax",
        "wrong_number_multiset",
        "non_integer_intermediate",
        "division_by_zero",
        "wrong_target",
        "truncated",
        "other_invalid",
    }
)
_TOKEN = re.compile(r"\s*(?:(\d+)|([()+*/-]))")
_ANSWER_SPAN = re.compile(r"<answer\s*>(.*?)</answer\s*>", flags=re.IGNORECASE | re.DOTALL)
_ANSWER_MARKER = re.compile(r"</?answer\b", flags=re.IGNORECASE)
_UNSUPPORTED_OPERATOR = re.compile(r"(?:\*\*|//|%|<<|>>|&|\||\^|~|=)")
_EOS_ARTIFACTS: Final[tuple[str, ...]] = ("</s>", "<|endoftext|>", "<|im_end|>")


@dataclass(frozen=True)
class VerificationResult:
    """Verdict for one expression or completion.

    ``reason`` is a stable failure category for experiment JSONL. ``detail``
    is human-readable parser context and is not used for scoring.
    """

    valid: bool
    value: Fraction | None = None
    reason: str = "other_invalid"
    detail: str = ""

    @property
    def failure_category(self) -> str:
        return self.reason


@dataclass(frozen=True)
class ExtractionResult:
    expression: str | None
    reason: str


class _VerificationError(ValueError):
    def __init__(self, category: str, detail: str) -> None:
        if category not in FAILURE_CATEGORIES:
            raise ValueError(f"unknown verification category: {category}")
        super().__init__(detail)
        self.category = category


class _Parser:
    def __init__(
        self,
        expression: str,
        allowed_numbers: list[int] | None,
        *,
        require_integer_intermediates: bool,
    ):
        self.tokens = self._tokenize(expression)
        self.position = 0
        self.used_numbers: list[int] = []
        self.allowed_numbers = (
            Counter(int(number) for number in allowed_numbers) if allowed_numbers is not None else None
        )
        self.require_integer_intermediates = require_integer_intermediates

    @staticmethod
    def _tokenize(expression: str) -> list[str]:
        text = expression.strip()
        if not text:
            raise _VerificationError("no_answer", "empty expression")
        if _UNSUPPORTED_OPERATOR.search(text):
            raise _VerificationError("unsupported_syntax", "unsupported operator")
        tokens: list[str] = []
        cursor = 0
        while cursor < len(text):
            match = _TOKEN.match(text, cursor)
            if not match:
                raise _VerificationError("unsupported_syntax", "expression contains unsupported text")
            tokens.append(match.group(1) or match.group(2))
            cursor = match.end()
        return tokens

    def _peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def _take(self, expected: str | None = None) -> str:
        token = self._peek()
        if token is None or (expected is not None and token != expected):
            raise _VerificationError("syntax_error", f"expected {expected or 'another token'}")
        self.position += 1
        return token

    def _integer(self, value: Fraction) -> Fraction:
        if self.require_integer_intermediates and value.denominator != 1:
            raise _VerificationError("non_integer_intermediate", "a binary operation produced a fraction")
        return value

    def parse(self) -> Fraction:
        value = self._parse_sum()
        if self._peek() is not None:
            raise _VerificationError("syntax_error", "trailing tokens")
        if self.allowed_numbers is not None and Counter(self.used_numbers) != self.allowed_numbers:
            raise _VerificationError(
                "wrong_number_multiset", "each supplied number must be used exactly once"
            )
        return value

    def _parse_sum(self) -> Fraction:
        value = self._parse_product()
        while self._peek() in {"+", "-"}:
            operator = self._take()
            rhs = self._parse_product()
            value = self._integer(value + rhs if operator == "+" else value - rhs)
        return value

    def _parse_product(self) -> Fraction:
        value = self._parse_atom()
        while self._peek() in {"*", "/"}:
            operator = self._take()
            rhs = self._parse_atom()
            if operator == "/":
                if rhs == 0:
                    raise _VerificationError("division_by_zero", "division by zero")
                value = self._integer(value / rhs)
            else:
                value = self._integer(value * rhs)
        return value

    def _parse_atom(self) -> Fraction:
        token = self._peek()
        if token == "(":
            self._take("(")
            value = self._parse_sum()
            self._take(")")
            return value
        if token == "-":
            raise _VerificationError("unsupported_syntax", "unary minus is not allowed")
        if token is None or not token.isdigit():
            raise _VerificationError("syntax_error", "expected a supplied integer")
        number = int(self._take())
        if self.allowed_numbers is not None and self.used_numbers.count(number) >= self.allowed_numbers[number]:
            raise _VerificationError("wrong_number_multiset", "a supplied number was reused")
        self.used_numbers.append(number)
        return Fraction(number)


def _strip_eos_artifacts(text: str) -> str:
    """Remove only known trailing tokenizer EOS renderings."""

    cleaned = text.strip()
    while True:
        lowered = cleaned.lower()
        artifact = next((item for item in _EOS_ARTIFACTS if lowered.endswith(item)), None)
        if artifact is None:
            return cleaned
        cleaned = cleaned[: -len(artifact)].rstrip()


def extract_answer(completion: str) -> ExtractionResult:
    """Select the last well-formed answer span, or a whole tag-free completion.

    Tag-free text is intentionally not reduced to its last line. The full text
    must parse as a legal expression, preventing prose from being rewarded.
    """

    text = _strip_eos_artifacts(str(completion))
    tagged = _ANSWER_SPAN.findall(text)
    if tagged:
        expression = _strip_eos_artifacts(tagged[-1])
        return ExtractionResult(expression or None, "ok" if expression else "no_answer")
    if _ANSWER_MARKER.search(text):
        return ExtractionResult(None, "malformed_answer")
    if not text:
        return ExtractionResult(None, "no_answer")
    return ExtractionResult(text, "ok")


def extract_expression(completion: str) -> str | None:
    """Compatibility helper returning only the selected expression, if any."""

    return extract_answer(completion).expression


def verify_expression(expression: str, target: int, nums: list[int]) -> VerificationResult:
    """Verify an expression against the locked exact Countdown contract."""

    try:
        value = _Parser(expression, nums, require_integer_intermediates=True).parse()
    except _VerificationError as error:
        return VerificationResult(valid=False, reason=error.category, detail=str(error))
    if value != Fraction(int(target)):
        return VerificationResult(
            valid=False,
            value=value,
            reason="wrong_target",
            detail=f"expression evaluates to {value}, not {target}",
        )
    return VerificationResult(valid=True, value=value, reason="ok")


def reaches_target_without_contract(expression: str, target: int) -> bool:
    """Return whether a syntactically legal arithmetic expression reaches ``target``.

    This is an analysis-only diagnostic for expressions that fail the locked
    Countdown contract. It still uses the same parser and exact rational
    arithmetic; it merely ignores the supplied-number multiset and permits
    fractional intermediate values. It must never be used for reward.
    """

    try:
        value = _Parser(
            expression,
            allowed_numbers=None,
            require_integer_intermediates=False,
        ).parse()
    except _VerificationError:
        return False
    return value == Fraction(int(target))


def verify_completion(
    completion: str,
    target: int,
    nums: list[int],
    *,
    truncated: bool = False,
) -> VerificationResult:
    """Extract and verify model output while retaining a stable failure category."""

    if truncated:
        return VerificationResult(valid=False, reason="truncated", detail="generation reached its limit")
    extracted = extract_answer(completion)
    if extracted.expression is None:
        return VerificationResult(valid=False, reason=extracted.reason, detail="no scorable expression")
    return verify_expression(extracted.expression, target, nums)
