"""Bounded public-task adapters for the Paper 2 executable transfer slice."""

from __future__ import annotations

import ast
import calendar
import re
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from typing import Any

from ccpu.common.schema import CoprocessorRequest, DetectionCandidate
from ccpu.paper1.arithmetic import ArithmeticNormalizer, BoundedCalculator

from .logic import HornEngine
from .state import TypedMicroState

_GSM_TRACE = re.compile(r"<<([^=<>]+)=([^<>]+)>>")
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?", re.IGNORECASE)
_MDY = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_MONTHS = {
    name.casefold(): index for index, name in enumerate(calendar.month_name) if name
} | {name.casefold(): index for index, name in enumerate(calendar.month_abbr) if name}


def _fraction(raw: str) -> Fraction:
    return Fraction(raw.strip().replace(",", "").replace("$", ""))


def _gsm8k_adapter(row: dict[str, Any]) -> dict[str, Any]:
    traces = _GSM_TRACE.findall(str(row["raw"]["answer"]))
    if not traces:
        raise ValueError("GSM8K row has no annotated arithmetic trace")
    calculator = BoundedCalculator()
    normalizer = ArithmeticNormalizer()
    calls = []
    for index, (expression, expected) in enumerate(traces):
        cleaned = expression.replace(",", "").replace("$", "").strip()
        candidate = DetectionCandidate(
            candidate_id=f"{row['example_id']}:public:{index}",
            family="compute",
            raw_text=cleaned,
            start_offset=0,
            end_offset=len(cleaned),
            detector="paper2.public.gsm8k_annotation",
        )
        request = normalizer.normalize(candidate)
        result = calculator.execute(request)
        if not result.ok or _fraction(result.display) != _fraction(expected):
            raise ValueError("GSM8K annotated operation failed exact execution")
        calls.append(
            {
                "expression": cleaned,
                "canonical_expression": request.payload["canonical_expression"],
                "result": result.display,
            }
        )
    if _fraction(calls[-1]["result"]) != _fraction(str(row["target"])):
        raise ValueError("GSM8K final annotation is not the benchmark target")
    return {
        "intent": "compute",
        "result": calls[-1]["result"],
        "formalization_source": "annotated_trace",
        "engine": "calculator",
        "operation_count": len(calls),
        "trace": calls,
    }


_PREFIXES = {
    "nano": Decimal("1e-9"),
    "micro": Decimal("1e-6"),
    "milli": Decimal("1e-3"),
    "centi": Decimal("1e-2"),
    "deci": Decimal("1e-1"),
    "deca": Decimal("1e1"),
    "hecto": Decimal("1e2"),
    "kilo": Decimal("1e3"),
    "mega": Decimal("1e6"),
    "giga": Decimal("1e9"),
}
_BASE_UNITS: dict[str, tuple[str, Decimal]] = {
    "meter": ("length", Decimal(1)),
    "inch": ("length", Decimal("0.0254")),
    "foot": ("length", Decimal("0.3048")),
    "yard": ("length", Decimal("0.9144")),
    "mile": ("length", Decimal("1609.344")),
    "gram": ("mass", Decimal(1)),
    "pound": ("mass", Decimal("453.59237")),
    "stone": ("mass", Decimal("6350.29318")),
    "tonne": ("mass", Decimal(1000000)),
    "second": ("time", Decimal(1)),
    "minute": ("time", Decimal(60)),
    "hour": ("time", Decimal(3600)),
    "day": ("time", Decimal(86400)),
    "week": ("time", Decimal(604800)),
    "month": ("time", Decimal(2592000)),
    "year": ("time", Decimal(31536000)),
    "liter": ("volume", Decimal(1)),
    "teaspoon": ("volume", Decimal("0.00492892159375")),
    "tablespoon": ("volume", Decimal("0.01478676478125")),
    "cup": ("volume", Decimal("0.2365882365")),
    "pint": ("volume", Decimal("0.473176473")),
    "quart": ("volume", Decimal("0.946352946")),
    "gallon": ("volume", Decimal("3.785411784")),
    "hertz": ("frequency", Decimal(1)),
    "rpm": ("frequency", Decimal(1) / Decimal(60)),
    "newton": ("force", Decimal(1)),
    "pound-force": ("force", Decimal("4.4482216152605")),
    "pascal": ("pressure", Decimal(1)),
    "torr": ("pressure", Decimal("133.32236842105263")),
    "atm": ("pressure", Decimal(101325)),
    "mole": ("amount", Decimal(1)),
    "kelvin": ("temperature_interval", Decimal(1)),
    "amp": ("current", Decimal(1)),
    "candela": ("luminous_intensity", Decimal(1)),
    "joule": ("energy", Decimal(1)),
    "calorie": ("energy", Decimal("4.184")),
    "are": ("area", Decimal(100)),
    "hectare": ("area", Decimal(10000)),
    "curie": ("radioactivity", Decimal("3.7e10")),
    "rutherford": ("radioactivity", Decimal("1e6")),
}
_UNIT_ALIASES = {
    "m": "meter",
    "cm": "centimeter",
    "mm": "millimeter",
    "km": "kilometer",
    "in": "inch",
    "ft": "foot",
    "yd": "yard",
    "mi": "mile",
    "g": "gram",
    "kg": "kilogram",
    "mg": "milligram",
    "lb": "pound",
    "s": "second",
    "min": "minute",
    "h": "hour",
    "hr": "hour",
    "yr": "year",
    "l": "liter",
    "ml": "milliliter",
    "hl": "hectoliter",
    "tsp": "teaspoon",
    "qt": "quart",
    "hz": "hertz",
    "khz": "kilohertz",
    "n": "newton",
    "cn": "centinewton",
    "hn": "hectonewton",
    "pa": "pascal",
    "j": "joule",
    "cal": "calorie",
    "mol": "mole",
    "mmol": "millimole",
    "a": "amp",
    "cd": "candela",
    "ci": "curie",
    "dag": "decagram",
}


def _singular(value: str) -> str:
    value = value.casefold().strip().rstrip(".")
    if value in {"feet": "foot", "inches": "inch", "years": "year"}:
        return {"feet": "foot", "inches": "inch", "years": "year"}[value]
    if value.endswith("ies"):
        candidate = value[:-3] + "y"
        if candidate in _BASE_UNITS:
            return candidate
    if value.endswith("s") and not value.endswith("ss"):
        value = value[:-1]
    return value


def _unit_factor(raw: str) -> tuple[str, Decimal, str]:
    value = _UNIT_ALIASES.get(raw.casefold().strip().rstrip("."), raw)
    value = _singular(value)
    if value in _BASE_UNITS:
        dimension, factor = _BASE_UNITS[value]
        return dimension, factor, value
    for prefix in sorted(_PREFIXES, key=len, reverse=True):
        if value.startswith(prefix) and value[len(prefix) :] in _BASE_UNITS:
            base = value[len(prefix) :]
            dimension, factor = _BASE_UNITS[base]
            return dimension, factor * _PREFIXES[prefix], value
    raise ValueError(f"unsupported public unit: {raw}")


_UNIT_INPUT = re.compile(
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+"
    r"(?P<source>[A-Za-z-]+).*?units of\s+(?P<target>[A-Za-z-]+)",
    re.IGNORECASE,
)


def _unit_adapter(row: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row["prompt"])
    match = _UNIT_INPUT.search(prompt)
    if match is None or " per " in prompt.casefold() or "^" in prompt:
        raise ValueError("public unit adapter supports scalar linear conversions only")
    source_dim, source_factor, source = _unit_factor(match["source"])
    target_dim, target_factor, target = _unit_factor(match["target"])
    if source_dim != target_dim:
        raise ValueError("public unit dimensions differ")
    with localcontext() as context:
        context.prec = 50
        result = Decimal(match["value"]) * source_factor / target_factor
    expected_match = _NUMBER.search(str(row["target"]))
    if expected_match is None:
        raise ValueError("public unit target has no numeric value")
    expected = Decimal(expected_match.group(0))
    tolerance = max(abs(expected) * Decimal("0.015"), Decimal("1e-12"))
    if abs(result - expected) > tolerance:
        raise ValueError("public unit execution does not match rounded benchmark target")
    return {
        "intent": "compute",
        "result": format(result.normalize(), "f"),
        "formalization_source": "prompt_parser",
        "engine": "units",
        "operation_count": 1,
        "trace": {
            "value": match["value"],
            "source_unit": source,
            "target_unit": target,
            "dimension": source_dim,
        },
    }


def _parse_mdy(match: re.Match[str]) -> date:
    return date(int(match[3]), int(match[1]), int(match[2]))


def _month_date(text: str) -> date | None:
    match = re.search(
        r"\b([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", text
    )
    if match is None or match[1].casefold() not in _MONTHS:
        return None
    return date(int(match[3]), _MONTHS[match[1].casefold()], int(match[2]))


def _shift_months(value: date, months: int) -> date:
    index = value.year * 12 + value.month - 1 + months
    year, month_index = divmod(index, 12)
    month = month_index + 1
    return date(year, month, min(value.day, calendar.monthrange(year, month)[1]))


def _date_base(prompt: str) -> date:
    lowered = prompt.casefold()
    choice = re.search(
        r"jane thinks today is (\d{1,2}/\d{1,2}/\d{4}), but john thinks today is "
        r"(\d{1,2}/\d{1,2}/\d{4})\. (jane|john) is correct",
        lowered,
    )
    if choice:
        selected = choice[1] if choice[3] == "jane" else choice[2]
        return _parse_mdy(_MDY.search(selected))  # type: ignore[arg-type]
    patterns = (
        (r"tomorrow is (\d{1,2}/\d{1,2}/\d{4})", -1),
        (r"yesterday was (\d{1,2}/\d{1,2}/\d{4})", 1),
        (r"day before yesterday was (\d{1,2}/\d{1,2}/\d{4})", 2),
        (r"today(?:,| is)?\s*(\d{1,2}/\d{1,2}/\d{4})", 0),
        (r"it is (\d{1,2}/\d{1,2}/\d{4}) today", 0),
    )
    for pattern, adjustment in patterns:
        match = re.search(pattern, lowered)
        if match:
            parsed = _MDY.search(match[1])
            if parsed:
                return _parse_mdy(parsed) + timedelta(days=adjustment)
    explicit = _month_date(prompt)
    if explicit and "today" in lowered:
        return explicit
    first_year = re.search(r"today is the first day of (\d{4})", lowered)
    if first_year:
        return date(int(first_year[1]), 1, 1)
    last_year = re.search(r"(?:today is|this is) the last day of (\d{4})", lowered)
    if last_year:
        return date(int(last_year[1]), 12, 31)
    quarter = re.search(r"last day of the first quarter of (\d{4})", lowered)
    if quarter:
        return date(int(quarter[1]), 3, 31)
    raise ValueError("unsupported public date reference form")


def _date_query_offset(prompt: str) -> tuple[str, int]:
    question = prompt.casefold().rsplit("what is the date", 1)[-1]
    if "day after tomorrow" in question:
        return "days", 2
    if "day before yesterday" in question:
        return "days", -2
    if "tomorrow" in question or "24 hours later" in question:
        return "days", 1
    if "yesterday" in question:
        return "days", -1
    if "one week ago" in question:
        return "days", -7
    if "one week from today" in question or "one week later" in question:
        return "days", 7
    if "one month ago" in question or "a month ago" in question:
        return "months", -1
    if "one month later" in question or "a month later" in question:
        return "months", 1
    if "one year ago" in question:
        return "months", -12
    if "one year later" in question:
        return "months", 12
    if "today" in question:
        return "days", 0
    raise ValueError("unsupported public date query offset")


def _date_adapter(row: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row["prompt"])
    base = _date_base(prompt)
    unit, offset = _date_query_offset(prompt)
    result = base + timedelta(days=offset) if unit == "days" else _shift_months(base, offset)
    expected_match = _MDY.search(str(row["target"]))
    if expected_match is None or result != _parse_mdy(expected_match):
        raise ValueError("public date execution does not match benchmark target")
    return {
        "intent": "compute",
        "result": result.strftime("%m/%d/%Y"),
        "formalization_source": "prompt_parser",
        "engine": "date_time",
        "operation_count": 1,
        "trace": {"base": base.isoformat(), "offset_unit": unit, "offset": offset},
    }


def _attribute_atom(subject: str, adjective: str, negative: bool = False) -> dict[str, Any]:
    predicate = f"not_{adjective}" if negative else adjective
    argument = "?x" if subject in {"someone", "something", "they", "it"} else subject.casefold()
    return {"predicate": predicate, "arguments": [argument]}


def _attribute_clause(text: str, default_subject: str | None = None) -> tuple[dict[str, Any], str]:
    text = text.strip().casefold()
    match = re.fullmatch(
        r"(someone|something|they|it|[a-z]+)\s+(?:is|are)\s+(not\s+)?([a-z]+)", text
    )
    if match:
        subject = match[1]
        return _attribute_atom(subject, match[3], bool(match[2])), subject
    match = re.fullmatch(r"(not\s+)?([a-z]+)", text)
    if match and default_subject:
        return _attribute_atom(default_subject, match[2], bool(match[1])), default_subject
    raise ValueError(f"unsupported ProofWriter attribute clause: {text}")


def _attribute_rule(sentence: str) -> dict[str, Any]:
    lowered = sentence.strip().casefold()
    if lowered.startswith("if "):
        body_text, head_text = lowered.removeprefix("if ").split(" then ", 1)
        parts = body_text.split(" and ")
        body = []
        subject = None
        for part in parts:
            atom, parsed_subject = _attribute_clause(part, subject)
            subject = subject or parsed_subject
            body.append(atom)
        head, _ = _attribute_clause(head_text, subject)
        return {"head": head, "body": body}
    match = re.fullmatch(
        r"(?:all\s+)?([a-z]+(?:,\s*[a-z]+)*)\s+(?:people|things)\s+are\s+"
        r"(not\s+)?([a-z]+)",
        lowered,
    )
    if not match:
        raise ValueError(f"unsupported ProofWriter attribute rule: {sentence}")
    adjectives = [value.strip() for value in match[1].split(",")]
    return {
        "head": _attribute_atom("something", match[3], bool(match[2])),
        "body": [_attribute_atom("something", adjective) for adjective in adjectives],
    }


def _attribute_fact(sentence: str) -> dict[str, Any]:
    match = re.fullmatch(r"([A-Z][a-z]+) is (not )?([a-z]+)", sentence.strip())
    if not match:
        raise ValueError(f"unsupported ProofWriter attribute fact: {sentence}")
    return _attribute_atom(match[1], match[3], bool(match[2]))


def _run_horn(facts: list[dict[str, Any]], rules: list[dict[str, Any]], query: dict[str, Any]) -> bool:
    request = CoprocessorRequest(
        request_id="paper2:public:proofwriter",
        candidate_id="paper2:public:proofwriter",
        family="reasoning",
        operation="horn.query",
        engine="horn",
        payload={"facts": facts, "rules": rules, "query": query},
    )
    result = HornEngine(TypedMicroState()).execute(request)
    if not result.ok:
        raise ValueError(result.error_message or "ProofWriter Horn execution failed")
    return bool(result.value)


def _proofwriter_adapter(row: dict[str, Any]) -> dict[str, Any]:
    if not str(row["raw"]["id"]).startswith("Att"):
        raise ValueError("public ProofWriter adapter is bounded to attribute theories")
    facts = []
    rules = []
    rule_phase = False
    sentences = [item.strip() for item in str(row["raw"]["theory"]).split(".") if item.strip()]
    for sentence in sentences:
        if sentence.casefold().startswith("if ") or re.search(
            r"\b(?:people|things) are\b", sentence.casefold()
        ):
            rule_phase = True
        if rule_phase:
            rules.append(_attribute_rule(sentence))
        else:
            facts.append(_attribute_fact(sentence))
    query = _attribute_fact(str(row["raw"]["question"]).rstrip("."))
    predicate = str(query["predicate"])
    opposite_predicate = (
        predicate.removeprefix("not_")
        if predicate.startswith("not_")
        else f"not_{predicate}"
    )
    opposite = {**query, "predicate": opposite_predicate}
    positive = _run_horn(facts, rules, query)
    negative = _run_horn(facts, rules, opposite)
    if positive and negative:
        raise ValueError("ProofWriter theory derives both query polarities")
    result = "TRUE" if positive else "FALSE" if negative else "UNCERTAIN"
    if result != str(row["target"]).upper():
        raise ValueError("ProofWriter execution does not match benchmark target")
    return {
        "intent": "verify",
        "result": result,
        "formalization_source": "prompt_parser",
        "engine": "horn",
        "operation_count": len(rules),
        "trace": {"fact_count": len(facts), "rule_count": len(rules)},
    }


def _clutrr_adapter(row: dict[str, Any]) -> dict[str, Any]:
    query = ast.literal_eval(str(row["raw"]["query"]))
    proof_states = ast.literal_eval(str(row["raw"]["proof_state"]))
    matches = []
    for state in proof_states:
        if not isinstance(state, dict):
            continue
        for key, premises in state.items():
            if len(key) == 3 and (key[0], key[2]) == tuple(query):
                matches.append((str(key[1]), premises))
    if not matches or len({match[0] for match in matches}) != 1:
        raise ValueError("CLUTRR annotation has no unique query proof")
    result, premises = matches[-1]
    if result != str(row["target"]).casefold():
        raise ValueError("CLUTRR proof replay does not match benchmark target")
    return {
        "intent": "retrieve",
        "result": result,
        "formalization_source": "annotated_proof_replay",
        "engine": "kinship_graph",
        "operation_count": len(premises),
        "trace": {"query": list(query), "premises": premises},
    }


def registered_assistance(row: dict[str, Any]) -> dict[str, Any]:
    """Build and validate one exact registered result for a materialized row."""

    benchmark = str(row["benchmark"])
    adapters = {
        "gsm8k": _gsm8k_adapter,
        "bigbench_unit_conversion": _unit_adapter,
        "bigbench_date_understanding": _date_adapter,
        "proofwriter_balanced": _proofwriter_adapter,
        "clutrr": _clutrr_adapter,
    }
    try:
        return adapters[benchmark](row)
    except (KeyError, InvalidOperation, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"unsupported public adapter row {row['example_id']}: {error}") from error
