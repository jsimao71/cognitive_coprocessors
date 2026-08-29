"""Bounded deterministic ISO date arithmetic."""

from __future__ import annotations

import re
from datetime import date, timedelta

from ccpu.common.schema import CoprocessorRequest, CoprocessorResult

_ADD = re.compile(r"^add\s+(\d{4}-\d{2}-\d{2})\s+([+-]?P\d+D)$", re.IGNORECASE)
_DIFF = re.compile(
    r"^diff\s+(\d{4}-\d{2}-\d{2})\s+(\d{4}-\d{2}-\d{2})$", re.IGNORECASE
)


def normalize_date_payload(text: str) -> tuple[str, dict[str, str | int]]:
    """Parse the intentionally small date language into typed payload data."""

    match = _ADD.fullmatch(text.strip())
    if match:
        base, duration = match.groups()
        sign = -1 if duration.startswith("-P") else 1
        days = int(duration.lstrip("+").removeprefix("P").removesuffix("D")) * sign
        if abs(days) > 36600:
            raise ValueError("date duration exceeds 100-year budget")
        date.fromisoformat(base)
        return "date.add_days", {"date": base, "days": days}
    match = _DIFF.fullmatch(text.strip())
    if match:
        left, right = match.groups()
        date.fromisoformat(left)
        date.fromisoformat(right)
        return "date.diff_days", {"left": left, "right": right}
    raise ValueError("date block must be 'add YYYY-MM-DD PnD' or 'diff DATE DATE'")


class DateTimeEngine:
    name = "date_time"

    def execute(self, request: CoprocessorRequest) -> CoprocessorResult:
        try:
            if request.operation == "date.add_days":
                value = date.fromisoformat(str(request.payload["date"])) + timedelta(
                    days=int(request.payload["days"])
                )
                display = value.isoformat()
            elif request.operation == "date.diff_days":
                value = date.fromisoformat(str(request.payload["right"])) - date.fromisoformat(
                    str(request.payload["left"])
                )
                display = str(value.days)
            else:
                raise ValueError(f"unsupported operation: {request.operation}")
            return CoprocessorResult(
                request_id=request.request_id,
                engine=self.name,
                ok=True,
                value=display,
                display=display,
                metadata={"exact": True},
            )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            return CoprocessorResult(
                request_id=request.request_id,
                engine=self.name,
                ok=False,
                error_code="invalid_or_bounded_ir",
                error_message=str(error),
            )
