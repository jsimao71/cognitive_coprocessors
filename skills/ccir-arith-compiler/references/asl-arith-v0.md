# ASL-Arith v0 Reference

## Surface

ASL is a compact model-facing language. JSON is only the API envelope and the
canonical parsed representation.

```text
name = expression
path.to.value = expression
RETURN expression
```

Expressions support numbers, strings, paths, parentheses, unary minus, infix
`+ - * /`, comparisons, and registered calls. Operator precedence is ordinary
arithmetic precedence.

Preferred arithmetic calls:

- `percent_of(value, percent)` computes a percentage of a value.
- `inc_pct(value, percent)` increases a value by a percentage.
- `dec_pct(value, percent)` decreases a value by a percentage.
- `rate_times_duration(rate, duration)` multiplies a rate by a duration.
- `sum(...)`, `mean(...)`, `min(...)`, `max(...)`, and `abs(...)` aggregate values.

Function and predicate syntax is grammar-open, but this skill may emit only the
operators listed in the request's active registry.

## State And References

Assignments create or replace a value in the current effective scope. A read
path must already be present in `state_before` or be established earlier in the
same response. Use stable semantic names based on the clause, not temporary
reasoning labels such as `tmp1` when a meaningful name is available.

The dataset supplies one root scope per example. Do not emit `SCOPE/END` merely
because a problem has multiple sentences. Explicit scopes are reserved for
meaningful nested subproblems not already represented by metadata.

## Examples

Source fact:

```text
month1.downloads = 60
```

Derived relation:

```text
month2.downloads = month1.downloads * 3
```

Percentage decrease:

```text
month3.downloads = dec_pct(month2.downloads, 30)
```

Final query:

```text
total = month1.downloads + month2.downloads + month3.downloads
RETURN total
```

Bad: `month2.downloads = 180` when the clause says it is three times month 1.
This erases the semantic relation and may leak a solved value.

Bad: `calculator(month1.downloads * 3)`. ASL names semantics; the runtime chooses
the concrete coprocessor.

Bad: reading `month1.downloads` when it is absent from `state_before`. Return
`ambiguous` instead of inventing the missing fact.

## Ambiguity

Pronouns or phrases such as "that amount" may be resolved only when exactly one
compatible state value is evident. Record a concise assumption when resolution
is safe but not explicit. Otherwise return `ambiguous`.
