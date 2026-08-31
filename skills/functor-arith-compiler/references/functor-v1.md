# Functor v1 Grammar

Every line is one allowlisted function call with positional arguments. Paths are
quoted lowercase dotted strings. Literals are unquoted numbers.

## F1: Isomorphic Low-Level Control

Top-level calls:

```text
set(path, expression)
query(path)
```

Expression calls:

```text
const(number)  ref(path)
add(a,b,...)   sub(a,b)   mul(a,b,...)   div(a,b)
abs(a)         sum(a,...) mean(a,...)    min(a,...) max(a,...)
percent_of(base,pct)
increase_percent(base,pct)
decrease_percent(base,pct)
rate_times_duration(rate,duration)
```

## F2: Semantic Relations

```text
given(target,value)                 same(target,source)
offset(target,base,signed_delta)    difference(target,left,right)
absolute_difference(target,left,right)
sum_of(target,a,b,...)              product_of(target,a,b,...)
quotient(target,numerator,denominator)
multiple(target,base,factor)        fraction_of(target,base,numerator,denominator)
percent_of(target,base,pct)         increase_percent(target,base,pct)
percentage_ratio(target,part,whole)
decrease_percent(target,base,pct)   rate_total(target,rate,duration)
per_unit_total(target,count,each)   remaining(target,whole,used,...)
mean_of(target,a,b,...)             minimum_of(target,a,b,...)
maximum_of(target,a,b,...)          query(path)
```

Arguments after `target` may be semantic paths or numeric source literals. Prefer
paths for explicitly named source facts so the blackboard retains provenance.
Use the most specific relation available: `percentage_ratio` for “A as a percentage
of B,” `percent_of` for “P percent of B,” and `remaining` rather than a generic
difference when the source describes consumption or removal.

## Fixed ICL Set

These examples are fixed across all records.

Input: `Mira has 12 cards. Jon has 5 more cards than Mira. How many cards does Jon have?`

F1:

```text
set("mira.cards", const(12))
set("jon.cards", add(ref("mira.cards"), const(5)))
query("jon.cards")
```

F2:

```text
given("mira.cards", 12)
offset("jon.cards", "mira.cards", 5)
query("jon.cards")
```

Input: `A crate has 8 rows with 6 jars in each row. How many jars are there?`

F1:

```text
set("crate.rows", const(8))
set("crate.jars_per_row", const(6))
set("crate.jars_total", mul(ref("crate.rows"), ref("crate.jars_per_row")))
query("crate.jars_total")
```

F2:

```text
given("crate.rows", 8)
given("crate.jars_per_row", 6)
per_unit_total("crate.jars_total", "crate.rows", "crate.jars_per_row")
query("crate.jars_total")
```
