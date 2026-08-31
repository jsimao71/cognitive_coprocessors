# Functor v1 Grammar

Every line is one allowlisted function call with positional arguments. Paths are
quoted lowercase dotted strings. Literals are unquoted numbers.

## F1: Isomorphic Low-Level Control

```text
value(target,number)                  copy(target,source)
add(target,a,b,...)                   subtract(target,a,b,...)
multiply(target,a,b,...)              divide(target,a,b)
absolute(target,a)                    sum_values(target,a,...)
mean_values(target,a,...)             minimum(target,a,...)
maximum(target,a,...)                 percent_of(target,base,pct)
increase_percent(target,base,pct)     decrease_percent(target,base,pct)
rate_times_duration(target,rate,duration)
query(path)
```

F1 calls are deliberately flat so F1 and F2 have the same one-call-per-relation
surface shape. F1 names arithmetic operations; F2 names source-level relations.

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
value("mira.cards", 12)
add("jon.cards", "mira.cards", 5)
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
value("crate.rows", 8)
value("crate.jars_per_row", 6)
multiply("crate.jars_total", "crate.rows", "crate.jars_per_row")
query("crate.jars_total")
```

F2:

```text
given("crate.rows", 8)
given("crate.jars_per_row", 6)
per_unit_total("crate.jars_total", "crate.rows", "crate.jars_per_row")
query("crate.jars_total")
```

Input: `A club has 60 junior members out of 240 members. What percentage are juniors?`

F1:

```text
value("club.junior_members", 60)
value("club.total_members", 240)
divide("club.junior_fraction", "club.junior_members", "club.total_members")
multiply("club.junior_percentage", "club.junior_fraction", 100)
query("club.junior_percentage")
```

F2:

```text
given("club.junior_members", 60)
given("club.total_members", 240)
percentage_ratio("club.junior_percentage", "club.junior_members", "club.total_members")
query("club.junior_percentage")
```
