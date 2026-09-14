---
aliases:
  - "WHERE 只保留 TRUE 而 FALSE 与 UNKNOWN 都被过滤"
  - WHERE retains only TRUE and filters both FALSE and UNKNOWN
  - WHERE 的三值过滤语义
student_os: knowledge-atom
atom_id: CS-SQL-008
atom_type: rule
status: source-checked
mastery_state: unassessed
part_of:
  - "[[关系模型与 SQL 查询.canvas]]"
requires:
  - "[[SELECT 逻辑处理顺序]]"
  - "[[NULL 与 UNKNOWN]]"
leads_to:
  - "[[LEFT JOIN 条件位置]]"
  - "[[NOT IN 与 NOT EXISTS]]"
---

# WHERE 只保留 TRUE 而 FALSE 与 UNKNOWN 都被过滤

<!-- bilingual-en:start -->
*WHERE retains only TRUE and filters both FALSE and UNKNOWN*
<!-- bilingual-en:end -->

> [!summary] 原子规则
> `WHERE` 对每一行计算布尔条件，只保留结果为 `TRUE` 的行。`FALSE` 和 `UNKNOWN` 都不会进入下一阶段。
>
> <!-- bilingual-en:start -->
> `WHERE` evaluates a Boolean condition for each row and retains only rows for which the result is `TRUE`. Both `FALSE` and `UNKNOWN` are discarded.
> <!-- bilingual-en:end -->

这解释了一个常见误会。若 `score` 为 `NULL`，表达式 `score = 5` 是 `UNKNOWN`；它的否定 `NOT (score = 5)` 仍是 `UNKNOWN`，不是 `TRUE`。所以：

```sql
WHERE score <> 5
```

不会保留缺失分数。若业务问题是“分数不是 5，或分数尚未知”，必须明确写：

```sql
WHERE score <> 5 OR score IS NULL
```
<!-- bilingual-en:start -->
If `score` is `NULL`, `score = 5` is `UNKNOWN`; its negation remains `UNKNOWN`, not `TRUE`. Therefore `WHERE score <> 5` does not retain missing scores. If the requirement is “not 5, or not yet known,” the null case must be stated explicitly.
<!-- bilingual-en:end -->

“未保留”不等于“条件为假”。查询结果只让我们知道该行没有通过 `TRUE` 检查，不能据此区分它是明确不符合，还是因为输入缺失而无法判断。
<!-- bilingual-en:start -->
“Not retained” does not necessarily mean “the predicate was false.” The result only tells us that the row failed the `TRUE` test; it does not distinguish definite failure from an indeterminate comparison caused by missing input.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> `WHERE NOT (x = 5)` 会不会保留 `x IS NULL` 的行？
>
> **答案：** 不会。`x = 5` 为 `UNKNOWN`，`NOT UNKNOWN` 仍为 `UNKNOWN`，而 `WHERE` 只保留 `TRUE`。

## 来源与核验

- PostgreSQL, [*Table Expressions: WHERE*](https://www.postgresql.org/docs/current/queries-table-expressions.html#QUERIES-WHERE)：核验 `WHERE` 保留真值为真的行，并丢弃假或空值结果。
- PostgreSQL, [*Logical Operators*](https://www.postgresql.org/docs/current/functions-logical.html)：核验 SQL 的 `TRUE`、`FALSE`、`UNKNOWN` 三值逻辑及其否定规则。
