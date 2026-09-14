---
aliases:
  - "COUNT(*) 计行而 COUNT(expr) 只计非 NULL 输入"
  - COUNT star counts rows while COUNT expression counts non-null inputs
  - COUNT(*) 与 COUNT(expr)
student_os: knowledge-atom
atom_id: CS-SQL-010
atom_type: distinction
status: source-checked
mastery_state: unassessed
part_of:
  - "[[关系模型与 SQL 查询.canvas]]"
requires:
  - "[[NULL 与 UNKNOWN]]"
leads_to:
  - "[[WHERE 与 HAVING]]"
---

# COUNT(*) 计行而 COUNT(expr) 只计非 NULL 输入

<!-- bilingual-en:start -->
*COUNT(*) counts rows, while COUNT(expr) counts non-null inputs*
<!-- bilingual-en:end -->

> [!summary] 原子区别
> `COUNT(*)` 统计进入聚合阶段的行数；`COUNT(expr)` 只统计表达式结果非 `NULL` 的行数。二者之差可以表示缺失数量，但前提是前面的连接和过滤没有改变目标总体。
>
> <!-- bilingual-en:start -->
> `COUNT(*)` counts rows entering the aggregate, while `COUNT(expr)` counts only rows whose expression is non-null. Their difference can measure missingness, provided earlier joins and filters have not changed the target population.
> <!-- bilingual-en:end -->

例如五名学生中三人的 `score` 已录入：

```sql
SELECT COUNT(*) AS students,
       COUNT(score) AS observed_scores,
       COUNT(*) - COUNT(score) AS missing_scores
FROM exam_results;
```

结果分别是 5、3、2。`COUNT(score)` 不是“成绩总数”的可靠同义词，除非一行确实对应一名目标学生，而且缺失成绩就是你有意排除的情况。
<!-- bilingual-en:start -->
With five student rows and three recorded scores, the three expressions return 5, 3, and 2. `COUNT(score)` is not a safe synonym for “number of students” unless one row truly represents one target student and excluding missing scores is intended.
<!-- bilingual-en:end -->

许多常见的标量聚合（如 `SUM`、`AVG`、`MIN`、`MAX`）会忽略 `NULL` 输入，但这不是所有聚合的共同规律；例如 PostgreSQL 的 `array_agg` 会把 `NULL` 收进数组。另外，`SUM` 等函数在没有可计算输入时通常返回 `NULL`，并不自动返回 0。需要把“没有值”解释为 0 时，应由查询明确写出 `COALESCE`，因为这已经是业务解释，不只是语法细节。
<!-- bilingual-en:start -->
Many familiar scalar aggregates, including `SUM`, `AVG`, `MIN`, and `MAX`, ignore null inputs, but this is not universal; PostgreSQL's `array_agg`, for example, retains nulls. Aggregates such as `SUM` also generally return `NULL`, not zero, when there is no value to compute over. Use `COALESCE` only when the domain deliberately interprets “no value” as zero; that is a business interpretation, not merely syntax.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 一张表有 100 行，其中 `email` 有 12 个 `NULL`。`COUNT(*)` 与 `COUNT(email)` 各是多少？
>
> **答案：** 分别是 100 和 88；前者计行，后者计非空表达式结果。

## 来源与核验

- PostgreSQL, [*Aggregate Functions*](https://www.postgresql.org/docs/current/functions-aggregate.html)：核验 `COUNT(*)`、`COUNT(expr)`、聚合对空值的处理，以及除 `count` 外聚合在无输入行时通常返回 `NULL`。
