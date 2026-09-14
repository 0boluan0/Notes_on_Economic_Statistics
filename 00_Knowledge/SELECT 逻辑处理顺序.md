---
aliases:
  - "SELECT 的书写顺序不等于查询的逻辑处理顺序"
  - SELECT syntax order differs from logical query processing order
  - SQL 逻辑查询处理顺序
student_os: knowledge-atom
atom_id: CS-SQL-007
atom_type: model
status: source-checked
mastery_state: unassessed
part_of:
  - "[[关系模型与 SQL 查询.canvas]]"
requires:
  - "[[关系模型结构]]"
leads_to:
  - "[[WHERE 与三值逻辑]]"
  - "[[WHERE 与 HAVING]]"
  - "[[窗口函数行身份]]"
  - "[[ORDER BY 的充分键]]"
---

# SELECT 的书写顺序不等于查询的逻辑处理顺序

<!-- bilingual-en:start -->
*SELECT syntax order differs from logical query processing order*
<!-- bilingual-en:end -->

> [!summary] 原子模型
> 查询虽然从 `SELECT` 开始写，理解名称可见性和行数变化时却应从数据来源开始：`FROM/JOIN → WHERE → GROUP BY/聚合 → HAVING → SELECT 输出（其中窗口函数在前述阶段之后计算）→ DISTINCT → ORDER BY → LIMIT`。
>
> <!-- bilingual-en:start -->
> Although a query is written beginning with `SELECT`, reason about name visibility and row-count changes from the data source forward: `FROM/JOIN → WHERE → GROUP BY/aggregation → HAVING → SELECT output (with window functions evaluated after those earlier stages) → DISTINCT → ORDER BY → LIMIT`.
> <!-- bilingual-en:end -->

这条顺序是语义模型，不是数据库必须照着逐步执行的物理计划。窗口函数只允许出现在 `SELECT` 列表或 `ORDER BY` 中，所以不应把它和 `SELECT` 输出误画成两个彼此独立的通用子句；准确边界是它看不到被 `WHERE`、分组或 `HAVING` 排除的行。优化器可以在保持结果等价的前提下改写、合并或提前执行操作，但用户仍应按逻辑阶段判断名称何时可见、某一过滤发生在聚合前还是聚合后。
<!-- bilingual-en:start -->
This order is a semantic model, not a physical execution recipe. Window functions are permitted in the `SELECT` list or `ORDER BY`, so they should not be drawn as a separate general-purpose clause before `SELECT`; the precise boundary is that they see only rows remaining after `WHERE`, grouping, and `HAVING`. An optimizer may rewrite, merge, or move operations while preserving the result, while the logical stages still govern name visibility and filtering semantics.
<!-- bilingual-en:end -->

例如：

```sql
SELECT customer_id, SUM(amount) AS total
FROM orders
WHERE status = 'paid'
GROUP BY customer_id
HAVING SUM(amount) > 1000
ORDER BY total DESC;
```

先由 `FROM` 提供订单，再由 `WHERE` 移除未付款行；剩余行按客户分组并求和，`HAVING` 删除总额不超过 1000 的组，`SELECT` 才形成输出列 `total`，最后 `ORDER BY` 使用这个输出名排序。把语法从上往下当作执行顺序，会误解为什么 `WHERE total > 1000` 通常不可用。
<!-- bilingual-en:start -->
`FROM` first supplies orders, then `WHERE` removes unpaid rows. The remaining rows are grouped and summed, `HAVING` removes groups whose total is at or below 1000, `SELECT` forms the output column `total`, and `ORDER BY` can finally use that output label. Reading syntax top-to-bottom as execution order obscures why `WHERE total > 1000` is generally invalid.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 为什么 `ORDER BY` 常能引用 `SELECT` 中刚定义的别名，而 `WHERE` 不能？
>
> **答案：** 在逻辑处理模型中，`WHERE` 早于输出列表，别名尚未形成；`ORDER BY` 晚于输出列表，可以使用输出名。

## 来源与核验

- PostgreSQL, [*SELECT*](https://www.postgresql.org/docs/current/sql-select.html)：核验 `FROM`、`WHERE`、分组、`HAVING`、输出、去重、排序与限制的处理阶段。
- PostgreSQL, [*Window Functions Tutorial*](https://www.postgresql.org/docs/current/tutorial-window.html)：核验窗口函数在 `WHERE`、`GROUP BY`、`HAVING` 和普通聚合之后处理。
