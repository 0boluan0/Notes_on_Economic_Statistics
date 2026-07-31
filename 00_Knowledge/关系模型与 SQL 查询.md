---
aliases:
  - "Relational Model and SQL Queries"
  - "SQL Querying"
  - "SQL查询"
status: source-checked
---

# 关系模型与 SQL 查询
<!-- bilingual-en:start -->
*The Relational Model and SQL Queries*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 用表、键和声明式查询从关系数据中筛选、连接、聚合，得到粒度明确且可解释的结果。
> **具体锚点：** `JOIN` 不是把两张表“贴上”，而是按条件形成匹配行；一对多连接会增加行数并可能重复聚合。
> **核心难点：** SQL 是集合/多重集语义，`NULL` 使用三值逻辑；查询结果正确性取决于粒度和键。
> **为什么重要：** 在分析前弄清一行代表什么，比写出能运行的查询更重要。
> **继续：** 先定义表粒度、主外键，再写 FROM/JOIN/WHERE/GROUP/HAVING/SELECT 的逻辑流程。
> <!-- bilingual-en:start -->
> **Problem addressed:** Use tables, keys, and declarative queries to filter, join, and aggregate relational data; transaction-level integrity continues in [[数据库事务、约束与索引|Database Transactions, Constraints, and Indexes]].
> **Concrete anchor:** A `JOIN` does not merely place two tables side by side; it creates matching rows under a predicate, so a one-to-many join increases row count and can duplicate an aggregate.
> **Central difficulty:** SQL uses set or multiset semantics, while `NULL` participates in three-valued logic; query correctness depends on row grain and keys.
> **Why it matters:** Establishing what one row represents matters more than producing a query that happens to run.
> **Continue with:** Define table grain and primary/foreign keys, then reason through FROM/JOIN/WHERE/GROUP/HAVING/SELECT in logical order.
> <!-- bilingual-en:end -->

> [!source] SQL 核验依据
> - [PostgreSQL SQL Language documentation](https://www.postgresql.org/docs/current/sql.html)：核验查询语义、NULL、连接、聚合与窗口函数。
> <!-- bilingual-en:start -->
> - The [PostgreSQL SQL Language documentation](https://www.postgresql.org/docs/current/sql.html) verifies query semantics, `NULL`, joins, aggregation, and window functions.
> <!-- bilingual-en:end -->

## 关系与键
<!-- bilingual-en:start -->
*Relations and Keys*
<!-- bilingual-en:end -->

表表示同一粒度的关系，列有域。主键唯一标识行，外键引用另一表主键并维护参照完整性。自然键有业务含义，surrogate key 简化引用但不能替代业务唯一约束。
<!-- bilingual-en:start -->
A table represents a relation at one grain, and each column has a domain. A primary key identifies a row uniquely, while a foreign key references another table and preserves referential integrity. A natural key has business meaning; a surrogate key simplifies references but cannot replace a business uniqueness constraint.
<!-- bilingual-en:end -->

查询前写一句“结果中一行代表什么”。若答案从“一个订单”变成“订单中的一件商品”，任何订单级金额都会随明细行重复，除非先聚合回正确粒度。
<!-- bilingual-en:start -->
Before querying, write one sentence stating what a result row represents. If the answer changes from “one order” to “one line item in an order,” every order-level amount is repeated across detail rows unless the data is first aggregated back to the intended grain.
<!-- bilingual-en:end -->

## SELECT 与逻辑顺序
<!-- bilingual-en:start -->
*SELECT and Logical Query Order*
<!-- bilingual-en:end -->

概念顺序是 FROM/JOIN → WHERE → GROUP BY → HAVING → SELECT → ORDER/LIMIT。WHERE 过滤聚合前行，HAVING 过滤组。无 ORDER BY 不保证返回顺序。
<!-- bilingual-en:start -->
The conceptual order is FROM/JOIN → WHERE → GROUP BY → HAVING → SELECT → ORDER/LIMIT. WHERE filters rows before aggregation, while HAVING filters groups. Without ORDER BY, result order is not guaranteed.
<!-- bilingual-en:end -->

## NULL 与三值逻辑
<!-- bilingual-en:start -->
*NULL and Three-Valued Logic*
<!-- bilingual-en:end -->

NULL 表示未知/缺失，不等于 0 或空串；比较用 `IS NULL`。`x = NULL` 得 unknown。聚合函数通常忽略 NULL，`COUNT(*)` 计行、`COUNT(col)` 计非 NULL。
<!-- bilingual-en:start -->
`NULL` represents unknown or missing information and is not zero or an empty string; test it with `IS NULL`. `x = NULL` evaluates to unknown. Aggregate functions usually ignore `NULL`; `COUNT(*)` counts rows, whereas `COUNT(col)` counts non-null values.
<!-- bilingual-en:end -->

在 WHERE 中，false 与 unknown 都不会保留行。因此 `NOT (x = 5)` 仍不会包含 `x IS NULL` 的行；需要时必须把缺失情况写进条件。
<!-- bilingual-en:start -->
In a WHERE clause, neither false nor unknown retains a row. Therefore `NOT (x = 5)` still excludes rows where `x IS NULL`; include the missing case explicitly when the requirement needs it.
<!-- bilingual-en:end -->

## JOIN
<!-- bilingual-en:start -->
*Joins*
<!-- bilingual-en:end -->

INNER 保留匹配，LEFT 保留左表全部。连接条件缺失会笛卡尔积；一对多会复制“一”侧值。聚合前先验证每表和连接后的行粒度、匹配率与重复数。
<!-- bilingual-en:start -->
An INNER join retains matches, while a LEFT join retains every row from the left table. A missing predicate produces a Cartesian product; a one-to-many relationship repeats values from the “one” side. Before aggregation, verify the grain, match rate, and duplicate count of each table and of the joined result.
<!-- bilingual-en:end -->

## 聚合、子查询与窗口
<!-- bilingual-en:start -->
*Aggregation, Subqueries, and Windows*
<!-- bilingual-en:end -->

GROUP BY 折叠为组；窗口函数在不折叠行的情况下计算排名、累计和或分组统计。子查询/CTE 用于分阶段表达，优化器可能内联，不应只为格式制造深层嵌套。
<!-- bilingual-en:start -->
GROUP BY collapses rows into groups. A window function computes ranks, running totals, or partition statistics without collapsing rows. Subqueries and CTEs express stages of reasoning, although an optimizer may inline them; do not create deep nesting solely for formatting.
<!-- bilingual-en:end -->

## Worked example：避免一对多重复求和
<!-- bilingual-en:start -->
*Worked Example: Avoid Double-Counting after a One-to-Many Join*
<!-- bilingual-en:end -->

`orders` 一行一个订单，`items` 一行一个商品。若直接连接后求和 `orders.total`，每个订单金额会按商品数重复。应先在订单粒度聚合明细，或直接在未连接的订单表求订单总额。
<!-- bilingual-en:start -->
`orders` has one row per order, while `items` has one row per line item. Summing `orders.total` after a direct join repeats each order total by its item count. Aggregate detail rows to order grain first, or sum order totals without joining when item data is unnecessary.
<!-- bilingual-en:end -->

```sql
WITH item_totals AS (
    SELECT order_id, SUM(quantity * unit_price) AS item_total
    FROM items
    GROUP BY order_id
)
SELECT o.customer_id, SUM(i.item_total) AS revenue
FROM orders AS o
JOIN item_totals AS i ON i.order_id = o.order_id
GROUP BY o.customer_id;
```

## 失败诊断
<!-- bilingual-en:start -->
*Failure Diagnosis*
<!-- bilingual-en:end -->

- 数值突然放大：比较连接前后行数，并按候选键统计重复数。
  <!-- bilingual-en:start -->
  Values suddenly inflate: compare row counts before and after the join and count duplicates by the candidate key.
  <!-- bilingual-en:end -->
- LEFT JOIN 后少行：检查右表条件是否写进 WHERE，从而把 unmatched 的 `NULL` 行重新过滤掉。
  <!-- bilingual-en:start -->
  Rows disappear after a LEFT JOIN: check whether a condition on the right table was placed in WHERE, thereby filtering unmatched `NULL` rows.
  <!-- bilingual-en:end -->
- `NOT IN` 意外为空：子查询若含 `NULL` 会传播 unknown；明确排除 `NULL` 或改用相关 `NOT EXISTS`。
  <!-- bilingual-en:start -->
  `NOT IN` unexpectedly returns nothing: a `NULL` in its subquery propagates unknown; exclude nulls explicitly or use a correlated `NOT EXISTS`.
  <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### 一对多 JOIN 后为什么总金额可能被放大？
<!-- bilingual-en:start -->
*Why can a total be inflated after a one-to-many JOIN?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 一侧金额会随多侧匹配行重复，若直接求和会重复计；应先按正确粒度聚合或明确分配。
> <!-- bilingual-en:start -->
> The value on the “one” side is repeated for each matching row on the “many” side, so direct summation double-counts it; aggregate to the intended grain first or define an explicit allocation.
> <!-- bilingual-en:end -->

### `COUNT(*)` 与 `COUNT(col)` 有何差别？
<!-- bilingual-en:start -->
*How do `COUNT(*)` and `COUNT(col)` differ?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 前者计所有行，后者只计该列非 NULL 的行。
> <!-- bilingual-en:start -->
> The former counts every row; the latter counts only rows where that column is not `NULL`.
> <!-- bilingual-en:end -->

### 为什么无 `ORDER BY` 的结果不能当成稳定顺序？
<!-- bilingual-en:start -->
*Why must a result without `ORDER BY` not be treated as stably ordered?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 关系本身没有顺序，执行计划、并行与存储布局都可改变返回次序；只有显式排序建立契约。
> <!-- bilingual-en:start -->
> A relation is unordered, and plan choice, parallelism, or storage layout can alter output order; only an explicit ordering creates that contract.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- [PostgreSQL SQL Language documentation](https://www.postgresql.org/docs/current/sql.html)：核验 SELECT 处理、数据定义、函数与类型语义。
  <!-- bilingual-en:start -->
  The [PostgreSQL SQL Language documentation](https://www.postgresql.org/docs/current/sql.html) verifies SELECT processing, data definition, functions, and type semantics.
  <!-- bilingual-en:end -->
