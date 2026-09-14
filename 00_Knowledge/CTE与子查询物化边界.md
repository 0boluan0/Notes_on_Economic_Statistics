---
aliases:
  - "CTE 和子查询表达阶段但不自动保证物化"
  - CTEs and subqueries express stages but do not automatically guarantee materialization
  - CTE 与子查询的物化边界
student_os: knowledge-atom
atom_id: CS-SQL-018
atom_type: boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[关系模型与 SQL 查询.canvas]]"
requires:
  - "[[SELECT 逻辑处理顺序]]"
  - "[[关系表粒度]]"
leads_to:
  - "[[一对多 JOIN 重复聚合]]"
---

# CTE 和子查询表达阶段但不自动保证物化

<!-- bilingual-en:start -->
*CTEs and subqueries express stages but do not automatically guarantee materialization*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 子查询和 `WITH` CTE 能给中间关系命名、固定每一步的列与粒度，让复杂查询按语义分段。它们不天然意味着数据库一定会“先完整写出一张临时表”。
>
> <!-- bilingual-en:start -->
> Subqueries and `WITH` CTEs can name intermediate relations and make each stage's columns and grain explicit. They do not inherently mean that the database must “fully write a temporary table first.”
> <!-- bilingual-en:end -->

下面的 CTE 有明确价值：它先把商品明细恢复到订单粒度，再与订单连接。

```sql
WITH item_totals AS (
  SELECT order_id, SUM(quantity * unit_price) AS item_total
  FROM items
  GROUP BY order_id
)
SELECT o.customer_id, SUM(i.item_total) AS revenue
FROM orders AS o
JOIN item_totals AS i USING (order_id)
GROUP BY o.customer_id;
```

这里 CTE 的核心作用是表达粒度边界，不是强迫某种执行方式。以 PostgreSQL 为例，一个非递归、无副作用的 `SELECT` CTE 在只被引用一次时默认会折叠进外层查询；被引用多次时默认不会折叠，并通常只计算一次。`MATERIALIZED` 与 `NOT MATERIALIZED` 可以改变这一选择，但后者可能导致重复计算。含副作用或易变函数的 CTE 还有不同边界；其他数据库也应查看各自规则。
<!-- bilingual-en:start -->
The CTE's main role here is to express a grain boundary, not to force an execution strategy. In PostgreSQL, a non-recursive, side-effect-free `SELECT` CTE referenced once is folded into the parent by default; when referenced more than once, it is not folded by default and is normally computed only once. `MATERIALIZED` and `NOT MATERIALIZED` can override that choice, although the latter can duplicate computation. Side effects and volatile functions add further boundaries, and other systems have their own rules.
<!-- bilingual-en:end -->

分层的理由应是给一个有意义的中间关系命名，例如“每个订单一行的商品合计”。如果新的一层既不澄清粒度、列或复用边界，也不服务必要运算，只是把同一查询再包一层，它通常增加的是阅读负担。
<!-- bilingual-en:start -->
A stage should name a meaningful intermediate relation, such as “one item total per order.” If a new layer clarifies neither grain, columns, reuse, nor a required operation and merely wraps the same query again, it usually adds reading cost rather than structure.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 把一段查询放进 CTE，是否就保证数据库先把它完整算成一张中间表？
>
> **答案：** 不保证。物化取决于查询性质、引用次数、数据库实现与计划选择；CTE 首先是表达中间关系和阶段的工具。

## 来源与核验

- PostgreSQL, [*WITH Queries: CTE Materialization*](https://www.postgresql.org/docs/current/queries-with.html#QUERIES-WITH-CTE-MATERIALIZATION)：核验 CTE 可被折叠或物化及 `MATERIALIZED` / `NOT MATERIALIZED` 的边界。
