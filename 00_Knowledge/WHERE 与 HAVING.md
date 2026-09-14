---
aliases:
  - "WHERE 过滤行而 HAVING 过滤分组结果"
  - WHERE filters rows while HAVING filters grouped results
  - WHERE 与 HAVING
student_os: knowledge-atom
atom_id: CS-SQL-014
atom_type: distinction
status: source-checked
mastery_state: unassessed
part_of:
  - "[[关系模型与 SQL 查询.canvas]]"
requires:
  - "[[SELECT 逻辑处理顺序]]"
  - "[[COUNT 与 NULL]]"
leads_to:
  - "[[窗口函数行身份]]"
---

# WHERE 过滤行而 HAVING 过滤分组结果

<!-- bilingual-en:start -->
*WHERE filters rows, while HAVING filters grouped results*
<!-- bilingual-en:end -->

> [!summary] 原子区别
> `WHERE` 在分组前决定哪些输入行参加计算；`HAVING` 在分组和聚合后决定哪些组留在结果中。它们即使写出相似条件，也回答不同层级的问题。
>
> <!-- bilingual-en:start -->
> `WHERE` decides which input rows participate before grouping. `HAVING` decides which groups remain after grouping and aggregation. Similar-looking predicates can therefore answer different questions.
> <!-- bilingual-en:end -->

```sql
SELECT customer_id, SUM(amount) AS paid_total
FROM orders
WHERE status = 'paid'
GROUP BY customer_id
HAVING SUM(amount) >= 1000;
```

这里 `WHERE` 先排除未付款订单，所以 `SUM` 只看到已付款行；`HAVING` 再保留已付款总额至少 1000 的客户。若把订单级条件挪到 `HAVING`，不仅可能语法不合法，也会混淆“哪些行参与汇总”和“哪些汇总结果达标”。
<!-- bilingual-en:start -->
`WHERE` first excludes unpaid orders, so the aggregate sees only paid rows. `HAVING` then retains customers whose paid total reaches 1000. Moving a row-level predicate into `HAVING` may be invalid and, more importantly, confuses which rows enter the aggregate with which aggregate results qualify.
<!-- bilingual-en:end -->

一个实用判断是：条件是否需要先算出 `SUM`、`COUNT`、`AVG` 等组级结果？若需要，通常属于 `HAVING`；若可以对每条原始行直接判断，通常应尽早放在 `WHERE`。但最终仍以问题语义为准，而不是机械看函数名。
<!-- bilingual-en:start -->
A useful test is whether the predicate requires a group result such as `SUM`, `COUNT`, or `AVG`. If so, it usually belongs in `HAVING`; if each raw row can be judged directly, it usually belongs in `WHERE`. The domain question remains authoritative rather than the mere presence of a function name.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> “只统计已付款订单”与“只显示订单数超过 10 的客户”分别放在哪里？
>
> **答案：** 已付款是输入行条件，放 `WHERE`；订单数是分组结果，放 `HAVING COUNT(*) > 10`。

## 来源与核验

- PostgreSQL, [*Table Expressions: GROUP BY and HAVING*](https://www.postgresql.org/docs/current/queries-table-expressions.html#QUERIES-GROUP)：核验 `WHERE` 先过滤输入行，`GROUP BY` 形成组，`HAVING` 再过滤组。
