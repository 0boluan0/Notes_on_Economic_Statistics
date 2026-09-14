---
aliases:
  - "LEFT JOIN 的右表条件放在 ON 与 WHERE 中会改变保留对象"
  - A right-side predicate in ON versus WHERE changes what a LEFT JOIN preserves
  - LEFT JOIN 中 ON 与 WHERE 的区别
student_os: knowledge-atom
atom_id: CS-SQL-013
atom_type: distinction
status: source-checked
mastery_state: unassessed
part_of:
  - "[[关系模型与 SQL 查询.canvas]]"
requires:
  - "[[JOIN 与关系基数]]"
  - "[[WHERE 与三值逻辑]]"
---

# LEFT JOIN 的右表条件放在 ON 与 WHERE 中会改变保留对象

<!-- bilingual-en:start -->
*A right-side predicate in ON versus WHERE changes what a LEFT JOIN preserves*
<!-- bilingual-en:end -->

> [!summary] 原子区别
> 对 `LEFT JOIN` 而言，右表条件写在 `ON` 中会限制“哪些右行可以匹配”，但仍保留所有左行；同一条件写在 `WHERE` 中会在连接后过滤，常把没有匹配的左行也删掉。
>
> <!-- bilingual-en:start -->
> In a `LEFT JOIN`, a right-side predicate in `ON` limits which right rows may match while preserving every left row. The same predicate in `WHERE` filters after the join and commonly removes unmatched left rows as well.
> <!-- bilingual-en:end -->

要列出每位客户及其已付款订单，包括没有已付款订单的客户，应写：

```sql
SELECT c.customer_id, o.order_id
FROM customers AS c
LEFT JOIN orders AS o
  ON o.customer_id = c.customer_id
 AND o.status = 'paid';
```

没有已付款订单的客户仍出现一行，右侧列为 `NULL`。若把状态条件移到 `WHERE o.status = 'paid'`，这些补出的空值行会得到 `UNKNOWN` 并被过滤，结果在该问题上退化成内连接效果。
<!-- bilingual-en:start -->
Customers without paid orders still appear with null-extended right columns. Moving the status predicate to `WHERE` makes those rows evaluate to `UNKNOWN` and removes them, producing inner-join behavior for this question.
<!-- bilingual-en:end -->

这不是“条件放哪里都一样”的格式选择。先说清保留总体：是在所有客户中附加符合条件的订单，还是只要拥有符合条件订单的客户？前者把条件放进连接匹配，后者可以在连接后过滤，甚至直接使用内连接。也要注意，这里“会删掉无匹配行”针对 `o.status = 'paid'` 这类对补出 `NULL` 不会返回 `TRUE` 的条件；`WHERE o.order_id IS NULL` 反而常被有意用来只找无匹配对象。
<!-- bilingual-en:start -->
Predicate placement is not merely formatting. First identify the preserved population: all customers with qualifying orders attached, or only customers who have a qualifying order? The first constrains matching; the second may filter afterward or use an inner join directly. The loss of unmatched rows applies to predicates such as `o.status = 'paid'` that are not true on the null-extended row; `WHERE o.order_id IS NULL`, by contrast, is often used deliberately to retain only unmatched objects.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 为什么 `LEFT JOIN ... WHERE right_col = 1` 常会丢掉无匹配的左行？
>
> **答案：** 无匹配左行的右列被补为 `NULL`，比较结果是 `UNKNOWN`；`WHERE` 只保留 `TRUE`。

## 来源与核验

- PostgreSQL, [*Table Expressions: Joined Tables*](https://www.postgresql.org/docs/current/queries-table-expressions.html#QUERIES-JOIN)：核验外连接补空行，以及同一右表限制置于 `ON` 与 `WHERE` 时的不同结果。
