---
aliases:
  - "一对多 JOIN 会复制一侧度量并可能导致重复聚合"
  - A one-to-many JOIN repeats one-side measures and can double-count aggregates
  - JOIN 后重复求和
student_os: knowledge-atom
atom_id: CS-SQL-012
atom_type: failure-mode
status: source-checked
mastery_state: unassessed
part_of:
  - "[[关系模型与 SQL 查询.canvas]]"
requires:
  - "[[JOIN 与关系基数]]"
  - "[[关系表粒度]]"
contrasts_with:
  - "[[SQL 重复行与 DISTINCT]]"
---

# 一对多 JOIN 会复制一侧度量并可能导致重复聚合

<!-- bilingual-en:start -->
*A one-to-many JOIN repeats one-side measures and can double-count aggregates*
<!-- bilingual-en:end -->

> [!summary] 原子故障
> 连接把“一”侧行复制到每个匹配的“多”侧行。若随后直接汇总“一”侧度量，该度量会按匹配数重复计算；正确做法取决于度量属于哪个粒度。
>
> <!-- bilingual-en:start -->
> A join repeats the one-side row for every matching many-side row. Summing a one-side measure afterward counts it once per match. The repair depends on the grain to which the measure belongs.
> <!-- bilingual-en:end -->

`orders` 一行一个订单，`items` 一行一个订单商品。订单 10 的 `order_total = 90`，有三个商品行。直接连接再 `SUM(order_total)` 会贡献 270，而不是 90。

```sql
-- 错误：订单总额在商品粒度重复
SELECT SUM(o.order_total)
FROM orders AS o
JOIN items AS i ON i.order_id = o.order_id;
```

如果问题只需要订单总额，根本不要连接 `items`。如果还需要商品条件，应先把商品表压到“一行一个订单”的中间结果，或用 `EXISTS` 只判断是否有符合条件的商品。若度量本来属于商品行，则应汇总商品度量，而不是重复使用订单总额。
<!-- bilingual-en:start -->
If the question needs only order totals, do not join items. If it needs an item-based condition, first reduce items to one row per order or use `EXISTS` to test for a match. If the measure belongs to item grain, aggregate the item measure rather than repeating an order-level total.
<!-- bilingual-en:end -->

`SUM(DISTINCT order_total)` 不是普遍修复：两个不同订单完全可以恰好都是 90，这时它会错误地只算一次。要消除的是重复的事实身份，不是相同的数值。
<!-- bilingual-en:start -->
`SUM(DISTINCT order_total)` is not a general fix: two different orders may legitimately both total 90, and the expression would count that value once. The object to deduplicate is the fact identity, not the numeric value.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 为什么 `SUM(DISTINCT amount)` 不能可靠修复连接倍增？
>
> **答案：** 它按数值去重，不按事实身份去重；不同事实可以有相同金额。应恢复正确粒度或避免不必要的多侧连接。

## 来源与核验

- PostgreSQL, [*Table Expressions: Joined Tables*](https://www.postgresql.org/docs/current/queries-table-expressions.html#QUERIES-JOIN)：核验一行对每个匹配行生成一条连接结果。
- PostgreSQL, [*Aggregate Functions*](https://www.postgresql.org/docs/current/functions-aggregate.html)：核验聚合在进入聚合阶段的输入行上计算；重复计数结论由两者结合推出。
