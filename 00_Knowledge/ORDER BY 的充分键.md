---
aliases:
  - "没有足够 ORDER BY 键就没有可依赖的结果顺序"
  - Without sufficient ORDER BY keys result order is not reliable
  - SQL 结果顺序必须显式且充分指定
student_os: knowledge-atom
atom_id: CS-SQL-016
atom_type: invariant
status: source-checked
mastery_state: unassessed
part_of:
  - "[[关系模型与 SQL 查询.canvas]]"
requires:
  - "[[SELECT 逻辑处理顺序]]"
---

# 没有足够 ORDER BY 键就没有可依赖的结果顺序

<!-- bilingual-en:start -->
*Without sufficient ORDER BY keys, result order is not reliable*
<!-- bilingual-en:end -->

> [!summary] 原子不变量
> 没有 `ORDER BY`，数据库不承诺返回顺序；只有不够区分并列行的 `ORDER BY`，也只承诺主排序键顺序，不承诺并列行之间的稳定次序。
>
> <!-- bilingual-en:start -->
> Without `ORDER BY`, the database makes no ordering guarantee. An `ORDER BY` that does not distinguish tied rows guarantees only the stated key order, not a stable order among ties.
> <!-- bilingual-en:end -->

索引扫描、顺序扫描、并行计划、连接算法或存储重写都可能改变无排序结果的呈现。某次运行“恰好按主键出来”不是契约，不能作为分页、取首行或对比相邻行的依据。
<!-- bilingual-en:start -->
Index scans, sequential scans, parallel plans, join algorithms, or storage rewrites can all change unsorted output. A run that happens to appear in primary-key order creates no contract for pagination, selecting a first row, or comparing adjacent rows.
<!-- bilingual-en:end -->

子查询或 CTE 内部的查询级 `ORDER BY` 可以决定与它配套的 `LIMIT` 保留哪些行，却不能替代最外层的输出排序。窗口计算需要在 `OVER (...)` 中声明自己的 `ORDER BY`；顺序敏感的聚合则应使用聚合调用内支持的排序语法。它们与查询输出排序是不同的约定，不能从其中一种自动推断另一种。需要这些区分时，可接着读 [[窗口函数行身份]]。
<!-- bilingual-en:start -->
Query-level `ORDER BY` inside a subquery or CTE can determine which rows its accompanying `LIMIT` retains, but it does not replace the outer output ordering. A window specifies its own order inside `OVER (...)`; an order-sensitive aggregate should use the ordering syntax supported inside the aggregate call. These are separate contracts, not guarantees that can be inferred from one another. See [[窗口函数行身份|window functions and row identity]] for the window distinction.
<!-- bilingual-en:end -->

即使写了：

```sql
ORDER BY score DESC
```

同分记录之间仍可任意排列。若需要可重复分页或唯一名次，应再加入稳定的决定键，例如：

```sql
ORDER BY score DESC, submitted_at ASC, submission_id ASC
```

最后一个唯一键把剩余并列消除。这里的目标不是“排序列越多越好”，而是让业务要求的先后关系被完整表达。
<!-- bilingual-en:start -->
Rows tied on `score` may still appear in any order. Reproducible pagination or a unique sequence requires tie-breakers, usually ending in a unique key. The aim is not more columns for their own sake but a complete expression of the required ordering.
<!-- bilingual-en:end -->

PostgreSQL 的 `DISTINCT ON` 还把“每组保留第一行”直接交给排序决定：若 `ORDER BY` 不足以唯一决定组内次序，被留下的那一行就不可预测。这是该扩展的专门规则，也再次说明“第一行”只有在排序契约充分时才有意义。
<!-- bilingual-en:start -->
PostgreSQL's `DISTINCT ON` makes ordering choose the first row retained from each group. If `ORDER BY` does not uniquely determine the within-group order, the surviving row is unpredictable. This extension-specific rule again shows that “first” is meaningful only under a sufficient ordering contract.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> `ORDER BY created_at DESC LIMIT 1` 在多个记录时间戳相同时是否唯一确定一行？
>
> **答案：** 不一定。还要加入能打破并列的稳定键，例如 `id DESC`，才能唯一决定“第一行”。

## 来源与核验

- PostgreSQL, [*Sorting Rows (ORDER BY)*](https://www.postgresql.org/docs/current/queries-order.html)：核验无排序时顺序未指定，以及多排序表达式按先后处理并列。
- PostgreSQL, [*Select Lists: DISTINCT ON*](https://www.postgresql.org/docs/current/queries-select-lists.html#QUERIES-DISTINCT)：核验“第一行”在排序键不能唯一决定次序时仍不可预测。
- PostgreSQL 18, [*Window Functions Tutorial*](https://www.postgresql.org/docs/18/tutorial-window.html) 与 [*Aggregate Expressions*](https://www.postgresql.org/docs/18/sql-expressions.html#SYNTAX-AGGREGATES)：核验 `OVER` 内窗口排序、聚合调用内排序和查询输出排序各自的作用范围。
