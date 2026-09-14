---
aliases:
  - "多列 B-tree 的列顺序决定能缩小哪一段索引而非简单决定能否使用"
  - Column order in a multicolumn B-tree determines the bounded index range rather than simple usability
  - PostgreSQL 复合索引列顺序
student_os: knowledge-atom
atom_id: CS-DB-016
atom_type: mechanism
status: source-checked
mastery_state: unassessed
part_of:
  - "[[数据库事务、约束与索引.canvas]]"
---

# 多列 B-tree 的列顺序决定能缩小哪一段索引而非简单决定能否使用

<!-- bilingual-en:start -->
*Column order in a multicolumn B-tree determines the bounded index range rather than simple usability*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> 对 PostgreSQL multicolumn B-tree，leading columns 上的 equality 条件，加上其后第一列的 inequality，最直接决定需要扫描的连续索引范围。后续列条件仍可在索引中检查、减少 heap 访问；当前 PostgreSQL 还可能用 skip scan，因此“没有最左前缀就绝对不能用索引”过于绝对。
>
> <!-- bilingual-en:start -->
> For a PostgreSQL multicolumn B-tree, equality constraints on leading columns plus an inequality on the first following column most directly bound the contiguous range to scan. Conditions on later columns can still be checked in the index and reduce heap visits, and current PostgreSQL may use skip scan. “No leftmost prefix means the index is impossible to use” is therefore too absolute.
> <!-- bilingual-en:end -->

## 自然解释

索引 `(tenant_id, created_at)` 很适合“某 tenant 在一段时间内的记录”：`tenant_id = ?` 先固定一个连续区域，`created_at >= ?` 再切范围。只按 `created_at` 查全体 tenant 时，这个顺序通常不能同样有效地缩小扫描，但 planner 仍可能在 tenant distinct values 很少时选择 skip scan。

<!-- bilingual-en:start -->
An index on `(tenant_id, created_at)` fits “records for one tenant over a time range”: `tenant_id = ?` fixes a contiguous region and `created_at >= ?` narrows it. A query on `created_at` alone usually cannot bound the scan as effectively, although the planner may choose skip scan when there are few distinct tenant values.
<!-- bilingual-en:end -->

列顺序还要结合排序、join、选择性、包含列和写入模式，而不是机械把“最有选择性”列放第一。一个索引服务的是访问形状，不是孤立列排行榜。

<!-- bilingual-en:start -->
Column order also depends on ordering, joins, selectivity, included columns, and write patterns. Do not mechanically place the “most selective” column first. An index serves an access shape, not a ranking of isolated columns.
<!-- bilingual-en:end -->

> [!warning] 边界
> 上述范围规则专指 B-tree。GiST、GIN 与 BRIN 的多列行为不同；即使 B-tree 逻辑可用，[[查询规划器与索引|planner]]也可能选择其他路径。
>
> <!-- bilingual-en:start -->
> This range rule is specific to B-tree. GiST, GIN, and BRIN behave differently for multiple columns. Even when a B-tree is logically usable, the [[查询规划器与索引|planner]] may choose another path.
> <!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 索引 `(a,b)` 面对 `WHERE b=7` 是否一定完全无用？
>
> **答案：** 不一定。它通常不能像约束 `a` 那样直接缩小连续范围，但 PostgreSQL 可能扫描整个索引或用 skip scan；是否划算由数据分布与成本决定。

## 来源与核验

- PostgreSQL 18, [*Multicolumn Indexes*](https://www.postgresql.org/docs/current/indexes-multicolumn.html)：说明 B-tree leading-column 范围规则、后续列过滤与 skip scan 边界。
