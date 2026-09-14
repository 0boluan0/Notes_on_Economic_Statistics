---
aliases:
  - "EXPLAIN 显示估算而 EXPLAIN ANALYZE 会执行查询并给出实际运行证据"
  - EXPLAIN shows estimates while EXPLAIN ANALYZE executes the query and reports runtime evidence
  - PostgreSQL EXPLAIN ANALYZE
student_os: knowledge-atom
atom_id: CS-DB-018
atom_type: diagnostic-procedure
status: source-checked
mastery_state: unassessed
part_of:
  - "[[数据库事务、约束与索引.canvas]]"
---

# EXPLAIN 显示估算而 EXPLAIN ANALYZE 会执行查询并给出实际运行证据

<!-- bilingual-en:start -->
*EXPLAIN shows estimates, while EXPLAIN ANALYZE executes the query and reports runtime evidence*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> `EXPLAIN` 展示 planner 选择的 plan、estimated rows 与 cost，但不执行查询。`EXPLAIN ANALYZE` 实际运行查询并加入 actual time、rows、loops 等证据；配合 `BUFFERS` 可看缓存与 I/O。诊断重点是估算与实际在哪一层开始分叉，而不只是寻找有没有 “Index Scan”。
>
> <!-- bilingual-en:start -->
> `EXPLAIN` displays the chosen plan, estimated rows, and costs without executing the query. `EXPLAIN ANALYZE` runs it and adds evidence such as actual time, rows, and loops; `BUFFERS` can expose cache and I/O activity. Diagnosis should locate where estimates first diverge from reality, not merely search for the phrase “Index Scan.”
> <!-- bilingual-en:end -->

## 自然解释

若 planner 估计过滤后 10 行，实际却是 100,000 行，后续 nested loop 的内层会被执行远多于预期。最慢节点未必是根因；更早的 cardinality estimate 错误可能让 planner 选择了整条错误路径。要沿 plan tree 同时读 rows、loops 和 inclusive time。

<!-- bilingual-en:start -->
If the planner estimates 10 rows after a filter but actually produces 100,000, an inner nested-loop node runs far more often than expected. The slowest node may not be the root cause; an earlier cardinality error may have selected the wrong plan shape. Read rows, loops, and inclusive time together along the plan tree.
<!-- bilingual-en:end -->

`EXPLAIN ANALYZE` 会真的执行 `INSERT`、`UPDATE`、`DELETE` 等写操作。需要只观察计划时可放在显式事务中随后 rollback，但还要注意 sequence 等某些数据库状态并不随事务回滚；生产数据上必须谨慎。

<!-- bilingual-en:start -->
`EXPLAIN ANALYZE` really executes `INSERT`, `UPDATE`, `DELETE`, and other writes. To inspect without retaining ordinary table changes, it can run inside an explicit transaction followed by rollback, but some state such as sequences is not transactional. Use great care on production data.
<!-- bilingual-en:end -->

> [!warning] 边界
> 一次实际运行只代表当时参数、缓存、并发与数据分布，而且 `EXPLAIN ANALYZE` 的节点计时本身会增加 profiling overhead；只需 actual rows 时可考虑 `TIMING OFF`。索引决策应在代表性参数和冷/热缓存场景下重复测量，并同时观察写入代价。
>
> <!-- bilingual-en:start -->
> One execution reflects its parameters, cache state, concurrency, and data distribution, and node timing in `EXPLAIN ANALYZE` adds profiling overhead; consider `TIMING OFF` when actual row counts are sufficient. Test index decisions across representative parameters and cold or warm cache states, while also measuring write cost.
> <!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 计划里显示 Index Scan，为什么仍不能证明新索引改善了整体工作负载？
>
> **答案：** 还要比较实际时间、rows/loops、buffers、代表性参数，以及索引给写入和存储增加的成本；一种查询的一次路径不足以代表整体。

## 来源与核验

- PostgreSQL 18, [*Using EXPLAIN*](https://www.postgresql.org/docs/current/using-explain.html)：定义 estimated plan、`EXPLAIN ANALYZE` 实际执行、rows/loops 与 `BUFFERS` 解释。
- PostgreSQL 18, [*EXPLAIN*](https://www.postgresql.org/docs/current/sql-explain.html)：明确 `ANALYZE` 会执行语句，给出在事务中回滚写操作的诊断方法，并说明 profiling overhead。
- PostgreSQL 18, [*Transaction Isolation*](https://www.postgresql.org/docs/current/transaction-iso.html)：说明 sequence 变更不会随事务 abort 回滚。
