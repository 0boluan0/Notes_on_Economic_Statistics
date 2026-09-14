---
aliases:
  - "PostgreSQL Read Committed 为每条语句取得新快照并在等待后重查更新条件"
  - PostgreSQL Read Committed takes a new snapshot per statement and rechecks update predicates after waits
  - PostgreSQL Read Committed
student_os: knowledge-atom
atom_id: CS-DB-005
atom_type: system-semantics
status: source-checked
mastery_state: unassessed
part_of:
  - "[[数据库事务、约束与索引.canvas]]"
---

# PostgreSQL Read Committed 为每条语句取得新快照并在等待后重查更新条件

<!-- bilingual-en:start -->
*PostgreSQL Read Committed takes a new snapshot per statement and rechecks update predicates after waits*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> PostgreSQL 默认的 Read Committed 让普通 `SELECT` 看见语句开始前已提交的数据，以及本事务先前尚未提交的更改；同一事务中的下一条语句会取得新快照，因此可能看见后来提交的更改。并发 `UPDATE` 等待目标行后，会在最新行版本上重新检查自己的 `WHERE` 条件。
>
> <!-- bilingual-en:start -->
> PostgreSQL's default Read Committed level lets an ordinary `SELECT` see data committed before that statement began as well as the transaction's own earlier uncommitted changes. The next statement in the same transaction takes a new snapshot and may see later commits. A concurrent `UPDATE` that waits for a target row rechecks its `WHERE` predicate against the updated row version.
> <!-- bilingual-en:end -->

## 自然解释

事务 A 先查询库存为 1；事务 B 扣减并提交；A 的第二次普通 `SELECT` 可以看到 0。若 A 直接执行 `UPDATE ... SET stock=stock-1 WHERE stock>0`，它在等待 B 后会对 B 提交的新版本重新检查 `stock>0`，条件失败就不更新。

<!-- bilingual-en:start -->
Transaction A first reads stock 1; transaction B decrements and commits; A's second ordinary `SELECT` may see 0. If A instead runs `UPDATE ... SET stock=stock-1 WHERE stock>0`, it waits for B and then rechecks `stock>0` on B's committed version; if false, the row is not updated.
<!-- bilingual-en:end -->

这正适合目标行明确、条件局部的原子更新，却不表示一个复杂多行判断会自动保持整体快照一致。PostgreSQL 文档明确提醒，Read Committed 中更新命令可能看见目标行的并发新版本，却看不见同一并发命令对其他行的全部影响。

<!-- bilingual-en:start -->
This works well for a local conditional update on a predetermined row, but it does not make a complex multi-row decision use one globally consistent snapshot. PostgreSQL notes that an updating command may see a concurrent new version of its target row without seeing all effects of that concurrent command on other rows.
<!-- bilingual-en:end -->

> [!warning] 边界
> “不会 dirty read”不等于“同一事务重复读结果不变”，也不等于“没有 serialization anomaly”。Read Committed 只排除其文档承诺排除的现象。
>
> <!-- bilingual-en:start -->
> “No dirty reads” does not imply repeatable reads within the transaction or absence of serialization anomalies. Read Committed excludes only the phenomena its contract excludes.
> <!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 为什么 PostgreSQL Read Committed 下，一条带 `stock > 0` 的 `UPDATE` 比“先 SELECT 再无条件 UPDATE”更安全？
>
> **答案：** 条件和写入在同一语句中；若目标行被并发事务先改，PostgreSQL 等待后会在新版本上重查条件，而分开的 SELECT 已留下竞态窗口。

## 来源与核验

- PostgreSQL 18, [*Read Committed Isolation Level*](https://www.postgresql.org/docs/current/transaction-iso.html#XACT-READ-COMMITTED)：说明逐语句 snapshot、并发更新等待和 `WHERE` 重新检查。
