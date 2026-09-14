---
aliases:
  - "PostgreSQL Repeatable Read 固定事务快照但仍可能出现序列化异常与重试"
  - PostgreSQL Repeatable Read fixes one transaction snapshot but can still admit serialization anomalies and retries
  - PostgreSQL Repeatable Read
student_os: knowledge-atom
atom_id: CS-DB-006
atom_type: system-semantics
status: source-checked
mastery_state: unassessed
part_of:
  - "[[数据库事务、约束与索引.canvas]]"
---

# PostgreSQL Repeatable Read 固定事务快照但仍可能出现序列化异常与重试

<!-- bilingual-en:start -->
*PostgreSQL Repeatable Read fixes one transaction snapshot but can still admit serialization anomalies and retries*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> PostgreSQL Repeatable Read 让事务中的查询使用事务首次非控制语句开始时的稳定 snapshot；后续查询不看见其他事务后来提交的更改。它在 PostgreSQL 中也防止 phantom read，但仍允许 serialization anomaly；更新已被并发提交修改的行还可能使本事务 abort，并要求从头重试。
>
> <!-- bilingual-en:start -->
> PostgreSQL Repeatable Read makes queries use a stable snapshot taken when the transaction's first non-control statement begins. Later queries do not see changes committed by other transactions afterward. PostgreSQL also prevents phantom reads at this level, but serialization anomalies remain possible; updating a row changed by a concurrent committed transaction can abort the transaction and require a full retry.
> <!-- bilingual-en:end -->

## 自然解释

稳定 snapshot 解决了“同一报表前后两次查询看见不同总体”的问题，但它不是串行执行。两个事务可以各自在自己的旧 snapshot 上判断“至少留一个值班医生”，随后分别让不同医生下班；若没有额外约束或锁，两边都可能提交，合起来破坏规则。这类 write skew 正是 snapshot 稳定却非 serializable 的边界。

<!-- bilingual-en:start -->
A stable snapshot prevents two reads in one report from seeing different populations, but it is not serial execution. Two transactions can each observe “at least one doctor remains on call” in its own old snapshot, then take different doctors off duty. Without an additional constraint or lock, both may commit and jointly break the rule. This write-skew pattern marks the boundary between snapshot stability and serializability.
<!-- bilingual-en:end -->

若 Repeatable Read 事务试图修改一个在其开始后已被另一事务提交修改的目标行，PostgreSQL 会报 serialization failure，而不是把新版本偷偷混入旧 snapshot。应用必须丢弃本次结果并从事务开头重试。

<!-- bilingual-en:start -->
If a Repeatable Read transaction tries to modify a target row changed and committed after its snapshot began, PostgreSQL raises a serialization failure rather than silently mixing the new version into the old snapshot. The application must discard the attempt and retry from the transaction start.
<!-- bilingual-en:end -->

> [!warning] 边界
> PostgreSQL 的 Repeatable Read 是 snapshot isolation，并且比 SQL 标准最低要求更强。其他数据库同名级别可能允许不同现象；必须查具体产品文档。
>
> <!-- bilingual-en:start -->
> PostgreSQL Repeatable Read is snapshot isolation and is stronger than the SQL minimum in some respects. The same name can permit different phenomena in another database; check the product contract.
> <!-- bilingual-en:end -->

> [!question]- 回忆与应用
> “同一事务中两次 SELECT 完全相同”能否证明该执行等价于某个串行顺序？
>
> **答案：** 不能。稳定 snapshot 阻止不可重复读，但多个事务的读写依赖仍可能组成没有串行等价顺序的异常。

## 来源与核验

- PostgreSQL 18, [*Repeatable Read Isolation Level*](https://www.postgresql.org/docs/current/transaction-iso.html#XACT-REPEATABLE-READ)：说明 transaction snapshot、PostgreSQL 的 phantom 边界、serialization anomaly 和完整重试要求。
