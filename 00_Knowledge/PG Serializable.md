---
aliases:
  - "PostgreSQL Serializable 只让可串行化结果提交并要求应用重试失败事务"
  - PostgreSQL Serializable commits only serializable outcomes and requires application retries
  - PostgreSQL Serializable
student_os: knowledge-atom
atom_id: CS-DB-007
atom_type: guarantee-boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[数据库事务、约束与索引.canvas]]"
---

# PostgreSQL Serializable 只让可串行化结果提交并要求应用重试失败事务

<!-- bilingual-en:start -->
*PostgreSQL Serializable commits only serializable outcomes and requires application retries*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> PostgreSQL Serializable 监测并发读写依赖，只允许其共同结果与某个逐个执行的 serial order 等价的事务集合成功提交。无法维持该保证时，数据库会让至少一个事务以 serialization failure 失败；应用必须把整个事务从头重试。
>
> <!-- bilingual-en:start -->
> PostgreSQL Serializable monitors concurrent read/write dependencies and permits a set of transactions to commit only when their joint effect matches some one-at-a-time serial order. When that guarantee cannot be preserved, the database aborts at least one transaction with a serialization failure, and the application must retry the entire transaction.
> <!-- bilingual-en:end -->

## 自然解释

Serializable 并不是让事务真的一个接一个排队。它们仍可并发执行；PostgreSQL 通过 Serializable Snapshot Isolation 追踪可能形成 serialization anomaly 的依赖环。若两个事务都提交会得到任何串行顺序都不可能产生的结果，其中一个就必须失败。

<!-- bilingual-en:start -->
Serializable does not literally queue all transactions one after another. They still execute concurrently; PostgreSQL's Serializable Snapshot Isolation tracks dependency patterns that could create a serialization anomaly. If committing both would produce an outcome impossible under every serial order, at least one must fail.
<!-- bilingual-en:end -->

因此，serialization failure 不是偶然的“数据库坏了”，而是保证的一部分。重试必须重新执行全部读取与业务判断，不能只重发最后一条 `UPDATE`；旧读取属于已失败历史，不能继续使用。

<!-- bilingual-en:start -->
A serialization failure is therefore part of the guarantee, not evidence that the database malfunctioned. A retry must repeat all reads and business decisions in the transaction, not merely resend the last `UPDATE`; reads from the failed history are no longer valid.
<!-- bilingual-en:end -->

> [!warning] 边界
> Serializable 只保证已成功提交事务的数据库效果可串行化。它不自动重试，也不回滚外部 API、邮件等事务外副作用；重试路径必须幂等，识别 SQLSTATE `40001`，按有限次数与退避从头重试，并在持续冲突时上报而不是无限循环。
>
> <!-- bilingual-en:start -->
> Serializable guarantees serializable database effects for transactions that commit. It neither retries automatically nor rolls back external APIs or email. Retry paths must be idempotent, recognise SQLSTATE `40001`, restart the whole transaction with bounded backoff, and surface persistent contention instead of looping forever.
> <!-- bilingual-en:end -->

> [!question]- 回忆与应用
> Serializable 事务最后一条写入收到 `40001`。为什么不能保留前面的查询结果，只重试这条写入？
>
> **答案：** 整个读取—判断—写入组合属于被判定不可提交的历史；新事务必须在新 snapshot 上重新做决定。

## 来源与核验

- PostgreSQL 18, [*Serializable Isolation Level*](https://www.postgresql.org/docs/current/transaction-iso.html#XACT-SERIALIZABLE)：定义 serial-order 等价、依赖监测与 serialization failure。
- PostgreSQL 18, [*Serialization Failure Handling*](https://www.postgresql.org/docs/current/mvcc-serialization-failure-handling.html)：明确 `40001`、完整事务重试以及重试仍不保证一次成功。
