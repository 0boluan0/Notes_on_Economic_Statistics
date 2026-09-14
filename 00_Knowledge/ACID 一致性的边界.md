---
aliases:
  - "ACID 一致性只保护已表达的不变量而不会自动理解业务"
  - ACID consistency protects expressed invariants rather than inventing business rules
  - 事务一致性不是业务魔法
student_os: knowledge-atom
atom_id: CS-DB-003
atom_type: distinction
status: source-checked
mastery_state: unassessed
part_of:
  - "[[数据库事务、约束与索引.canvas]]"
---

# ACID 一致性只保护已表达的不变量而不会自动理解业务

<!-- bilingual-en:start -->
*ACID consistency protects expressed invariants rather than inventing business rules*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> ACID 中的 consistency（“一致性”）是：一次正确事务应把满足不变量的数据库状态变成另一个满足不变量的状态。数据库会实施已声明的 schema constraint；其他业务不变量只有在触发器或事务逻辑正确表达，并配上足够的隔离或锁时才受保护。没有表达的业务含义不会被引擎自动猜出。
>
> <!-- bilingual-en:start -->
> In ACID, consistency means that a correct transaction takes a database state satisfying its invariants to another state satisfying them. The database enforces declared schema constraints; other business invariants are protected only when triggers or transaction logic express them correctly under sufficient isolation or locking. The engine cannot infer unstated business meaning.
> <!-- bilingual-en:end -->

## 自然解释

数据库知道 `stock >= 0`，前提是 schema 有相应 `CHECK`，或更新语句把它写进条件。它不会因为列名叫 `stock` 就知道库存不能为负，也不会自动知道“同一用户只能有一个当前订阅”。后一个规则至少要有适当的 `UNIQUE` constraint、unique partial index，或在正确隔离与锁策略下执行的事务逻辑。

<!-- bilingual-en:start -->
The database knows `stock >= 0` only if the schema has the corresponding `CHECK` or the update encodes that condition. It does not infer nonnegative inventory from a column name, nor does it know that a user may have only one current subscription. The latter needs an appropriate `UNIQUE` constraint, a unique partial index, or transaction logic under suitable isolation and locking.
<!-- bilingual-en:end -->

因此，“用了事务”与“业务状态一定正确”是两回事。事务给规则提供原子与并发边界；规则本身仍要被设计、表达和测试。

<!-- bilingual-en:start -->
Therefore, “uses a transaction” and “business state is correct” are different claims. A transaction provides atomic and concurrency boundaries for rules; the rules themselves still need to be designed, expressed, and tested.
<!-- bilingual-en:end -->

> [!warning] 边界
> consistency 也不是“所有副本每时每刻读到相同值”的同义词；后者属于复制一致性模型。这里讨论的是事务前后数据库不变量。
>
> <!-- bilingual-en:start -->
> Consistency here is not synonymous with every replica always returning the same value; that is a replication consistency model. This atom concerns database invariants before and after a transaction.
> <!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 表允许任意整数余额，应用用一个事务把余额改成 -50。事务成功是否违反 ACID？
>
> **答案：** 数据库会接受它，因为 schema 没有表达非负规则；从业务不变量看，这个事务并不一致。原子性、隔离性和持久性即使都成立，也不会让数据库凭 ACID 这个名字发现未声明的规则。

## 来源与核验

- PostgreSQL 18, [*Glossary — Consistency*](https://www.postgresql.org/docs/current/glossary.html#GLOSSARY-CONSISTENCY)：把一致性界定为数据库 constraints 的满足状态。
- PostgreSQL 18, [*Constraints*](https://www.postgresql.org/docs/current/ddl-constraints.html)：列出数据库能直接实施的非空、检查、唯一、主键和参照约束。
- PostgreSQL 18, [*Data Consistency Checks at the Application Level*](https://www.postgresql.org/docs/current/applevel-consistency.html)：说明跨事务业务规则还需要适当隔离或显式锁。
