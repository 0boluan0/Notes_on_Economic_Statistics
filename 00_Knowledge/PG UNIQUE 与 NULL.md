---
aliases:
  - "PostgreSQL UNIQUE 约束跨行判重而 NULL 是否互异必须显式决定"
  - PostgreSQL UNIQUE checks across rows while NULL distinctness must be chosen explicitly
  - PostgreSQL UNIQUE 与 NULL
student_os: knowledge-atom
atom_id: CS-DB-014
atom_type: constraint-semantics
status: source-checked
mastery_state: unassessed
part_of:
  - "[[数据库事务、约束与索引.canvas]]"
---

# PostgreSQL UNIQUE 约束跨行判重而 NULL 是否互异必须显式决定

<!-- bilingual-en:start -->
*PostgreSQL UNIQUE checks across rows, while NULL distinctness must be chosen explicitly*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> PostgreSQL `UNIQUE` 约束要求指定列组合在各行之间不重复，并自动建立对应的 unique B-tree index。默认情况下，两个 `NULL` 被视为彼此不同，所以可出现多行 NULL；`NULLS NOT DISTINCT` 才把 NULL 当成相同值参与唯一性判定。
>
> <!-- bilingual-en:start -->
> A PostgreSQL `UNIQUE` constraint prevents duplicate combinations across rows and automatically creates a corresponding unique B-tree index. By default, two NULL values are treated as distinct, so several NULL rows can coexist; `NULLS NOT DISTINCT` makes NULL values compare as equal for uniqueness.
> <!-- bilingual-en:end -->

## 自然解释

`UNIQUE (country_code, phone)` 表示完整列组合不能重复，而不是每一列各自唯一。若任一列可为 NULL，默认 NULL 语义可能让看似重复的“未知号码”出现多次；这是否正确取决于业务中 NULL 表示“尚未知”还是“只允许一个空占位”。

<!-- bilingual-en:start -->
`UNIQUE (country_code, phone)` makes the whole combination unique, not each column separately. If either column is nullable, default NULL semantics can allow several apparently duplicate “unknown numbers.” Whether that is correct depends on whether NULL means “not known yet” or a single permitted empty placeholder.
<!-- bilingual-en:end -->

只让满足条件的行唯一，例如“每个用户最多一个 active subscription”，在 PostgreSQL 中常用 unique partial index：只索引 `status='active'` 的行。它是索引层能力，不等同于普通 SQL `UNIQUE` constraint 的语法。

<!-- bilingual-en:start -->
To make only qualifying rows unique, such as “at most one active subscription per user,” PostgreSQL commonly uses a unique partial index over rows with `status='active'`. This is an index feature rather than ordinary SQL `UNIQUE` constraint syntax.
<!-- bilingual-en:end -->

> [!warning] 边界
> UNIQUE 证明的是列值组合唯一，不证明一对一业务关系的所有方向，也不替代[[主键约束|主键的行身份契约]]。NULL 的默认唯一语义在数据库产品间也可能不同。
>
> <!-- bilingual-en:start -->
> UNIQUE proves uniqueness of a value combination; it does not establish every direction of a one-to-one business relationship or replace [[主键约束|the primary-key identity contract]]. Default NULL uniqueness can also differ across database products.
> <!-- bilingual-en:end -->

> [!question]- 回忆与应用
> `email text UNIQUE` 为什么默认可能允许多行 `email = NULL`？
>
> **答案：** PostgreSQL 默认把 NULL 彼此视为 distinct；若业务只允许一个 NULL，要显式使用 `NULLS NOT DISTINCT` 或重新建模。

## 来源与核验

- PostgreSQL 18, [*Unique Constraints*](https://www.postgresql.org/docs/current/ddl-constraints.html#DDL-CONSTRAINTS-UNIQUE-CONSTRAINTS)：定义多列唯一、NULL distinctness、`NULLS NOT DISTINCT` 与自动 unique B-tree index。
- PostgreSQL 18, [*Partial Indexes*](https://www.postgresql.org/docs/current/indexes-partial.html)：说明 unique partial index 对子集实施唯一性。
