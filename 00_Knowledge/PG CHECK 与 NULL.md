---
aliases:
  - "PostgreSQL CHECK 只拒绝 FALSE 而 NULL 与跨行规则需要另行处理"
  - PostgreSQL CHECK rejects only false while NULL and cross-row rules need separate treatment
  - PostgreSQL CHECK 约束边界
student_os: knowledge-atom
atom_id: CS-DB-013
atom_type: constraint-boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[数据库事务、约束与索引.canvas]]"
---

# PostgreSQL CHECK 只拒绝 FALSE 而 NULL 与跨行规则需要另行处理

<!-- bilingual-en:start -->
*PostgreSQL CHECK rejects only false, while NULL and cross-row rules need separate treatment*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> PostgreSQL `CHECK` 在表达式为 `FALSE` 时拒绝一行；结果为 `TRUE` 或因 `NULL` 得到 UNKNOWN 时都通过。因此“值必须非负且必须存在”通常要同时写 `CHECK (x >= 0)` 与 `NOT NULL`。PostgreSQL 还假定 `CHECK` 对同一行输入给出稳定结果，不支持用它可靠约束其他行或表。
>
> <!-- bilingual-en:start -->
> PostgreSQL `CHECK` rejects a row when its expression is `FALSE`; both `TRUE` and UNKNOWN caused by `NULL` pass. “The value must exist and be nonnegative” therefore commonly needs both `CHECK (x >= 0)` and `NOT NULL`. PostgreSQL also assumes a `CHECK` gives a stable result for the same row and does not support using it to enforce cross-row or cross-table conditions reliably.
> <!-- bilingual-en:end -->

## 自然解释

`price numeric CHECK (price > 0)` 仍允许 `price IS NULL`，因为 `NULL > 0` 不是 false，而是 unknown。若缺失也非法，应写 `price numeric NOT NULL CHECK (price > 0)`。这与[[WHERE 与三值逻辑|WHERE 的三值逻辑]]不同：WHERE 只保留 true，CHECK 只拒绝 false。

<!-- bilingual-en:start -->
`price numeric CHECK (price > 0)` still allows `price IS NULL`, because `NULL > 0` is unknown rather than false. If absence is illegal, write `price numeric NOT NULL CHECK (price > 0)`. This differs from [[WHERE 与三值逻辑|WHERE under three-valued logic]]: WHERE keeps only true, whereas CHECK rejects only false.
<!-- bilingual-en:end -->

“全表最多一行 active”不是普通 CHECK 的工作，因为新行是否合法取决于其他行。应考虑 `UNIQUE`/partial unique index、exclusion constraint、foreign key 或在清楚并发边界下的触发器与事务逻辑。

<!-- bilingual-en:start -->
“At most one active row in the table” is not an ordinary CHECK rule because validity depends on other rows. Consider a `UNIQUE` or partial unique index, an exclusion constraint, a foreign key, or trigger and transaction logic with an explicit concurrency boundary.
<!-- bilingual-en:end -->

> [!warning] 边界
> PostgreSQL 不会在被引用函数的实现后来改变时自动重新验证既有 CHECK 数据。constraint expression 中的函数应视为对相同输入不可变；要改语义，应删除、更新数据并重建约束。
>
> <!-- bilingual-en:start -->
> PostgreSQL does not automatically revalidate existing rows when a function used by a CHECK later changes implementation. Treat functions in constraint expressions as immutable for the same input; to change semantics, drop, repair data, and recreate the constraint.
> <!-- bilingual-en:end -->

> [!question]- 回忆与应用
> `CHECK (age >= 18)` 为什么没有阻止 `age = NULL`？
>
> **答案：** 比较结果是 UNKNOWN，不是 FALSE；若必须有年龄，还要 `NOT NULL`。

## 来源与核验

- PostgreSQL 18, [*Check Constraints*](https://www.postgresql.org/docs/current/ddl-constraints.html#DDL-CONSTRAINTS-CHECK-CONSTRAINTS)：说明 true/NULL 通过、跨行引用不受支持及表达式不变性假设。
