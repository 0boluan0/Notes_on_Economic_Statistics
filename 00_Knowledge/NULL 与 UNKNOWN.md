---
aliases:
  - "NULL 表示未知或缺失而普通比较会产生 UNKNOWN"
  - NULL marks unknown or missing information and ordinary comparisons yield UNKNOWN
  - NULL 与三值逻辑
student_os: knowledge-atom
atom_id: CS-SQL-009
atom_type: language-semantics
status: source-checked
mastery_state: unassessed
part_of:
  - "[[关系模型与 SQL 查询.canvas]]"
leads_to:
  - "[[WHERE 与三值逻辑]]"
  - "[[COUNT 与 NULL]]"
  - "[[NOT IN 与 NOT EXISTS]]"
---

# NULL 表示未知或缺失而普通比较会产生 UNKNOWN

<!-- bilingual-en:start -->
*NULL marks unknown or missing information, so ordinary comparisons yield UNKNOWN*
<!-- bilingual-en:end -->

> [!summary] 原子语义
> `NULL` 是“这里没有一个可供普通运算的已知值”的标记，不是数字 0、空字符串或一个可与自身正常相等的值。普通比较只要涉及 `NULL`，通常得到 `UNKNOWN`。
>
> <!-- bilingual-en:start -->
> `NULL` marks the absence of a known value available to ordinary operations. It is neither zero nor an empty string nor an ordinary value equal to itself. Comparisons involving `NULL` ordinarily yield `UNKNOWN`.
> <!-- bilingual-en:end -->

因此不能写 `x = NULL` 判断缺失，而要写 `x IS NULL`。若确实要把两个 `NULL` 当作“在此次比较中相同”，PostgreSQL 提供 `IS NOT DISTINCT FROM`；这和普通 `=` 的空值语义不同，意图也应明确。
<!-- bilingual-en:start -->
Use `x IS NULL`, not `x = NULL`, to test for missingness. If a comparison must deliberately treat two nulls as equal for that operation, PostgreSQL provides `IS NOT DISTINCT FROM`; this is intentionally different from ordinary equality.
<!-- bilingual-en:end -->

`NULL` 也不能自动说明缺失原因。“尚未测量”“不适用”“被拒绝提供”可能都被存成同一个标记，但它们在业务上并不等价。若原因会影响分析或流程，应另设状态字段，而不是要求一个 `NULL` 同时表达所有含义。
<!-- bilingual-en:start -->
`NULL` does not record why information is absent. “Not yet measured,” “not applicable,” and “declined to provide” may all be stored as null while having different business meanings. If the reason matters, model it explicitly rather than asking one marker to carry several states.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 为什么 `NULL = NULL` 不是 `TRUE`？
>
> **答案：** 两边都没有已知值可供普通相等比较，结果是 `UNKNOWN`。检查缺失用 `IS NULL`；需要空值安全相等时用明确的相应谓词。

## 来源与核验

- PostgreSQL, [*Comparison Functions and Operators*](https://www.postgresql.org/docs/current/functions-comparison.html)：核验普通比较遇到 `NULL` 返回未知，以及 `IS NULL`、`IS DISTINCT FROM` 的语义。
- PostgreSQL, [*Logical Operators*](https://www.postgresql.org/docs/current/functions-logical.html)：核验三值逻辑。
