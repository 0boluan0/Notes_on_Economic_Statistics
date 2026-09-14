---
aliases:
  - "NOT IN 遇到 NULL 可能得到 UNKNOWN 而 NOT EXISTS 只检查行是否存在"
  - NOT IN can become UNKNOWN with NULL while NOT EXISTS checks row existence
  - NOT IN 的 NULL 陷阱
student_os: knowledge-atom
atom_id: CS-SQL-017
atom_type: distinction
status: source-checked
mastery_state: unassessed
part_of:
  - "[[关系模型与 SQL 查询.canvas]]"
requires:
  - "[[NULL 与 UNKNOWN]]"
  - "[[WHERE 与三值逻辑]]"
---

# NOT IN 遇到 NULL 可能得到 UNKNOWN 而 NOT EXISTS 只检查行是否存在

<!-- bilingual-en:start -->
*NOT IN can become UNKNOWN in the presence of NULL, while NOT EXISTS checks row existence*
<!-- bilingual-en:end -->

> [!summary] 原子区别
> `x NOT IN (subquery)` 必须确认 `x` 与右侧每个值都不相等；若没有相等值但右侧含 `NULL`，结论可能是 `UNKNOWN`。相关 `NOT EXISTS` 只问是否存在满足明确匹配条件的行，不会因无关的空值候选自动变成未知。
>
> <!-- bilingual-en:start -->
> `x NOT IN (subquery)` must establish that `x` differs from every returned value. If no equal value exists but the subquery contains `NULL`, the result may be `UNKNOWN`. A correlated `NOT EXISTS` asks only whether a row satisfying an explicit match exists, so an unrelated null candidate does not automatically make the result unknown.
> <!-- bilingual-en:end -->

要找没有被封禁的用户，若封禁表的 `user_id` 意外含 `NULL`：

```sql
WHERE u.user_id NOT IN (SELECT user_id FROM bans)
```

可能一行也不返回，因为对每个没有匹配的用户，仍无法证明他“不等于那个未知值”。更直接的反连接写法是：

```sql
WHERE NOT EXISTS (
  SELECT 1
  FROM bans AS b
  WHERE b.user_id = u.user_id
)
```
<!-- bilingual-en:start -->
If `bans.user_id` contains a null, `NOT IN` may return no users: for every nonmatching user, SQL still cannot prove inequality to the unknown value. The correlated `NOT EXISTS` formulation directly asks whether a matching ban row exists.
<!-- bilingual-en:end -->

这不意味着任何 `NOT IN` 都错误。若右侧列由 `NOT NULL` 约束保证非空，而且左侧也不为空，它可以完全合适。关键是先证明空值边界，而不是靠样例数据碰巧没有 `NULL`。
<!-- bilingual-en:start -->
This does not make every `NOT IN` wrong. It can be appropriate when schema constraints guarantee non-null values on the relevant sides. The key is to prove the null boundary rather than rely on sample data that happens not to contain nulls.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 右侧子查询返回 `(2, NULL)` 时，`1 NOT IN (...)` 为什么不是 `TRUE`？
>
> **答案：** 能确认 `1 <> 2`，却无法确认 `1 <> NULL`；整体结果是 `UNKNOWN`，在 `WHERE` 中不会保留该行。

## 来源与核验

- PostgreSQL, [*Subquery Expressions: NOT IN*](https://www.postgresql.org/docs/current/functions-subquery.html#FUNCTIONS-SUBQUERY-NOTIN)：核验右侧含 `NULL` 且无相等值时 `NOT IN` 返回空值而非真。
- PostgreSQL, [*Subquery Expressions: EXISTS*](https://www.postgresql.org/docs/current/functions-subquery.html#FUNCTIONS-SUBQUERY-EXISTS)：核验 `EXISTS` 只依赖子查询是否返回至少一行。
