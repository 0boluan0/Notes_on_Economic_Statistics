---
aliases:
  - "SQL 查询默认保留重复行而 DISTINCT 显式去重"
  - SQL queries retain duplicate rows by default and DISTINCT removes them explicitly
  - SQL 多重集语义
  - SQL bag semantics
student_os: knowledge-atom
atom_id: CS-SQL-006
atom_type: language-semantics
status: source-checked
mastery_state: unassessed
part_of:
  - "[[关系模型与 SQL 查询.canvas]]"
requires:
  - "[[关系模型结构]]"
leads_to:
  - "[[JOIN 与关系基数]]"
  - "[[一对多 JOIN 重复聚合]]"
---

# SQL 查询默认保留重复行而 DISTINCT 显式去重

<!-- bilingual-en:start -->
*SQL queries retain duplicate rows by default, while DISTINCT removes them explicitly*
<!-- bilingual-en:end -->

> [!summary] 原子语义
> `SELECT` 默认采用 `ALL` 语义：投影后相同的结果行仍可出现多次。`DISTINCT` 才按整个选择列表消除重复；它不是修复错误连接的通用按钮。
>
> <!-- bilingual-en:start -->
> `SELECT` uses `ALL` semantics by default, so identical projected rows may appear several times. `DISTINCT` removes duplicates across the entire select list; it is not a general repair button for an incorrect join.
> <!-- bilingual-en:end -->

若三笔订单来自同一城市，

```sql
SELECT city
FROM orders;
```

可以返回三行 `London`。这不是数据库“忘了去重”，而是三条输入事实投影到同一列以后仍保留各自行。改为 `SELECT DISTINCT city` 才是在问“出现过哪些不同城市”。两个查询的问题不同。
<!-- bilingual-en:start -->
If three orders come from the same city, selecting `city` may return `London` three times. The database did not forget to deduplicate; three input facts projected to the same value remain three result rows. `SELECT DISTINCT city` asks a different question: which distinct cities occurred?
<!-- bilingual-en:end -->

当连接意外制造倍增时，最外层加 `DISTINCT` 可能让行数看起来正常，却仍可能把金额等度量算错，也可能误删本来就有意义的重复事实。先核对键、粒度和连接基数，再决定业务问题是否真的要求去重。
<!-- bilingual-en:start -->
When a join accidentally multiplies rows, an outer `DISTINCT` may make the row count look plausible while measures remain wrong, or it may erase legitimate repeated facts. Check keys, grain, and join cardinality first; deduplicate only if the question genuinely asks for distinct rows.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 查询出现重复行时，为什么不能第一反应就加 `DISTINCT`？
>
> **答案：** 重复可能来自真实的多条事实，也可能暴露错误连接。`DISTINCT` 只删除相同输出行，不解释重复来源，也不修复已经发生的重复聚合。

## 来源与核验

- PostgreSQL, [*Select Lists: DISTINCT*](https://www.postgresql.org/docs/current/queries-select-lists.html#QUERIES-DISTINCT)：核验 `ALL` 是默认行为，`DISTINCT` 在选择列表处理后消除重复行。
- CMU 15-445/645, [*Modern SQL*](https://15445.courses.cs.cmu.edu/fall2025/notes/02-modernsql.pdf)：核验 SQL 与纯关系集合语义之间的多重集差异。
