---
aliases:
  - "JOIN 按匹配谓词生成行对并由关系基数决定结果行数"
  - JOIN forms row pairs under a predicate and cardinality determines result size
  - JOIN 的行对与基数
student_os: knowledge-atom
atom_id: CS-SQL-011
atom_type: model
status: source-checked
mastery_state: unassessed
part_of:
  - "[[关系模型与 SQL 查询.canvas]]"
requires:
  - "[[关系表粒度]]"
  - "[[外键与关系基数]]"
leads_to:
  - "[[一对多 JOIN 重复聚合]]"
  - "[[LEFT JOIN 条件位置]]"
---

# JOIN 按匹配谓词生成行对并由关系基数决定结果行数

<!-- bilingual-en:start -->
*JOIN forms row pairs under a predicate, and relationship cardinality determines result size*
<!-- bilingual-en:end -->

> [!summary] 原子模型
> `JOIN` 不是把两张表按视觉位置横向贴合。在匹配阶段，它考察左右行对，并为每一个使连接条件为真的行对生成结果；一行匹配几行，就会贡献几行匹配结果。内连接到此为止，外连接还会为没有匹配的一侧补出含 `NULL` 的行。
>
> <!-- bilingual-en:start -->
> A `JOIN` does not paste two tables side by side by visual position. During matching, it considers pairs of left and right rows and emits one result for every pair whose join condition is true. An inner join stops there; an outer join additionally emits null-extended rows for the preserved side when no match exists.
> <!-- bilingual-en:end -->

若客户 7 有三笔订单，下面的连接会产生三行客户 7，而不是一行：

```sql
SELECT c.customer_id, c.segment, o.order_id
FROM customers AS c
JOIN orders AS o
  ON o.customer_id = c.customer_id;
```

这是正确的一对多结果，不是重复错误。错误与否取决于目标粒度：如果问题是一行一个订单，这三行正确；如果问题是一行一个客户，还需要聚合或存在性判断。
<!-- bilingual-en:start -->
If customer 7 has three orders, the join produces three rows for that customer. This is a correct one-to-many result, not automatically a duplicate error. It is correct at order grain; a customer-grain question still needs aggregation or an existence test.
<!-- bilingual-en:end -->

连接条件缺失或始终为真会形成笛卡尔积：$N$ 行与 $M$ 行恰好产生 $N\times M$ 个行对。即使写了条件，若所谓“键”实际重复，也会出现多对多倍增。因此，审计连接应同时检查两侧粒度、连接列唯一性、未匹配率和连接前后行数。
<!-- bilingual-en:start -->
A missing or always-true predicate forms a Cartesian product: $N$ and $M$ rows produce exactly $N\times M$ pairs. Even with a predicate, duplicated supposed keys can create many-to-many multiplication. Audit both grains, key uniqueness, unmatched rates, and row counts before and after the join.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 左表一行在右表匹配四行，`INNER JOIN` 后它会出现几次？
>
> **答案：** 四次，每个满足谓词的行对各生成一行。

## 来源与核验

- PostgreSQL, [*Table Expressions: Joined Tables*](https://www.postgresql.org/docs/current/queries-table-expressions.html#QUERIES-JOIN)：核验笛卡尔积、内连接、外连接以及“每个满足条件的行对产生结果”的语义。
