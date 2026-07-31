# 00 课程概览
<!-- bilingual-en:start -->
*00 Course Overview*
<!-- bilingual-en:end -->
## 1）学习目标
<!-- bilingual-en:start -->
*1) Learning Objectives*
<!-- bilingual-en:end -->
- 理解 [[关系模型与 SQL 查询|SQL 的作用与关系型数据库的表-行-列模型]]
- 能写出基础 `SELECT` 查询并读懂结果表
- 能使用 `WHERE` 约束筛选目标行
- 完成 SQLBolt 交互练习并形成可复用的查询模板
<!-- bilingual-en:start -->
- Understand [[关系模型与 SQL 查询|SQL's role and the table–row–column model of relational databases]].
- Write basic `SELECT` queries and interpret the resulting tables.
- Use `WHERE` constraints to retain only target rows.
- Complete the interactive SQLBolt exercises and build reusable query templates.
<!-- bilingual-en:end -->

## 2）练习环境与数据集
<!-- bilingual-en:start -->
*2) Practice Environment and Dataset*
<!-- bilingual-en:end -->
- SQLBolt 提供浏览器内交互式 SQL 编辑器与练习
- 早期练习主要使用 `movies` 表（包含 `id`、`title`、`director`、`year` 等列）
- 介绍页用车辆表示例说明“表、行、列”的含义
<!-- bilingual-en:start -->
- SQLBolt provides an interactive in-browser SQL editor and exercises.
- Early exercises mainly use a `movies` table with columns such as `id`, `title`, `director`, and `year`.
- The introduction uses a vehicle table to explain the meaning of tables, rows, and columns.
<!-- bilingual-en:end -->

## 3）笔记记录规范（统一模板）
<!-- bilingual-en:start -->
*3) Standard Note-Taking Template*
<!-- bilingual-en:end -->
- 每节课固定模块：本课要点 / 概念卡片 / 查询模板清单 / 练习题解 / 反例/陷阱 / 小结
- 概念卡片采用列表格式：`概念：定义/作用；例子（如需）`
- 查询模板清单与题解统一放在 `sql` 代码块中，便于复用
- 练习题解以“题目 + 参考解”记录，必要时附一行反思
<!-- bilingual-en:start -->
- Each lesson uses the same sections: key points, concept cards, query templates, exercise solutions, counterexamples or pitfalls, and a summary.
- Concept cards use the list format `Concept: definition/purpose; example (if needed)`.
- Query templates and solutions appear in `sql` code blocks for reuse.
- Exercise notes record each problem with a reference solution and, where useful, one line of reflection.
<!-- bilingual-en:end -->

## 4）进度与完成情况
<!-- bilingual-en:start -->
*4) Progress and Completion Status*
<!-- bilingual-en:end -->
- [x] 00 课程概览
- [ ] 01 Introduction to SQL
- [ ] 02 SQL Lesson 1: SELECT queries 101
- [ ] 03 SQL Lesson 2: Queries with constraints (Pt. 1)
- [ ] 04 SQL Lesson 3: Queries with constraints (Pt. 2)
- [ ] 05 SQL Lesson 4: Filtering and sorting Query results
- [ ] 06 SQL Review: Simple SELECT Queries
- [ ] 07 SQL Lesson 6: Multi-table queries with JOINs
- [ ] 08 SQL Lesson 7: OUTER JOINs
- [ ] 09 SQL Lesson 8: A short note on NULLs
- [ ] 10 SQL Lesson 9: Queries with expressions
- [ ] 11 SQL Lesson 10: Queries with aggregates (Pt. 1)
- [ ] 12 SQL Lesson 11: Queries with aggregates (Pt. 2)
- [ ] 13 SQL Lesson 12: Order of execution of a Query
- [ ] 14 SQL Lesson 13: Inserting rows
- [ ] 15 SQL Lesson 14: Updating rows
- [ ] 16 SQL Lesson 15: Deleting rows
- [ ] 17 SQL Lesson 16: Creating tables
- [ ] 18 SQL Lesson 17: Altering tables
- [ ] 19 SQL Lesson 18: Dropping tables
- [ ] 20 SQL Lesson X: To infinity and beyond!
<!-- bilingual-en:start -->
- [x] 00 Course Overview
- [ ] 01 Introduction to SQL
- [ ] 02 SQL Lesson 1: SELECT Queries 101
- [ ] 03 SQL Lesson 2: Queries with Constraints (Part 1)
- [ ] 04 SQL Lesson 3: Queries with Constraints (Part 2)
- [ ] 05 SQL Lesson 4: Filtering and Sorting Query Results
- [ ] 06 SQL Review: Simple SELECT Queries
- [ ] 07 SQL Lesson 6: Multi-Table Queries with JOINs
- [ ] 08 SQL Lesson 7: OUTER JOINs
- [ ] 09 SQL Lesson 8: A Short Note on NULLs
- [ ] 10 SQL Lesson 9: Queries with Expressions
- [ ] 11 SQL Lesson 10: Queries with Aggregates (Part 1)
- [ ] 12 SQL Lesson 11: Queries with Aggregates (Part 2)
- [ ] 13 SQL Lesson 12: Query Execution Order
- [ ] 14 SQL Lesson 13: Inserting Rows
- [ ] 15 SQL Lesson 14: Updating Rows
- [ ] 16 SQL Lesson 15: Deleting Rows
- [ ] 17 SQL Lesson 16: Creating Tables
- [ ] 18 SQL Lesson 17: Altering Tables
- [ ] 19 SQL Lesson 18: Dropping Tables
- [ ] 20 SQL Lesson X: To Infinity and Beyond!
<!-- bilingual-en:end -->

# 01 Introduction to SQL
## 1）本课要点
<!-- bilingual-en:start -->
*1) Key Points*
<!-- bilingual-en:end -->
- SQL 用于在关系型数据库中查询、操作与转换数据
- 关系型数据库由多张二维表组成，列是属性，行是记录
- SQLBolt 的课程以查询为起点，后续再讲表结构的创建与修改
<!-- bilingual-en:start -->
- SQL is used to query, manipulate, and transform data in relational databases.
- A relational database contains two-dimensional tables whose columns are attributes and whose rows are records.
- SQLBolt begins with querying and later covers creating and modifying table structures.
<!-- bilingual-en:end -->

## 2）概念卡片
<!-- bilingual-en:start -->
*2) Concept Cards*
<!-- bilingual-en:end -->
- SQL：用于在关系型数据库中查询、操作与转换数据
- 关系型数据库：由多张二维表组成，列是属性，行是记录
- 课程路线：先学查询，再学习表结构的创建与修改
<!-- bilingual-en:start -->
- SQL: a language for querying, manipulating, and transforming relational data.
- Relational database: a collection of two-dimensional tables with attributes as columns and records as rows.
- Course path: learn queries first, then learn to create and modify table structures.
<!-- bilingual-en:end -->

## 3）查询模板清单
<!-- bilingual-en:start -->
*3) Query Template List*
<!-- bilingual-en:end -->
```sql
SELECT * FROM mytable;
```

## 4）练习题解
<!-- bilingual-en:start -->
*4) Exercise Solutions*
<!-- bilingual-en:end -->
本节为导读页，无交互练习。
<!-- bilingual-en:start -->
This is an introductory page and has no interactive exercise.
<!-- bilingual-en:end -->

## 5）反例/陷阱
<!-- bilingual-en:start -->
*5) Counterexamples and Pitfalls*
<!-- bilingual-en:end -->
- 混淆表、行、列的含义，导致对查询结果理解错误
<!-- bilingual-en:start -->
- Confusing tables, rows, and columns leads to misinterpreting query results.
<!-- bilingual-en:end -->

## 6）小结
<!-- bilingual-en:start -->
*6) Summary*
<!-- bilingual-en:end -->
明确了 SQL 与关系型数据库的基本模型，后续从 `SELECT` 语句入手。
<!-- bilingual-en:start -->
This lesson establishes SQL and the relational model; the next step begins with `SELECT` statements.
<!-- bilingual-en:end -->

# 02 SQL Lesson 1: SELECT queries 101
## 1）本课要点
<!-- bilingual-en:start -->
*1) Key Points*
<!-- bilingual-en:end -->
- `SELECT` 用来声明要取哪些列、从哪张表取
- 表可视作“实体类型”，行是实例，列是共有属性
- `*` 可以一次性取回表的所有列，适合快速查看数据
<!-- bilingual-en:start -->
- `SELECT` declares which columns to retrieve and the table from which to retrieve them.
- A table can represent an entity type, with rows as instances and columns as shared attributes.
- `*` retrieves every column at once and is useful for quick inspection.
<!-- bilingual-en:end -->

## 2）概念卡片
<!-- bilingual-en:start -->
*2) Concept Cards*
<!-- bilingual-en:end -->
- SELECT：声明需要返回的列
- FROM：指定数据来源的表
- `*`：一次性选择所有列，适合快速查看
<!-- bilingual-en:start -->
- `SELECT`: declares the columns to return.
- `FROM`: identifies the source table.
- `*`: selects all columns at once for quick inspection.
<!-- bilingual-en:end -->

## 3）查询模板清单
<!-- bilingual-en:start -->
*3) Query Template List*
<!-- bilingual-en:end -->
```sql
SELECT column, another_column
FROM mytable;
```

```sql
SELECT *
FROM mytable;
```

## 4）练习题解
<!-- bilingual-en:start -->
*4) Exercise Solutions*
<!-- bilingual-en:end -->
题目1：查询每部电影的标题
参考解：
<!-- bilingual-en:start -->
Problem 1: Retrieve the title of every movie.

Reference solution:
<!-- bilingual-en:end -->
```sql
SELECT title
FROM movies;
```

题目2：查询每部电影的导演
参考解：
<!-- bilingual-en:start -->
Problem 2: Retrieve the director of every movie.

Reference solution:
<!-- bilingual-en:end -->
```sql
SELECT director
FROM movies;
```

题目3：查询每部电影的标题与导演
参考解：
<!-- bilingual-en:start -->
Problem 3: Retrieve each movie's title and director.

Reference solution:
<!-- bilingual-en:end -->
```sql
SELECT title, director
FROM movies;
```

题目4：查询每部电影的标题与上映年份
参考解：
<!-- bilingual-en:start -->
Problem 4: Retrieve each movie's title and release year.

Reference solution:
<!-- bilingual-en:end -->
```sql
SELECT title, year
FROM movies;
```

题目5：查询每部电影的所有列
参考解：
<!-- bilingual-en:start -->
Problem 5: Retrieve every column for every movie.

Reference solution:
<!-- bilingual-en:end -->
```sql
SELECT *
FROM movies;
```

## 5）反例/陷阱
<!-- bilingual-en:start -->
*5) Counterexamples and Pitfalls*
<!-- bilingual-en:end -->
- 选择多列时忘记逗号或漏写 `FROM`
- 在真实系统中滥用 `SELECT *`，导致结果冗余
<!-- bilingual-en:start -->
- Forgetting a comma between selected columns or omitting `FROM`.
- Overusing `SELECT *` in real systems and returning unnecessary data.
<!-- bilingual-en:end -->

## 6）小结
<!-- bilingual-en:start -->
*6) Summary*
<!-- bilingual-en:end -->
掌握了最基础的 `SELECT ... FROM ...` 以及 `*` 的用法。
<!-- bilingual-en:start -->
This lesson covers the basic `SELECT ... FROM ...` form and the use of `*`.
<!-- bilingual-en:end -->

# 03 SQL Lesson 2: Queries with constraints (Pt. 1)
## 1）本课要点
<!-- bilingual-en:start -->
*1) Key Points*
<!-- bilingual-en:end -->
- 使用 WHERE 子句过滤行，避免返回无关数据
- 条件可通过  AND / OR 组合形成更复杂约束
- 常见数值约束操作符：=, !=, <, <=, >, >=, BETWEEN, IN, NOT IN
- 约束能减少结果集规模，提升查询可读性与效率
- SQL 关键字大小写不敏感，但建议大写书写
<!-- bilingual-en:start -->
- A `WHERE` clause filters rows and excludes irrelevant data.
- `AND` and `OR` combine conditions into more complex constraints.
- Common numeric operators include `=`, `!=`, `<`, `<=`, `>`, `>=`, `BETWEEN`, `IN`, and `NOT IN`.
- Constraints reduce the result set and can improve readability and efficiency.
- SQL keywords are case-insensitive, though uppercase is a common convention.
<!-- bilingual-en:end -->

## 2）概念卡片
<!-- bilingual-en:start -->
*2) Concept Cards*
<!-- bilingual-en:end -->
- WHERE：用于过滤行，限制结果集
- AND/OR：组合多个条件
- BETWEEN/IN：表达区间或离散值约束
- 约束：减少结果集规模并提升可读性
<!-- bilingual-en:start -->
- `WHERE`: filters rows and restricts the result set.
- `AND`/`OR`: combine conditions.
- `BETWEEN`/`IN`: express ranges or sets of discrete values.
- Constraint: reduces result size and improves readability.
<!-- bilingual-en:end -->

## 3）查询模板清单
<!-- bilingual-en:start -->
*3) Query Template List*
<!-- bilingual-en:end -->
```sql
SELECT column, another_column
FROM mytable
WHERE condition AND/OR another_condition;
```

```sql
SELECT *
FROM movies
WHERE year BETWEEN 2000 AND 2010;
```

```sql
SELECT *
FROM movies
WHERE id IN (1, 2, 3);
```

## 4）练习题解
<!-- bilingual-en:start -->
*4) Exercise Solutions*
<!-- bilingual-en:end -->
题目1：查询 `id = 6` 的电影
参考解：
<!-- bilingual-en:start -->
Problem 1: Retrieve the movie with `id = 6`.

Reference solution:
<!-- bilingual-en:end -->
```sql
SELECT *
FROM movies
WHERE id = 6;
```

题目2：查询上映年份在 2000–2010 之间的电影
参考解：
<!-- bilingual-en:start -->
Problem 2: Retrieve movies released between 2000 and 2010.

Reference solution:
<!-- bilingual-en:end -->
```sql
SELECT *
FROM movies
WHERE year BETWEEN 2000 AND 2010;
```

题目3：查询上映年份不在 2000–2010 之间的电影
参考解：
<!-- bilingual-en:start -->
Problem 3: Retrieve movies released outside 2000–2010.

Reference solution:
<!-- bilingual-en:end -->
```sql
SELECT *
FROM movies
WHERE year NOT BETWEEN 2000 AND 2010;
```

题目4：查询前 5 部 Pixar 电影及其上映年份
参考解：
<!-- bilingual-en:start -->
Problem 4: Retrieve the first five Pixar movies and their release years.

Reference solution:
<!-- bilingual-en:end -->
```sql
SELECT title, year
FROM movies
WHERE id BETWEEN 1 AND 5;
```

## 5）反例/陷阱
<!-- bilingual-en:start -->
*5) Counterexamples and Pitfalls*
<!-- bilingual-en:end -->
- `BETWEEN` 是包含边界的区间
- `IN`/`NOT IN` 用于一组离散值，注意括号与逗号
<!-- bilingual-en:start -->
- `BETWEEN` includes both endpoints.
- `IN` and `NOT IN` operate on a discrete set; take care with parentheses and commas.
<!-- bilingual-en:end -->

## 6）小结
<!-- bilingual-en:start -->
*6) Summary*
<!-- bilingual-en:end -->
通过 `WHERE` 约束实现了对结果集的精确筛选。
<!-- bilingual-en:start -->
This lesson uses `WHERE` constraints to filter a result set precisely.
<!-- bilingual-en:end -->

# 04 SQL Lesson 3: Queries with constraints (Pt. 2)
## 1）本课要点
<!-- bilingual-en:start -->
*1) Key Points*
<!-- bilingual-en:end -->
## 2）语法与示例
<!-- bilingual-en:start -->
*2) Syntax and Examples*
<!-- bilingual-en:end -->
## 3）练习记录
<!-- bilingual-en:start -->
*3) Exercise Record*
<!-- bilingual-en:end -->
## 4）易错点
<!-- bilingual-en:start -->
*4) Common Pitfalls*
<!-- bilingual-en:end -->
## 5）小结
<!-- bilingual-en:start -->
*5) Summary*
<!-- bilingual-en:end -->

# 05 SQL Lesson 4: Filtering and sorting Query results
## 1）本课要点
<!-- bilingual-en:start -->
*1) Key Points*
<!-- bilingual-en:end -->
## 2）语法与示例
<!-- bilingual-en:start -->
*2) Syntax and Examples*
<!-- bilingual-en:end -->
## 3）练习记录
<!-- bilingual-en:start -->
*3) Exercise Record*
<!-- bilingual-en:end -->
## 4）易错点
<!-- bilingual-en:start -->
*4) Common Pitfalls*
<!-- bilingual-en:end -->
## 5）小结
<!-- bilingual-en:start -->
*5) Summary*
<!-- bilingual-en:end -->

# 06 SQL Review: Simple SELECT Queries
## 1）本课要点
<!-- bilingual-en:start -->
*1) Key Points*
<!-- bilingual-en:end -->
## 2）语法与示例
<!-- bilingual-en:start -->
*2) Syntax and Examples*
<!-- bilingual-en:end -->
## 3）练习记录
<!-- bilingual-en:start -->
*3) Exercise Record*
<!-- bilingual-en:end -->
## 4）易错点
<!-- bilingual-en:start -->
*4) Common Pitfalls*
<!-- bilingual-en:end -->
## 5）小结
<!-- bilingual-en:start -->
*5) Summary*
<!-- bilingual-en:end -->

# 07 SQL Lesson 6: Multi-table queries with JOINs
## 1）本课要点
<!-- bilingual-en:start -->
*1) Key Points*
<!-- bilingual-en:end -->
## 2）语法与示例
<!-- bilingual-en:start -->
*2) Syntax and Examples*
<!-- bilingual-en:end -->
## 3）练习记录
<!-- bilingual-en:start -->
*3) Exercise Record*
<!-- bilingual-en:end -->
## 4）易错点
<!-- bilingual-en:start -->
*4) Common Pitfalls*
<!-- bilingual-en:end -->
## 5）小结
<!-- bilingual-en:start -->
*5) Summary*
<!-- bilingual-en:end -->

# 08 SQL Lesson 7: OUTER JOINs
## 1）本课要点
<!-- bilingual-en:start -->
*1) Key Points*
<!-- bilingual-en:end -->
## 2）语法与示例
<!-- bilingual-en:start -->
*2) Syntax and Examples*
<!-- bilingual-en:end -->
## 3）练习记录
<!-- bilingual-en:start -->
*3) Exercise Record*
<!-- bilingual-en:end -->
## 4）易错点
<!-- bilingual-en:start -->
*4) Common Pitfalls*
<!-- bilingual-en:end -->
## 5）小结
<!-- bilingual-en:start -->
*5) Summary*
<!-- bilingual-en:end -->

# 09 SQL Lesson 8: A short note on NULLs
## 1）本课要点
<!-- bilingual-en:start -->
*1) Key Points*
<!-- bilingual-en:end -->
## 2）语法与示例
<!-- bilingual-en:start -->
*2) Syntax and Examples*
<!-- bilingual-en:end -->
## 3）练习记录
<!-- bilingual-en:start -->
*3) Exercise Record*
<!-- bilingual-en:end -->
## 4）易错点
<!-- bilingual-en:start -->
*4) Common Pitfalls*
<!-- bilingual-en:end -->
## 5）小结
<!-- bilingual-en:start -->
*5) Summary*
<!-- bilingual-en:end -->

# 10 SQL Lesson 9: Queries with expressions
## 1）本课要点
<!-- bilingual-en:start -->
*1) Key Points*
<!-- bilingual-en:end -->
## 2）语法与示例
<!-- bilingual-en:start -->
*2) Syntax and Examples*
<!-- bilingual-en:end -->
## 3）练习记录
<!-- bilingual-en:start -->
*3) Exercise Record*
<!-- bilingual-en:end -->
## 4）易错点
<!-- bilingual-en:start -->
*4) Common Pitfalls*
<!-- bilingual-en:end -->
## 5）小结
<!-- bilingual-en:start -->
*5) Summary*
<!-- bilingual-en:end -->

# 11 SQL Lesson 10: Queries with aggregates (Pt. 1)
## 1）本课要点
<!-- bilingual-en:start -->
*1) Key Points*
<!-- bilingual-en:end -->
## 2）语法与示例
<!-- bilingual-en:start -->
*2) Syntax and Examples*
<!-- bilingual-en:end -->
## 3）练习记录
<!-- bilingual-en:start -->
*3) Exercise Record*
<!-- bilingual-en:end -->
## 4）易错点
<!-- bilingual-en:start -->
*4) Common Pitfalls*
<!-- bilingual-en:end -->
## 5）小结
<!-- bilingual-en:start -->
*5) Summary*
<!-- bilingual-en:end -->

# 12 SQL Lesson 11: Queries with aggregates (Pt. 2)
## 1）本课要点
<!-- bilingual-en:start -->
*1) Key Points*
<!-- bilingual-en:end -->
## 2）语法与示例
<!-- bilingual-en:start -->
*2) Syntax and Examples*
<!-- bilingual-en:end -->
## 3）练习记录
<!-- bilingual-en:start -->
*3) Exercise Record*
<!-- bilingual-en:end -->
## 4）易错点
<!-- bilingual-en:start -->
*4) Common Pitfalls*
<!-- bilingual-en:end -->
## 5）小结
<!-- bilingual-en:start -->
*5) Summary*
<!-- bilingual-en:end -->

# 13 SQL Lesson 12: Order of execution of a Query
## 1）本课要点
<!-- bilingual-en:start -->
*1) Key Points*
<!-- bilingual-en:end -->
## 2）语法与示例
<!-- bilingual-en:start -->
*2) Syntax and Examples*
<!-- bilingual-en:end -->
## 3）练习记录
<!-- bilingual-en:start -->
*3) Exercise Record*
<!-- bilingual-en:end -->
## 4）易错点
<!-- bilingual-en:start -->
*4) Common Pitfalls*
<!-- bilingual-en:end -->
## 5）小结
<!-- bilingual-en:start -->
*5) Summary*
<!-- bilingual-en:end -->

# 14 SQL Lesson 13: Inserting rows
## 1）本课要点
<!-- bilingual-en:start -->
*1) Key Points*
<!-- bilingual-en:end -->
## 2）语法与示例
<!-- bilingual-en:start -->
*2) Syntax and Examples*
<!-- bilingual-en:end -->
## 3）练习记录
<!-- bilingual-en:start -->
*3) Exercise Record*
<!-- bilingual-en:end -->
## 4）易错点
<!-- bilingual-en:start -->
*4) Common Pitfalls*
<!-- bilingual-en:end -->
## 5）小结
<!-- bilingual-en:start -->
*5) Summary*
<!-- bilingual-en:end -->

# 15 SQL Lesson 14: Updating rows
## 1）本课要点
<!-- bilingual-en:start -->
*1) Key Points*
<!-- bilingual-en:end -->
## 2）语法与示例
<!-- bilingual-en:start -->
*2) Syntax and Examples*
<!-- bilingual-en:end -->
## 3）练习记录
<!-- bilingual-en:start -->
*3) Exercise Record*
<!-- bilingual-en:end -->
## 4）易错点
<!-- bilingual-en:start -->
*4) Common Pitfalls*
<!-- bilingual-en:end -->
## 5）小结
<!-- bilingual-en:start -->
*5) Summary*
<!-- bilingual-en:end -->

# 16 SQL Lesson 15: Deleting rows
## 1）本课要点
<!-- bilingual-en:start -->
*1) Key Points*
<!-- bilingual-en:end -->
## 2）语法与示例
<!-- bilingual-en:start -->
*2) Syntax and Examples*
<!-- bilingual-en:end -->
## 3）练习记录
<!-- bilingual-en:start -->
*3) Exercise Record*
<!-- bilingual-en:end -->
## 4）易错点
<!-- bilingual-en:start -->
*4) Common Pitfalls*
<!-- bilingual-en:end -->
## 5）小结
<!-- bilingual-en:start -->
*5) Summary*
<!-- bilingual-en:end -->

# 17 SQL Lesson 16: Creating tables
## 1）本课要点
<!-- bilingual-en:start -->
*1) Key Points*
<!-- bilingual-en:end -->
## 2）语法与示例
<!-- bilingual-en:start -->
*2) Syntax and Examples*
<!-- bilingual-en:end -->
## 3）练习记录
<!-- bilingual-en:start -->
*3) Exercise Record*
<!-- bilingual-en:end -->
## 4）易错点
<!-- bilingual-en:start -->
*4) Common Pitfalls*
<!-- bilingual-en:end -->
## 5）小结
<!-- bilingual-en:start -->
*5) Summary*
<!-- bilingual-en:end -->

# 18 SQL Lesson 17: Altering tables
## 1）本课要点
<!-- bilingual-en:start -->
*1) Key Points*
<!-- bilingual-en:end -->
## 2）语法与示例
<!-- bilingual-en:start -->
*2) Syntax and Examples*
<!-- bilingual-en:end -->
## 3）练习记录
<!-- bilingual-en:start -->
*3) Exercise Record*
<!-- bilingual-en:end -->
## 4）易错点
<!-- bilingual-en:start -->
*4) Common Pitfalls*
<!-- bilingual-en:end -->
## 5）小结
<!-- bilingual-en:start -->
*5) Summary*
<!-- bilingual-en:end -->

# 19 SQL Lesson 18: Dropping tables
## 1）本课要点
<!-- bilingual-en:start -->
*1) Key Points*
<!-- bilingual-en:end -->
## 2）语法与示例
<!-- bilingual-en:start -->
*2) Syntax and Examples*
<!-- bilingual-en:end -->
## 3）练习记录
<!-- bilingual-en:start -->
*3) Exercise Record*
<!-- bilingual-en:end -->
## 4）易错点
<!-- bilingual-en:start -->
*4) Common Pitfalls*
<!-- bilingual-en:end -->
## 5）小结
<!-- bilingual-en:start -->
*5) Summary*
<!-- bilingual-en:end -->

# 20 SQL Lesson X: To infinity and beyond!
## 1）本课要点
<!-- bilingual-en:start -->
*1) Key Points*
<!-- bilingual-en:end -->
## 2）语法与示例
<!-- bilingual-en:start -->
*2) Syntax and Examples*
<!-- bilingual-en:end -->
## 3）练习记录
<!-- bilingual-en:start -->
*3) Exercise Record*
<!-- bilingual-en:end -->
## 4）易错点
<!-- bilingual-en:start -->
*4) Common Pitfalls*
<!-- bilingual-en:end -->
## 5）小结
<!-- bilingual-en:start -->
*5) Summary*
<!-- bilingual-en:end -->

# 附录
<!-- bilingual-en:start -->
*Appendix*
<!-- bilingual-en:end -->
## 1）常用 SQL 速查
<!-- bilingual-en:start -->
*1) Common SQL Quick Reference*
<!-- bilingual-en:end -->
## 2）常见错误清单
<!-- bilingual-en:start -->
*2) Common Error Checklist*
<!-- bilingual-en:end -->
## 3）个人总结与复盘
<!-- bilingual-en:start -->
*3) Personal Summary and Review*
<!-- bilingual-en:end -->
