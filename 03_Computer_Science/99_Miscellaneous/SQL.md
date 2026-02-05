# 00 课程概览
## 1）学习目标
- 理解 SQL 的作用与关系型数据库的表-行-列模型
- 能写出基础 `SELECT` 查询并读懂结果表
- 能使用 `WHERE` 约束筛选目标行
- 完成 SQLBolt 交互练习并形成可复用的查询模板

## 2）练习环境与数据集
- SQLBolt 提供浏览器内交互式 SQL 编辑器与练习
- 早期练习主要使用 `movies` 表（包含 `id`、`title`、`director`、`year` 等列）
- 介绍页用车辆表示例说明“表、行、列”的含义

## 3）笔记记录规范（统一模板）
- 每节课固定模块：本课要点 / 概念卡片 / 查询模板清单 / 练习题解 / 反例/陷阱 / 小结
- 概念卡片采用列表格式：`概念：定义/作用；例子（如需）`
- 查询模板清单与题解统一放在 `sql` 代码块中，便于复用
- 练习题解以“题目 + 参考解”记录，必要时附一行反思

## 4）进度与完成情况
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

# 01 Introduction to SQL
## 1）本课要点
- SQL 用于在关系型数据库中查询、操作与转换数据
- 关系型数据库由多张二维表组成，列是属性，行是记录
- SQLBolt 的课程以查询为起点，后续再讲表结构的创建与修改

## 2）概念卡片
- SQL：用于在关系型数据库中查询、操作与转换数据
- 关系型数据库：由多张二维表组成，列是属性，行是记录
- 课程路线：先学查询，再学习表结构的创建与修改

## 3）查询模板清单
```sql
SELECT * FROM mytable;
```

## 4）练习题解
本节为导读页，无交互练习。

## 5）反例/陷阱
- 混淆表、行、列的含义，导致对查询结果理解错误

## 6）小结
明确了 SQL 与关系型数据库的基本模型，后续从 `SELECT` 语句入手。

# 02 SQL Lesson 1: SELECT queries 101
## 1）本课要点
- `SELECT` 用来声明要取哪些列、从哪张表取
- 表可视作“实体类型”，行是实例，列是共有属性
- `*` 可以一次性取回表的所有列，适合快速查看数据

## 2）概念卡片
- SELECT：声明需要返回的列
- FROM：指定数据来源的表
- `*`：一次性选择所有列，适合快速查看

## 3）查询模板清单
```sql
SELECT column, another_column
FROM mytable;
```

```sql
SELECT *
FROM mytable;
```

## 4）练习题解
题目1：查询每部电影的标题
参考解：
```sql
SELECT title
FROM movies;
```

题目2：查询每部电影的导演
参考解：
```sql
SELECT director
FROM movies;
```

题目3：查询每部电影的标题与导演
参考解：
```sql
SELECT title, director
FROM movies;
```

题目4：查询每部电影的标题与上映年份
参考解：
```sql
SELECT title, year
FROM movies;
```

题目5：查询每部电影的所有列
参考解：
```sql
SELECT *
FROM movies;
```

## 5）反例/陷阱
- 选择多列时忘记逗号或漏写 `FROM`
- 在真实系统中滥用 `SELECT *`，导致结果冗余

## 6）小结
掌握了最基础的 `SELECT ... FROM ...` 以及 `*` 的用法。

# 03 SQL Lesson 2: Queries with constraints (Pt. 1)
## 1）本课要点
- 使用 `WHERE` 子句过滤行，避免返回无关数据
- 条件可通过 `AND` / `OR` 组合形成更复杂约束
- 常见数值约束操作符：`=`, `!=`, `<`, `<=`, `>`, `>=`, `BETWEEN`, `IN`, `NOT IN`
- 约束能减少结果集规模，提升查询可读性与效率
- SQL 关键字大小写不敏感，但建议大写书写

## 2）概念卡片
- WHERE：用于过滤行，限制结果集
- AND/OR：组合多个条件
- BETWEEN/IN：表达区间或离散值约束
- 约束：减少结果集规模并提升可读性

## 3）查询模板清单
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
题目1：查询 `id = 6` 的电影
参考解：
```sql
SELECT *
FROM movies
WHERE id = 6;
```

题目2：查询上映年份在 2000–2010 之间的电影
参考解：
```sql
SELECT *
FROM movies
WHERE year BETWEEN 2000 AND 2010;
```

题目3：查询上映年份不在 2000–2010 之间的电影
参考解：
```sql
SELECT *
FROM movies
WHERE year NOT BETWEEN 2000 AND 2010;
```

题目4：查询前 5 部 Pixar 电影及其上映年份
参考解：
```sql
SELECT title, year
FROM movies
WHERE id BETWEEN 1 AND 5;
```

## 5）反例/陷阱
- `BETWEEN` 是包含边界的区间
- `IN`/`NOT IN` 用于一组离散值，注意括号与逗号

## 6）小结
通过 `WHERE` 约束实现了对结果集的精确筛选。

# 04 SQL Lesson 3: Queries with constraints (Pt. 2)
## 1）本课要点
## 2）语法与示例
## 3）练习记录
## 4）易错点
## 5）小结

# 05 SQL Lesson 4: Filtering and sorting Query results
## 1）本课要点
## 2）语法与示例
## 3）练习记录
## 4）易错点
## 5）小结

# 06 SQL Review: Simple SELECT Queries
## 1）本课要点
## 2）语法与示例
## 3）练习记录
## 4）易错点
## 5）小结

# 07 SQL Lesson 6: Multi-table queries with JOINs
## 1）本课要点
## 2）语法与示例
## 3）练习记录
## 4）易错点
## 5）小结

# 08 SQL Lesson 7: OUTER JOINs
## 1）本课要点
## 2）语法与示例
## 3）练习记录
## 4）易错点
## 5）小结

# 09 SQL Lesson 8: A short note on NULLs
## 1）本课要点
## 2）语法与示例
## 3）练习记录
## 4）易错点
## 5）小结

# 10 SQL Lesson 9: Queries with expressions
## 1）本课要点
## 2）语法与示例
## 3）练习记录
## 4）易错点
## 5）小结

# 11 SQL Lesson 10: Queries with aggregates (Pt. 1)
## 1）本课要点
## 2）语法与示例
## 3）练习记录
## 4）易错点
## 5）小结

# 12 SQL Lesson 11: Queries with aggregates (Pt. 2)
## 1）本课要点
## 2）语法与示例
## 3）练习记录
## 4）易错点
## 5）小结

# 13 SQL Lesson 12: Order of execution of a Query
## 1）本课要点
## 2）语法与示例
## 3）练习记录
## 4）易错点
## 5）小结

# 14 SQL Lesson 13: Inserting rows
## 1）本课要点
## 2）语法与示例
## 3）练习记录
## 4）易错点
## 5）小结

# 15 SQL Lesson 14: Updating rows
## 1）本课要点
## 2）语法与示例
## 3）练习记录
## 4）易错点
## 5）小结

# 16 SQL Lesson 15: Deleting rows
## 1）本课要点
## 2）语法与示例
## 3）练习记录
## 4）易错点
## 5）小结

# 17 SQL Lesson 16: Creating tables
## 1）本课要点
## 2）语法与示例
## 3）练习记录
## 4）易错点
## 5）小结

# 18 SQL Lesson 17: Altering tables
## 1）本课要点
## 2）语法与示例
## 3）练习记录
## 4）易错点
## 5）小结

# 19 SQL Lesson 18: Dropping tables
## 1）本课要点
## 2）语法与示例
## 3）练习记录
## 4）易错点
## 5）小结

# 20 SQL Lesson X: To infinity and beyond!
## 1）本课要点
## 2）语法与示例
## 3）练习记录
## 4）易错点
## 5）小结

# 附录
## 1）常用 SQL 速查
## 2）常见错误清单
## 3）个人总结与复盘
