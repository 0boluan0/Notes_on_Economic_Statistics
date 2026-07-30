---
aliases:
  - "SQL and Relational Data"
  - "Structured Query Language"
  - "SQL"
status: source-checked
---

# SQL 查询与关系数据

> [!summary] 快速恢复
> **它解决什么：** 用表、键和声明式查询从关系数据中筛选、连接、聚合，并在并发更新下保持约束。
> **具体锚点：** `JOIN` 不是把两张表“贴上”，而是按条件形成匹配行；一对多连接会增加行数并可能重复聚合。
> **核心难点：** SQL 是集合/多重集语义，`NULL` 使用三值逻辑；查询结果正确性取决于粒度和键。
> **为什么重要：** 在分析前弄清一行代表什么，比写出能运行的查询更重要。
> **继续：** 先定义表粒度、主外键，再写 FROM/JOIN/WHERE/GROUP/HAVING/SELECT 的逻辑流程。

> [!source] 本节依据
> - [MIT Missing Semester official course](https://missing.csail.mit.edu/2020/) 与本地练习：支持 shell、Git、调试、构建与安全工作流。

## 关系与键

表表示同一粒度的关系，列有域。主键唯一标识行，外键引用另一表主键并维护参照完整性。自然键有业务含义，surrogate key 简化引用但不能替代业务唯一约束。

## SELECT 与逻辑顺序

概念顺序是 FROM/JOIN → WHERE → GROUP BY → HAVING → SELECT → ORDER/LIMIT。WHERE 过滤聚合前行，HAVING 过滤组。无 ORDER BY 不保证返回顺序。

## NULL 与三值逻辑

NULL 表示未知/缺失，不等于 0 或空串；比较用 `IS NULL`。`x = NULL` 得 unknown。聚合函数通常忽略 NULL，`COUNT(*)` 计行、`COUNT(col)` 计非 NULL。

## JOIN

INNER 保留匹配，LEFT 保留左表全部。连接条件缺失会笛卡尔积；一对多会复制“一”侧值。聚合前先验证每表和连接后的行粒度、匹配率与重复数。

## 聚合、子查询与窗口

GROUP BY 折叠为组；窗口函数在不折叠行的情况下计算排名、累计和或分组统计。子查询/CTE 用于分阶段表达，优化器可能内联，不应只为格式制造深层嵌套。

## 事务与并发

事务把更新作为原子单元，ACID 涵盖原子、一致、隔离、持久。隔离级别权衡并发与脏读/不可重复读/幻读；约束放数据库可防所有写入路径破坏规则。

## 索引与计划

索引加速特定查找/排序但增加存储和写成本。使用 query plan 检查扫描、连接顺序和估计；先建正确查询，再针对真实瓶颈加最小索引。

## 最小自检

### 一对多 JOIN 后为什么总金额可能被放大？

> [!answer]- 答案
> 一侧金额会随多侧匹配行重复，若直接求和会重复计；应先按正确粒度聚合或明确分配。
### `COUNT(*)` 与 `COUNT(col)` 有何差别？

> [!answer]- 答案
> 前者计所有行，后者只计该列非 NULL 的行。
### 为什么业务唯一性应有数据库约束？

> [!answer]- 答案
> 应用可能有多个写入路径或并发，数据库约束在共同边界阻止所有路径产生重复/非法状态。

## 来源与核验

- [MIT Missing Semester official course](https://missing.csail.mit.edu/2020/) 与本地练习：支持 shell、Git、调试、构建与安全工作流。
