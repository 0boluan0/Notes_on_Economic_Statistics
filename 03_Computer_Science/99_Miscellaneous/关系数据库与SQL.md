---
aliases:
  - 关系数据库与 SQL 连续阅读
  - Relational databases and SQL
---

# 关系数据库与 SQL

<!-- bilingual-en:start -->
*Relational databases and SQL*
<!-- bilingual-en:end -->

学 SQL，需要同时看清两件事：查询怎样把输入事实变成答案，数据库又怎样让这些事实在修改和并发访问中保持有效。本页沿这两条线连续展开，必要的共享原子直接嵌入正文；地图用来回看它们之间的联系。

<!-- bilingual-en:start -->
Learning SQL means understanding both how a query turns input facts into an answer and how the database keeps those facts valid during updates and concurrent access. This page develops both threads through embedded shared atoms; use the maps to revisit their relationships.
<!-- bilingual-en:end -->

[[关系模型与 SQL 查询.canvas|查询关系图]] · [[数据库事务、约束与索引.canvas|事务、约束与索引关系图]] · [SQLBolt 练习](https://sqlbolt.com/) · [[知识原子.base#定义与概念入口|定义与概念入口]]

第一次从第一部分读起，先做到能解释“一行代表什么”和“为什么保留或丢掉这行”。第二部分再处理提交、并发与性能。语法练习仍在 SQLBolt 进行；产品特有的语义在这里以 PostgreSQL 18 为准。

<!-- bilingual-en:start -->
On a first pass, begin with Part I and learn to explain what a row represents and why it is retained or discarded. Part II then covers commits, concurrency, and performance. Use SQLBolt for syntax practice; product-specific semantics here refer to PostgreSQL 18.
<!-- bilingual-en:end -->

## 第一部分：沿着结果表理解 SELECT

<!-- bilingual-en:start -->
*Part I: Understand SELECT by following the result table*
<!-- bilingual-en:end -->

读 SQL 时，始终带着一个问题：**这一步的一行代表什么，为什么会出现在结果里？** 下面先从关系的结构与身份出发，再经过筛选、连接和汇总，最后延伸到分析查询。基础语法与交互题可在 [SQLBolt](https://sqlbolt.com/) 练习，具体语义按 PostgreSQL 18 文档核对。

<!-- bilingual-en:start -->
Keep one question in mind while reading SQL: **What does a row represent at this stage, and why does it appear?** Begin with structure and identity, follow filtering, joining, and summarising, then continue into analytical queries. Practise basic syntax and interactive problems on [SQLBolt](https://sqlbolt.com/), with precise semantics checked against PostgreSQL 18 documentation.
<!-- bilingual-en:end -->

### 从“一行是什么”走到“怎样识别它”

<!-- bilingual-en:start -->
*From what a row means to how it is identified*
<!-- bilingual-en:end -->

先把熟悉的表格外观换成对事实的描述：哪些属性共同表达一件事？这决定后续查询究竟在处理什么。

<!-- bilingual-en:start -->
Begin by describing the facts behind the familiar grid: which attributes together express one fact? This establishes what the query will operate on.
<!-- bilingual-en:end -->

![[00_Knowledge/关系模型结构]]

表结构告诉我们有哪些属性，粒度则把它们合起来解释成“一行一个什么”。先分别说清输入与目标结果的粒度，稍后才能判断连接增加的行是否符合问题。

<!-- bilingual-en:start -->
The structure lists the attributes; grain explains what they mean together as one row. State the input and intended output grains separately so that extra rows produced by a join can later be judged against the question.
<!-- bilingual-en:end -->

![[00_Knowledge/关系表粒度]]

粒度确定后，再选择能够识别这个事实单位的属性组合。读主键时，特别注意“组合唯一”与“每列都唯一”的区别。

<!-- bilingual-en:start -->
Once the grain is clear, choose the attributes that identify that unit of fact. When reading a primary key, distinguish a unique combination from individually unique columns.
<!-- bilingual-en:end -->

![[00_Knowledge/主键约束]]

识别自己的行之后，表还需要引用其他表中的对象。外键给这种引用加上检查；它能保证什么、还缺少什么约束，会直接影响我们对连接行数的预期。

<!-- bilingual-en:start -->
After identifying its own rows, a table may need to reference objects in another table. A foreign key checks those references. Its guarantees and remaining constraints shape our expectations about join size.
<!-- bilingual-en:end -->

![[00_Knowledge/外键与关系基数]]

如果身份来自自动生成的 `id`，还要回到业务事实再检查一次：两个不同编号是否可能描述同一件事？

<!-- bilingual-en:start -->
If identity comes from a generated `id`, return to the domain facts for another check: could two different IDs describe the same fact?
<!-- bilingual-en:end -->

![[00_Knowledge/代理键与业务唯一性]]

### 从输入事实走到查询结果

<!-- bilingual-en:start -->
*From input facts to query results*
<!-- bilingual-en:end -->

有了行身份，再看最简单的 `SELECT`：当它只输出部分属性时，不同事实可能显示成相同的一行。接下来要分清“结果看起来相同”与“应当只计一次”。

<!-- bilingual-en:start -->
With row identity established, consider a simple `SELECT`: projecting only some attributes can make different facts look identical. We now need to distinguish identical-looking output from facts that should count only once.
<!-- bilingual-en:end -->

![[00_Knowledge/SQL 重复行与 DISTINCT]]

多一个子句，结果就多一个需要解释的阶段。下面的逻辑顺序是一张阅读地图：先找到数据来源，再追踪哪些行留下、何时形成组、何时产生输出列。

<!-- bilingual-en:start -->
Each added clause introduces another stage to explain. The logical sequence below is a reading map: start with the source, then track retained rows, group formation, and output columns.
<!-- bilingual-en:end -->

![[00_Knowledge/SELECT 逻辑处理顺序]]

正式判断哪些行留下之前，先处理“不知道这个值”的情况。否则，把条件取反时就容易漏掉本来想保留的记录。

<!-- bilingual-en:start -->
Before deciding which rows survive, account for values that are unknown. Otherwise, negating a condition can miss records that the question intended to retain.
<!-- bilingual-en:end -->

![[00_Knowledge/NULL 与 UNKNOWN]]

现在把三值逻辑放进 `WHERE`：逐行判断时，能进入下一阶段的究竟是哪一种结果？

<!-- bilingual-en:start -->
Now apply three-valued logic to `WHERE`: which predicate result actually allows a row into the next stage?
<!-- bilingual-en:end -->

![[00_Knowledge/WHERE 与三值逻辑]]

### 连接之后，重新确认保留了谁

<!-- bilingual-en:start -->
*After joining, check who remains*
<!-- bilingual-en:end -->

当答案需要两张表的信息时，回到前面的键与粒度，按匹配行对解释 `JOIN`。把一张表中的某一行代入连接条件，数清另一张表有几个匹配，通常比盯着最终结果猜原因更容易。

<!-- bilingual-en:start -->
When an answer needs information from two tables, return to keys and grain and interpret `JOIN` through matching pairs. Take one input row and count its matches on the other side before trying to explain the final result.
<!-- bilingual-en:end -->

![[00_Knowledge/JOIN 与关系基数]]

接着增加一个业务要求：“即使没有匹配，也要显示这个对象。”这时不仅连接类型重要，条件放在匹配阶段还是放在连接之后也会改变答案。

<!-- bilingual-en:start -->
Now add a requirement: show the object even when it has no match. Both the join type and the placement of the condition—during matching or afterwards—matter to the answer.
<!-- bilingual-en:end -->

![[00_Knowledge/LEFT JOIN 条件位置]]

### 汇总之前，确认正在数什么

<!-- bilingual-en:start -->
*Before summarising, establish what is being counted*
<!-- bilingual-en:end -->

外连接补出的那一行，让“有一行结果”与“有一个真实匹配”成为两个问题。开始统计时，要选出能够区分它们的计数表达式。

<!-- bilingual-en:start -->
A null-extended outer-join row makes “there is an output row” different from “there is a real match.” Choose a counting expression that distinguishes the two.
<!-- bilingual-en:end -->

![[00_Knowledge/COUNT 与 NULL]]

当问题从“哪些订单符合条件”转成“哪些客户的汇总达标”，判断对象已经从输入行变成了组。用这个变化理解 `WHERE` 与 `HAVING`，比仅凭关键字记位置更可靠。

<!-- bilingual-en:start -->
When the question changes from qualifying orders to qualifying customer totals, the object being judged changes from an input row to a group. Use that change to understand `WHERE` and `HAVING`.
<!-- bilingual-en:end -->

![[00_Knowledge/WHERE 与 HAVING]]

把这些判断放进一个小例子：`customers` 一行一个客户，主键为 `customer_id`，现有客户 1、2、3；`orders` 一行一个订单，主键为 `order_id`，金额均非空。订单 101、102 属于客户 1，状态均为 `paid`，金额各为 90；订单 103 属于客户 2，状态为 `pending`，金额为 50；客户 3 没有订单。目标是**每位客户一行，显示已付款订单数与金额合计**，并约定没有已付款订单时金额合计为 0。

<!-- bilingual-en:start -->
Combine these decisions in a small example. `customers` has one row per customer, keyed by `customer_id`, containing customers 1, 2, and 3. `orders` has one row per order, keyed by `order_id`, with non-null amounts. Orders 101 and 102 belong to customer 1, are both `paid`, and each has amount 90. Order 103 belongs to customer 2, is `pending`, and has amount 50. Customer 3 has no orders. The target is **one row per customer, showing paid-order count and total**, with no paid orders interpreted as a total of zero.
<!-- bilingual-en:end -->

```sql
SELECT c.customer_id,
       COUNT(o.order_id) AS paid_orders,
       COALESCE(SUM(o.amount), 0) AS paid_total
FROM customers AS c
LEFT JOIN orders AS o
  ON o.customer_id = c.customer_id
 AND o.status = 'paid'
GROUP BY c.customer_id
ORDER BY c.customer_id;
```

| customer_id | paid_orders | paid_total |
| --- | --- | --- |
| 1 | 2 | 180 |
| 2 | 0 | 0 |
| 3 | 0 | 0 |

客户 2 与 3 都被保留；只有真实订单的非空 `order_id` 被计数。若把 `COUNT(o.order_id)` 换成 `COUNT(*)`，二人的计数都会变成 1，因为补空行也被数了进去；若增加 `HAVING COUNT(o.order_id) >= 2`，则是在新的问题下只显示客户 1。[连接规则](https://www.postgresql.org/docs/18/queries-table-expressions.html#QUERIES-JOIN)与[聚合规则](https://www.postgresql.org/docs/18/functions-aggregate.html)共同解释这些结果。

<!-- bilingual-en:start -->
Customers 2 and 3 remain, but only non-null IDs of actual orders are counted. Replacing `COUNT(o.order_id)` with `COUNT(*)` would give both customers a count of 1 because their null-extended rows would count. Adding `HAVING COUNT(o.order_id) >= 2` asks a new question and retains only customer 1. These results follow from the [join rules](https://www.postgresql.org/docs/18/queries-table-expressions.html#QUERIES-JOIN) together with the [aggregate rules](https://www.postgresql.org/docs/18/functions-aggregate.html).
<!-- bilingual-en:end -->

假如这个查询继续连接商品明细，订单级金额就可能被多次带入汇总。下一张卡把前面关于身份、重复行和连接基数的判断合在一起，检查金额为何会算大。

<!-- bilingual-en:start -->
If the query also joins item details, an order-level amount may enter the total several times. The next card combines identity, duplicate rows, and join cardinality to explain inflated totals.
<!-- bilingual-en:end -->

![[00_Knowledge/一对多 JOIN 重复聚合]]

### 延伸：从不存在的匹配走到分阶段分析

<!-- bilingual-en:start -->
*Continuation: from missing matches to staged analysis*
<!-- bilingual-en:end -->

基础筛选、连接和分组之后，可以继续处理更复杂的问题。下面的反连接、CTE 与窗口函数是进一步阅读的内容。

<!-- bilingual-en:start -->
After basic filtering, joins, and grouping, continue to more complex questions. The anti-join, CTE, and window material below is further reading.
<!-- bilingual-en:end -->

若只想知道某对象“有没有符合条件的关联行”，无需把每个匹配都铺开成结果。先看否定存在性与逐值比较的差别，尤其检查右侧含 `NULL` 时会发生什么。

<!-- bilingual-en:start -->
If the question only asks whether a qualifying related row exists, there is no need to expand every match into output. Compare negated existence with value-by-value comparison, especially when the right-hand side contains `NULL`.
<!-- bilingual-en:end -->

![[00_Knowledge/NOT IN 与 NOT EXISTS]]

若问题确实需要先汇总再连接，便给这个中间结果一个能说出粒度的名字。CTE 帮助读者看清阶段；至于这些阶段如何执行，需要另读数据库实现的规则。

<!-- bilingual-en:start -->
If the question needs aggregation before joining, give the intermediate result a name that expresses its grain. A CTE exposes that stage to the reader; its execution behaviour depends on the database's rules.
<!-- bilingual-en:end -->

![[00_Knowledge/CTE与子查询物化边界]]

分阶段查询还解决另一类需求：既要显示当前行，又想把它与同组其他行比较。这一次先保留当前粒度，让窗口计算为每行增加一个分析结果。

<!-- bilingual-en:start -->
Staged queries also help when each current row must remain visible while being compared with related rows. Keep the current grain and let a window calculation attach an analytical result to each row.
<!-- bilingual-en:end -->

![[00_Knowledge/窗口函数行身份]]

最后检查答案里的“第一”“相邻”和“前几名”究竟由什么决定。窗口内的 `ORDER BY` 决定窗口计算的先后，最外层 `ORDER BY` 决定展示顺序；需要稳定区分并列行时，要在相应位置补足决定键。[PostgreSQL 18 窗口教程](https://www.postgresql.org/docs/18/tutorial-window.html)明确区分这两个顺序。

<!-- bilingual-en:start -->
Finally, determine what “first,” “adjacent,” and “top rows” mean in the answer. Window ordering governs the calculation; the outermost `ORDER BY` governs presentation. Add sufficient tie-breakers at the relevant level when tied rows must be distinguished consistently. The [PostgreSQL 18 window tutorial](https://www.postgresql.org/docs/18/tutorial-window.html) distinguishes these two orders.
<!-- bilingual-en:end -->

![[00_Knowledge/ORDER BY 的充分键]]

走完这条查询路径后，可以把注意力从“怎样得到答案”转向“数据怎样持续保持有效”：模式与约束如何表达业务规则，事务如何处理并发修改，索引又如何影响访问成本。

<!-- bilingual-en:start -->
After following the query path, turn from obtaining an answer to keeping the underlying data valid: how schemas and constraints express domain rules, how transactions handle concurrent changes, and how indexes affect access cost.
<!-- bilingual-en:end -->

## 第二部分：让数据在修改中保持有效

<!-- bilingual-en:start -->
*Part II: Keeping data valid through updates*
<!-- bilingual-en:end -->

### 事务：把一次业务改变作为整体

<!-- bilingual-en:start -->
*Transactions: treating a business change as a whole*
<!-- bilingual-en:end -->

从这里开始进入数据库实现与应用设计的延伸，超出 SQLBolt 最初的 `SELECT` 练习。以下具体行为以 PostgreSQL 18 为准。先把注意力从“查出哪些行”移到“订单写入和库存扣减必须怎样一起成立”：这需要明确一次改变的提交边界。

<!-- bilingual-en:start -->
This part extends into database implementation and application design, beyond SQLBolt's initial `SELECT` exercises. Product-specific behavior below refers to PostgreSQL 18. Shift attention from which rows a query returns to how an order and its inventory deduction must succeed together: this requires an explicit commit boundary.
<!-- bilingual-en:end -->

![[事务提交原子性]]

整组更改同成同败之后，还要判断数据库何时确认了提交，以及这份确认覆盖哪些故障。

<!-- bilingual-en:start -->
After establishing all-or-nothing behavior, ask when the database acknowledged the commit and which failures that acknowledgement covers.
<!-- bilingual-en:end -->

![[持久性承诺边界]]

能够整体提交并保存下来，仍需要规则说明什么状态才合法。库存非负、订单必须指向已有商品，都是需要具体表达的不变量。

<!-- bilingual-en:start -->
A change that commits and survives still needs rules defining a valid state. Nonnegative inventory and orders referencing existing products are invariants that must be expressed explicitly.
<!-- bilingual-en:end -->

![[ACID 一致性的边界]]

### 并发：读取依据能保持到什么时候

<!-- bilingual-en:start -->
*Concurrency: how long a read remains a valid basis for a decision*
<!-- bilingual-en:end -->

一个请求读到库存为 1 时，另一个请求可能正在购买同一商品。先确定数据库允许哪些交错，再判断业务规则是否受到了足够保护。

<!-- bilingual-en:start -->
When one request reads a stock of 1, another may be buying the same product. Identify the permitted interleavings before deciding whether the business rule is sufficiently protected.
<!-- bilingual-en:end -->

![[隔离级别与业务规则]]

先读 PostgreSQL 的默认级别，把普通读取的快照与等待后的更新条件重查分开理解。

<!-- bilingual-en:start -->
Start with PostgreSQL's default level, distinguishing an ordinary read's snapshot from predicate rechecking after an update waits.
<!-- bilingual-en:end -->

![[PG Read Committed]]

如果两次查询必须面对同一批已提交数据，可以继续看稳定的事务快照；同时保留对跨行规则的检查。

<!-- bilingual-en:start -->
When successive queries need the same view of committed data, consider a stable transaction snapshot while continuing to examine cross-row rules.
<!-- bilingual-en:end -->

![[PG Repeatable Read]]

若正确性依赖串行等价，应让相关读写都采用 Serializable，并把整个事务的重试纳入应用流程。一个事务单独正确，是用这份并发保证保护业务规则的起点。[PostgreSQL 应用级一致性](https://www.postgresql.org/docs/18/applevel-consistency.html)

<!-- bilingual-en:start -->
If correctness relies on serial equivalence, use Serializable for the relevant reads and writes and handle whole-transaction retries. The starting point is a transaction that preserves the rule when run alone. [PostgreSQL application-level consistency](https://www.postgresql.org/docs/18/applevel-consistency.html)
<!-- bilingual-en:end -->

![[PG Serializable]]

### 从先查后写走到受保护的更新

<!-- bilingual-en:start -->
*From check-then-write to a protected update*
<!-- bilingual-en:end -->

现在回到库存为 1 的场景，沿着两个请求的读取和写入顺序，找出检查结果可能过期的位置。

<!-- bilingual-en:start -->
Return to the stock of 1 and follow both requests' reads and writes to locate where an observation can become stale.
<!-- bilingual-en:end -->

![[先查后写竞态]]

下面的库存例子假定 `product_id` 是[[主键约束|主键]]，`stock` 是非空整数，并采用普通表读写，没有改变更新行为的额外机制。需要守住的局部规则是“只有当前库存大于 0，才能扣减 1”。

<!-- bilingual-en:start -->
The inventory example assumes that `product_id` is the [[主键约束|primary key]], `stock` is a non-null integer, and ordinary table operations have no additional mechanisms altering update behavior. The local rule is to deduct 1 only when current stock is positive.
<!-- bilingual-en:end -->

![[条件更新]]

纸上推演：A 先扣减并持有行锁，B 的同一条件更新等待；A 提交后，B 在库存 0 上重查条件，返回零行。应用必须识别这个结果；若更新属于更大的显式事务，返回一行也只是语句成功，整个业务操作仍要等提交确认。[PostgreSQL Read Committed](https://www.postgresql.org/docs/18/transaction-iso.html#XACT-READ-COMMITTED)

<!-- bilingual-en:start -->
Trace it on paper: A deducts and holds the row lock; B waits. After A commits, B rechecks stock 0 and returns no rows. The application must handle that outcome; within a larger explicit transaction, one returned row confirms statement success, while the business operation still awaits commit acknowledgement. [PostgreSQL Read Committed](https://www.postgresql.org/docs/18/transaction-iso.html#XACT-READ-COMMITTED)
<!-- bilingual-en:end -->

需要在多条语句之间保护一个已有资源时，接着考虑显式锁，并确认锁住的资源覆盖了判断所依赖的状态。

<!-- bilingual-en:start -->
When an existing resource needs protection across several statements, consider explicit locking and check that the locked resources cover the state on which the decision depends.
<!-- bilingual-en:end -->

![[行级锁的边界]]

多个资源一起参与事务时，锁的获取顺序也成为正确执行流程的一部分。

<!-- bilingual-en:start -->
When a transaction touches several resources, lock acquisition order becomes part of its execution design.
<!-- bilingual-en:end -->

![[死锁与事务重试]]

### 约束：让合法性规则进入数据库

<!-- bilingual-en:start -->
*Constraints: putting validity rules in the database*
<!-- bilingual-en:end -->

并发策略之外，还要让所有写入入口遵守相同规则。回看[[主键约束|主键的行身份]]与[[外键与关系基数|外键的参照保证]]，再补上值域与跨行判重。

<!-- bilingual-en:start -->
Alongside the concurrency strategy, every write path must respect the same rules. Revisit [[主键约束|row identity through a primary key]] and [[外键与关系基数|reference integrity through a foreign key]], then add value restrictions and uniqueness across rows.
<!-- bilingual-en:end -->

![[数据库约束]]

这里可以把查询部分的三值逻辑重新用起来：同一个 UNKNOWN，在筛选行和验证 `CHECK` 时承担不同角色。

<!-- bilingual-en:start -->
Reuse three-valued logic from the query section: UNKNOWN has different consequences when filtering rows and validating a `CHECK`.
<!-- bilingual-en:end -->

![[PG CHECK 与 NULL]]

从单行的合法取值走到多行的唯一组合，还要明确缺失值是否允许重复，以及规则覆盖全表还是某个子集。

<!-- bilingual-en:start -->
Moving from valid values within a row to unique combinations across rows requires deciding how nulls participate and whether the rule covers the whole table or a subset.
<!-- bilingual-en:end -->

![[PG UNIQUE 与 NULL]]

### 查询正确之后：索引与执行计划

<!-- bilingual-en:start -->
*After correctness: indexes and execution plans*
<!-- bilingual-en:end -->

现在已有查询语义、写入规则和并发边界，可以进一步看执行代价。索引是否有价值，要放回实际的读写工作负载中判断。

<!-- bilingual-en:start -->
With query semantics, write rules, and concurrency boundaries established, examine execution cost. Evaluate an index against the actual mix of reads and writes.
<!-- bilingual-en:end -->

![[索引的读写代价]]

对于多列索引，先把查询描述成“固定哪些列，再在哪一列取范围”，再分析 B-tree 的列顺序。

<!-- bilingual-en:start -->
For a multicolumn index, describe which columns the query fixes and where its range begins, then examine the B-tree column order.
<!-- bilingual-en:end -->

![[多列 B-tree 列顺序]]

适合查询形状的索引提供了一条候选访问路径；接下来要看规划器如何比较它与其他路径。

<!-- bilingual-en:start -->
An index matching the query shape offers a candidate access path. Next, examine how the planner compares it with the alternatives.
<!-- bilingual-en:end -->

![[查询规划器与索引]]

最后把“预计便宜”与实际运行证据对照。初次阅读先辨认计划字段；实际试验应在练习数据库中进行，并先确认 `ANALYZE` 选项会执行的语句及其副作用。

<!-- bilingual-en:start -->
Finally, compare estimated cost with execution evidence. On a first reading, identify the plan fields; use a practice database for experiments and inspect the statement and side effects that the `ANALYZE` option will execute.
<!-- bilingual-en:end -->

![[EXPLAIN 估算与实测]]

## 把整条路径用起来

<!-- bilingual-en:start -->
*Putting the whole path to work*
<!-- bilingual-en:end -->

读完查询部分后，合上正文，重新写出三位客户的已付款订单统计，并解释为什么缺少匹配的客户仍出现、为什么计数为 0。再把问题改成“只显示至少两笔已付款订单的客户”，说清应在哪个阶段加条件。

<!-- bilingual-en:start -->
After the query section, close the notes and reconstruct the three-customer paid-order report. Explain why customers without matching orders remain and why their counts are zero. Then change the question to customers with at least two paid orders and identify the stage where that condition belongs.
<!-- bilingual-en:end -->

读完数据库部分后，在纸上画出库存为 1 时两个请求的交错过程，分别标出条件检查、等待、更新和提交。最后拿一条慢查询，先说明结果应该是什么，再决定要检查哪项计划证据；真实执行留在练习数据库中。

<!-- bilingual-en:start -->
After the database section, draw how two requests interleave when stock is 1, marking condition checks, waits, updates, and commits. For a slow query, establish the intended result before choosing which execution-plan evidence to inspect. Keep actual experiments in a practice database.
<!-- bilingual-en:end -->

## 来源与核验

- [SQLBolt 课程](https://sqlbolt.com/)：提供入门课程顺序与练习语境；窗口、CTE 和数据库并发是后续延伸。
- PostgreSQL 18 [表表达式](https://www.postgresql.org/docs/18/queries-table-expressions.html)、[聚合函数](https://www.postgresql.org/docs/18/functions-aggregate.html)与[窗口教程](https://www.postgresql.org/docs/18/tutorial-window.html)：核验查询衔接、三客户算例、空值计数与排序作用域。
- PostgreSQL 18 [事务隔离](https://www.postgresql.org/docs/18/transaction-iso.html)、[应用级一致性](https://www.postgresql.org/docs/18/applevel-consistency.html)、[约束](https://www.postgresql.org/docs/18/ddl-constraints.html)与[EXPLAIN](https://www.postgresql.org/docs/18/using-explain.html)：核验并发推演、提交边界和性能诊断；各原子末尾保留更细的来源及适用条件。
