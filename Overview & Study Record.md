
────────────────────────────────

Phase 1｜课程组合（Curriculum Stack）

────────────────────────────────

  

A. Python 编程与数据处理（面向 DS）

主选 1：MIT 6.100L（Python 入门到可写可测）

来源：csdiy 条目与公开视频/讲次（共 26 讲）

理由：你 Python 自评 2/10，最缺的是“能定义、能解释、能实现、能跑通”。6.100L 强调测试、调试、复杂度等，直接补你“能写能跑”的底座。

先修：无（英文阅读）。

预计总时长：50h+（csdiy估计）

产出：每讲 1 个可运行脚本/函数 + 1–3 个测试；每周 1 个小型整合脚本（数据读写/可视化/模拟）。

  

主选 2（接力）：UCB Data 100（数据科学工作流：pandas/EDA/推断/初级ML）

来源：csdiy 条目 + 官方 schedule（Lecture/Lab/HW 清单）

理由：把 Python 迅速落到 DS 典型任务：清洗、可视化、抽样/推断、简单预测。

先修：基础 Python、线代（你会同步补）

预计总时长：80h（csdiy估计）

产出：Lab/HW（可复现 notebook 或 .py），并把关键函数抽出来做成可测试模块。

  

替代方案（若你更想“更硬核编程训练”）：CS61A（强训练，但对你当前 DS 目标性略弱）。csdiy 首页提到 CS61A 作为 Python 入门路线之一

  

B. SQL/MySQL（面向 DS 分析 + Python 联动 + 简单 ETL）

主选：SQLBolt（快速补齐查询肌肉）

来源：SQLBolt 官方 lessons 列表（Lesson 1.. 等）

理由：交互式练习，适合你“每天必须能跑通”的节奏。

先修：无。

预计总时长：15–25h（按练习进度）。

产出：每周≥2天 SQL 练习记录（同一份 db 数据集），并沉淀成“查询模板库”。

  

Python↔MySQL 联动：MySQL Connector/Python 官方开发者指南（连接/查询/插入/游标 API）

来源：官方文档示例（连接、查询、fetchall、insert）

产出：每周≥1天做一次“CSV→清洗→入库→SQL分析→结果回到 pandas”的最小 ETL。

  

C. 机器学习（实用 + 必要理论）

主选：Stanford CS229（系统 ML）

来源：课程 schedule 页面 + syllabus pdf（讲次/主题）

理由：你理论占比 40%，CS229 的“推导 + 训练建议”覆盖面广；同时你会在每天“从零实现”里把核心算法撸成可跑模块。

先修：线代/微积分/概率统计。

预计总时长：100h+（按你作业深度）。

产出：每个主题一套：从零实现（numpy）+ sklearn 对照 + sanity checks。

  

D. 深度学习（高优先级，PyTorch）

主选：Dive into Deep Learning（D2L，含 PyTorch 代码与章节式目录）

来源：官方 Table of Contents（章/节完整目录）

理由：你想“能写能跑”，D2L每节就是可执行 notebook，适合每天 MVP + sanity checks。

先修：Python、线代/微积分（你同步补）。

预计总时长：120h（按你做到“从零实现”的严格程度）。

产出：手写训练循环/模块化模型；每周一个小实验（不是大项目）。

  

辅助（选读/对照）：CS231n 2024 schedule（看目录与 slides 追前沿主题）

注意：部分视频不对外开放，计划里只用“讲次标题 + slides/notes”作为阅读任务，不依赖视频访问。

  

E. 时间序列（金融场景优先）

主选：Forecasting: Principles and Practice (3rd ed, FPP3)

来源：官方目录页（章/节列表齐全，且标注最近更新时间）

理由：非常贴“DS/业务可用”的预测流程：探索→基线→CV→ETS/ARIMA/回归。你会用 Python/statsmodels 做等价实现（不被 R 限死）。

先修：统计、回归、基本线代。

预计总时长：60–80h。

产出：每章一个“可复现预测管线”脚本（含滚动验证）。

  

F. 图数据（优先级 2）

主选：Stanford CS224W（Graph ML）

来源：公开 schedule（Lecture 1..19 + 作业/colab）

理由：你想做“多关系图/超图/金融图”，CS224W覆盖 node embedding→GNN→异构图→KG。

先修：ML、线代、Python。

预计总时长：80–120h。

产出：至少 2 个小实验：GCN/GAT 从零实现 + PyG/DGL 对照。

  

G. 文本/NLP（优先级 3）

主选：Stanford CS224N（NLP with DL）

来源：公开 schedule（Week 1.. 主题与 slides/notes）

理由：你后续做金融文本/公告/舆情会直接用到：word vectors→RNN→seq2seq→transformers→pretraining。

先修：DL 基础、PyTorch、微积分/线代。

预计总时长：60–100h（你不一定做完全部大作业，重点做核心实现）。

产出：至少完成：word2vec（负采样）/attention/mini-transformer/微调一条端到端流程。

  

H. 工程化最低基线（Shell/Git/测试/调试）

主选：MIT Missing Semester（10 讲）

来源：2020 Lectures 列表（Lecture 1..10 主题清晰）

理由：你要的是“研究/实习能用”，工程化是底盘：shell、git、调试、测试、脚本化。

预计总时长：20–30h + 持续实践。

产出：把你所有课程代码都工程化：统一项目结构、Makefile/justfile、pytest、pre-commit（最小版也行）。

  

数学（系统重学，每周 6h）

线代：MIT 18.06（视频讲次列表公开）

微积分：MIT 18.01（视频讲次列表公开）

说明：数学不做“刷满全部题库”的幻想目标；每个你在 ML/DL/TS 用到的点，都必须能推到“可检验定义 + 反例 + 数值验证”。

  

────────────────────────────────

Phase 2｜把课程“结构化为可排程大纲”

（这里只列出你后面排程会引用的“讲次/章节编号与主题”，不编造内容）

────────────────────────────────

1. MIT 6.100L（Python）
    
    讲次：YouTube playlist 提供 Lecture 1–26（标题可见）
    
    排程粒度：每次 1 讲（必要时拆半讲）。
    
    练习：每讲自己写“同主题变体”+ 1–3 tests（pytest）。
    
    重点阶段：数据结构、函数/类、调试与复杂度。
    
2. UCB Data 100
    
    官方 schedule 按 Week 列出 Lecture/Lab/HW（可直接对齐周计划）
    
    例：Week1 Lecture1 Intro, Lecture2 Pandas I；配 Lab1/HW1A/1B 等
    
    练习来源：对应 Lab/HW（你本地下载后做）。
    
    排程粒度：Lecture + Lab/HW 的子任务拆到“每天可跑通”。
    
3. MIT Missing Semester
    
    10 讲主题列表公开：Shell、脚本、Vim、Data Wrangling、环境、Git、Debugging、Metaprogramming、Security、Potpourri
    
    练习来源：每讲配套 exercises（你把结果固化到 dotfiles/脚本仓库）。
    
4. SQLBolt
    
    Lessons 列表在 Review 页可见：Lesson 1–N（含 joins/null/expressions/aggregates 等）
    
    排程粒度：每天 1 个 lesson 或 1 个 review + 10 道变体题（同一库）。
    
    额外：每周做 1 次“把 SQL 写成可复用视图/CTE 模板”。
    
5. MySQL Connector/Python（联动/ETL）
    
    连接与查询示例：官方“Connecting”与“Querying”章节
    
    API：execute/fetchall 说明与示例
    
    插入：官方 insert 示例
    
    排程粒度：每周一次 ETL：建表→插入→查询→聚合→回写 pandas。
    
6. Stanford CS229（ML）
    
    官方 schedule 页面列出 lecture 主题与作业节点 （也可对照 syllabus pdf）
    
    排程粒度：每周 2–3 个 lecture 主题 + 1 次从零实现 + 1 次 sklearn 对照。
    
7. D2L（DL, PyTorch）
    
    官方 TOC 完整列出章节（2.1 Data Manipulation、3.4 Linear Regression from Scratch、4.4 Softmax from Scratch、5 MLP、6 Modules 等）
    
    排程粒度：每天 1–2 小节（保证“从零实现”与“简洁实现”都写一遍）。
    
8. FPP3（时间序列/预测）
    
    官方目录页列出 Chapter 1–13 及每节标题，适合按章排周
    
    排程粒度：每周 1 章（或半章）+ Python 等价实现 + 滚动 CV。
    
9. CS224W（图 ML）
    
    公开 schedule（Lecture 1–19）+ 作业/colab 节点齐全
    
    排程粒度：每周 2 个 lecture + 1 次 coding（Colab/自写）+ 1 次从零实现（不依赖 PyG 的核心层）。
    
10. CS224N（NLP）
    
    公开 schedule（至少 Week1–Week6 的 lecture 主题与 slides/notes/作业节点）可直接引用
    
    排程粒度：每周 2 个 lecture 主题 + 1 个核心算法从零实现（word2vec/attention/transformer）。
    

  

────────────────────────────────

你要求的“最小掌握阈值”（全程通用，写进计划）

────────────────────────────────

  

讲明白阈值（每个知识单元必须过）：

- 定义：一句话可检验定义（含符号/输入输出）。
    
- 边界：至少 2 条（适用条件 + 不适用/失败条件）。
    
- 例子：1 个最小例子（能算/能跑）。
    
- 常见误解/反例：至少 1 个，并解释为什么错。
    

  

实现阈值（每天都有，哪怕很小）：

- MVP：从零写出核心函数/模块（不复制粘贴整段答案）。
    
- Sanity checks：至少 2 条（维度/数值范围/单调性/对称性/梯度检查等）。
    
- 对照：尽可能与库函数（numpy/sklearn/torch/statsmodels）结果对照，误差在你写明的容许范围内。
    
- 可复现命令：给出 python -m ... 或 pytest -q 的运行方式与预期输出。
    

  

未达阈值处理：

- 任何一天未达阈值，必须标记为“欠债卡（Debt Card）”，并在周末毕业设计时间里留出 30–60 分钟“清债”（计划里已内置）。
    

  

────────────────────────────────

Obsidian 最小知识闭环模板（计划每天会引用）

────────────────────────────────

  

每天至少 1 条笔记，标题必须中英双语术语，例如：

“Softmax 回归（Softmax Regression）| 交叉熵（Cross-Entropy）”

  

笔记必填段落清单（你要求的那套）：

- What / Why
    
- 定义与边界（Definition & Assumptions）
    
- 最小例子（Minimal Example）
    
- 从零实现（From-Scratch MVP）
    
- 常见坑（Pitfalls）
    
- 自测清单（Self-check）
    
- 连接（Links：和哪些概念相连）
    

  

公式：$$…$$；行内变量：$…$；矩阵换行用 \\。

  

────────────────────────────────

Phase 3｜26周×按天计划（每周一表）

说明：周一到周五为“DS 主线 4–6h”；周末为“毕业设计主导 4–6h + 30–60min 轻量 DS 复盘/最小实现”。

  

表格字段保证齐全：时长/时间块、听课、练习、从零实现、笔记、当日验收（讲明白+跑通）。

  

下面开始从 第1周 周一 起排。

  第一周第一天是2026/2/2 后续以此类推,

# **============================================================**

#   

# **第1–4周（第1月）：把“能写能跑”立起来（Python + 工程化 + SQL 起步 + 数学重启）**

#   

# **完成标准：你能独立写一个小脚本完成“读数据→清洗→SQL入库→查询→画图”，并有基本测试；能把“函数/类/复杂度/调试”讲清楚。**

  

第1周（主课：MIT 6.100L L1–L4；Missing Semester L1）

| **日程** | **时长/时间块**                         | **听课任务**                      | **练习任务**                          | **从零实现（含测试/调试）**                                     | **Obsidian笔记 & 当日验收**      |
| ------ | ---------------------------------- | ----------------------------- | --------------------------------- | ---------------------------------------------------- | -------------------------- |
| 周一     | 5h：90m课/60m练/60m实现/30m测/30m记/30m复盘 | 6.100L L1                     | 写 3 个 I/O 小程序变体                   | 实现 cli_sumstats.py：读 stdin 数字→输出均值/方差；pytest 2例      | 笔记：Python 输入输出（I/O）        |
| 周二     | 5h                                 | Missing Semester L1 Shell     | shell 基础：ls/cd/grep 20条命令         | 写 run_all.sh 批量运行本周脚本；检查返回码                          | 笔记：Shell 命令行（Shell Basics） |
| 周三     | 5h                                 | 6.100L L2                     | 10道字符串/分支小题（自拟）                   | 实现 parse_prices.py：解析“symbol,price”→清洗异常；pytest 3例   | 笔记：分支（Branching）           |
| 周四     | 5h                                 | SQLBolt Lesson 1–2            | 完成 Lesson1–2 + 自改 10 查询           | 用 sqlite 或本地 MySQL（先用 sqlite）建 movies 表并导入样例         | 笔记：SELECT 与约束（Constraints） |
| 周五     | 5h                                 | 6.100L L3–L4                  | 循环题 15 道（含 off-by-one）            | 实现 rolling_mean.py（窗口均值）+ 2个sanity checks（常数序列/窗口=1） | 笔记：迭代（Iteration）           |
| 周六     | 4–6h（毕业设计主）+45m DS                 | 复盘本周 6.100L 任意1讲              | SQLBolt Review: Simple SELECT（做完） | 在毕业设计仓库加 pytest 与 1 个测试（最小）                          | 笔记：复盘（Weekly Review #1）    |
| 周日     | 4–6h（毕业设计主）+45m DS                 | 18.06 Linear Algebra L1（几何直观） | 线代题：向量/线性组合 5题                    | 实现 proj_onto_vector(a,b) 并验证投影性质（正交残差）               | 笔记：投影（Projection）          |

第2周（主课：6.100L L5–L8；Missing Semester L2；SQLBolt L3–4）

|**日程**|**时长/时间块**|**听课任务**|**练习任务**|**从零实现（含测试/调试）**|**Obsidian笔记 & 当日验收**|
|---|---|---|---|---|---|
|周一|5h|6.100L L5|函数题 10道（默认参数/返回）|写 stats.py：mean/var/cov（纯python）+ pytest 3例|笔记：函数（Functions）|
|周二|5h|Missing Semester L2 Shell Tools & Scripting|用 awk/sed 清洗 CSV 练 8条|写 clean_csv.sh + 对比 python 清洗结果一致|笔记：数据清洗（Data Wrangling）|
|周三|5h|6.100L L6|递归/分治基础题 8道|实现 binary_search + 反例测试（未排序输入）|笔记：递归（Recursion）|
|周四|5h|SQLBolt Lesson 3–4|完成 + 自拟 10 查询（ORDER/LIMIT）|实现 sql_runner.py（sqlite）支持读取 .sql 文件并输出表格|笔记：排序与分页（ORDER/LIMIT）|
|周五|5h|6.100L L7–L8|debugging 小题：给 5 段错代码找bug|实现 pytest 参数化测试（至少2组）到本周模块|笔记：调试（Debugging）|
|周六|4–6h毕业设计+45m DS|18.01 Calculus L1 Rate of Change|导数概念题 6题|写 finite_diff(f,x,h) 并对比解析导数（误差随 h 变化）|笔记：导数（Derivative）|
|周日|4–6h毕业设计+45m DS|复盘 6.100L 任意1讲|清债：本周未过阈值项|毕设仓库：加 1 个日志（logging）+ 1 个断言|笔记：清债记录（Debt Card Log）|

第3周（主课：6.100L L9–L12；Missing Semester L3；SQLBolt L6）

|**日程**|**时长/时间块**|**听课任务**|**练习任务**|**从零实现（含测试/调试）**|**Obsidian笔记 & 当日验收**|
|---|---|---|---|---|---|
|周一|5h|6.100L L9|数据结构题：list/dict 12题|实现 groupby_key(items,key_fn) + pytest 2例|笔记：字典（Dictionary）|
|周二|5h|Missing Semester L3 Editors (Vim)|vim 20个操作（宏/搜索/替换）|写 vim_cheatsheet.md + 在 repo 配 .editorconfig|笔记：编辑器（Editor）|
|周三|5h|6.100L L10|OOP 基础题 6道|实现 PriceSeries 类（append/rolling_mean）+ pytest|笔记：类与对象（OOP）|
|周四|5h|SQLBolt Lesson 6 JOIN|完成 + 自拟 10 JOIN 查询|实现 join_check.sql：用两种写法（JOIN/子查询）得同结果|笔记：JOIN（Inner Join）|
|周五|5h|6.100L L11–L12|复杂度直觉题 8道|写 timeit_bench.py 比较 O(n) vs O(n^2) 并画图|笔记：复杂度（Big-O）|
|周六|4–6h毕业设计+45m DS|18.06 L2 消元|线代消元 5题|实现 gauss_elim(A,b) + sanity checks（可逆/奇异）|笔记：高斯消元（Gaussian Elimination）|
|周日|4–6h毕业设计+45m DS|复盘 Missing Semester L1–3|清债|毕设：把一个脚本改成 python -m pkg.module 形式|笔记：工程化结构（Project Layout）|

第4周（里程碑#1，主课：6.100L L13–L20；Missing Semester L4–L6；SQLBolt L7–9）

|**日程**|**时长/时间块**|**听课任务**|**练习任务**|**从零实现（含测试/调试）**|**Obsidian笔记 & 当日验收**|
|---|---|---|---|---|---|
|周一|5h|6.100L L13|文件读写题 8道|实现 csv_loader.py + dtype/缺失值处理 + pytest|笔记：文件I/O（File IO）|
|周二|5h|Missing Semester L4 Data Wrangling|grep/sort/uniq 管道练 10条|写 pipeline.sh 生成统计报表（top-k）|笔记：管道（Pipes）|
|周三|5h|SQLBolt L7–9（OUTER/NULL/expressions）|完成 + 自拟 10查询|写 null_sanity.sql 验证 NULL 三值逻辑（反例）|笔记：NULL（Three-valued Logic）|
|周四|5h|Missing Semester L5 环境 + L6 Git|练 git：rebase/branch/tag（10操作）|建 ds_bootcamp 总仓库：src/tests/notebooks|笔记：Git 工作流（Git Workflow）|
|周五（里程碑日）|6h|复盘 6.100L 关键点（函数/类/复杂度）|SQL 综合 20题（含 JOIN/NULL）|从零做“小闭环”：CSV→sqlite→SQL聚合→matplotlib图；pytest≥3|笔记：里程碑#1（Milestone 1）|
|周六|4–6h毕业设计+45m DS|18.01 L2–L3 Limits/Derivatives|极限/导数题 6题|grad_check.py：数值导数验证（线性/二次）|笔记：数值梯度（Finite Difference）|
|周日|4–6h毕业设计+45m DS|清债/复盘|补齐未过阈值|毕设：加 1 个 Makefile 或 justfile（最小）|笔记：清债#1|

（到这里第1月结束：你已经有“Python可测+SQL基础+工程化骨架”。）

  

# **============================================================**

#   

# **第5–8周（第2月）：Data 100 上身（pandas/EDA/推断）+ MySQL 联动 + 数学继续**

#   

# **完成标准：你能用 pandas 做一次完整 EDA + 置换检验/抽样推断，并把核心数据写入 MySQL 做分析查询，再回到 Python 出图。**

  

第5周（主课：Data100 Week1–2；MySQL环境搭建）

参考 Data100 schedule：Week1 Lecture1 Intro/Lecture2 Pandas I；Week2 Lecture3 Pandas II/Lecture4 Pandas III

|**日程**|**时长/时间块**|**听课任务**|**练习任务**|**从零实现（含测试/调试）**|**Obsidian笔记 & 当日验收**|
|---|---|---|---|---|---|
|周一|5h|Data100 Lec1|按 Lec1 Note 复写关键代码|写 eda_skeleton.py：统一 EDA 报告函数（head/describe/missing) + pytest|笔记：EDA 生命周期（DS Lifecycle）|
|周二|5h|Data100 Lec2 Pandas I|Lab1/2A（按课程）|抽出 clean_columns(df) 做成库函数 + 2 tests|笔记：pandas 基础（Pandas I）|
|周三|5h|MySQL Connector 连接示例|安装 MySQL + 建库 + 建表 1 张|写 mysql_connect.py（读取 env 配置）+ 连接失败重试|笔记：Python↔MySQL连接（Connector）|
|周四|5h|Data100 Lec3 Pandas II|Discussion/Notebook（按课程）|实现 assert_schema(df, schema) + pytest|笔记：数据校验（Schema Validation）|
|周五|5h|SQLBolt Lesson10（Aggregates Pt.1）起|聚合题 15道|写 load_to_mysql.py：CSV→MySQL（insertmany） + sanity check 行数|笔记：GROUP BY（Aggregation）|
|周六|4–6h毕业设计+45m DS|18.06 L3–L4 矩阵乘/ LU|线代题 6题|lu_factor.py（不追求最优）+ 对照 scipy/numpy|笔记：LU分解（LU）|
|周日|4–6h毕业设计+45m DS|复盘 Data100 本周|清债|毕设：把数据处理逻辑抽成函数+1测试|笔记：清债#2|

第6周（主课：Data100 Week3–4：EDA/Regex/Viz；SQL 每周2天）

Data100 schedule 显示 Week3 进入 Data Wrangling/EDA，Week4 有 Regex/Viz

|**日程**|**时长/时间块**|**听课任务**|**练习任务**|**从零实现（含测试/调试）**|**Obsidian笔记 & 当日验收**|
|---|---|---|---|---|---|
|周一|5h|Data100 Lec5 EDA(Part1)|Lab/Notebook（按课程）|写 plot_univariate.py（直方/箱线）+ 2 checks（空列/全常数）|笔记：EDA图形（EDA Plots）|
|周二|5h|SQLBolt Aggregates Pt.2 + Review|15题（含 HAVING）|写 agg_templates.sql：常用聚合模板（CTE）|笔记：HAVING（Having vs Where）|
|周三|5h|Data100 Lec6 Regex|Regex 练习 20条（清洗文本列）|实现 regex_clean(series,pattern,repl) + tests|笔记：正则（Regex）|
|周四|5h|MySQL 查询示例 + fetchall|用 MySQL 写 10 条分析查询（聚合+JOIN）|query_to_df.py：SQL→pandas DataFrame（含参数化）|笔记：参数化查询（Parameterized Query）|
|周五|5h|Data100 Lec7 Viz I|HW（按课程）|写 viz_style.py 统一绘图风格 + 1测试（文件存在）|笔记：可视化原则（Visualization）|
|周六|4–6h毕业设计+45m DS|18.01 L4–L6 Chain/Implicit/ExpLog|微积分题 6题|写 logistic_grad.py 推导并数值验证梯度|笔记：链式法则（Chain Rule）|
|周日|4–6h毕业设计+45m DS|清债/复盘|补齐未过阈值|毕设：增加 README 的可复现命令段落|笔记：复盘#6|

第7周（主课：Data100 推断/抽样；MySQL ETL 固化）

|**日程**|**时长/时间块**|**听课任务**|**练习任务**|**从零实现（含测试/调试）**|**Obsidian笔记 & 当日验收**|
|---|---|---|---|---|---|
|周一|5h|Data100 Sampling 相关讲次（按 schedule Week3/4 后续）|练抽样/Bootstrap 5个小实验|bootstrap_ci.py + sanity（固定随机种子/覆盖率粗检）|笔记：Bootstrap（Bootstrap CI）|
|周二|5h|SQLBolt Subqueries（Topic）|子查询 15题|subquery_vs_join.sql 比较两种写法性能（EXPLAIN 可选）|笔记：子查询（Subquery）|
|周三|5h|Data100 假设检验（Permutation Test HW1A提到）|做一次置换检验（自选数据）|perm_test.py + 2 sanity（交换不变性/极端值）|笔记：置换检验（Permutation Test）|
|周四|5h|MySQL 插入/事务示例|设计一张 fact 表 + 2 dim 表|etl_daily.py：增量入库（按日期）+ 回滚测试|笔记：事务（Transaction）|
|周五|5h|Data100 小型ML（线性/逻辑）预告|用 sklearn 做 baseline（1个分类）|从零实现 logreg.py（梯度下降）+ 对照 sklearn AUC|笔记：逻辑回归（Logistic Regression）|
|周六|4–6h毕业设计+45m DS|18.06 L5–L6 转置/空间|线代题 6题|orth_proj.py：投影到子空间（用 QR）|笔记：子空间（Subspace）|
|周日|4–6h毕业设计+45m DS|清债/复盘|补齐未过阈值|毕设：加 1 个 profiling/计时记录|笔记：复盘#7|

第8周（里程碑#2：Data100 阶段闭环 + MySQL联动演示）

|**日程**|**时长/时间块**|**听课任务**|**练习任务**|**从零实现（含测试/调试）**|**Obsidian笔记 & 当日验收**|
|---|---|---|---|---|---|
|周一|5h|Data100 复盘：Pandas/EDA/推断|补齐未做的Lab/HW|把 Data100 关键函数抽成 ds_utils 包 + pytest≥5|笔记：Data100 总结（Data100 Wrap-up）|
|周二|5h|SQLBolt Unions/Intersections Topic|15题|写 report.sql：多表汇总报表（CTE+union）|笔记：集合运算（UNION）|
|周三|5h|MySQL 查询+fetchall 深入|写 10条窗口/分组查询（如支持）|sql_to_features.py：SQL聚合→特征表→训练集|笔记：特征工程（Feature Table）|
|周四|5h|18.01 L7–L8 积分入门|微积分题 6题|integrate_trapz.py + 对照解析积分|笔记：数值积分（Numerical Integration）|
|周五（里程碑日）|6h|口头彩排：Data100+SQL+ETL|SQL综合 30题|从零做：选择一份金融时序CSV→ETL入MySQL→SQL生成特征→python训练logreg→出图+报告|笔记：里程碑#2（Milestone 2）|
|周六|4–6h毕业设计+45m DS|清债|补齐|毕设：重构一个模块 + 2 tests|笔记：清债#2|
|周日|4–6h毕业设计+45m DS|休整+轻复盘|—|最小实现：写 notes_indexer.py 扫描 Obsidian 笔记生成索引|笔记：知识库索引（Notes Index）|

# **============================================================**

#   

# **第9–12周（第3月）：ML 核心（CS229）+ 时间序列预测入门（FPP3 1–5）+ SQL/ETL不掉线**

#   

# **完成标准：你能从零实现线性回归/逻辑回归/GDA/朴素贝叶斯/k-means/PCA，并能解释 bias-variance 与 CV；能做滚动时间序列CV与基线预测。**

  

第9周（主课：CS229 Lec1–3；FPP3 Ch1–2）

CS229 schedule：Lecture topics 在官方页 ；FPP3目录见官方页

|**日程**|**时长/时间块**|**听课任务**|**练习任务**|**从零实现（含测试/调试）**|**Obsidian笔记 & 当日验收**|
|---|---|---|---|---|---|
|周一|5h|CS229 Lec1 监督学习概览|推导：最小二乘（手写）|linreg_gd.py（GD/正规方程）+ 对照 sklearn|笔记：线性回归（Linear Regression）|
|周二|5h|SQL（每周第1天）|SQL 20题（聚合+JOIN）|feature_query.sql：用SQL做滞后特征（lag）近似|笔记：滞后特征（Lag Feature）|
|周三|5h|FPP3 Ch1–2（图形/白噪声）|做 3 个时序图/ACF|acf_from_scratch.py + 对照 statsmodels acf|笔记：ACF（Autocorrelation）|
|周四|5h|CS229 Lec2–3（逻辑回归/GLM）|推导 logloss/梯度|logreg_newton.py（牛顿法）+ 梯度检验|笔记：GLM（Generalized Linear Model）|
|周五|5h|MySQL联动（本周ETL日）|CSV→MySQL→SQL→pandas|etl_ts_features.py：滚动窗口特征入库 + 2 tests|笔记：滚动特征（Rolling Features）|
|周六|4–6h毕业设计+45m DS|18.06 L7–L8 正交/最小二乘|线代题 6题|qr_ls.py：QR解最小二乘 + 对照 numpy|笔记：最小二乘几何（LS Geometry）|
|周日|4–6h毕业设计+45m DS|清债/复盘|—|毕设：加 1 个 end-to-end 运行脚本（make run）|笔记：复盘#9|

第10周（主课：CS229 GDA/Naive Bayes/评估；FPP3 Ch3）

|**日程**|**时长/时间块**|**听课任务**|**练习任务**|**从零实现**|**笔记 & 验收**|
|---|---|---|---|---|---|
|周一|5h|CS229 GDA|手推：高斯判别边界|gda.py + 对照 sklearn QDA/LDA|笔记：GDA（Gaussian Discriminant）|
|周二|5h|SQL（每周第2天）|SQLBolt 复盘 joins/null|sql_tests.py：用 pytest 校验SQL查询结果（行数/聚合值）|笔记：SQL可测试性（Test SQL）|
|周三|5h|FPP3 Ch3 分解（STL）|分解 2 条时序|stl_like.py（简化：移动平均）+ sanity|笔记：分解（Decomposition）|
|周四|5h|CS229 Naive Bayes + Metrics|做混淆矩阵/PR-AUC|naive_bayes.py（文本词袋小样本）|笔记：朴素贝叶斯（Naive Bayes）|
|周五|5h|调试/测试训练（本周）|给本周3个模型补 tests|加 pre-commit（格式+pytest hook）|笔记：工程化检查（Pre-commit）|
|周六|4–6h毕业设计+45m DS|18.01 积分技巧（继续）|6题|softmax_stable.py：数值稳定实现 + 溢出反例|笔记：数值稳定（Numerical Stability）|
|周日|4–6h毕业设计+45m DS|清债|—|毕设：写一段 profiling 结果说明|笔记：复盘#10|

第11周（主课：CS229 k-means/GMM/EM；FPP3 Ch4–5）

|**日程**|**时长**|**听课**|**练习**|**从零实现**|**笔记 & 验收**|
|---|---|---|---|---|---|
|周一|5h|CS229 k-means|手推：目标函数|kmeans.py + 收敛曲线 + 对照 sklearn|笔记：k-means（Clustering）|
|周二|5h|SQL + MySQL ETL|写 10 条分组统计|daily_refresh.sql + etl_refresh.py（增量）|笔记：增量ETL（Incremental ETL）|
|周三|5h|FPP3 Ch4 特征；Ch5 workflow|做时序特征 5个|ts_cv.py：滚动CV拆分 + sanity（无泄露）|笔记：时序CV（Time Series CV）|
|周四|5h|CS229 EM for GMM|推导 E/M 步|gmm_em.py + 对照 sklearn GMM|笔记：EM算法（EM Algorithm）|
|周五|5h|小综合练习（2–4h）|选：聚类用于市场状态划分|端到端脚本：特征→聚类→可视化→报告|笔记：市场状态（Regime）|
|周六|4–6h毕业设计+45m DS|18.06 特征值直觉|6题|power_iteration.py + 对照 numpy eig|笔记：特征值（Eigenvalue）|
|周日|4–6h毕业设+45m DS|清债/复盘|—|毕设：写 1 个回归测试（防止旧bug复发）|笔记：复盘#11|

第12周（里程碑#3：ML+TS 入门闭环）

|**日程**|**时长**|**听课**|**练习**|**从零实现**|**笔记 & 验收**|
|---|---|---|---|---|---|
|周一|5h|CS229 PCA/SVD|推导：PCA目标|pca_from_scratch.py + 对照 sklearn|笔记：PCA（Dim Reduction）|
|周二|5h|SQL（保持）|20题（含子查询/聚合）|feature_store.sql：把因子/特征存成视图|笔记：视图（View）|
|周三|5h|FPP3 Ch5 评估与残差诊断|做残差白噪声检验|residual_diag.py（ACF+Ljung-Box）|笔记：残差诊断（Residual Diagnostics）|
|周四|5h|CS229 Decision Trees/Boosting（按 schedule 后续）|用 sklearn 跑 1 次GBDT|decision_stump.py（最小）+ sanity|笔记：Boosting（AdaBoost）|
|周五（里程碑日）|6h|口头彩排：ML+TS+SQL|综合题 40|从零做：滚动CV预测（基线/ARIMA简化/ML回归）+ MySQL特征 + 报告|笔记：里程碑#3|
|周六|4–6h毕业设+45m DS|清债|—|毕设：文档化“如何复现”|笔记：清债#3|
|周日|4–6h毕业设+45m DS|休整+轻复盘|—|最小实现：写 metrics.py（MAE/RMSE/MAPE）+ tests|笔记：预测指标（Forecast Metrics）|

# **============================================================**

#   

# **第13–16周（第4月）：DL 基础（D2L 2–6）+ 经典时间序列模型（FPP3 7–9）+ 工程化升级**

#   

# **完成标准：你能从零写出训练循环、softmax/MLP、正则化/初始化；能用 ETS/ARIMA 做滚动预测并做残差诊断；代码具备最小工程化（结构+测试+配置）。**

  

第13周（主课：D2L 2.1–3.5；FPP3 Ch7）

D2L TOC：2.1 Data Manipulation、3.4 Linear Regression from Scratch 等 ；FPP3 Ch7 目录

|**日程**|**时长**|**听课**|**练习**|**从零实现**|**笔记 & 验收**|
|---|---|---|---|---|---|
|周一|5h|D2L 2.1–2.2（Data Manip/Preprocess）|tensor/ndarray练 20小题|tensor_ops.py（broadcast/reshape）+ 3 tests|笔记：张量操作（Tensor Ops）|
|周二|5h|SQL（第1天）|20题|dataset_to_mysql.py：把训练数据入库+统计|笔记：数据入库（Load Data）|
|周三|5h|D2L 3.1–3.4（Linear Reg + From Scratch）|跟做 notebook|train_loop_v0.py（mini-batch+SGD）+ sanity（loss下降）|笔记：训练循环（Training Loop）|
|周四|5h|FPP3 Ch7 回归与预测|做一条回归预测（含季节哑变量）|ts_regression.py + 时间切分CV|笔记：时序回归（TS Regression）|
|周五|5h|D2L 3.5（Concise impl）|用 torch.nn 复写|model_compare.py：from-scratch vs nn.Module 误差对比|笔记：两种实现（Scratch vs Module）|
|周六|4–6h毕业设+45m DS|18.06 SVD（继续）|6题|svd_recon.py：低秩重构 + 误差曲线|笔记：SVD（SVD）|
|周日|4–6h毕业设+45m DS|清债|—|毕设：把实验参数改为 yaml 配置读取|笔记：配置管理（Config）|

第14周（主课：D2L 4.1–5.3；FPP3 Ch8 ETS）

|**日程**|**时长**|**听课**|**练习**|**从零实现**|**笔记 & 验收**|
|---|---|---|---|---|---|
|周一|5h|D2L 4.1–4.4（Softmax + scratch）|跟做+改数据集|softmax_scratch.py + 梯度检验|笔记：Softmax（Softmax）|
|周二|5h|SQL（第2天）|20题（含NULL/聚合）|sql_feature_tests.py：对关键聚合加断言|笔记：数据质量（Data Quality）|
|周三|5h|FPP3 Ch8 指数平滑|跑 ETS baseline|ets_simple.py（SES/Holt简化）+ 对照 statsmodels|笔记：ETS（Exponential Smoothing）|
|周四|5h|D2L 5.1–5.3（MLP+反传）|手推一层反传|mlp_scratch.py（ReLU）+ 梯度检查|笔记：反向传播（Backprop）|
|周五|5h|工程化训练（测试日）|给本周DL/TS补 tests≥5|加 pytest fixtures + seed 固定|笔记：可复现（Reproducibility）|
|周六|4–6h毕业设+45m DS|18.01 无穷级数（看进度）|6题|exp_series.py：级数近似 exp 并对照|笔记：级数（Series）|
|周日|4–6h毕业设+45m DS|清债|—|毕设：写一份“实验日志模板”|笔记：实验日志（Experiment Log）|

第15周（主课：D2L 5.4–6；FPP3 Ch9 ARIMA(1)）

|**日程**|**时长**|**听课**|**练习**|**从零实现**|**笔记 & 验收**|
|---|---|---|---|---|---|
|周一|5h|D2L 5.4 初始化与数值稳定|复现实验|init_compare.py：不同初始化训练曲线|笔记：初始化（Initialization）|
|周二|5h|FPP3 9.1–9.4（差分/AR/MA）|画 ACF/PACF|ar_ma_sim.py：模拟AR(1)/MA(1)并估计参数|笔记：ARMA（AR/MA）|
|周三|5h|SQL+MySQL ETL（日）|10查询|arima_features.sql：差分/滞后特征表|笔记：差分（Differencing）|
|周四|5h|D2L 6 Modules/Parameters/GPU|跟做|nn_modules_play.py：自定义Layer+单测|笔记：Module（nn.Module）|
|周五|5h|SQL（第2天）|20题|把一条复杂查询改成可复用 CTE 模板|笔记：CTE（Common Table Expression）|
|周六|4–6h毕业设+45m DS|清债|—|毕设：加 1 个“端到端回归测试”|笔记：回归测试（Regression Test）|
|周日|4–6h毕业设+45m DS|复盘 D2L 本周|—|最小实现：seed_everything.py + 文档|笔记：随机种子（Random Seed）|

第16周（里程碑#4：DL+TS 模块化与对照）

|**日程**|**时长**|**听课**|**练习**|**从零实现**|**笔记 & 验收**|
|---|---|---|---|---|---|
|周一|5h|FPP3 9.5–9.8（ARIMA估计/预测）|跑 ARIMA baseline|arima_fit.py：手写极简AR估计 + 对照 statsmodels|笔记：ARIMA（ARIMA）|
|周二|5h|D2L 5.6 Dropout|跟做|dropout_scratch.py + 对照 torch dropout|笔记：Dropout（Dropout）|
|周三|5h|SQL（保持）|20题|model_registry.py：保存模型/指标到 MySQL 表|笔记：实验登记（Model Registry）|
|周四|5h|调试/测试训练（本周）|给DL/TS管线补 tests≥5|加 Makefile：make data/train/eval|笔记：Makefile（Make）|
|周五（里程碑日）|6h|口头讲解彩排|综合题|从零做：选择一条金融时序→SQL特征→(ARIMA vs MLP)对照→报告（含误差与残差）|笔记：里程碑#4|
|周六|4–6h毕业设+45m DS|清债|—|毕设：整理一个“数据→模型→评估”流水线|笔记：清债#4|
|周日|4–6h毕业设+45m DS|休整+轻复盘|—|最小实现：cli_train.py（argparse）|笔记：命令行接口（CLI）|

# **============================================================**

#   

# **第17–20周（第5月）：序列模型与 Transformer（D2L + CS224N选讲）+ 时间序列强化（FPP3 10/12）**

#   

# **完成标准：你能从零实现 RNN/LSTM（最小版）、attention、mini-transformer，并能把它们用于一条“金融时序/文本”小任务；能做动态回归与多变量（至少理解VAR思想）。**

  

（从这里开始：主课仍≤2门。主课1：D2L序列与注意力章节；主课2：CS224N 对应讲次做“讲解+实现”。CS224N讲次来自公开 schedule Week1–Week6。 ）

  

第17周（CS224N Week3 RNN/LM；D2L 序列模型入门）

|**日程**|**时长**|**听课**|**练习**|**从零实现**|**笔记 & 验收**|
|---|---|---|---|---|---|
|周一|5h|CS224N Week3 “Recurrent Neural Networks and Language Models”|跑一个字符级LM（小数据）|rnn_scratch.py（tanh RNN）+ 梯度裁剪 sanity|笔记：RNN（RNN）|
|周二|5h|SQL（第1天）|20题|seq_data.sql：构造序列样本（按时间排序）|笔记：序列抽样（Sequence Sampling）|
|周三|5h|D2L（对应RNN/LSTM章节，按TOC定位）|跟做+改超参|lstm_scratch.py（最小门控）+ sanity（输出维度/梯度裁剪）|笔记：LSTM（LSTM）|
|周四|5h|FPP3 Ch10 动态回归|做一个带外生变量预测|dyn_reg.py（ARIMA errors简化）|笔记：动态回归（Dynamic Regression）|
|周五|5h|MySQL ETL（本周）|入库：价格+成交量+因子|etl_exog.py：外生变量对齐 + 2 tests|笔记：对齐（Alignment）|
|周六|4–6h毕业设+45m DS|清债|—|毕设：实现 1 个可复现实验脚本|笔记：清债#5|
|周日|4–6h毕业设+45m DS|复盘本周RNN|—|最小实现：grad_clip.py + 单测|笔记：梯度裁剪（Gradient Clipping）|

第18周（CS224N Seq2Seq；Attention）

|**日程**|**时长**|**听课**|**练习**|**从零实现**|**笔记 & 验收**|
|---|---|---|---|---|---|
|周一|5h|CS224N “Seq2Seq, MT, Subword Models”|toy 翻译/序列复制任务|seq2seq_scratch.py（encoder-decoder最小）|笔记：Seq2Seq（Seq2Seq）|
|周二|5h|SQL（第2天）|20题|token_stats.sql：统计词频/长度分布（简化）|笔记：词频（Token Frequency）|
|周三|5h|Attention（自实现）|做对齐可视化|attention_dot.py（scaled dot-product最小）+ shape checks|笔记：注意力（Attention）|
|周四|5h|调试/测试训练|给 seq2seq/attn 补 tests≥4|增加 torch.no_grad() 评估脚本|笔记：训练/评估（Train vs Eval）|
|周五|5h|FPP3 Ch12.4 Neural Nets（只读关键段）|做一条“MLP预测”对照|mlp_forecast.py：滚动预测对照 ARIMA|笔记：NN预测（NN Forecast）|
|周六|4–6h毕业设+45m DS|清债|—|毕设：补 2 个单测|笔记：复盘#18|
|周日|4–6h毕业设+45m DS|休整+轻复盘|—|最小实现：positional_encoding.py|笔记：位置编码（Positional Encoding）|

第19周（CS224N Transformers；D2L Transformer对应章节）

|**日程**|**时长**|**听课**|**练习**|**从零实现**|**笔记 & 验收**|
|---|---|---|---|---|---|
|周一|5h|CS224N “Self-Attention and Transformers”|读 paper 摘要（不超30min）|transformer_block.py（单层block）+ shape tests|笔记：Transformer Block|
|周二|5h|SQL（第1天）|20题|feature_seq.sql：构造序列batch索引表|笔记：Batching（Batching）|
|周三|5h|D2L Transformer 训练技巧（按TOC定位）|跟做|mini_transformer_train.py（toy任务）+ loss 曲线|笔记：训练技巧（Warmup）|
|周四|5h|MySQL ETL（本周）|入库：文本+标签（简化）|etl_text.py：清洗→分词→入库→统计|笔记：文本ETL（Text ETL）|
|周五|5h|时间序列：对照实验|比较 ARIMA/ETS/MLP/Transformer toy|compare_models.py + 统一指标|笔记：模型对照（Model Comparison）|
|周六|4–6h毕业设+45m DS|清债|—|毕设：整理“实验结果表”模板|笔记：结果表（Results Table）|
|周日|4–6h毕业设+45m DS|休整+轻复盘|—|最小实现：masking.py（causal mask）|笔记：Mask（Causal Mask）|

第20周（里程碑#5：序列模型可用化）

|**日程**|**时长**|**听课**|**练习**|**从零实现**|**笔记 & 验收**|
|---|---|---|---|---|---|
|周一|5h|CS224N Week5 Pretraining（了解）|只做“概念卡”|mlm_data.py：MLM掩码数据生成|笔记：预训练（Pretraining）|
|周二|5h|SQL（第2天）|20题|experiment_store.sql：指标入库与查询|笔记：实验查询（Query Experiments）|
|周三|5h|调试/测试训练|给 transformer 训练脚本补 tests|加 argparse + seed + config|笔记：训练脚手架（Training Scaffold）|
|周四|5h|FPP3 Ch13 实务问题（缺失/异常）|做缺失/异常处理实验|impute_outliers.py + 对照策略|笔记：缺失与异常（Missing/Outliers）|
|周五（里程碑日）|6h|口头讲解彩排|综合题|从零做：选择“金融时序或小文本”→ETL→Transformer/基线→对照→报告（含失败分析）|笔记：里程碑#5|
|周六|4–6h毕业设+45m DS|清债|—|毕设：补 1 个数据校验脚本|笔记：清债#5|
|周日|4–6h毕业设+45m DS|休整|—|最小实现：runbook.md（故障排查清单）|笔记：Runbook|

# **============================================================**

#   

# **第21–24周（第6月前半）：图学习主线（CS224W）+ SQL/ETL 维持 + 小型实验频繁**

#   

# **完成标准：你能从零实现 GCN/GAT 的核心传播（不用PyG也能跑小图），并能在一个金融关系图/异构图 toy 数据上完成训练与评估；能清晰解释 message passing 的假设与失败模式。**

  

CS224W schedule 公开且含 Lecture 1–10 等

  

第21周（CS224W Lec1–2）

|**日程**|**时长**|**听课**|**练习**|**从零实现**|**笔记 & 验收**|
|---|---|---|---|---|---|
|周一|5h|CS224W Lec1 Intro|读slides+做概念题5个|graph_basics.py：邻接表/矩阵互转+tests|笔记：图表示（Graph Representation）|
|周二|5h|SQL（第1天）|20题|edges.sql：从表构造边表（source,target,weight）|笔记：边表（Edge Table）|
|周三|5h|CS224W Lec2 Feature Eng|做 2 种节点特征|node_features.py：degree/centrality(简化)|笔记：节点特征（Node Features）|
|周四|5h|MySQL ETL（本周）|入库：节点/边/特征|etl_graph.py：CSV→节点表/边表→统计|笔记：图数据ETL（Graph ETL）|
|周五|5h|小实验|用特征+逻辑回归做节点分类 toy|node_clf_baseline.py + 对照|笔记：图任务（Graph Tasks）|
|周六|4–6h毕业设+45m DS|清债|—|毕设：抽出“图相关模块”雏形|笔记：复盘#21|
|周日|4–6h毕业设+45m DS|休整+轻复盘|—|最小实现：negative_sampling.py|笔记：负采样（Neg Sampling）|

第22周（CS224W Lec3 Node Embeddings；Lec4 GNN）

|**日程**|**时长**|**听课**|**练习**|**从零实现**|**笔记 & 验收**|
|---|---|---|---|---|---|
|周一|5h|CS224W Lec3 Node Embeddings|做 toy random walk|deepwalk_scratch.py（最小）|笔记：DeepWalk（DeepWalk）|
|周二|5h|SQL（第2天）|20题|walk_store.sql：存 walk 统计（可选）|笔记：随机游走（Random Walk）|
|周三|5h|CS224W Lec4 GNN|读 GCN 论文摘要|gcn_layer.py（纯numpy/torch）+ shape tests|笔记：GCN（Graph Convolution）|
|周四|5h|调试/测试训练|给GNN层补 tests≥4|加小图单元测试（2–3节点）|笔记：小图验证（Tiny Graph Test）|
|周五|5h|小实验|GCN 做节点分类 toy（对照MLP）|gcn_train.py + baseline对照|笔记：Message Passing|
|周六|4–6h毕业设+45m DS|清债|—|毕设：补 2 tests|笔记：清债#22|
|周日|4–6h毕业设+45m DS|休整|—|最小实现：adj_norm.py（对称归一化）|笔记：归一化（Normalization）|

第23周（CS224W Lec5–6：GNN视角/训练）

|**日程**|**时长**|**听课**|**练习**|**从零实现**|**笔记 & 验收**|
|---|---|---|---|---|---|
|周一|5h|CS224W Lec5 General Perspective|总结 3 种GNN差异|gat_layer.py（最小 attention）|笔记：GAT（Graph Attention）|
|周二|5h|SQL（第1天）|20题|hetero_edges.sql：构造异构边类型字段|笔记：异构图（Hetero Graph）|
|周三|5h|CS224W Lec6 Augmentation/Training|做 dropout/edge-drop 实验|edge_dropout.py + 2 sanity|笔记：图增强（Graph Augment）|
|周四|5h|MySQL ETL（本周）|入库：实验结果|store_metrics.py：把GNN实验指标入库|笔记：指标入库（Store Metrics）|
|周五|5h|小实验|GAT vs GCN 对照|gnn_compare.py|笔记：GNN对照（GNN Compare）|
|周六|4–6h毕业设+45m DS|清债|—|毕设：整理“图模块”接口|笔记：复盘#23|
|周日|4–6h毕业设+45m DS|休整|—|最小实现：batch_graphs.py（小批）|笔记：Batching Graphs|

第24周（里程碑#6：图学习闭环演示）

|**日程**|**时长**|**听课**|**练习**|**从零实现**|**笔记 & 验收**|
|---|---|---|---|---|---|
|周一|5h|CS224W Lec7 Theory（选读）|总结 2 个理论结论|wl_test.py（WL思想最小）|笔记：表达力（Expressivity）|
|周二|5h|SQL（第2天）|20题|写一条“图特征汇总报表SQL”|笔记：图报表（Graph Report SQL）|
|周三|5h|CS224W Lec8 Label Prop（了解）|做 label propagation toy|label_prop.py|笔记：标签传播（Label Propagation）|
|周四|5h|调试/测试训练|给图模块补 tests≥6|增加 CI（GitHub Actions 最小）|笔记：CI（CI Basics）|
|周五（里程碑日）|6h|口头讲解彩排|综合|从零做：金融关系图 toy（公司-行业-新闻）→ETL→GCN/GAT→对照baseline→报告|笔记：里程碑#6|
|周六|4–6h毕业设+45m DS|清债|—|毕设：把图模块接到你的毕设部分（若相关）|笔记：清债#6|
|周日|4–6h毕业设+45m DS|休整|—|最小实现：repro_manifest.md（环境清单）|笔记：环境清单（Env Manifest）|

# **============================================================**

#   

# **第25–26周（第6月后半）：NLP收口 + DS开学即用的“战备包”整理**

#   

# **完成标准：你有一套可复用的 repo 模板（ETL/训练/评估/SQL/测试/报告），并能用 10 分钟讲清楚：你会什么、怎么验证、怎么复现。**

  

第25周（CS224N Transformers/Pretraining/NLG 选讲）

CS224N schedule 中：Transformers、Pretraining、NLG 等条目公开

|**日程**|**时长**|**听课**|**练习**|**从零实现**|**笔记 & 验收**|
|---|---|---|---|---|---|
|周一|5h|CS224N Pretraining（BERT概念）|写概念自测 10问|bert_mlm_mvp.py（极简embedding+mlm头）|笔记：BERT/MLM|
|周二|5h|SQL（第1天）|20题|text_features.sql：文本统计特征（长度/频次）|笔记：文本特征（Text Features）|
|周三|5h|CS224N NLG（了解）|beam search概念题|beam_search.py（最小）|笔记：Beam Search|
|周四|5h|MySQL ETL（本周）|结果入库|papertrail.sql：记录实验与摘要|笔记：实验摘要（Experiment Summary）|
|周五|5h|工程化收口（模板化）|整理 repo|生成 cookiecutter 或脚手架（最小也行）|笔记：项目模板（Project Template）|
|周六|4–6h毕业设+45m DS|清债|—|毕设：把模板用到毕设仓库一部分|笔记：清债#25|
|周日|4–6h毕业设+45m DS|休整|—|最小实现：demo_notebook.ipynb（一页演示）|笔记：演示脚本（Demo）|

第26周（最终里程碑#7：开学战备包）

|**日程**|**时长**|**听课**|**练习**|**从零实现**|**笔记 & 验收**|
|---|---|---|---|---|---|
|周一|5h|总复盘：ML/DL/TS/GNN/NLP|口头讲解稿（10分钟）|talk_track.md：10分钟脚本+反例|笔记：最终复盘（Final Review）|
|周二|5h|SQL（第2天）|30题综合|sql_playbook.sql：最常用20条查询模板|笔记：SQL手册（SQL Playbook）|
|周三|5h|复盘 MySQL Connector 关键API|写 5 个“联动任务”脚本|db_client.py：统一DB接口（connect/query/insert）+ tests|笔记：DB客户端（DB Client）|
|周四|5h|复盘 FPP3 workflow+diagnostics|做一条最终预测报告|forecast_report.py：滚动CV+残差诊断+表格|笔记：预测报告（Forecast Report）|
|周五（最终里程碑日）|6h|终极演示彩排|综合|从零做：任选一个（金融时序/图/文本）→ETL(MySQL)→模型训练→对照→报告→入库→可复现命令|笔记：里程碑#7（Final Milestone）|
|周六|4–6h毕业设+45m DS|清债（最后清仓）|—|毕设：补齐文档/测试|笔记：清债#Final|
|周日|4–6h毕业设+45m DS|休整+计划下阶段|—|最小实现：写“开学后30天学习地图”|笔记：Next 30 Days|

────────────────────────────────

你接下来执行时的“唯一要求”（很硬，但能救命）

────────────────────────────────

1. 每天的“从零实现”再小也要有，而且要有 tests/sanity；否则你会回到“模糊知道但写不出”的老路。
    
2. 所有脚本必须能被一句命令复现：python -m ... 或 make ...。周末毕业设计更要做到这一点。
    
3. Obsidian 笔记宁可少写废话，也要写“定义+边界+反例+最小实现+自测清单”。你要训练的是“可检验知识”，不是“看过就算”。
    

  
