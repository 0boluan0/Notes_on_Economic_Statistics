好的，你的回答把关键不确定性基本消掉了。我现在按这些“已锁定的项目约束”推进（后面每周都以此为默认，除非你明确改）：

- 比例：数学30% / 编程40% / DS工程与ML30%（你说默认按我建议）
    
- 方向：数据科学 + 机器学习（Data Science + Machine Learning）
    
- 微积分：查漏补缺型（补“能用的工具”，不做全量系统重学）
    
- 6.042J：允许降优先级（作为严谨性/离散概率补强，不抢主线）
    
- MySQL：Docker 和 Homebrew 都要会用（默认主线用Docker保证可复现；每周留一个小任务让你把同样流程在本地Homebrew跑一遍，形成“双栈能力”）
    
- Python：目标是“能写项目”（所以首月主线偏 CS50P + Missing Semester，CS61A作为强化课后置）
    

  

下面按你规定的交付顺序来。

---

B) 本周计划（W1：Python + 工程化入门；按课程进度推进）

本周你会同时推进两条课程线 + 一条工程线。每一块都有对应“闭环笔记标题”和“实现任务”。

1. 编程主线：Harvard CS50P（Python基础，目标：能写项目）
    

  

- 课程推进（按你自己的CS50P章节顺序）：变量/控制流 → 函数 → 异常 → 文件I/O → 基础数据结构（list/dict/set）→ 简单测试思想
    
- 闭环笔记（本周至少2篇）
    
    - 《Python脚本到模块 / From Script to Module》
        
    - 《异常处理与数据质量 / Exceptions & Data Quality》
        
    
- 实现任务（MVP）
    
    - MVP-A：etl_csv_to_parquet.py（读CSV→清洗→聚合→导出；带sanity checks）
        
    - MVP-B：把MVP-A重构成 src/etl.py 可被导入调用，并加pytest测试
        
    

  

2. 工具链主线：MIT Missing Semester（Shell + Git + 自动化）
    

  

- 课程推进：shell基本命令 + 文件系统 + 管道；git基本流（branch/commit/merge）；最小自动化（Makefile或task脚本二选一）
    
- 闭环笔记（本周至少1篇）
    
    - 《最小可复现项目骨架 / Minimal Reproducible Project Skeleton》
        
    
- 实现任务（MVP）
    
    - MVP-C：建立模板仓库 ds-lab：src/ tests/ data/ scripts/ + pytest跑通 + ruff检查 + README一键运行命令
        
    

  

3. 工程化主线：MySQL（Docker优先 + Homebrew补一遍）+ Python联动
    

  

- 课程推进：SQL基础查询（SELECT/WHERE/GROUP BY/JOIN）+ Python连接 + 写入/读出 + 对照校验
    
- 闭环笔记（本周至少1篇）
    
    - 《SQL聚合查询 / SQL Aggregation Queries》
        
    - 《Python连接MySQL与最小ETL / Python-MySQL Minimal ETL》
        
    
- 实现任务（MVP）
    
    - MVP-D：Docker起MySQL，Python写入聚合结果
        
    - MVP-E：SQL做同样聚合，与Pandas结果逐字段对照（误差=0）
        
    - MVP-F（本地补强）：用Homebrew装MySQL，把同样流程再跑一遍（只需跑通，不追求完美配置）
        
    

  

本周验收总目标（硬标准）

- 你能在新环境从零复现：git clone → uv venv → 安装依赖 → 一条命令跑完整ETL+入库+对照 → pytest全绿。
    
- 你能用60秒讲清楚两件事（CN+EN术语）：什么是“可复现（Reproducibility）”，什么是“数据质量（Data Quality）与异常（Exceptions）”。
    

---

C) 今日执行脚本（Day1，4–6小时，按时间块+完成标准）

假设你今天做“仓库骨架 + 工具链跑通”。如果你今天已经有仓库或环境了，也照这个脚本做一次“从零复现”，否则后面会反复出血。

  

Block 1（90min）：建立项目仓库与目录结构（Project Skeleton）

- 你做什么：
    
    1. 新建仓库 ds-lab（本地文件夹即可，后续再推GitHub也行）
        
    2. 创建目录：src/ tests/ data/raw/ data/processed/ scripts/
        
    3. 写一个最小入口：python -m src.hello 能打印一句话
        
    
- 完成标准：
    
    - python -m src.hello 能运行
        
    - 目录结构固定下来（后续所有MVP都放这套结构里）
        
    

  

Block 2（60min）：依赖与虚拟环境（uv + venv）

- 你做什么：
    
    1. 建立虚拟环境（Python建议3.11）
        
    2. 安装本周最低依赖：pandas pyarrow sqlalchemy pymysql pytest ruff
        
    
- 完成标准：
    
    - python -c "import pandas" 不报错
        
    - pytest 能跑（即便此时还没有测试，也要能运行不崩）
        
    

  

Block 3（90min）：最小测试与静态检查（pytest + ruff）

- 你做什么：
    
    1. 写第一个测试：tests/test_smoke.py（断言True或导入模块成功）
        
    2. 加ruff配置（最小即可）
        
    
- 完成标准：
    
    - pytest -q 全绿
        
    - ruff check . 不报错（允许少量warning，但不能一堆）
        
    

  

Block 4（60–90min）：写README的“一键命令”

- 你做什么：
    
    1. README写三段：如何建环境、如何运行、如何测试
        
    
- 完成标准：
    
    - 你把README复制给任何人，对方照做能跑到 pytest 全绿（这是你后面所有项目的生命线）
        
    

  

Block 5（30min）：记录复盘与卡点

- 你做什么：在Obsidian写一条日志：今天遇到的报错/耗时点/你不理解的概念
    
- 完成标准：至少记录3条（哪怕很小），下一次我会据此调计划
    

---

D) 本次涉及的 Obsidian 笔记草稿（按你的模板，四篇一次给全）

说明：下面四篇是“本周闭环笔记”的初稿。你需要补充的“人类步骤”我会在每篇末尾单列出来（推导/解释/编码/自测）。你今天先用第一篇就够；其余三篇按日程推进时再填充与执行实现即可。

---

---

## **title: 最小可复现项目骨架 / Minimal Reproducible Project Skeleton**

##   

## **tags: [ds, missing_semester, topic_engineering, lv0]**

##   

## **source: MIT Missing Semester (shell/git/project setup)**

##   

## **status: draft**

##   

## **prereq: [Python基础/Python basics, 命令行/CLI basics]**

##   

## **created: 2026-02-01**

1. 这是什么（What it is）
    
    可检验定义：一个“最小可复现项目骨架（Minimal Reproducible Project Skeleton）”是指：在一台全新机器上，仅依赖项目仓库内容与明确的安装命令，就能稳定得到相同的可运行程序与测试结果的项目结构。直觉上，它解决的是“我机器上能跑但别人跑不了”的问题，把环境、入口、依赖、测试和运行方式都固定住。
    
2. 为什么重要（Why it matters）
    
    数据科学里你会频繁经历：写脚本→堆功能→改到不可维护→结果不可复现。一个最小骨架让你把每一次学习都沉淀成可复用工程资产：同样的结构可以承载ETL、训练、评估、可视化、数据库联动。什么时候别用？当你只是临时试验一行代码时不需要；但一旦你要“交作业/交给同学/提交仓库/做项目”，就必须用。
    
3. 精确定义与边界（Definition & assumptions）
    
    假设：你有一个Python解释器与包管理工具；你愿意把“入口命令、依赖版本、测试”写清楚。边界条件：骨架不保证科学结果完全一致（比如浮点、并行），但要保证“代码能跑、流程一致、主要指标一致”。失败场景：只在notebook里跑、没有固定依赖、没有入口命令、没有最小测试，都会导致“你以为学会了，实际上无法迁移”。
    
4. 关键公式/推导（Key math, if any）
    
    这里不需要数学推导。核心是工程不变量：入口（entrypoint）、依赖（dependencies）、测试（tests）、数据路径（data paths）、运行命令（commands）。
    
5. 最小例子（Minimal example）
    
    一个最小骨架至少包含：
    

  

- src/：代码
    
- tests/：测试
    
- README.md：运行方式
    
    如果别人拿到仓库只做三件事：建环境、装依赖、运行测试，就应当得到同样结果。
    

  

6. 从零实现（From-scratch implementation）
    
    建议Python版本：3.11
    

  

文件结构（最小）：

```
ds-lab/
  README.md
  pyproject.toml        # 若你用uv/现代Python构建
  src/
    __init__.py
    hello.py
  tests/
    test_smoke.py
  data/
    raw/
    processed/
```

最小代码：src/hello.py

```
def main() -> None:
    print("ds-lab: hello")

if __name__ == "__main__":
    main()
```

最小测试：tests/test_smoke.py

```
from src.hello import main

def test_smoke(capsys):
    main()
    out = capsys.readouterr().out
    assert "hello" in out
```

运行与sanity checks（你今天就做）：

- 运行：python -m src.hello
    
- 测试：pytest -q
    
- 静态检查：ruff check .
    

  

7. 常见坑与误区（Pitfalls）
    
    第一坑：只有notebook没有入口。发现方式：你不知道“从哪运行”。修正：把核心逻辑移到src/，notebook只做调用与展示。
    
    第二坑：依赖不锁定。发现方式：换电脑/换环境就报错。修正：用uv/requirements锁住最低依赖，至少确保版本范围明确。
    
    第三坑：没有测试。发现方式：你改了一点东西，全流程悄悄变错。修正：至少写一个smoke test和一个数据一致性测试（sanity check）。
    
8. 自测清单（Mastery checks）
    
    60秒脚本（CN+EN术语）
    
    “我做的是最小可复现项目骨架（Minimal Reproducible Project Skeleton）。它的标准是：别人拿到仓库，在新机器上按README建环境（environment）、装依赖（dependencies）、运行入口（entrypoint）并跑测试（tests），就能得到同样的运行结果。它解决‘我能跑你不能跑’的问题，是把学习变成可复用工程资产的前提。”
    

  

自测题（含答案要点）

Q1：为什么只写notebook会降低复现性？A：入口不清晰、依赖散落、执行顺序隐含、难以测试。

Q2：最小测试应该测什么？A：至少导入成功+入口能跑（smoke），再加一个关键输出/关键指标不变的断言。

Q3：什么情况下“可复现”不等于“结果完全相同”？A：浮点误差、随机性、并行与硬件差异；但流程与主要指标应一致。

  

实现自测

把hello.py改成接收一个字符串参数，再让测试覆盖这个变化；保证你改完后pytest依旧全绿。

9. 与其他概念的连接（Connections）
    
    这个骨架会被后续所有模块复用：数据清洗（ETL）、SQL联动、训练评估、时间序列walk-forward。它也和Missing Semester里的shell脚本、git工作流、自动化任务直接关联：你最终会把“学习进度”固化成“可运行命令+可验证输出”。
    

  

你需要补充/你必须亲手做的步骤

- 你亲手建一次全新环境跑通（不要跳过），并记录任何报错与解决方式。
    
- 你把README写成“别人照做就能跑”的程度（这一步不能让AI替代你完成理解）。
    

---

---

## **title: Python脚本到模块 / From Script to Module**

##   

## **tags: [ds, cs50p, topic_python, lv0]**

##   

## **source: CS50P (functions/modules/files) + 本周MVP工程化**

##   

## **status: draft**

##   

## **prereq: [函数/Function, 导入/import, 文件路径/paths]**

##   

## **created: 2026-02-01**

1. 这是什么（What it is）
    
    可检验定义：把“脚本（script）”改造成“模块（module）”，指的是把可复用逻辑写成可被import的函数/类，并提供一个清晰入口（entrypoint）来运行全流程。直觉上，脚本是“一次性执行”，模块是“可组合、可测试、可复用”。
    
2. 为什么重要（Why it matters）
    
    数据科学工作最常见的失败是：脚本越写越长，改动导致不可控，无法写测试，无法复用。模块化让你能把ETL、特征工程、训练、评估拆成稳定接口，便于在硕士课程lab/科研原型里快速迭代。什么时候别用？当你只是验证一个很小想法且不需要复用时；但一旦要交付或复跑，就必须模块化。
    
3. 精确定义与边界（Definition & assumptions）
    
    假设：你能描述清楚输入输出（I/O）。边界：模块化不等于面向对象；你完全可以只用函数。常见误解：把所有东西塞进一个类里就叫工程化。反例：一个“巨型类”仍然难测难维护。
    
4. 关键公式/推导（Key math, if any）
    
    无。
    
5. 最小例子（Minimal example）
    
    你有一个脚本：读CSV→清洗→聚合→保存。如果模块化，你应当至少抽出：
    

  

- load_data(path) -> DataFrame
    
- clean(df) -> df_clean
    
- aggregate(df_clean) -> df_out
    
- save(df_out, path)
    

  

6. 从零实现（From-scratch implementation）
    
    最小文件：
    

```
src/
  etl.py
  cli.py
tests/
  test_etl.py
```

src/etl.py（示意）

```
import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 例：去掉全空行
    out = out.dropna(how="all")
    return out

def aggregate_demo(df: pd.DataFrame) -> pd.DataFrame:
    # 例：按某列分组求均值
    if "group" not in df.columns or "value" not in df.columns:
        raise ValueError("require columns: group, value")
    return df.groupby("group", as_index=False)["value"].mean()

def run_pipeline(raw_path: str, out_path: str) -> None:
    df = load_csv(raw_path)
    df = clean_basic(df)
    out = aggregate_demo(df)
    out.to_parquet(out_path, index=False)
```

tests/test_etl.py（sanity）

```
import pandas as pd
from src.etl import aggregate_demo

def test_aggregate_demo():
    df = pd.DataFrame({"group": ["a","a","b"], "value":[1,3,2]})
    out = aggregate_demo(df)
    assert set(out["group"]) == {"a","b"}
    assert float(out.loc[out["group"]=="a","value"].iloc[0]) == 2.0
```

7. 常见坑与误区（Pitfalls）
    
    坑1：函数没有清晰I/O。发现：你调用时不知道传什么、返回什么。修正：写类型提示与docstring。
    
    坑2：把路径写死在代码里。发现：换机器就找不到文件。修正：入口通过参数传路径。
    
    坑3：清洗逻辑不可追踪。发现：你不知道哪一步把数据改坏了。修正：每一步加sanity check（行数、缺失率、范围）。
    
8. 自测清单（Mastery checks）
    
    60秒脚本（CN+EN术语）
    
    “脚本（script）是一次性执行的文件，模块（module）是可被导入（import）的可复用逻辑。我会把核心流程拆成函数：加载（load）、清洗（clean）、聚合（aggregate）、保存（save），再用入口（entrypoint）把它们串起来。模块化的检验标准是：我能写单元测试（unit tests），并在不同数据/不同路径下重复运行得到一致输出。”
    

  

自测题

Q1：为什么模块化会提升可测试性？A：核心逻辑被拆成纯函数/可调用接口，测试可以直接对输入输出断言。

Q2：入口与模块的职责差异？A：入口负责参数与编排；模块负责业务逻辑。

Q3：什么是“可复用”的最小标准？A：能在另一个脚本/项目里import并工作，且有测试兜底。

  

实现自测

把aggregate_demo改成支持传入聚合函数（mean/sum），并写两条测试覆盖。

9. 与其他概念的连接（Connections）
    
    模块化会直接服务后续：SQL写入/读出、训练脚手架、walk-forward验证。它也与Git工作流强绑定：每一次重构都要保证pytest不破。
    

  

你需要补充/你必须亲手做的步骤

- 你要明确写出你本周用的数据字段含义（哪列是group、哪列是value），否则ETL会漂。
    
- 你要亲手把脚本重构一次并写测试，体会“改代码不怕”的感觉。
    

---

---

## **title: SQL聚合查询 / SQL Aggregation Queries**

##   

## **tags: [ds, mysql, topic_sql, lv0]**

##   

## **source: 本周MySQL实战（GROUP BY/JOIN）+ 数据分析场景**

##   

## **status: draft**

##   

## **prereq: [表/rows-columns, SELECT/WHERE basics]**

##   

## **created: 2026-02-01**

1. 这是什么（What it is）
    
    可检验定义：SQL聚合查询是指使用GROUP BY将行按键（key）分组，并对组内数据应用聚合函数（aggregate functions，如COUNT/SUM/AVG）得到组级别结果。直觉上，它把“明细表（transactions）”变成“汇总表（summary/mart）”。
    
2. 为什么重要（Why it matters）
    
    数据科学常见工作是做指标与特征：每个用户的订单数、每只股票的均值波动、每个日期的成交额。这些本质上都是聚合。什么时候别用？当你需要逐行复杂变换且SQL难写时可以在Python做，但最理想的分工是：能在数据库里聚合的尽量在数据库里做，减少数据搬运。
    
3. 精确定义与边界（Definition & assumptions）
    
    假设：你的数据是表结构，字段类型正确。边界：SQL聚合需要注意NULL处理（AVG忽略NULL）、重复行、以及连接导致的行数膨胀。失败场景：JOIN后再GROUP BY如果不理解基数（cardinality），指标会被放大。
    
4. 关键公式/推导（Key math, if any）
    
    无公式。核心是“分组键决定粒度（granularity）”。
    
5. 最小例子（Minimal example）
    
    表trades(group, value)：
    

  

- 你想要每个group的平均value：
    
    SELECT group, AVG(value) FROM trades GROUP BY group;
    

  

6. 从零实现（From-scratch implementation）
    
    建表与插入（示意）：
    

```
CREATE TABLE trades (
  id INT PRIMARY KEY AUTO_INCREMENT,
  `group` VARCHAR(10),
  value DOUBLE
);

INSERT INTO trades(`group`, value) VALUES
('a',1),('a',3),('b',2);
```

聚合查询：

```
SELECT `group`, AVG(value) AS avg_value
FROM trades
GROUP BY `group`;
```

sanity checks（发现错得早）

- 检查分组后的行数是否等于不同group数：
    
    SELECT COUNT(DISTINCT group) FROM trades;
    
- 检查总行数与聚合前一致（不涉及JOIN时）：
    
    SELECT COUNT(*) FROM trades;
    

  

7. 常见坑与误区（Pitfalls）
    
    坑1：忘了GROUP BY粒度。发现：结果行数不对。修正：先写COUNT(DISTINCT key)预测结果行数。
    
    坑2：JOIN导致重复。发现：聚合结果比预期大很多。修正：先检查JOIN后的行数变化，必要时先去重或先在子查询里聚合再JOIN。
    
    坑3：NULL处理误判。发现：均值与你在Python算的不一致。修正：明确是否需要COALESCE(value,0)，以及是否应该丢弃NULL。
    
8. 自测清单（Mastery checks）
    
    60秒脚本（CN+EN术语）
    
    “SQL聚合（aggregation）就是用GROUP BY按键分组，然后用COUNT/SUM/AVG得到组级指标。它的关键是粒度（granularity）：分组键决定输出每一行代表什么对象。我会先用COUNT(DISTINCT key)预测输出行数，再做sanity check防止JOIN重复和NULL导致的偏差。”
    

  

自测题

Q1：什么时候应该“先聚合再JOIN”？A：当JOIN会引入一对多导致重复计数时。

Q2：如何快速发现JOIN重复？A：比较JOIN前后行数与关键键的重复度。

Q3：为什么SQL与Pandas聚合会不一致？A：NULL处理、类型转换、重复行、浮点格式。

  

实现自测

用同一份数据，在SQL和Pandas里分别计算AVG/SUM/COUNT，写一个脚本逐字段比对误差为0（浮点可用容忍度，但你先追求完全一致）。

9. 与其他概念的连接（Connections）
    
    SQL聚合直接连到“特征工程（Feature Engineering）”与“数据集市（Data Mart）”。后面你做时间序列特征时，会用窗口函数（window functions）把聚合从“组”扩展到“滑动窗口”。
    

  

你需要补充/你必须亲手做的步骤

- 你必须亲手画出一张“粒度表”：每张表一行代表什么对象（用户/交易/日频），否则你写SQL会不断犯错。
    

---

---

## **title: Python连接MySQL与最小ETL / Python-MySQL Minimal ETL**

##   

## **tags: [ds, mysql, topic_engineering, lv1]**

##   

## **source: 本周MVP：Docker+Homebrew 双栈 + Python写入读出**

##   

## **status: draft**

##   

## **prereq: [SQL基础/SQL basics, pandas, 环境管理/venv]**

##   

## **created: 2026-02-01**

1. 这是什么（What it is）
    
    可检验定义：Python-MySQL最小ETL（Minimal ETL）指的是：用Python从原始数据抽取（Extract）、转换（Transform）、加载（Load）到MySQL，并能用SQL复现关键聚合结果，最后与Python计算结果对照一致。直觉上，它把“分析脚本”升级为“可落库、可复用的数据管道”。
    
2. 为什么重要（Why it matters）
    
    硕士课程/科研/实习里，数据很少永远只在CSV里。你需要会把数据落到数据库，便于复用、共享与查询，也便于做训练数据集市。什么时候别用？数据量极小且一次性时可以只用文件，但一旦需要反复查询或多人协作，就应该落库。
    
3. 精确定义与边界（Definition & assumptions）
    
    假设：你能启动MySQL服务（Docker或本地），并知道连接信息。边界：本MVP不讨论复杂权限、安全与高并发，只追求“能用且可复现”。失败场景：表结构不清晰、字段类型随意、没有主键/唯一键导致重复写入，都会让ETL变得不可控。
    
4. 关键公式/推导（Key math, if any）
    
    无。
    
5. 最小例子（Minimal example）
    
    你从data/raw/sample.csv读入，清洗后得到df_out，写入表agg_result。SQL再聚合或直接查询agg_result，与df_out对照一致。
    
6. 从零实现（From-scratch implementation）
    
    建议：主线用Docker（可复现），补强用Homebrew（熟悉本地服务管理）。
    

  

Docker方式（推荐主线）

docker-compose.yml（最小示意，你让Claude Code生成更完整版本）

```
services:
  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: root
      MYSQL_DATABASE: ds_lab
    ports:
      - "3306:3306"
```

Python依赖建议（Python 3.11）：

- sqlalchemy
    
- pymysql
    
- pandas
    
- pyarrow
    

  

最小Python写入（示意）

```
import pandas as pd
from sqlalchemy import create_engine

def get_engine():
    # Docker默认：root/root@localhost:3306
    url = "mysql+pymysql://root:root@127.0.0.1:3306/ds_lab"
    return create_engine(url)

def write_df(df: pd.DataFrame, table: str):
    engine = get_engine()
    df.to_sql(table, engine, if_exists="replace", index=False)

def read_df(table: str) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(f"SELECT * FROM {table}", engine)

if __name__ == "__main__":
    df = pd.DataFrame({"group":["a","b"], "avg_value":[2.0,2.0]})
    write_df(df, "agg_result")
    back = read_df("agg_result")
    assert back.shape == df.shape
```

sanity checks（必须做）

- 写入后行数一致：COUNT(*)等于Python dataframe行数
    
- 聚合指标一致：SQL与Pandas同口径计算
    
- 防重复：后续你会加唯一键/主键策略（本周先replace覆盖）
    

  

Homebrew方式（补强）

你要能做到：本地启动/停止MySQL服务、修改端口/密码、能用同样Python连接字符串连上。你本周只需要“跑通一次同样流程”。

7. 常见坑与误区（Pitfalls）
    
    坑1：字符集/时区导致数据错。发现：时间字段偏移或乱码。修正：显式设置时区与编码，入库前统一datetime。
    
    坑2：重复写入。发现：表行数越来越大但你以为是覆盖。修正：先用if_exists="replace"，后续再进阶成“upsert（插入或更新）”。
    
    坑3：类型不一致。发现：数值列变成字符串。修正：在Pandas里显式转类型，并在建表时固定字段类型。
    
8. 自测清单（Mastery checks）
    
    60秒脚本（CN+EN术语）
    
    “我做的是Python-MySQL最小ETL（Minimal ETL）：Python抽取（Extract）原始数据，转换（Transform）得到清洗/聚合结果，再加载（Load）进MySQL。验收标准是：我能用SQL得到同口径结果，并与Pandas输出逐字段对照一致。Docker版保证可复现（reproducible），Homebrew版保证我能在本地环境工作。”
    

  

自测题

Q1：为什么主线用Docker？A：环境隔离、版本固定、别人更容易复现。

Q2：如何保证不重复写入？A：最简单先覆盖写；进阶用主键+upsert。

Q3：SQL与Pandas不一致最常见原因？A：NULL、类型、JOIN重复、浮点与时间处理。

  

实现自测

把数据换成你真实的小数据（哪怕50行），保证ETL依旧跑通；再加一个测试：故意引入缺失值，验证你的清洗逻辑能发现并处理。

9. 与其他概念的连接（Connections）
    
    这个ETL模板会直接复用到：特征工程（Features）、训练数据集市（Training Data Mart）、时间序列滚动窗口特征、实验记录与可复现训练。后面引入索引（Index）与查询计划（EXPLAIN）时，你会用同一套例子验证“为什么加索引能变快”。
    

  

你需要补充/你必须亲手做的步骤

- 你必须亲手在Docker与Homebrew各跑通一次连接，并记录连接信息差异（host、port、user、password）。
    
- 你必须写一条“SQL vs Pandas对照脚本”，这是你避免自欺欺人的核心武器。
    

---

E) Claude Code 子 prompts 套件（可直接喂；每条含目标/路径/验收/运行）

你把下面每条分别丢给Claude Code即可。你自己只负责“运行+验收+记录报错”。

  

Prompt 1：初始化项目骨架

- 目标：在ds-lab/生成最小可复现项目结构，包含src/ tests/ data/ scripts/，并能pytest跑通
    
- 文件路径：README.md, src/hello.py, tests/test_smoke.py, 可选pyproject.toml
    
- 验收标准：
    
    - python -m src.hello 输出包含“hello”
        
    - pytest -q 全绿
        
    - README包含：建环境、运行、测试三条命令
        
    
- 运行与验证：
    
    - pytest -q
        
    - python -m src.hello
        
    

  

Prompt 2：生成ETL最小MVP（CSV→清洗→聚合→Parquet）

- 目标：实现src/etl.py与一个CLI入口scripts/run_etl.py，若data/raw/sample.csv不存在则自动生成一份小样本数据
    
- 文件路径：src/etl.py, scripts/run_etl.py, tests/test_etl.py
    
- 验收标准：
    
    - python scripts/run_etl.py --raw data/raw/sample.csv --out data/processed/out.parquet 成功生成输出
        
    - 至少2个sanity checks：行数合理、缺失率下降或数值范围合理
        
    - pytest -q 全绿
        
    

  

Prompt 3：Docker MySQL一键启动 + Python写入读出

- 目标：提供docker-compose.yml启动MySQL 8.0；实现src/db.py封装engine；实现scripts/load_to_mysql.py把Parquet写入表
    
- 文件路径：docker-compose.yml, src/db.py, scripts/load_to_mysql.py
    
- 验收标准：
    
    - docker compose up -d 后MySQL可连接
        
    - python scripts/load_to_mysql.py --table agg_result --in data/processed/out.parquet 成功
        
    - 写入后用Python读回行数一致（assert）
        
    
- 运行与验证：
    
    - docker compose up -d
        
    - python scripts/load_to_mysql.py ...
        
    

  

Prompt 4：SQL vs Pandas对照校验脚本

- 目标：实现scripts/sql_vs_pandas_check.py，用SQL聚合得到结果，与Pandas聚合结果按键合并后逐字段比对（误差容忍度可设为0或很小）
    
- 文件路径：scripts/sql_vs_pandas_check.py, tests/test_sql_check.py
    
- 验收标准：
    
    - 脚本运行输出“PASS”
        
    - 故意篡改数据库某一行能被检测出“FAIL”
        
    
- 运行与验证：
    
    - python scripts/sql_vs_pandas_check.py --tolerance 0
        
    

  

Prompt 5（补强）：Homebrew MySQL本地模式说明与连接适配

- 目标：在README.md增加“Homebrew安装与启动/停止/连接信息”的段落，并让src/db.py支持通过环境变量切换Docker/本地连接
    
- 文件路径：README.md, src/db.py
    
- 验收标准：
    
    - 你只改环境变量即可连接不同MySQL来源
        
    - README写清楚最少命令与常见错误处理（端口占用、权限）
        
    

---

F) 本次验收 checklist + 下次你要反馈的3类信息

本次验收 checklist（你今天做完至少打勾前3项）

1. pytest -q 全绿（哪怕只有smoke test）
    
2. python -m src.hello 可运行
    
3. README包含“建环境/运行/测试”三条命令
    
4. ruff check . 通过或只有你能解释的少量warning
    
5. 你在Obsidian记录了至少3条：报错/耗时点/不理解概念
    

  

你下次需要反馈给我的三类信息（越具体越好，直接复制粘贴即可）

1. 学习卡点：你觉得最模糊的概念是什么？比如“模块/包/导入路径到底怎么找”
    
2. 实现报错：完整报错堆栈（traceback）+ 你运行的命令
    
3. 笔记空缺：哪一段你写不出来？是定义、边界、反例、还是自测题？
    

  

你现在只要回一句：你今天准备做Day1还是你已经有一个仓库/环境了，我就按你的现状把“Day1脚本”微调到最合适的起点。