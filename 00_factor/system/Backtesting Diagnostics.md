---
aliases:
- VaR backtesting diagnostics
- VaR回测诊断
- 回溯检验诊断
- Backtesting system
tags:
- system
- risk-management
---
# Backtesting Diagnostics

## 诊断目标

这张 system 卡不再定义回测；定义见 [[Backtesting|Backtesting concept]]。这里记录如何判断一个 VaR 模型是否在真实损益上失效，以及失效后该往哪里修。

## 输入

- 每日预测 VaR 序列。
- 同期实际损益或损失序列。
- 置信水平 $\alpha$ 与理论尾部概率 $p=1-\alpha$。
- 回测天数 $n$。

## 诊断步骤

### Step 1：标记例外

定义：

$$
I_t=\mathbf{1}\{L_t>\operatorname{VaR}_t\}
$$

统计例外次数 $x=\sum I_t$ 和例外率 $\hat p=x/n$。

### Step 2：检查例外频率

用 [[Kupiec Test]] 检验 $\hat p$ 是否和 $p$ 一致。

- 例外过多：模型低估风险。
- 例外过少：模型过度保守，资本占用可能偏高。

### Step 3：检查例外聚束

用 [[Clustering Test]] 或 [[Christoffersen Test]] 检查例外是否独立。

- 例外集中出现：模型没有捕捉 [[Volatility Clustering|波动聚集]] 或市场状态切换。
- 例外随机分布：至少没有明显连续失效。

### Step 4：用监管信号灯做快速分层

99% VaR、250 天回测常用口径：

| 例外次数 | 信号 | 解释 |
| --- | --- | --- |
| 0-4 | 绿色 | 结果通常可接受 |
| 5-9 | 黄色 | 需要关注并解释 |
| 10+ | 红色 | 模型显著低估风险 |

### Step 5：定位修正方向

| 诊断现象 | 可能原因 | 修正方向 |
| --- | --- | --- |
| 例外过多 | 波动率估计偏低、厚尾忽略 | 引入 [[GARCH]]、[[EWMA]]、厚尾分布或 [[EVT]] |
| 例外聚集 | 风险因子状态变化、波动聚集 | 缩短 [[Observation Window]]、改用时变波动模型 |
| 例外过少 | 过度保守 | 检查参数、窗口和分位数设定 |
| 只在极端时期失效 | 压力情景缺失 | 补 [[Scenario Analysis]] 与 [[Stress Testing]] |

## 稳健性检查

- 分别看不同资产类别、交易台和风险因子，不只看总组合。
- 同时报告例外频率、聚束和最大单日超越幅度。
- 回测样本必须和 VaR 模型的使用期一致，避免事后挑窗口。

## 常见错误

- 只数例外次数，不看例外是否连续。
- 把回测通过等同于未来有效。
- 用同一段数据既调模型又证明模型有效。

## 来自课程位置

- [[12_VAR风险]]
- [[22_情景分析和压力测试]]

## 关联卡片

## 复现规范

记录输入数据与样本区间、模型/检验设定、阈值或显著性水平、软件版本和处理决策；保留诊断图表与原始输出，使“发现—判断—修正”链条可复核。

- [[VaR]]
- [[Kupiec Test]]
- [[Clustering Test]]
- [[Christoffersen Test]]
- [[VaR Standard Error]]
