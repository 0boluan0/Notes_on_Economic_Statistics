---
aliases:
- Autocorrelation Diagnosis
- 自相关诊断
- 序列相关诊断
tags:
- system
- econometrics
---
# Autocorrelation Diagnosis

## 诊断目标

判断回归残差是否存在序列相关，并决定是修正标准误、加入动态结构，还是用 FGLS 类方法。

## 诊断流程

### Step 1：看残差时间图

连续为正/负通常提示正自相关；正负交替可能提示负自相关。

### Step 2：选择检验

| 情况 | 检验 |
| --- | --- |
| 只关心一阶自相关，模型无滞后因变量 | [[Durbin-Watson Statistic]] |
| 高阶自相关或有滞后因变量 | [[Breusch-Godfrey Test]] |
| 检查序列是否接近白噪声 | [[Q Test]] / Ljung-Box |

### Step 3：判断后果

在严格外生仍成立时：

- OLS 系数通常仍无偏。
- OLS 不再 BLUE。
- 经典标准误通常错误。
- t/F 检验不可靠。

若模型含滞后因变量且误差自相关，可能出现内生性问题。

### Step 4：选择处理方式

| 问题 | 处理 |
| --- | --- |
| 只需要稳健推断 | [[Newey-West]] |
| 一阶 AR(1) 误差结构可信 | [[Cochrane-Orcutt]] 或 Prais-Winsten |
| 自相关来自遗漏动态 | 增加滞后变量或改模型设定 |
| 非平稳导致伪相关 | 先做 [[Unit Root Test]] 和差分/协整分析 |

## 常见错误

- 用 DW 检验含滞后因变量的模型。
- 发现自相关后只机械差分，导致长期关系丢失。
- 把变量自相关和残差自相关混为一谈。

## 来自课程位置

- [[08_自相关]]

## 关联卡片

## 复现规范

记录输入数据与样本区间、模型/检验设定、阈值或显著性水平、软件版本和处理决策；保留诊断图表与原始输出，使“发现—判断—修正”链条可复核。

- [[Autocorrelation]]
- [[Durbin-Watson Statistic]]
- [[Breusch-Godfrey Test]]
- [[Newey-West]]
