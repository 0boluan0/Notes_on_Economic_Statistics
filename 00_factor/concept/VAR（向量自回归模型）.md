---
aliases:
- VAR
- 向量自回归
- 向量自回归模型
tags:
- concept
---
# VAR（向量自回归模型）

## 它是什么
- VAR（Vector Autoregression）是多变量时间序列模型，用每个变量自身滞后项和其他变量滞后项来解释当前值。

## 最小可检索信息
- 形式：$y_t = c + A_1 y_{t-1} + \\cdots + A_p y_{t-p} + u_t$
- 适用：多个变量相互影响、需要做动态联动分析
- 常见产出：脉冲响应函数、方差分解

## 关联卡片
- [[Time Series Analysis-hub]]
- [[脉冲响应函数]]
- [[方差分解]]
- [[Johansen协整检验步骤]]
