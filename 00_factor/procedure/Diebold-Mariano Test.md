---
aliases:
- Diebold-Mariano Test
- DM Test
- Diebold Mariano Test
- Diebold-Mariano检验
- DM检验
tags:
- procedure
- 时间序列
---
# Diebold-Mariano Test

## 这张卡什么时候用

当你要比较两个模型的预测能力，并且希望允许一般损失函数和预测误差自相关时，用 Diebold-Mariano 检验。

## 输入

- 两个模型在同一预测期的误差 $e_{1i},e_{2i}$；
- 一个损失函数 $g(e)$，例如 $e^2$ 或 $|e|$；
- 预测期长度 $H$。

## 输出

- 两个模型的平均预测损失是否显著不同；
- 哪个模型的平均损失更低。

## Step 1. 构造损失差

$$
d_i=g(e_{1i})-g(e_{2i}).
$$

## Step 2. 计算平均损失差

$$
\bar d=\frac{1}{H}\sum_{i=1}^{H}d_i.
$$

若 $\bar d<0$，模型 1 的平均损失更低；若 $\bar d>0$，模型 2 更低。

## Step 3. 估计 $\bar d$ 的方差

若 $d_i$ 没有自相关，可用普通方差。

若 $d_i$ 有自相关，使用 Newey-West 型长期方差估计。

## Step 4. 构造统计量

$$
DM=\frac{\bar d}{\sqrt{\widehat{\operatorname{Var}}(\bar d)}}.
$$

大样本下用标准正态分布近似判断。

## Step 5. 写结论

- 拒绝平均损失相等：两个模型预测表现有显著差异。
- 不拒绝：样本中损失较小的一方不一定稳定更好。

## 常见错误

- 两个模型预测期不一致。
- 多步预测时忘记处理损失差的自相关。
- 只看 $\bar d$ 正负，不报告显著性。

## 来自课程位置

- [[03_平稳时间序列模型#5.5. 预测效果评估|时间序列 03：DM 检验]]

## 关联卡片

- [[Forecast Evaluation]]
- [[Granger-Newbold Test]]
- [[Newey-West]]

