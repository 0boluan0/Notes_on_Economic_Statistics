---
aliases:
- Cross-Correlation Function
- CCF
- Cross-correlation
- 交叉相关函数
- 互相关函数
tags:
- concept
- 时间序列
---

# Cross-Correlation Function

## 先记一句话

交叉相关函数就是：**看一个序列和另一个序列的领先/滞后值之间有多强线性关系**。

## 它是什么

课程中的写法：
$$
\rho_{yz}(i)=\frac{\operatorname{cov}(y_t,z_{t-i})}{\sigma_y\sigma_z}.
$$

$i>0$ 时，比较的是 $y_t$ 和过去的 $z_{t-i}$。

## 它解决什么判断

- $z$ 是否领先于 $y$。
- 影响大约从第几期滞后开始。
- 传递函数 $C(L)$ 的候选滞后结构。

## 最小例子

如果 $\rho_{yz}(2)$ 显著，而 $\rho_{yz}(0)$、$\rho_{yz}(1)$ 不显著，说明 $z$ 对 $y$ 的影响可能两期后才出现。

## 易混点

- CCF 是识别线索，不是最终模型。
- 如果 $z_t$ 自身高度自相关，CCF 的尖刺可能来自 $z$ 的内部动态。
- 需要结合 [[Transfer Function Model]] 和残差诊断使用。

## 来自课程位置

- [[05_多方程模型Multi-equation Time Series Models#3.2. ADL的模型性质|时间序列 05：交叉相关函数]]

## 关联卡片

- [[Transfer Function Model]]
- [[ADL]]
- [[Leading Indicator]]
- [[Autocorrelation Function]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Transfer Function Model]]、[[05_多方程模型Multi-equation Time Series Models]]、[[ADL]]、[[Leading Indicator]]、[[Autocorrelation Function]]。
