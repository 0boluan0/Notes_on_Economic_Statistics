---
aliases:
- White Noise
- 白噪声
- 白噪声过程
tags:
- 时间序列
- 概率论
- concept
---
白噪声过程（White Noise）是均值为 0、方差为常数且不同期不相关的随机过程。常见设定为高斯白噪声 $\varepsilon_t \sim i.i.d.\,N(0,\sigma^2)$。

## 定义

满足以下条件即为白噪声：

1. $E(\varepsilon_t)=0$
2. $\mathrm{Var}(\varepsilon_t)=\sigma^2$（与 $t$ 无关）
3. $\mathrm{Cov}(\varepsilon_t,\varepsilon_{t-s})=0$（任意非零滞后 $s$）

## 备注

- 若进一步假定正态分布，则为“高斯白噪声”。
- 白噪声可作为 ARMA 创新项与残差检验基准。

## Connections

- 用途：[[ARMA]] 创新项、[[白噪声检验]] 残差诊断
- 相关：[[自相关函数]]、[[偏自相关函数]]

## source_notes

- [[03_平稳时间序列模型#1.1 自回归移动平均模型ARMA(p,q) model]]（创新项设为白噪声）
- [[03_平稳时间序列模型#1.2 三种‘没有关系’的辨析]]（白噪声定义）
