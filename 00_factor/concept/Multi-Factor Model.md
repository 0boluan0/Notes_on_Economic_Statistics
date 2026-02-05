---
aliases:
- 多因子模型
- Multi-Factor Model
- Multi
tags:
- 经济
- concept
---
**多因子模型：**可以推广到 $M$ 个因子。假设有因子 $F_1, \ldots, F_M$ 彼此独立且均为 $N(0,1)$，每个变量 $U_i$ 有对应的加载向量 $(a_{i1}, a_{i2}, \dots, a_{iM})$，则：
$$
U_i = a_{i1}F_1 + a_{i2}F_2 + \cdots + a_{iM}F_M \;+\; \sqrt{\,1 - \sum_{m=1}^M a_{im}^2\,}\;Z_i \,.
$$ 
在保证 $1 - \sum_{m}a_{im}^2 \ge 0$ 的前提下，每个 $U_i$ 方差仍为1。任意两变量的相关系数是各自对公共因子加载的**逐因子乘积之和**：
$$
Corr(U_i, U_j) = \sum_{m=1}^M a_{im}\,a_{jm} \,.
$$ 
例如，在两因子模型下 $Corr(U_i, U_j) = a_{i1}a_{j1} + a_{i2}a_{j2}$。单因子模型是 $M=1$ 的特例。
