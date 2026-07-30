---
aliases:
- 单因子模型
- Single-Factor Model
- Single
tags:
- 经济
- concept
---

# Single-Factor Model
**单因子模型：**假设存在一个公共因子 $F$，以及每个变量各自的独立特异因素 $Z_i$。令 $U_i$ 表示标准化后的第 $i$ 个变量（均值0，方差1，例如资产收益的标准化），模型表示为：
$$
U_i = a_i\,F \;+\; \sqrt{\,1 - a_i^2\,}\;Z_i \,, \qquad i=1,2,\ldots,N,
$$ 
其中 $F \sim N(0,1)$，各 $Z_i \sim N(0,1)$ 彼此独立且与 $F$ 独立，$a_i$ 是第 $i$ 个变量对公共因子的加载系数（$-1 \le a_i \le 1$）。在该模型下，任意两变量的相关系数可由因子加载计算得出：
$$
\mathrm{Corr}(U_i, U_j) = \mathrm{Cov}(U_i, U_j) = a_i a_j \,,
$$ 
因为 $\mathrm{Cov}(U_i, U_j) = a_i a_j\,\mathrm{Var}(F) + 0 = a_i a_j$（公共因子部分贡献相关，特异部分独立无协方差）。单因子模型将原本 $N(N-1)/2$ 个相关参数简化为 $N$ 个因子加载参数 $\{a_i\}$。

## 关联卡片

- [[Factor Analysis]]：单因子模型是因子模型的最小情形。
- [[Factor Loadings]]：$a_i$ 是变量与公共因子的加载系数。
- [[Specific Variance]]：特异项对应的不可由公共因子解释的波动。
- [[Systematic Risk]]：金融语境下，公共因子对应系统性风险来源。

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
