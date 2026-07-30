---
aliases:
  - "Multivariate Normal Distribution"
  - "Wishart Distribution"
  - "多元正态"
status: source-checked
---

# 多元正态分布与 Wishart 分布

> [!summary] 快速恢复
> **它解决什么：** 为多个连续变量提供一个可计算的联合分布，并描述样本协方差在正态抽样下如何波动。
> **具体锚点：** 身高和体重可各自近似正态，但真正的多元模型还要说明椭圆方向和相关性。
> **核心难点：** 每个边际都正态不必推出联合正态；条件分布与独立性结论依赖协方差块结构。
> **为什么重要：** Hotelling $T^2$、MANOVA、判别分析和协方差推断都建立在这里。
> **继续：** 均值检验见 [[多元均值推断与 MANOVA]]；分类见 [[判别分析与聚类]]。

> [!source] 本节依据
> - [Penn State STAT 505](https://online.stat.psu.edu/stat505/)：核验多元正态、均值推断、MANOVA、PCA、因子分析、判别与聚类的定义和使用条件。
> - Johnson & Wichern, *Applied Multivariate Statistical Analysis*, 6th ed.：核验矩阵公式与抽样分布。

## 密度与椭球几何

非奇异 $p$ 维正态写作 $N_p(\mu,\Sigma)$，密度等高线由 Mahalanobis 距离常数定义，是以 $\mu$ 为中心的椭球。$Sigma$ 的特征向量给主轴，特征值给各轴方差。奇异协方差对应分布落在低维仿射子空间上，不能使用普通密度公式。

## 线性组合与刻画

多元正态最有用的刻画是：任意线性组合 $a^TX$ 都是一元正态。线性变换仍正态，$AX+b\sim N(A\mu+b,A\Sigma A^T)$。仅逐个检查边际正态无法验证这一联合性质。

## 边际、条件与独立

分块正态的边际仍正态，条件分布的均值对已知块线性调整，条件协方差为 Schur complement。联合正态下零协方差等价于独立；离开正态族后这一等价一般失效。

## 二次型与卡方

若 $X\sim N_p(\mu,Sigma)$ 且 $Sigma$ 可逆，则 $(X-\mu)^T\Sigma^{-1}(X-\mu)\sim\chi_p^2$。这把椭球距离转为概率阈值，是置信区域和异常检测的基础。

## Wishart 与样本协方差

正态随机样本下，$(n-1)S$ 服从以 $Sigma$ 为尺度、自由度 $n-1$ 的 Wishart 分布，且样本均值与 S 独立。Wishart 是多元版卡方；不同教材参数化可能不同，使用密度和期望公式前必须核对尺度约定。

## 正态性诊断

诊断结合单变量图、Mahalanobis 距离 Q–Q 图和异常点检查。大样本检验容易把微小偏离判显著，小样本又缺乏功效；应同时判断方法对偏离的敏感性。

## 最小自检

### 各分量都服从正态，为什么仍可能不是多元正态？

> [!answer]- 答案
> 联合依赖结构可能使某些线性组合不正态；多元正态要求所有线性组合都正态。
### 联合正态中协方差为 0 为什么更强？

> [!answer]- 答案
> 正态联合分布完全由均值和协方差决定，零交叉协方差使联合密度分解，因此得到独立。
### Wishart 分布在多元推断中扮演什么角色？

> [!answer]- 答案
> 它描述正态样本协方差矩阵的抽样波动，类似卡方分布描述一元样本方差。

## 来源与核验

- [Penn State STAT 505](https://online.stat.psu.edu/stat505/)：核验多元正态、均值推断、MANOVA、PCA、因子分析、判别与聚类的定义和使用条件。
- Johnson & Wichern, *Applied Multivariate Statistical Analysis*, 6th ed.：核验矩阵公式与抽样分布。
