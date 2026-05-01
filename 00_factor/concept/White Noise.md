---
aliases:
- White Noise
- white noise
- WN
- 白噪声
- 白噪声过程
tags:
- concept
- 时间序列
- 概率论
---
# White Noise

## 先记一句话

白噪声就是：**均值为 0、方差稳定、不同期不相关的随机扰动**。

它是时间序列模型里最常见的“剩下不能解释的部分”。

## 它是什么

序列 $\varepsilon_t$ 是 white noise，通常要求：

$$
E(\varepsilon_t)=0,
$$

$$
\operatorname{Var}(\varepsilon_t)=\sigma^2,
$$

且对任意非零滞后 $s$：
$$
\operatorname{Cov}(\varepsilon_t,\varepsilon_{t-s})=0.
$$

如果还服从正态分布，就叫 Gaussian white noise。

## 它解决什么判断

白噪声是模型诊断的底线：

> 如果残差还不是白噪声，说明均值模型还漏掉了可预测结构。

在 ARMA 里，创新项通常假设为白噪声。

在建模后，要用 [[White Noise Test]] 或 [[Ljung-Box Test]] 检查残差。

## 和 IID / MDS 的关系

在零均值且二阶矩存在时，常见强弱关系是：
$$
[[IID]] \Rightarrow [[Martingale Difference Sequence]] \Rightarrow [[White Noise]].
$$

反过来一般不成立。

特别注意：白噪声只要求不相关，不要求独立。

## 常见误区

- 白噪声不是“没有方差”，而是方差稳定、不可由线性相关预测。
- 白噪声不等于 i.i.d.；i.i.d. 更强。
- 有条件异方差的序列可能仍然均值不可预测，但平方项有结构，这时要看 [[ARCH]] / [[GARCH]]。

## 来自课程位置

- [[03_平稳时间序列模型#1.1 自回归移动平均模型ARMA(p,q) model|时间序列 03：ARMA 创新项]]
- [[03_平稳时间序列模型#1.2 三种‘没有关系’的辨析|时间序列 03：IID/MDS/白噪声关系]]

## 关联卡片

- [[IID]]
- [[Martingale Difference Sequence]]
- [[White Noise Test]]
- [[ARMA]]
- [[ARCH]]
- [[GARCH]]

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
