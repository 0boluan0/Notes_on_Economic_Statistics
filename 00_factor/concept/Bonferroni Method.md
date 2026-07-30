---
aliases:
- Bonferroni Method
- Bonferroni Correction
- Bonferroni 多重比较
- Bonferroni 方法
tags:
- concept
- multivariate statistics
---

# Bonferroni Method

>[!note] 一句话记忆
> Bonferroni 方法把整体显著性水平分摊到多个比较上，用简单保守的方式控制全家错误率。

## 它是什么

若要同时做 $m$ 个区间或检验，可以把每个比较的显著性水平调成约 $\alpha/m$。

均值分量区间常写作
$$
\bar X_i\pm t_{n-1}\left(\frac{\alpha}{2m}\right)\sqrt{\frac{s_{ii}}{n}}.
$$

## 解决什么判断

- 多个均值分量或线性组合要同时报告时如何控制整体错误。
- 比较数量较少时能否得到比 Hotelling 区间更短的区间。
- 多重比较是否需要调整显著性水平。

## 最小例子

同时构造 4 个 95% 区间时，每个区间可按更严格的临界值计算，以保证整体覆盖率。

## 易混点

- Bonferroni 通常保守，比较数很多时区间会明显变宽。
- 它不依赖完整协方差结构，和 Hotelling 椭球思路不同。

## 来自课程位置

- [[05_ 总体平均向量的推论#1.5. 置信区域与同时置信区间|第5章 3 Bonferroni 方法]]

## 关联卡片

- [[Simultaneous Confidence Intervals]]
- [[Confidence Interval]]
- [[P-value]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[05_ 总体平均向量的推论]]、[[Simultaneous Confidence Intervals]]、[[Confidence Interval]]、[[P-value]]。
