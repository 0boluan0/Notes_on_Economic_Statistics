---
aliases:
- 牛顿-莱布尼茨公式
- 微积分基本定理
- Fundamental Theorem of Calculus
tags:
- 数学
- 微积分
- concept
---
微积分基本定理是联系微分和积分的核心定理，是微积分学的基石。

## 牛顿-莱布尼茨公式

如果 f(x) 在 [a, b] 上连续，且 F(x) 是 f(x) 的一个原函数，则：

$\int_a^b f(x)dx = F(b) - F(a)$

## 意义

1. **建立联系**：建立了微分和积分之间的联系
2. **计算工具**：提供了计算定积分的简便方法
3. **理论基石**：是微积分学的核心定理

## 第一部分定理

如果 f(x) 在 [a, b] 上连续，定义：

$F(x) = \int_a^x f(t)dt$

则 F(x) 在 [a, b] 上可导，且：

$F'(x) = f(x)$

## 第二部分定理

如果 f(x) 在 [a, b] 上连续，且 F(x) 是 f(x) 的一个原函数，则：

$\int_a^b f(x)dx = F(b) - F(a)$

## 推论

1. **变限积分求导**

   如果 F(x) = \int_{a(x)}^{b(x)} f(t)dt，则：
$$

   $F'(x) = f(b(x))b'(x) - f(a(x))a'(x)$

2. **积分中值定理**

   如果 f(x) 在 [a, b] 上连续，则存在 c ∈ [a, b]，使得：

   $\int_a^b f(x)dx = f(c)(b-a)$

## 应用

1. **计算定积分**：通过原函数计算定积分值
2. **求原函数**：通过定积分构造原函数
3. **证明定理**：微积分中很多定理的证明依赖于它
4. **数值计算**：为数值积分提供理论基础

## 历史意义

- 牛顿和莱布尼茨独立发现了这个定理
- 标志着微积分的正式诞生
- 使微积分成为统一的理论体系

## 相关链接
[[Derivative|导数]]
[[Integral|积分]]
[[Limit|极限]]

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
