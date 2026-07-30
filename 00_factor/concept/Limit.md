---
aliases:
- 收敛
- 极限
- Limit
tags:
  - 数学
  - 微积分
  - concept
---
# Limit

极限是描述函数在某点附近行为趋势的数学概念，是微积分的基石。

>[!note] 定义
>
> ### 函数极限
>
> 当 x 无限趋近于 a 时，f(x) 无限趋近于 L，记作：
>
> $\lim_{x \to a} f(x) = L$
>
> 严格定义（ε-δ定义）：
> 对任意 ε > 0，存在 δ > 0，使得当 0 < |x - a| < δ 时，有 |f(x) - L| < ε
>
> ### 单侧极限
>
> - 左极限：$\lim_{x \to a^-} f(x)$
> - 右极限：$\lim_{x \to a^+} f(x)$
>
## 极限存在条件

$\lim_{x \to a} f(x)$ 存在的充要条件是左右极限都存在且相等：

$\lim_{x \to a^-} f(x) = \lim_{x \to a^+} f(x)$

## 无穷极限

- 无限趋近于无穷：$\lim_{x \to \infty} f(x) = L$
- 函数值无限趋近于无穷：$\lim_{x \to a} f(x) = \infty$

## 极限运算性质

1. $\lim [f(x) ± g(x)] = \lim f(x) ± \lim g(x)$
2. $\lim [f(x) · g(x)] = \lim f(x) · \lim g(x)$
3. $\lim \frac{f(x)}{g(x)} = \frac{\lim f(x)}{\lim g(x)}（分母不为零）$
4. $\lim [f(x)]^n = [\lim f(x)]^n$

## 重要极限

1. $\lim_{x \to 0} \frac{\sin x}{x} = 1$
2. $\lim_{x \to 0} \frac{1 - \cos x}{x} = 0$
3. $\lim_{x \to \infty} (1 + \frac{1}{x})^x = e$
4. $\lim_{x \to 0} (1 + x)^{1/x} = e$

## 连续性

函数 f(x) 在点 a 处连续的条件：

1. $\lim_{x \to a} f(x)$ 存在
2. $f(a)$ 存在
3. $\lim_{x \to a} f(x) = f(a)$

## 求极限的方法

1. **代入法**：直接代入求值
2. **因式分解法**：约去零因子
3. **有理化法**：分子分母有理化
4. **洛必达法则**：处理 0/0 或 ∞/∞ 型
5. **等价无穷小替换**：利用无穷小性质
6. **泰勒展开**：复杂函数的近似
7. **夹逼定理**：用不等式确定极限

## 应用

1. **导数定义**：导数是极限的一种形式
2. **定积分定义**：定积分是和的极限
3. **级数收敛**：级数和是部分和的极限
4. **连续性判定**：用极限判断函数连续性

## 相关链接
[[Derivative|导数]]
[[Integral|积分]]


## 最小例子

把 **Limit** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
## 概念边界

本卡片只回答“它是什么”，集中在定义、核心关系与最小例子；具体估计步骤、证明和诊断流程应分别放在 procedure、proof 或 system 卡片中。
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
