---
aliases:
  - "劣等品只要求收入上升时需求下降，而Giffen品还要求自身价格上升时普通需求增加"
  - Inferior goods and Giffen goods
  - Inferiority does not imply Giffen behaviour
student_os: knowledge-atom
atom_id: MICRO-CONS-005
atom_set: income-substitution
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[Slutsky方程]]"
related:
  - "[[价格效应分解]]"
  - "[[普通与补偿需求]]"
part_of:
  - "[[收入效应与替代效应.canvas]]"
---

# 劣等品只要求收入上升时需求下降，而Giffen品还要求自身价格上升时普通需求增加
<!-- bilingual-en:start -->
*An inferior good is defined by demand falling with income, whereas a Giffen good additionally has Marshallian demand rising with its own price*
<!-- bilingual-en:end -->

> [!summary] 两个定义看的是不同导数
> 在给定价格与收入附近，劣等品满足 $\partial x_i/\partial m<0$；Giffen 品满足 $\partial x_i/\partial p_i>0$。前者描述收入变化，后者描述自身价格变化。劣等性只是 Giffen 机制的必要一环，不足以推出价格与需求同向。
> <!-- bilingual-en:start -->
> Inferiority concerns the income derivative of Marshallian demand; Giffen behaviour concerns its own-price derivative. An inferior good becomes Giffen only when the positive income-effect component of an own-price increase dominates the negative compensated substitution effect.
> <!-- bilingual-en:end -->

对需求为正的商品，Slutsky 方程写成
$$
\frac{\partial x_i}{\partial p_i}
=
\underbrace{\frac{\partial h_i}{\partial p_i}}_{\le0}
-
\underbrace{x_i\frac{\partial x_i}{\partial m}}_{\text{收入项的相反数}}.
$$
若商品正常，$\partial x_i/\partial m>0$，收入项也为负，自身价格上涨必使普通需求弱下降。若商品劣等，$\partial x_i/\partial m<0$，第二项转为正，收入效应与替代效应方向相反。

但“方向相反”不等于“收入效应获胜”。例如补偿价格导数为 $-2$、当前需求 $x_i=3$：

- 若 $\partial x_i/\partial m=-0.2$，总价格导数为 $-2-3(-0.2)=-1.4$。它是劣等品，却仍遵守通常的向下需求。
- 若 $\partial x_i/\partial m=-1$，总价格导数为 $-2-3(-1)=1$。正向收入效应超过替代效应，才在这个局部区间表现为 Giffen 品。

因此，在标准的非负消费、单值可微需求与自身补偿需求不增条件下，局部 Giffen 行为意味着该商品在相应点是劣等品；反过来不成立。“Giffen”也不是商品永恒不变的标签：同一种主食可能只在特定收入与消费区间满足条件。观察到夏季汽油价格和消费量同时上升，也不能据此判定 Giffen，因为共同的季节性需求移动并不是沿同一需求曲线的自身价格效应。

> [!question]- 自检
> 已知某商品是劣等品，能否推出它涨价后需求增加？还缺什么信息？
>
> **答案：** 不能。还要知道负向的补偿替代效应和正向的收入效应各有多大；只有后者严格占优，普通需求的自身价格导数才为正。

## 来源与核验

- [MIT 14.03, Lecture Note 7](https://ocw.mit.edu/courses/14-03-microeconomic-theory-and-public-policy-fall-2016/c4cc33011be7b8f56cab3b0203148aa2_MIT14_03F16_lec7.pdf)，pp. 1–3, 9：核对正常品、劣等品与 Giffen 品的导数定义，以及 Giffen 情形要求收入效应压过替代效应。
- [MIT 14.121, Consumer Theory slides](https://ocw.mit.edu/courses/14-121-microeconomic-theory-i-fall-2015/ea9f11b15ace05e7bfd31d58ae48beb9_MIT14_121F15_2S.pdf)，slides 34–40：核对补偿自身需求定律、Slutsky 方程和 normal/inferior/Giffen 的局部定义。
