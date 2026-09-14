---
aliases:
  - "Slutsky方程把普通需求的价格导数分解为补偿价格导数减去原需求量乘收入导数"
  - Slutsky equation for consumer demand
  - Slutsky decomposition
student_os: knowledge-atom
atom_id: MICRO-CONS-003
atom_set: income-substitution
atom_type: identity
status: source-checked
mastery_state: unassessed
requires:
  - "[[普通与补偿需求]]"
  - "[[多元链式法则]]"
related:
  - "[[价格效应分解]]"
  - "[[Hicks与Slutsky补偿]]"
  - "[[劣等品与Giffen品]]"
part_of:
  - "[[收入效应与替代效应.canvas]]"
---

# Slutsky方程把普通需求的价格导数分解为补偿价格导数减去原需求量乘收入导数
<!-- bilingual-en:start -->
*The Slutsky equation decomposes the price derivative of Marshallian demand into a compensated price derivative minus initial demand times the income derivative*
<!-- bilingual-en:end -->

> [!summary] 局部价格反应的精确恒等式
> 令 $x_i(p,m)$ 为商品 $i$ 的普通需求，$h_i(p,u)$ 为补偿需求，并在 $u=v(p,m)$ 处比较二者。在需求单值且可微时，
> $$
> \frac{\partial x_i(p,m)}{\partial p_j}
> =
> \frac{\partial h_i(p,u)}{\partial p_j}
> -x_j(p,m)\frac{\partial x_i(p,m)}{\partial m}.
> $$
> 左边是总价格效应；右边第一项是保持效用的替代效应，第二项是价格变化通过购买力产生的收入效应。
> <!-- bilingual-en:start -->
> At the matched utility level $u=v(p,m)$, the Slutsky equation writes the Marshallian price derivative as the Hicksian price derivative minus the income derivative scaled by the amount of good $j$ initially consumed.
> <!-- bilingual-en:end -->

这个式子来自对偶恒等式
$$
h_i(p,u)=x_i(p,e(p,u)).
$$
对 $p_j$ 求偏导，同时把 $u$ 固定，链式法则给出
$$
\frac{\partial h_i}{\partial p_j}
=
\frac{\partial x_i}{\partial p_j}
+
\frac{\partial x_i}{\partial m}
\frac{\partial e}{\partial p_j}.
$$
Shephard 引理说明 $\partial e/\partial p_j=h_j$；在匹配点 $h_j(p,u)=x_j(p,m)$。移项后就是 Slutsky 方程。因而收入项为什么乘以 $x_j$ 并非记忆规则：消费者原本买得越多，$p_j$ 的微小上涨对维持原效用所需支出的一级影响越大。

对自身价格 $j=i$，补偿需求定律给出
$$
\frac{\partial h_i}{\partial p_i}\le0.
$$
若 $i$ 是正常品，$\partial x_i/\partial m>0$，则收入项 $-x_i\partial x_i/\partial m\le0$，两项都使涨价后的需求下降。若 $i$ 是劣等品，收入项为正，与替代项对抗；是否最终向上，要比较大小。

对交叉价格 $j\ne i$，不能套用“替代项必为负”。$\partial h_i/\partial p_j$ 对补偿替代品可以为正，对补偿互补品可以为负。方程本身也不保证普通需求的交叉效应对称，因为收入效应通常不对称。

这是**微分、局部**恒等式。连续且局部非饱和的偏好保证基本对偶关系，但写成上述导数形式还要求相关需求在考察点单值并可微。有限价格变化应使用明确的 Hicks 或 Slutsky 补偿路径，不能把导数乘一个很大的价格差当作精确分解。

> [!question]- 自检
> 若某正常品的当前需求为正，它自身价格小幅上升时，Slutsky 方程中的两项各是什么方向？
>
> **答案：** 补偿替代项不为正；正常品的收入导数为正，所以 $-x_i\partial x_i/\partial m$ 也不为正。两项同向，因此普通需求不会因这次自身价格上涨而增加。

## 来源与核验

- [MIT 14.121, Consumer Theory slides](https://ocw.mit.edu/courses/14-121-microeconomic-theory-i-fall-2015/ea9f11b15ace05e7bfd31d58ae48beb9_MIT14_121F15_2S.pdf)，slides 30–39：核对 Shephard 引理、对偶恒等式、Slutsky 方程的 $i,j$ 形式及唯一、可微条件。
- [MIT 14.03, Lecture Note 7](https://ocw.mit.edu/courses/14-03-microeconomic-theory-and-public-policy-fall-2016/c4cc33011be7b8f56cab3b0203148aa2_MIT14_03F16_lec7.pdf)，pp. 4–10：核对从 $h=x(p,e)$ 到方程的推导，以及收入项按初始购买量缩放的解释。
