---
aliases:
  - 广义Pareto分布是由尺度和形状参数定义的非负超额分布族
  - 广义帕累托分布
  - Generalized Pareto Distribution
  - GPD
student_os: knowledge-atom
atom_id: RM-EVT-007
atom_type: definition
status: source-checked
part_of:
  - "[[极值理论 EVT 与尾部风险.canvas]]"
requires:
  - "[[累积分布函数]]"
related:
  - "[[超阈值法]]"
  - "[[广义极值分布]]"
leads_to:
  - "[[超阈值极限定理]]"
  - "[[广义Pareto矩存在条件]]"
  - "[[GPD阈值稳定性]]"
---

# 广义Pareto分布是由尺度和形状参数定义的非负超额分布族
<!-- bilingual-en:start -->
*The generalized Pareto family defines nonnegative excess distributions through scale and shape parameters*
<!-- bilingual-en:end -->

广义 Pareto 分布（GPD）是下列 CDF 定义的非负随机变量分布族。尺度 $\beta>0$ 与超额具有相同单位，形状 $\xi\in\mathbb R$ 无量纲；这里采用超额从 0 开始的两参数形式，不另加入位置参数。
<!-- bilingual-en:start -->
The generalized Pareto distribution is the family of nonnegative laws defined by the following CDF. The scale $\beta>0$ has the same units as the excess, while the shape $\xi\in\mathbb R$ is dimensionless. This is the two-parameter form starting at zero, without an additional location parameter.
<!-- bilingual-en:end -->

$$
G_{\xi,\beta}(y)=
\begin{cases}
1-(1+\xi y/\beta)^{-1/\xi},&\xi\ne0,\quad y\ge0,\quad1+\xi y/\beta>0,\\
1-e^{-y/\beta},&\xi=0,\quad y\ge0.
\end{cases}
$$

对 $y<0$，CDF 为 0。若 $\xi<0$，还须对 $y\ge-\beta/\xi$ 将 CDF 置为 1，包含有限右端点处的连续延拓。因此支持集为：$\xi\ge0$ 时 $[0,\infty)$；$\xi<0$ 时 $[0,-\beta/\xi]$。不能在负形状的端点以外继续使用内部幂函数表达式。
<!-- bilingual-en:start -->
The CDF is zero for $y<0$. For $\xi<0$, it is one for $y\ge-\beta/\xi$, including the continuous extension at the finite right endpoint. The support is $[0,\infty)$ for $\xi\ge0$ and $[0,-\beta/\xi]$ for $\xi<0$. The interior power expression must not be extended beyond a negative-shape endpoint.
<!-- bilingual-en:end -->

形状的三种情况是：$\xi>0$ 时生存概率呈幂律尾；$\xi=0$ 时恰为指数分布，这是公式在 $\xi\to0$ 时的连续极限；$\xi<0$ 时右尾有界。例如 $\xi=-1,\beta=3$ 给出 $G(y)=y/3$（$0\le y\le3$），即 $U(0,3)$。
<!-- bilingual-en:start -->
For $\xi>0$ the survival probability has a power-law tail. At $\xi=0$ the law is exactly exponential, obtained as the continuous limit as $\xi\to0$. For $\xi<0$ the right tail is bounded. For example, $\xi=-1,\beta=3$ gives $G(y)=y/3$ on $0\le y\le3$, the uniform law on $[0,3]$.
<!-- bilingual-en:end -->

GPD 的尺度不是一般意义下的标准差：某些形状下方差并不有限，见[[广义Pareto矩存在条件]]。GPD 拟合的是给定阈值后的超额，不能仅凭这个定义就把整条损失分布宣称为 GPD。阈值改变后的参数关系见[[GPD阈值稳定性]]。
<!-- bilingual-en:start -->
The GPD scale is not generally its standard deviation: variance need not be finite, as explained by [[广义Pareto矩存在条件|GPD moment conditions]]. A tail application fits excesses above a specified threshold; the definition does not make the entire loss distribution GPD. See [[GPD阈值稳定性|GPD threshold stability]] for parameter changes at a higher threshold.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- Martin Haugh，[Extreme Value Theory，PDF 第 19、21、31 页](https://www.columbia.edu/~mh2078/QRM/EVT_MasterSlides.pdf#page=19)：核对 CDF、尺度正值、不同形状的支持集、阈值变化及无限方差例。端点取值由 CDF 连续延拓补全，均匀分布例独立代入核对。
- McNeil–Frey，[1999-06-19 作者稿，PDF／文内第 7 页](https://statmath.wu.ac.at/~frey/publications/evt-garch.pdf#page=7)：逐式核对 GPD 的两分支和支持集；不把该页对 Gumbel 吸引域的宽泛尾速率描述当作所有母分布都是指数尾的定理。
<!-- bilingual-en:start -->
- Haugh's notes, PDF pp. 19, 21, and 31, support the CDF, positive scale, shape-dependent support, threshold changes, and an infinite-variance example. Endpoint values are completed by continuity, and the uniform example is checked independently.
- The McNeil–Frey author draft, 19 June 1999, PDF and internal p. 7, is checked for both GPD branches and support. Its broad tail-rate description of the Gumbel domain is not treated as an exponential-tail theorem for every parent law.
<!-- bilingual-en:end -->
