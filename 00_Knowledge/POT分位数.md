---
aliases:
  - POT分位数通过反解阈值以上的无条件尾概率求得
student_os: knowledge-atom
atom_id: RM-EVT-022
atom_type: method
status: source-checked
requires:
  - "[[POT尾概率]]"
  - "[[VaR定义]]"
leads_to:
  - "[[POT尾部ES]]"
part_of:
  - "[[极值理论 EVT 与尾部风险.canvas]]"
---

# POT分位数通过反解阈值以上的无条件尾概率求得
<!-- bilingual-en:start -->
*A POT quantile inverts the unconditional tail probability above the threshold*
<!-- bilingual-en:end -->

给定连续的 GPD 尾部模型，先检查目标分位是否确实位于阈值以上，再解“尾概率等于 $1-\alpha$”。POT 改变的是 [[VaR定义|VaR 的估计方法]]，不是 VaR 的定义；模型外推可以超过历史最大损失，但仍受尾部假设和参数误差约束。
<!-- bilingual-en:start -->
For a continuous GPD tail model, first check that the target quantile lies above the threshold, then solve for tail probability $1-\alpha$. POT changes the [[VaR定义|estimation method for VaR]], not its definition. Extrapolation may exceed the historical maximum but remains conditional on tail assumptions and uncertain parameters.
<!-- bilingual-en:end -->

设阈值为 $u$、超越概率为 $p_u>0$、GPD 参数为 $\beta>0,\xi$。采用严格的上尾范围 $1-p_u<\alpha<1$，则 $v_\alpha>u$。由 [[POT尾概率]]，当 $\xi\ne0$ 时：
<!-- bilingual-en:start -->
Let the threshold be $u$, its exceedance probability $p_u>0$, and the GPD parameters $\beta>0,\xi$. Restrict attention to $1-p_u<\alpha<1$, so that $v_\alpha>u$. From [[POT尾概率|the POT tail probability]], for $\xi\ne0$:
<!-- bilingual-en:end -->

$$
p_u\left(1+\xi\frac{v_\alpha-u}{\beta}\right)^{-1/\xi}=1-\alpha
\quad\Longrightarrow\quad
v_\alpha=u+\frac{\beta}{\xi}\left[\left(\frac{p_u}{1-\alpha}\right)^\xi-1\right].
$$

若 $\xi=0$，用指数尾直接反解，或对上式取极限：
<!-- bilingual-en:start -->
For $\xi=0$, invert exponential survival directly or take the continuous limit:
<!-- bilingual-en:end -->

$$v_\alpha=u+\beta\log\frac{p_u}{1-\alpha}.$$

实际估计用 $\widehat p_u=N_u/n,\widehat\xi,\widehat\beta$ 代入。若 $1-\alpha>\widehat p_u$，目标在这段尾部以下，不能用公式向下延伸来代替缺失的中心分布。等号给出模型尾部的接合值 $u$；若原分布在阈值附近有间隙或原子质量，左分位点本身仍应按定义核对。
<!-- bilingual-en:start -->
For estimation, substitute $\widehat p_u=N_u/n,\widehat\xi,\widehat\beta$. If $1-\alpha>\widehat p_u$, the target lies below the modelled tail; downward extrapolation cannot replace the missing central distribution. Equality gives the tail model's join value $u$, but gaps or probability atoms near the threshold require checking the left quantile itself.
<!-- bilingual-en:end -->

## 把结果代回概率
<!-- bilingual-en:start -->
*Check the result by substituting it back*
<!-- bilingual-en:end -->

沿用课程题设：$n=1000,N_u=50,u=100$ 万元，$\widehat\xi=0.2,\widehat\beta=10$ 万元，求一天 99% VaR。因为 $0.01<0.05$，目标在阈值以上：
<!-- bilingual-en:start -->
Using the course parameters $n=1000,N_u=50,u=100$ and $\widehat\beta=10$ in CNY 10,000, with $\widehat\xi=0.2$, calculate one-day 99% VaR. Since $0.01<0.05$, the target is above the threshold:
<!-- bilingual-en:end -->

$$
\widehat v_{0.99}=100+50(5^{0.2}-1)
=118.986483\ldots\ \text{万元}.
$$

不用中途舍入的值代回，得到 $1+0.2(\widehat v-100)/10=5^{0.2}$，因此尾概率为 $0.05\times5^{-1}=0.01$。若把 95% 当作本例的目标，公式只回到阈值 100；不能把条件 GPD 的 95% 分位误当整个损失分布的 95% VaR。
<!-- bilingual-en:start -->
Substituting the unrounded value gives $1+0.2(\widehat v-100)/10=5^{0.2}$ and hence tail probability $0.05\times5^{-1}=0.01$. A 95% target returns the threshold 100 in this model. The conditional GPD's 95th percentile is not the full loss distribution's 95% VaR.
<!-- bilingual-en:end -->

当 $\xi<0$ 时，这个分位数对每个 $\alpha<1$ 都在有限端点 $u-\beta/\xi$ 以下，并在 $\alpha\to1$ 时趋近端点。$\xi\ge1$ 也不妨碍这些有限置信水平的分位数有限；此时失效的是有限尾均值，见 [[POT尾部ES]]。
<!-- bilingual-en:start -->
For $\xi<0$, the quantile stays below the finite endpoint $u-\beta/\xi$ for every $\alpha<1$, approaching it as $\alpha\to1$. Even $\xi\ge1$ permits finite quantiles at such levels; what fails then is a finite tail mean, as explained in [[POT尾部ES|POT expected shortfall]].
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Haugh，*Extreme Value Theory*，PDF 第 27–29 页](https://www.columbia.edu/~mh2078/QRM/EVT_MasterSlides.pdf#page=28)：已重开并目视尾概率反解、高分位及超阈值比例估计。本页单独核对了 $\xi=0$ 极限、负形状端点和目标在尾部以上的条件。
- [[02_Economy/07_金融机构与风险管理/13_历史模拟法和极值理论|第 13 章计算题 1]]：题设与损失单位来源；本页保留未过早舍入的结果并代回概率检查。
<!-- bilingual-en:start -->
- Haugh, PDF pp. 27–29, was reopened and visually checked for tail inversion, high quantiles and exceedance-frequency estimation. The zero-shape limit, negative-shape endpoint and target-range restriction were independently checked here.
- [[02_Economy/07_金融机构与风险管理/13_历史模拟法和极值理论|Chapter 13, Calculation Question 1]], supplies the parameters and loss units. The unrounded result was checked by substituting it back into the tail probability.
<!-- bilingual-en:end -->
