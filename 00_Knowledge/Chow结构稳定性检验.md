---
aliases:
  - "已知分组或断点下的经典 Chow 检验用联合 F 统计量检验两组线性回归系数是否相同"
  - Classical Chow test for coefficient stability
  - Chow 检验
  - 邹检验
  - 已知断点结构稳定性检验
student_os: knowledge-atom
atom_id: ECON-OLS-024
atom_set: regression-inference
atom_type: procedure
status: source-checked
mastery_state: unassessed
part_of:
  - "[[回归推断.canvas|回归推断]]"
requires:
  - "[[嵌套模型F检验]]"
related:
  - "[[线性组合推断]]"
  - "[[Wald联合检验]]"
  - "[[标准误口径匹配]]"
leads_to:
  - "[[回归联合显著边界]]"
---

# 已知分组或断点下的经典 Chow 检验用联合 F 统计量检验两组线性回归系数是否相同
<!-- bilingual-en:start -->
*For a known grouping or break point, the classical Chow test uses a joint F statistic to test whether two linear regressions have the same coefficients*
<!-- bilingual-en:end -->

> [!summary] 原子方法
> 若同一个线性关系可能在两个事先给定的 groups 或断点两侧改变，Chow 检验的原假设是两组的整组回归系数相同。它把“分别拟合两组”当作无约束模型，把“强迫两组共用系数”当作受限模型，再用经典嵌套 F 检验衡量这些相等约束带来的 RSS 增量。
>
> <!-- bilingual-en:start -->
> When the same linear relationship may differ across two prespecified groups or on either side of a known break, the Chow null states that the complete coefficient vectors are equal. Separate regressions form the unrestricted model; a pooled common-coefficient regression is the restricted model. A classical nested-model F test measures the RSS cost of imposing equality.
> <!-- bilingual-en:end -->

设两组都使用含 $p$ 个参数的同一组 regressors，样本量分别为 $n_1,n_2$。令 $RSS_1,RSS_2$ 为分别估计的残差平方和，$RSS_P$ 为强迫两组共用系数的 pooled RSS，则检验全部 $p$ 个系数相等的统计量是
$$
F=
\frac{\bigl[RSS_P-(RSS_1+RSS_2)\bigr]/p}
{(RSS_1+RSS_2)/(n_1+n_2-2p)}.
$$
在经典正态、独立且两组共享同一误差方差的线性模型下，原假设成立时它服从 $F_{p,n_1+n_2-2p}$。若只检验部分系数稳定，要把 $RSS_P$ 换成只施加那些相等限制的 $RSS_R$，并把分子自由度改成独立限制的秩 $q$；不能继续使用“全部系数共用”模型的 RSS。
<!-- bilingual-en:start -->
With $p$ parameters in each group, the numerator is the pooled-model RSS increase per equality restriction and the denominator is the unrestricted residual variance. Under the classical normal linear model with independent errors and a common variance across groups, the full-coefficient statistic has an $F_{p,n_1+n_2-2p}$ null distribution. For a subset test, replace $RSS_P$ by the RSS from the model imposing only those equalities and replace the numerator degrees of freedom by the restriction rank $q$.
<!-- bilingual-en:end -->

## 用交互项写法看清它究竟检验什么
<!-- bilingual-en:start -->
*The interaction form shows exactly what is being tested*
<!-- bilingual-en:end -->

令 $D_i$ 指示第二组，把两组数据合并后写成
$$
y_i=x_i'\beta+D_i x_i'\delta+u_i.
$$
$\delta$ 收集第二组相对第一组的截距和斜率变化；Chow 原假设就是 $H_0:\delta=0$。因此它本质上是一个明确的联合线性限制检验。拒绝原假设只说明至少一个被检验的系数发生变化，不能自动定位是哪一个，也不能说明变化由政策、危机或其他机制造成。
<!-- bilingual-en:start -->
After pooling the data, interact the group indicator with every tested regressor. The vector $\delta$ then collects the second group's intercept and slope changes, and the Chow null is $H_0:\delta=0$. Rejection says that at least one tested coefficient differs; it neither identifies which one without further contrasts nor attributes the difference to a policy, crisis, or other mechanism.
<!-- bilingual-en:end -->

## 三个不能省略的边界
<!-- bilingual-en:start -->
*Three boundaries that cannot be omitted*
<!-- bilingual-en:end -->

1. 两组或 break point 应由研究设计、制度时间或事先假设给出。若先在很多断点中挑出统计量最大的一个，再使用普通单次 F 临界值，选择过程已经改变参考分布。
2. 传统 RSS 版本要求两组使用同一响应、同一 regressors，并依赖共同误差方差等经典条件。若存在异方差、序列相关或 cluster dependence，应在合并交互模型中使用与依赖结构匹配的 robust Wald/structural-break procedure，不能只给经典公式换标签。
3. “系数稳定”是关于所写线性模型和所检验参数的命题，不等于真实机制完全不变。遗漏变量、函数形式变化或样本构成变化都可能驱动拒绝。

<!-- bilingual-en:start -->
**1.** The groups or break point should come from the design, institutional timing, or a prespecified hypothesis. Searching over break points and then using an ordinary one-test F critical value ignores selection.<br>
**2.** The classical RSS version requires the same response and regressors and relies on a common error variance and the other classical conditions. Heteroskedastic, serially correlated, or clustered errors require a matching robust Wald or structural-break procedure.<br>
**3.** Coefficient stability concerns the stated linear model and tested parameters. A rejection can be driven by omitted variables, functional-form change, or a changing sample composition rather than the substantive mechanism named by the researcher.<br>
<!-- bilingual-en:end -->

> [!question]- 自检
> 研究者在 20 个候选月份中挑出 Chow F 最大的月份，再把它与预先指定一个断点时的普通 $F_{p,n_1+n_2-2p}$ 临界值比较。这个 p 值为什么不再是原来的单次检验 p 值？
>
> **答案：** 报告的统计量已经是 20 个候选统计量的最大值，而不是一个固定断点下的 F；搜索提高了纯噪声产生极端值的机会，必须使用纳入断点选择的结构突变校准。

## 来源与核验

- [[02_Economy/01_Econometrics/03_多元线性回归.md#3.2.5. F 检验的一个例子：Chow 结构稳定性检验|本地课程：Chow 结构稳定性检验]]：核对课程中的两组 RSS 写法、自由度与结构稳定语境。
- Chow (1960), [*Tests of Equality Between Sets of Coefficients in Two Linear Regressions*](https://doi.org/10.2307/1910133)：核验两组系数相等问题及其一般线性假设框架。
- Stata, [*Computing the Chow statistic*](https://www.stata.com/support/faqs/statistics/computing-chow-statistic/)：交叉核验 pooled-versus-separate RSS 公式、与合并样本全交互回归的等价，以及传统检验中的共同方差条件。
- Stata, [*Test for a structural break with a known break date*](https://www.stata.com/manuals15/tsestatsbknown.pdf)：核验已知断点原假设，并区分传统 Chow 与异方差稳健的 Wald/LR 实现。
<!-- bilingual-en:start -->
- The local course supplies the RSS implementation. Chow's original article establishes the equality-of-coefficients problem; Stata's official documentation verifies the pooled-interaction equivalence, classical variance boundary, and distinction between a known-break Chow test and robust structural-break procedures.
<!-- bilingual-en:end -->
