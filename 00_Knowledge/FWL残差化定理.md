---
aliases:
  - "FWL 定理把多元回归系数化为残差间斜率"
  - Frisch-Waugh-Lovell theorem
  - FWL theorem
  - Partialling-out theorem
  - 弗里施-沃-洛维尔定理
student_os: knowledge-atom
atom_id: ECON-OLS-009
atom_type: theorem
status: source-checked
mastery_state: unassessed
part_of:
  - "[[OLS 线性回归.canvas|OLS 线性回归]]"
  - "[[多重共线性.canvas|多重共线性与设计矩阵诊断]]"
requires:
  - "[[OLS正规方程]]"
explains:
  - "[[多元回归系数]]"
---

# FWL 定理把多元回归系数化为残差间斜率
<!-- bilingual-en:start -->
*The FWL theorem turns a multiple-regression coefficient into a slope between residuals*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> 在 $y$ 对焦点变量 $x$ 和控制矩阵 $Z$ 的回归中，先分别把 $y$ 与 $x$ 对 $Z$ 回归并取残差 $\tilde y$、$\tilde x$，再把 $\tilde y$ 对 $\tilde x$ 回归；所得斜率与完整多元回归中 $x$ 的 OLS 系数完全相同。
> <!-- bilingual-en:start -->
> In a regression of $y$ on a focal regressor $x$ and control matrix $Z$, first regress both $y$ and $x$ on $Z$ and retain residuals $\tilde y$ and $\tilde x$. Regressing $\tilde y$ on $\tilde x$ gives exactly the same slope as the OLS coefficient on $x$ in the full multiple regression.
> <!-- bilingual-en:end -->

## 使用条件与含义边界
<!-- bilingual-en:start -->
*Use condition and meaning boundary*
<!-- bilingual-en:end -->

两次残差化必须使用相同的控制变量、函数形式、权重和估计样本；若模型含截距，残差化回归也应含相应截距。FWL 是 OLS 的代数等价，不会让遗漏控制变量消失，也不会把条件线性关联自动变成因果效应。
<!-- bilingual-en:start -->
Both residualisations must use the same controls, functional form, weights, and estimation sample. If the full model contains an intercept, the residualisation regressions must include the corresponding intercept. FWL is an algebraic equivalence for OLS; it neither removes omitted controls nor converts a conditional linear association into a causal effect.
<!-- bilingual-en:end -->

FWL 保证焦点系数以及完整模型残差的代数等价；若还要用残差回归复现完整模型的标准误，必须沿用完整模型的自由度和同一种协方差估计规则。直接读取第三步简化回归默认报告的标准误，可能因为自由度或 leverage 修正不同而不相同。
<!-- bilingual-en:start -->
FWL guarantees algebraic equality of the focal coefficient and the full-model residuals. To reproduce the full model's standard error from the residual regression, one must also retain the full model's degrees of freedom and the same covariance-estimation rule. A default standard error reported by the reduced third-stage regression can differ because its degrees-of-freedom or leverage correction differs.
<!-- bilingual-en:end -->

## 自然解释
<!-- bilingual-en:start -->
*Natural interpretation*
<!-- bilingual-en:end -->

$\tilde x$ 是焦点变量中不能被其他已含回归量线性解释的部分，$\tilde y$ 是结果中不能被同一组控制线性解释的部分。多元系数只比较这两部分剩余变化，所以它通常不同于原始 $x$ 与 $y$ 的双变量斜率。
<!-- bilingual-en:start -->
$\tilde x$ is the part of the focal regressor not linearly explained by the other included regressors, and $\tilde y$ is the part of the outcome not linearly explained by those same controls. The multiple-regression coefficient compares only these remaining variations, so it ordinarily differs from the raw bivariate slope between $x$ and $y$.
<!-- bilingual-en:end -->

> [!question]- 应用提示
> 若教育几乎完全由已含的家庭背景变量预测，FWL 视角下教育系数为什么会很不精确？
> <!-- bilingual-en:start -->
> If included family-background variables almost completely predict education, why is the education coefficient imprecise from the FWL perspective?
> <!-- bilingual-en:end -->
>
> **答案：** 残差化后的教育 $\tilde x$ 几乎没有变动，模型缺少区分教育系数的独立样本信息。
> <!-- bilingual-en:start -->
> **Answer:** Residualised education $\tilde x$ has almost no variation, leaving little independent sample information with which to identify its coefficient.
> <!-- bilingual-en:end -->

## 继续

- [[多元回归系数]]：把残差化代数翻译成系数语言。
- [[近似共线性精度]]：查看残差化后焦点变量几乎无变动时的精度后果。
<!-- bilingual-en:start -->
- [[多元回归系数|A multiple-regression coefficient describes a linear association conditional on included controls]] translates the residualisation algebra into coefficient meaning.
- [[近似共线性精度|Precision under near multicollinearity]] develops the consequences when little focal-regressor variation remains after residualisation.
<!-- bilingual-en:end -->

## 来源与核验

- Wooldridge, *Introductory Econometrics*, 6th ed., Chapter 3 §3.2f 及 Appendix 3A.2：核验排除其他变量影响的系数公式与一般 Frisch–Waugh 结果。
- MIT OpenCourseWare, [14.382 Lecture 1: Least Squares and Adaptive Partialling Out](https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/c62d33e015c910b0d126bcc9344cf2c5_MIT14_382S17_lec1.pdf)：核验总体与样本的 partialling-out、偏线性预测含义及其非因果边界。
- [[02_Economy/01_Econometrics/03_多元线性回归.md#2.1.3. 系数计算：以二元回归模型为例|本地课程：FWL 直觉]]：核对对 $Y$ 和焦点 $X$ 同时残差化的课程解释。
<!-- bilingual-en:start -->
- Wooldridge, *Introductory Econometrics*, 6th ed., Chapter 3 §3.2f and Appendix 3A.2, supports the partialling-out coefficient formula and the general Frisch–Waugh result.
- [MIT OpenCourseWare 14.382 Lecture 1](https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/c62d33e015c910b0d126bcc9344cf2c5_MIT14_382S17_lec1.pdf) supports population and sample partialling out, its predictive-association interpretation, and its non-causal boundary.
- [[02_Economy/01_Econometrics/03_多元线性回归.md#2.1.3. 系数计算：以二元回归模型为例|The local course intuition for FWL]] supports residualising both $Y$ and the focal $X$ against the same controls.
<!-- bilingual-en:end -->
