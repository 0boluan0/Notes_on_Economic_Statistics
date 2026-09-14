---
aliases:
  - "调整 R² 用残差自由度缩放 RSS，从而在同一响应和样本内对新增回归参数施加有限样本惩罚"
  - "Adjusted R-squared scales RSS by residual degrees of freedom and thereby penalises added regression parameters within the same outcome and sample"
  - "Adjusted R-squared"
student_os: knowledge-atom
atom_id: ECON-SEL-004
atom_set: regression-model-selection
atom_type: definition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[回归模型比较与选择.canvas|回归模型比较与选择]]"
requires:
  - "[[训练拟合单调性]]"
  - "[[模型比较可比性]]"
related:
  - "[[嵌套模型F检验]]"
---

# 调整 R² 用残差自由度缩放 RSS，从而在同一响应和样本内对新增回归参数施加有限样本惩罚
<!-- bilingual-en:start -->
*Adjusted R-squared scales RSS by residual degrees of freedom and thereby penalises added regression parameters within the same outcome and sample*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 对含截距的 OLS，若 $p$ 是包括截距在内的估计系数个数，调整 $R^2$ 定义为
> $$
> \bar R^2
> =1-\frac{RSS/(n-p)}{TSS/(n-1)}.
> $$
> 分子是残差均方，分母是结果围绕样本均值的方差估计。与普通 $R^2$ 不同，加入变量只有在 RSS 的下降足以补偿损失的残差自由度时，$\bar R^2$ 才会上升。
> <!-- bilingual-en:start -->
> For OLS with an intercept, let $p$ denote the number of estimated regression coefficients including that intercept. Adjusted $R^2$ is $1-[RSS/(n-p)]/[TSS/(n-1)]$. The numerator is the residual mean square and the denominator estimates outcome variation around its sample mean. Unlike ordinary $R^2$, it rises after variables are added only when the fall in RSS compensates for the lost residual degrees of freedom.
> <!-- bilingual-en:end -->

## 怎样读它
<!-- bilingual-en:start -->
*How to read it*
<!-- bilingual-en:end -->

在同一响应、同一批观测和同一权重口径下，较高的 $\bar R^2$ 表示“按这个自由度修正后，样本内残差均方更小”。它可以帮助粗略筛选普通线性回归候选，但不是概率，也不是“被真正解释的比例”；有限样本中它甚至可以为负。

对嵌套模型一次加入 $q$ 个变量时，调整 $R^2$ 上升等价于这组变量对应的经典 partial $F$ 大于 1，而不是达到常见显著性临界值。这说明它的惩罚较轻：某组变量可以提高调整 $R^2$，却远未达到一项预先设定检验的拒绝标准。

<!-- bilingual-en:start -->
With the same outcome, observations, and weighting convention, a larger adjusted $R^2$ means a smaller in-sample residual mean square after this degrees-of-freedom correction. It is neither a probability nor a literal proportion of variation “truly explained,” and it can be negative. For nested models adding $q$ variables at once, adjusted $R^2$ rises exactly when the corresponding classical partial $F$ exceeds 1—not when it reaches a conventional significance critical value—so the implicit penalty is mild.
<!-- bilingual-en:end -->

## 适用边界
<!-- bilingual-en:start -->
*Use boundary*
<!-- bilingual-en:end -->

调整 $R^2$ 没有把模型拿到新数据上测试，也不检查残差曲率、异方差、内生性、数据泄漏或外推。不同响应尺度或不同样本上的数值通常不能直接比较。预测选择应补充结构匹配的验证；因果模型仍须先满足识别与控制变量边界。

<!-- bilingual-en:start -->
Adjusted $R^2$ does not test a model on new data and does not diagnose curvature, heteroskedasticity, endogeneity, leakage, or extrapolation. Values from different outcome scales or samples are generally not directly comparable. Predictive selection needs structurally appropriate validation, while causal models must first satisfy identification and control-variable requirements.
<!-- bilingual-en:end -->

> [!question]- 自检
> 两个模型使用相同工资样本；较大模型的调整 $R^2$ 略高。能否据此断言它对明年求职者预测更准？
>
> **答案：** 不能。调整 $R^2$ 只修正当前样本内的残差均方；外样本表现要由模拟真实部署的验证数据判断。

## 来源与核验

- Penn State STAT 501, [Lesson 5: Multiple Linear Regression](https://online.stat.psu.edu/stat501/Lesson05)：核验公式、普通 $R^2$ 的单调性以及调整 $R^2$ 的样本内模型构建用途。
- Penn State STAT 501, [Lesson 10: Model Building](https://online.stat.psu.edu/stat501/Lesson10)：核验其作为候选子集准则的使用方式和限制。
- [[02_Economy/01_Econometrics/03_多元线性回归.md#2.3. 判定系数的调整|本地课程：判定系数的调整]]：核对课程以 $k$ 表示不含截距解释变量时的等价公式。
