---
aliases:
  - "RESET 拒绝只表明所检非线性探针发现设定不足而不会定位唯一遗漏结构"
  - "Ramsey RESET"
  - "RESET specification test"
student_os: knowledge-atom
atom_id: ECON-SPEC-007
atom_type: diagnostic
status: source-checked
mastery_state: unassessed
part_of:
  - "[[回归模型设定与函数形式.canvas|回归模型设定与函数形式]]"
requires:
  - "[[残差图边界]]"
contrasts_with:
  - "[[LM设定检验]]"
---

# RESET 拒绝只表明所检非线性探针发现设定不足而不会定位唯一遗漏结构
<!-- bilingual-en:start -->
*RESET rejection shows inadequacy along the tested nonlinear probes but does not identify a unique omitted structure*
<!-- bilingual-en:end -->

> [!summary] 原子诊断
> 常见 Ramsey RESET 在原模型后加入 $\hat y^2,\hat y^3,\ldots$，再联合检验这些项的系数为零。拟合值幂汇总了已含回归变量的某些非线性组合，因此能探测一族设定不足，但不是待解释的实质变量。
> <!-- bilingual-en:start -->
> A common Ramsey RESET augments the original model with $\hat y^2,\hat y^3,\ldots$ and jointly tests whether their coefficients are zero. These powers collect certain nonlinear combinations of the included regressors, so they probe a family of misspecifications but are not substantive explanatory variables.
> <!-- bilingual-en:end -->

## 拒绝与不拒绝分别说明什么
<!-- bilingual-en:start -->
*What rejection and non-rejection mean*
<!-- bilingual-en:end -->

- **拒绝：** 当前规格沿所选 probes 存在可检测不足；来源可能是遗漏平方项、交互、错误变换或其他与 probes 相关的遗漏结构。
- **不拒绝：** 当前样本对这些 probes 没有足够反对证据；低功效、其他方向的设定错误或识别失败仍可能存在。

<!-- bilingual-en:start -->
- **Rejection:** the current specification is detectably inadequate along the chosen probes. The source may be an omitted square, interaction, wrong transformation, or another omitted structure related to those probes.
- **Non-rejection:** the sample provides insufficient evidence against these probes. Low power, misspecification in other directions, or identification failure can remain.
<!-- bilingual-en:end -->

拒绝后把 $\hat y^2$ 直接留在最终模型通常没有清楚的实质解释。正确下一步是回到研究机制、原始变量和残差模式，提出少量可解释候选项，再用与目标相符的诊断或验证比较。
<!-- bilingual-en:start -->
Keeping $\hat y^2$ as a final substantive regressor after rejection usually lacks a clear interpretation. The next step is to return to mechanism, original variables, and residual patterns, propose a small set of interpretable candidates, and compare them with diagnostics or validation matched to the goal.
<!-- bilingual-en:end -->

常规 RESET 的同方差 $F$ 形式依赖经典误差方差条件；若要把有限样本统计量精确按 $F$ 分布解释，还需要正态误差等附加条件。若存在异方差或聚类，应对新增 probes 做与数据结构匹配的稳健联合 Wald/score 检验；更换协方差估计只修正检验的参考不确定性，并不会让 RESET 获得定位唯一遗漏结构的能力。
<!-- bilingual-en:start -->
The conventional homoskedastic $F$ form of RESET relies on classical error-variance conditions; an exact finite-sample $F$ reference additionally requires normal errors and the other usual conditions. With heteroskedasticity or clustering, the added probes require a robust joint Wald or score test matched to the data structure. Changing the covariance estimator repairs the reference uncertainty; it does not give RESET the ability to identify a unique omitted structure.
<!-- bilingual-en:end -->

> [!question]- 回忆提示
> RESET 拒绝后，为什么不能说“正确模型就是加入 $\hat y^2$ 的模型”？
>
> **答案：** 因为拟合值幂只是汇总型探针；同一拒绝可由多种遗漏结构产生，它不提供唯一的实质修复。

## 继续

- [[LM设定检验]]：已有明确候选遗漏项时使用更可解释的指定方向。
- [[设定修复循环]]：把检验结果转成可追踪的模型修订，而不是自动选模。

## 来源与核验

- Ramsey, J. B. (1969), [Tests for Specification Errors in Classical Linear Least-Squares Regression Analysis](https://doi.org/10.1111/j.2517-6161.1969.tb00796.x)：核验 RESET 的原始 specification-testing 问题与通用替代方向。
- [[02_Economy/01_Econometrics/04_模型设定.md#3.3. 拉姆齐 RESET 检验]]：核对课程中的拟合值幂辅助回归和联合原假设。
- Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., Chapter 9：核验 RESET 的非定位性以及异方差下采用稳健联合检验的边界。
