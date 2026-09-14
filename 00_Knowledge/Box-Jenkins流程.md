---
aliases:
  - "Box–Jenkins 在识别估计诊断与重设之间迭代"
  - Box-Jenkins Method
  - Box–Jenkins modelling
  - ARMA Model Identification Steps
  - Box Jenkins 三阶段法
student_os: knowledge-atom
atom_id: TS-ARMA-015
atom_set: arma-modeling
atom_type: workflow
status: source-checked
mastery_state: unassessed
requires:
  - "[[ACF-PACF阶数识别]]"
  - "[[ARMA似然初值处理]]"
related:
  - "[[ARMA信息准则]]"
  - "[[Ljung-Box检验]]"
  - "[[滚动起点评估]]"
part_of:
  - "[[ARMA 模型：识别、估计、诊断与预测.canvas]]"
---

# Box–Jenkins 在识别估计诊断与重设之间迭代
<!-- bilingual-en:start -->
*Box–Jenkins iterates among identification, estimation, diagnosis, and respecification*
<!-- bilingual-en:end -->

> [!summary] 原子流程
> Box–Jenkins 不是“一次看图、一次拟合、一次检验”后结束，而是循环：
> $$\text{明确并平稳化建模对象}\to\text{提出少量候选}\to\text{估计}\to
> \text{诊断}\to\text{必要时重设}.$$
> 只有诊断尚可的候选才进入预测与时间顺序评估。
> <!-- bilingual-en:start -->
> Box–Jenkins is an iterative cycle: define and stationarize the modelling target, propose a small candidate set, estimate, diagnose, and respecify when diagnostics reveal remaining structure. Adequate candidates then proceed to forecasting and chronological evaluation.
> <!-- bilingual-en:end -->

识别阶段结合时序图、领域机制、差分/变换与 ACF/PACF；估计阶段明确似然或平方和方法，并检查因果、可逆、最小性；诊断阶段查看残差路径、ACF、portmanteau 检验、平方/绝对残差、异常点和结构变化。若线性相关仍留在残差中，应回到模型设定，而不是把一次未拒绝当作永久认证。

自动 ARIMA 搜索只替代其中一部分候选比较，不能替代数据检查、诊断与外样本验证。流程目标是得到面向具体用途、足够简单且经证据支持的模型，不是寻找一个由图形“读出来”的真模型。
<!-- bilingual-en:start -->
Identification combines plots, mechanism, transformations, and ACF/PACF. Estimation states the objective and checks causal, invertible, minimal parameter regions. Diagnosis examines residual linear and nonlinear structure, outliers, and breaks. Automated order search covers only part of this loop; it does not replace diagnosis or out-of-sample evaluation.
<!-- bilingual-en:end -->

> [!question]- 自检
> Ljung–Box 拒绝当前模型的残差白噪声原假设，流程下一步是什么？
>
> **答案：** 回到识别/设定，查遗漏的滞后、季节性、差分、异常点或结构变化，再估计并重新诊断；不是只报告 p 值后继续预测。

## 来源与核验

- [[01_Math/06_时间序列分析/lecture.pdf#page=122|课程讲义 pp. 122–123]]：核对三阶段名称、前置平稳化与残差诊断。
- [Hyndman & Athanasopoulos, FPP3 §9.7](https://otexts.com/fpp3/arima-r.html)：核对“诊断失败即修改模型”的迭代流程与自动算法边界。
