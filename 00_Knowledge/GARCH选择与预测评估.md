---
aliases:
  - "GARCH 模型比较需要可比似然而预测选择需要时序外样本损失"
  - GARCH model comparison
  - Volatility forecast evaluation
  - 波动模型选择
student_os: knowledge-atom
atom_id: TS-VOL-015
atom_set: conditional-volatility
atom_type: model-selection-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[GARCH残差双层诊断]]"
  - "[[ARMA信息准则]]"
related:
  - "[[模型比较可比性]]"
  - "[[训练验证测试划分]]"
  - "[[验证结构匹配]]"
  - "[[滚动起点评估]]"
  - "[[方差断点伪GARCH持久性]]"
  - "[[实现波动率]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# GARCH 模型比较需要可比似然而预测选择需要时序外样本损失
<!-- bilingual-en:start -->
*GARCH likelihood comparison requires comparable fits, while forecast selection requires ordered out-of-sample losses*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 按 [[模型比较可比性]]，AIC/BIC 只能比较同一观测响应与尺度、同一有效 likelihood 样本，并使用完整且可比的概率密度与计分约定的候选。常数项、变换 Jacobian、条件/初值处理必须相容；每个候选都要分别最大化自己的 likelihood，并把均值、方差和分布参数完整计入复杂度。均值动态与创新分布可以是候选之间的合法差异，“likelihood 可比”不等于“模型规格相同”。未明确归一化的准似然目标不能直接冒充可比的完整 likelihood。AIC/BIC 评价样本内拟合与复杂度折衷，不保证未来方差预测最好。
> <!-- bilingual-en:start -->
> Under the [[模型比较可比性|model-comparability contract]], AIC and BIC may compare candidates with different mean dynamics or innovation distributions when they model the same observed response on the same scale and effective likelihood sample, use mutually comparable full probability densities, maximise each candidate's own likelihood, and count every mean, variance, and distribution parameter. Constants, transformation Jacobians, and conditional or presample conventions must be compatible. An unnormalised quasi-likelihood score cannot be presented as a comparable full likelihood. Information criteria remain in-sample fit-complexity measures, not guarantees of forecast performance.
> <!-- bilingual-en:end -->

预测任务应按时间顺序做 rolling/expanding-origin 评估，用未来实现方差或与业务目标一致的代理计算损失。平方收益是噪声很大的潜在方差代理；MSE、MAE、QLIKE、VaR exceptions 或区间覆盖服务不同目标，不能只挑最有利指标。

按 [[训练验证测试划分]]，模型选择样本与最终 holdout 要分开。若反复查看同一“外样本”来调阶数、分布和断点，它已经成为验证集，不再是独立最终证据。风险管理还需同时检查尾部覆盖与经济损失，不能让较低 AIC 取代诊断。

> [!question]- 自检
> Gaussian GARCH 和 Student-$t$ GARCH 在不同有效样本上各有一个 AIC，可以直接取较小者吗？
>
> **答案：** 不可以。分布不同本身不妨碍比较，但这里有效样本不同；还必须确认两者都是同一响应尺度上的已最大化完整 likelihood，常数、初始化和参数计数相容。若目标是预测，还需在保序外样本上用预先规定的损失比较。

## 来源与核验

- [Engle & Ng (1993)](https://doi.org/10.1111/j.1540-6261.1993.tb05127.x)：核对多种不对称波动模型的拟合与诊断比较。
- [Patton (2011), *Volatility Forecast Comparison Using Imperfect Volatility Proxies*](https://doi.org/10.1016/j.jeconom.2010.03.034)：核对潜在方差代理含噪时，损失函数选择会影响预测排序的边界，并支持使用对代理噪声稳健的损失。
- [Hyndman & Athanasopoulos, FPP3 §5.10](https://otexts.com/fpp3/tscv.html)：核对时间序列滚动起点评估。
- [R `stats::AIC` 官方文档](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/AIC.html)：核对同一数据、最大化 log likelihood、相容常数与完整参数计数的比较条件；其示例也直接比较不同回归均值规格。
- [`arch` 单变量波动模型官方文档](https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html)：核对同一收益样本上 GARCH、GJR、TARCH/ZARCH 以及 Normal 与 standardized Student-$t$ 候选分别最大化 likelihood 并报告 IC；固定参数示例不作为已最大化 ML 证据。
- [[滚动起点评估]]：复用已核验的保序验证边界。
