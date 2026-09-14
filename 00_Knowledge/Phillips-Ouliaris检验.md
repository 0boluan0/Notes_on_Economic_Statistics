---
aliases:
  - "Phillips-Ouliaris 用长期协方差修正残差型协整统计量"
  - Phillips-Ouliaris residual cointegration tests
  - PO 协整检验
student_os: knowledge-atom
atom_id: TS-CI-015
atom_set: cointegration-error-correction
atom_type: test-method
status: source-checked
mastery_state: unassessed
requires:
  - "[[EG残差检验]]"
  - "[[PP检验]]"
related:
  - "[[单位根证据组合]]"
part_of:
  - "[[协整与误差修正模型.canvas]]"
---

# Phillips-Ouliaris 用长期协方差修正残差型协整统计量
<!-- bilingual-en:start -->
*Phillips-Ouliaris tests use long-run covariance corrections for residual-based cointegration statistics*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> Phillips–Ouliaris 检验同样从水平协整回归的估计残差出发，以“无协整”为原假设；它构造 $Z_\alpha$、$Z_t$ 等统计量，用非参数长期协方差估计修正残差的序列相关与相关创新影响。

它与 EG-ADF 的分工类似于 PP 与 ADF：EG-ADF 在残差回归里加入滞后差分来参数化短期相关，Phillips–Ouliaris 保持较简洁的残差回归并修正统计量。二者都不是普通单变量单位根检验，因为协整向量由同一数据估计；确定性项和协整回归中的 $I(1)$ 变量个数会改变参考分布。

“Phillips–Ouliaris 检验”不是单一个对归一化都不变的统计量。由某一归一化回归残差构造的 $Z_\alpha$ 和 $Z_t$ 会随被解释变量的选择改变；Phillips–Ouliaris 另外提出了对归一化不变的多变量 $P_z$ 类统计量。因此必须报告具体统计量与归一化，不能把整个方法族概括成“自动对称”。

长期协方差估计的核与带宽会影响有限样本结果，因此 PO 不是“无需选择”的万能稳健版本。它可作为与 EG-ADF 不同的残差型证据；若结论冲突，应回到规格、样本、结构突变和功效，而不是多数表决。

> [!question]- 自检
> Phillips–Ouliaris 与 EG-ADF 的核心差异是什么？
>
> **答案：** 两者都检验估计协整残差；EG-ADF 用滞后差分处理短期相关，PO 以长期协方差作非参数统计量修正。

## 来源与核验

- [Phillips & Ouliaris (1990)](https://cowles.yale.edu/sites/default/files/2022-08/d0847-r.pdf)：核对 $Z_\alpha$、$Z_t$ 残差型检验、长期协方差修正与归一化不变统计量。
- [MacKinnon (2010)](https://qed.econ.queensu.ca/working_papers/papers/qed_wp_1227.pdf)：核对常用残差型协整临界值的规格依赖。
