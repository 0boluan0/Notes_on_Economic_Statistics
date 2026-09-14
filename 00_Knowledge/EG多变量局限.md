---
aliases:
  - "Engle-Granger 在多变量中通常只能识别一个协整方向且依赖归一化"
  - Engle-Granger multivariate limitation
  - EG 单一协整向量局限
student_os: knowledge-atom
atom_id: TS-CI-014
atom_set: cointegration-error-correction
atom_type: method-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Engle-Granger两步法]]"
  - "[[协整向量归一化]]"
related:
  - "[[Johansen检验]]"
  - "[[协整秩与共同趋势]]"
part_of:
  - "[[协整与误差修正模型.canvas]]"
---

# Engle-Granger 在多变量中通常只能识别一个协整方向且依赖归一化
<!-- bilingual-en:start -->
*In multivariate systems, Engle-Granger usually identifies only one cointegrating direction and depends on normalization*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 单方程 Engle–Granger 先指定一个左边变量，因此自然估计一条归一化后的长期关系；当真实协整秩大于 1 时，它不能恢复整个协整空间，换归一化还可能改变有限样本结果。

若三个变量有两条独立协整关系，一次水平回归只给出其中某个线性组合。把另一个变量放到左边，不一定只是把同一向量机械变形：动态内生性、有限样本误差和不同残差过程都可能让检验结论不同。第一阶段的误差还会被带入第二阶段 ECM，虽然经典条件下两步估计有良好渐近性质，有限样本不确定性仍不能忽略。

因此 EG 最适合理论明确指向单一长期关系、变量数较少的场景。需要联合确定协整秩、同时估计多条关系或检验空间限制时，约化秩 VECM/Johansen 框架通常更合适；这不是说 Johansen 在小样本中自动更可靠。

> [!question]- 自检
> 四变量系统可能有两条协整关系，一次 EG 回归能否证明“协整秩恰好为 2”？
>
> **答案：** 不能。它通常只检验并估计一条归一化关系，无法完整决定多维协整空间的秩。

## 来源与核验

- [Engle & Granger (1987)](https://doi.org/10.2307/1913236)：核对单方程两步程序。
- [Johansen (1988)](https://doi.org/10.1016/0165-1889(88)90041-3)：核对多协整向量的系统估计动机。
