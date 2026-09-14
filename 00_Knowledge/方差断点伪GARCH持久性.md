---
aliases:
  - "未建模方差结构突变会伪装成 GARCH 高持久性"
  - Structural breaks and spurious GARCH persistence
  - Variance breaks in GARCH
  - 结构突变伪高持久性
student_os: knowledge-atom
atom_id: TS-VOL-019
atom_set: conditional-volatility
atom_type: failure-mode
status: source-checked
mastery_state: unassessed
requires:
  - "[[GARCH参数持久性]]"
related:
  - "[[GARCH选择与预测评估]]"
  - "[[GARCH-X因果边界]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# 未建模方差结构突变会伪装成 GARCH 高持久性
<!-- bilingual-en:start -->
*Unmodelled variance breaks can masquerade as high GARCH persistence*
<!-- bilingual-en:end -->

> [!summary] 原子失效机制
> 若样本跨越两个长期方差水平，而模型强迫一个固定 $\omega,\alpha,\beta$ 覆盖全期，GARCH 常会用很大的 $\hat\alpha+\hat\beta$ 缓慢吸收制度切换。于是估得的“持久性”可能部分来自遗漏断点，而非同一机制下冲击真的长期衰减。

这不是说接近一的 $\hat\rho$ 必然虚假，而是必须比较：分样本估计、已知或候选断点、滚动参数、异常期处理、regime-switching 或 time-varying parameter 规格。断点日期若由样本选择，推断还要反映选择过程。

诊断线索包括标准化残差在一段时期系统性过大/过小、滚动方差水平改变、不同子样本的 $\omega$ 明显不一致，以及加入断点后 $\hat\rho$ 显著下降。预测时，过度持久模型会把旧制度的冲击拖入过远期限。

> [!question]- 自检
> 全样本 $\hat\rho=0.99$，按两个制度分样本后分别为 0.75 与 0.80。最谨慎的解释是什么？
>
> **答案：** 全样本高持久性很可能吸收了方差水平变化；需要正式断点/稳定性检查，不能把 0.99 直接当作结构性长期记忆。

## 来源与核验

- [Lamoureux & Lastrapes (1990), *Persistence in Variance, Structural Change, and the GARCH Model*](https://doi.org/10.1080/07350015.1990.10509794)：核对未建模确定性方差位移会高估 GARCH 持久性的实证与模拟证据。
- [[01_Math/06_时间序列分析/lecture.pdf#page=155|课程讲义 Lecture 4]]：核对课程以 Great Moderation 等方差变化作为波动建模背景。
