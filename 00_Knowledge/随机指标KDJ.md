---
student_os: knowledge-atom
atom_id: FI-TA-025
atom_type: definition
aliases:
  - KDJ是对收盘区间位置连续平滑并放大快慢差的随机指标
  - KDJ stochastic oscillator
  - KDJ
status: source-checked
---

# KDJ是对收盘区间位置连续平滑并放大快慢差的随机指标
<!-- bilingual-en:start -->
*KDJ successively smooths the close's position within a price range and amplifies the difference between its fast and slow lines.*
<!-- bilingual-en:end -->

随机指标 KDJ 先计算收盘价在近期最高、最低价之间的位置 RSV，再依次平滑得到快线 K 和慢线 D，最后以 J 放大 K 相对 D 的偏离。这里的“随机指标”是指标名称，计算过程本身是确定的。
<!-- bilingual-en:start -->
KDJ first calculates RSV, the close's position between a recent high and low, then smooths it into a fast K line and a slow D line. J amplifies K's deviation from D. “Stochastic oscillator” is the indicator's name; the calculation itself is deterministic.
<!-- bilingual-en:end -->

本卡固定为九期窗口，两次平滑的当期权重均为 $1/3$，J 采用 $3K-2D$。以当期为终点，$H^{(9)}_t$、$L^{(9)}_t$ 分别为九期最高价与最低价，$C_t$ 为当期收盘价，各价格使用相同调整口径：
<!-- bilingual-en:start -->
This card uses a nine-period window, a current-observation weight of $1/3$ in each smoothing stage, and $J=3K-2D$. Let $H^{(9)}_t$ and $L^{(9)}_t$ be the highest high and lowest low of the nine periods ending at $t$, and $C_t$ the current close, all on a consistent adjustment basis:
<!-- bilingual-en:end -->

$$
RSV_t=100\frac{C_t-L^{(9)}_t}{H^{(9)}_t-L^{(9)}_t},\qquad H^{(9)}_t>L^{(9)}_t,
$$
$$
K_t=\frac23K_{t-1}+\frac13RSV_t,\qquad
D_t=\frac23D_{t-1}+\frac13K_t,
$$
$$
J_t=3K_t-2D_t=K_t+2(K_t-D_t).
$$

首个完整且区间宽度为正的九期窗口出现前不输出；在首次更新之前设置 K、D 的内部状态均为 $50$，再用首个有效 RSV 更新 K，随后用当期 K 更新 D。平滑权重是 $1/3$，不是三期算术平均，也不是通常三期 EMA 的 $1/2$。
<!-- bilingual-en:start -->
Produce no output until a complete nine-period window has a positive range. Immediately before the first update, set the internal K and D states to $50$. Update K from the first valid RSV, then D from the current K. The $1/3$ smoothing weight is neither a three-period arithmetic average nor the usual three-period EMA weight of $1/2$.
<!-- bilingual-en:end -->

若某期最高等于最低，RSV 为 $0/0$。本卡约定该期 K、D、J 输出 NA，保留最近有效的 K、D 内部状态；下一有效窗口用这两个保存值替代递推式中的前值，再恢复更新。这是明确的数据处理约定，不宣称所有软件如此。行情缺失也不能用填零制造 RSV。
<!-- bilingual-en:start -->
If a period's range is zero, RSV is $0/0$. This card outputs NA for K, D, and J while retaining the last valid internal K and D states. At the next valid window, those saved states supply the preceding values in the recursion. This is an explicit data-handling convention, not a claim about every platform. Missing market data must not be replaced with zeros to manufacture RSV.
<!-- bilingual-en:end -->

若上一有效 K、D 均为 $50$，窗口最高 $110$、最低 $90$、收盘 $106$，则 $RSV=80$、$K=60$、$D=53.3333$、$J=73.3333$。把 $0.8$ 当成 RSV 输入会得到 $K=33.6$，属于尺度错误。
<!-- bilingual-en:start -->
With previous valid K and D states of $50$, a high of $110$, low of $90$, and close of $106$ give $RSV=80$, $K=60$, $D=53.3333$, and $J=73.3333$. Entering $0.8$ as RSV instead produces $K=33.6$, a scaling error.
<!-- bilingual-en:end -->

在有效输入下，RSV、K、D 保持在 $0$–$100$，因为 K、D 是区间内数值的凸组合。J 含有外推项，可以高于 $100$ 或低于 $0$，不应擅自截断。RSV 与同窗口 [[威廉指标]] 的负值口径满足 $RSV=100+\%R$，不意味着所有名为 KD 或 stochastic 的软件采用本卡的平滑与初值。
<!-- bilingual-en:start -->
For valid inputs, RSV, K, and D remain within $0$–$100$, since K and D are convex combinations of values in that range. J extrapolates and can exceed $100$ or fall below $0$; do not clip it silently. RSV equals $100+\%R$ for the same-window [[威廉指标|Williams %R]], but other indicators named KD or stochastic need not share this card's smoothing and seeds.
<!-- bilingual-en:end -->

原 PDF 第848页确实写 $J=3D-2K$，并非仅有笔记识别错误。对上述例子，该式得到 $40$，与本卡的 $73.3333$ 不同；本卡采用下列正式文档的 $3K-2D$ 变体，并保留课程的 $50$ 初值约定。
<!-- bilingual-en:start -->
The original PDF on page 848 explicitly gives $J=3D-2K$; this is not merely a note-extraction error. It yields $40$ for the example, rather than this card's $73.3333$. This card uses the $3K-2D$ variant documented below while retaining the course's initialization at $50$.
<!-- bilingual-en:end -->

## 来源与核验

- [[02_Economy/06_证券投资学/证券投资学.pdf#page=842|《证券投资学》PDF 第842页]]、[[02_Economy/06_证券投资学/证券投资学.pdf#page=848|第848页]]：核对九期区间位置输入、两次 $1/3$ 平滑、$50$ 初值及原 J 公式。
- [富途帮助中心：KDJ 随机指标](https://support.futunn.com/topic149?lang=zh-cn)：第2节核对 RSV、K、D 和 $J=3K-2D$。本卡未将其应用段的预测性表述用作收益证据；区间退化处理由本卡明确指定，数例独立复算。
