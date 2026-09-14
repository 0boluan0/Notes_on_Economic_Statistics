---
student_os: knowledge-atom
atom_id: FI-TA-026
atom_type: definition
aliases:
  - OBV是按收盘涨跌方向累计成交量的指标
  - On-balance volume
  - 能量潮指标
status: source-checked
---

# OBV是按收盘涨跌方向累计成交量的指标
<!-- bilingual-en:start -->
*On-balance volume accumulates trading volume signed by the direction of close-to-close price changes.*
<!-- bilingual-en:end -->

能量潮指标（on-balance volume，OBV）将每一期的全部成交量，按该期收盘价相对前收盘价的涨跌方向加入或减去累计值；收盘持平时累计值保持不变。它把价的方向与量的大小结合为一条累计序列。
<!-- bilingual-en:start -->
On-balance volume (OBV) adds or subtracts the entire period's trading volume according to whether its close rises or falls relative to the preceding close. An unchanged close leaves the accumulated value unchanged. The series combines price direction with volume magnitude.
<!-- bilingual-en:end -->

设 $C_t$ 为同一证券的可比收盘价，$V_t\ge0$ 为对应期成交量。本卡在第一个观测期设置 $OBV_1=0$，从第二期开始递推；首期没有前收盘价时，不人为指定涨跌。
<!-- bilingual-en:start -->
Let $C_t$ be comparable closing prices for one security and $V_t\ge0$ its corresponding period volumes. This card sets $OBV_1=0$ at the first observation and updates from period two onward; it does not assign a price direction when no preceding close exists.
<!-- bilingual-en:end -->

$$
OBV_t=OBV_{t-1}+s_tV_t,\qquad
s_t=\begin{cases}
1,&C_t>C_{t-1},\\
0,&C_t=C_{t-1},\\
-1,&C_t<C_{t-1}.
\end{cases}
$$

例如收盘价为 $10,11,11,9$，成交量为 $100,200,300,150$，零种子下 OBV 依次为 $0,200,200,50$。第三期虽然成交 $300$，但收盘持平，因此既不累加，也不把此前累计值归零。
<!-- bilingual-en:start -->
For closes of $10,11,11,9$ and volumes of $100,200,300,150$, zero-seeded OBV is $0,200,200,50$. The third period trades $300$ units but closes unchanged, so it neither adds that volume nor resets the existing total to zero.
<!-- bilingual-en:end -->

改变起始常数只会平移同一数据段的 OBV。TA-Lib 使用首期成交量作为种子，上例会显示 $100,300,300,150$。比较两条曲线时须对齐起点和量的单位；股、手、合约等口径不能混用，缺失量也不等于真实零成交。
<!-- bilingual-en:start -->
Changing the starting constant shifts OBV without changing increments over the same data segment. TA-Lib seeds with the first volume, producing $100,300,300,150$ in this example. Align the starting point and volume units when comparing curves; shares, lots, and contracts cannot be mixed, and missing volume is not observed zero volume.
<!-- bilingual-en:end -->

价格涨跌的比较还须处理拆股等造成的不可比前收，量的历史单位亦须声明。OBV 给整期量赋一个符号，并不观测每笔交易的主动买卖方向，所以它不是实际净资金流入或主动买入量减主动卖出量。
<!-- bilingual-en:start -->
Close comparisons must account for events such as stock splits, and historical volume units must be specified. OBV assigns one sign to the entire period's volume; it does not observe trade-by-trade initiation. It therefore does not measure actual net cash inflows or buyer-initiated minus seller-initiated volume.
<!-- bilingual-en:end -->

## 来源与核验

- [[02_Economy/06_证券投资学/证券投资学.pdf#page=859|《证券投资学》PDF 第859页]]、[[02_Economy/06_证券投资学/证券投资学.pdf#page=860|第860页]]：教材明确该项为自学，给出涨、跌、平盘的签名累计规则；原课笔记仅有标题。
- [Fidelity：On Balance Volume](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/obv)：核对三种收盘方向下的加、减和保持。
- [TA-Lib：OBV 实现](https://github.com/TA-Lib/ta-lib/blob/972c5cc934fe78ba08fcdf2e805bc1c4318fb6cf/src/ta_func/ta_OBV.c)：核对其首量种子与递推，本卡采用零种子。数例、常数平移及无法识别逐笔资金方向的边界按定义独立核验。
