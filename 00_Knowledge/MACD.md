---
student_os: knowledge-atom
atom_id: FI-TA-022
atom_type: definition
aliases:
  - MACD是由快慢指数移动平均之差及其信号线构成的价格指标
  - Moving average convergence divergence
  - 平滑异同移动平均线
status: source-checked
---

# MACD是由快慢指数移动平均之差及其信号线构成的价格指标
<!-- bilingual-en:start -->
*MACD is a price indicator formed from the difference between fast and slow exponential moving averages and its signal line.*
<!-- bilingual-en:end -->

平滑异同移动平均线（MACD）比较快、慢两条 [[指数移动平均]]，并用差值的平滑线作为信号线。它包含差值线、信号线及二者差距的柱图；计算前必须明确期间、平滑、初值与柱图倍数。
<!-- bilingual-en:start -->
Moving average convergence divergence (MACD) compares fast and slow [[指数移动平均|exponential moving averages]] and smooths their difference into a signal line. Its outputs are the difference line, signal line, and a histogram of their gap. Reproduction requires periods, smoothing rules, seeds, and histogram scaling.
<!-- bilingual-en:end -->

本卡以收盘价 $C_t$ 采用 $(12,26,9)$ 期间：快慢价格 EMA 的系数分别为 $2/13$、$2/27$，信号线系数为 $2/10$。沿课程记号，差值线称 DIF，信号线称 DEA，柱图称 $H$：
<!-- bilingual-en:start -->
This card applies periods $(12,26,9)$ to closing prices $C_t$. The fast and slow price EMA coefficients are $2/13$ and $2/27$; the signal coefficient is $2/10$. Following course notation, DIF is the difference line, DEA the signal line, and $H$ the histogram:
<!-- bilingual-en:end -->

$$
DIF_t=E_{12,t}-E_{26,t},\qquad
DEA_t=\frac{2}{10}DIF_t+\frac{8}{10}DEA_{t-1},\qquad
H_t=2(DIF_t-DEA_t).
$$

初始化固定如下：$E_{12,12}$ 取 $C_1,\ldots,C_{12}$ 的均值，之后连续递推；$E_{26,26}$ 取 $C_1,\ldots,C_{26}$ 的均值，之后连续递推。DIF 从 $t=26$ 开始有效，DEA 在 $t=34$ 取首九个有效 DIF 的均值，此后按上式递推；DEA 和柱图在此之前不输出。
<!-- bilingual-en:start -->
Initialize $E_{12,12}$ with the mean of $C_1,\ldots,C_{12}$ and continue updating it. Initialize $E_{26,26}$ with the mean of $C_1,\ldots,C_{26}$. DIF first exists at $t=26$. Seed DEA at $t=34$ with the mean of the first nine valid DIF values, then apply the recursion above. DEA and the histogram are unavailable before that point.
<!-- bilingual-en:end -->

许多英语平台把 DIF 本身称为 MACD line，并将柱图画成 $DIF-DEA$。本卡的二倍柱图与它们只差正比例缩放，零点和穿零方向相同，并未增加一个独立信号。DIF、DEA 和柱图均继承价格单位，没有 RSI 那样固定的 $0$–$100$ 区间。
<!-- bilingual-en:start -->
Many English-language platforms call DIF the MACD line and plot $DIF-DEA$ as the histogram. This card's doubled histogram is a positive rescaling: zero crossings and their directions are unchanged, so it adds no independent signal. All three outputs retain price units and have no fixed $0$–$100$ range like RSI.
<!-- bilingual-en:end -->

例如上一期 $E_{12}=102$、$E_{26}=100$、$DEA=1.5$，今天收盘为 $104$，则今天 $E_{12}=102.307692$、$E_{26}=100.296296$，所以 $DIF=2.011396$、$DEA=1.602279$、$H=0.818234$。这是给定有效历史状态的一步更新，不是用单个价格初始化整套指标。
<!-- bilingual-en:start -->
Given valid previous states $E_{12}=102$, $E_{26}=100$, and $DEA=1.5$, a close of $104$ produces $E_{12}=102.307692$, $E_{26}=100.296296$, $DIF=2.011396$, $DEA=1.602279$, and $H=0.818234$. This is a one-step update from existing states, not initialization from a single price.
<!-- bilingual-en:end -->

原课 PDF 第829页把 DEA 写为未给期数的 DIF 算术平均，并明确柱图乘二；本卡补全为九期 EMA 信号线，保留二倍柱图。TA-Lib 的 MACD 默认初始化又将快慢均线种子对齐于慢线起始日，快线取该日向前十二期的均值，因此早期值不必与本卡各自初始化的价格 EMA 完全一致；比较软件输出还须对齐这些约定。
<!-- bilingual-en:start -->
Course PDF page 829 describes DEA as an arithmetic mean of DIF without a period and explicitly doubles the histogram. This card specifies a nine-period EMA signal while retaining that scale. TA-Lib's default MACD instead aligns both price EMA seeds at the slow EMA's starting date, seeding the fast EMA with the twelve closes ending there. Its early values can therefore differ from this card's separately initialized EMAs; software comparisons must align these conventions.
<!-- bilingual-en:end -->

## 来源与核验

- [[02_Economy/06_证券投资学/证券投资学.pdf#page=829|《证券投资学》PDF 第829页]]：原始 DIF、价格 EMA 权重、DEA 表述及二倍柱图；上文明确本卡补足和不同口径。
- [Fidelity：MACD](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/macd)：核对 12/26 价格 EMA 之差、九期 EMA 信号线及可变期间。
- [TA-Lib：MACD 实现](https://github.com/TA-Lib/ta-lib/blob/972c5cc934fe78ba08fcdf2e805bc1c4318fb6cf/src/ta_func/ta_MACD.c)：核对信号系数、首九个差值的均值种子、未乘二的柱图及其对齐初始化；本卡采用的独立价格 EMA 初始化已另行写明。一步数例独立复算。
