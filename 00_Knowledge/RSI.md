---
student_os: knowledge-atom
atom_id: FI-TA-023
atom_type: definition
aliases:
  - RSI是上涨幅度平滑值占上涨与下跌幅度平滑值之和的百分比指标
  - Relative strength index
  - 相对强弱指标
status: source-checked
---

# RSI是上涨幅度平滑值占上涨与下跌幅度平滑值之和的百分比指标
<!-- bilingual-en:start -->
*RSI measures smoothed upward price changes as a percentage of total smoothed upward and downward changes.*
<!-- bilingual-en:end -->

相对强弱指标（relative strength index，RSI）将同一价格序列的上涨幅度与下跌幅度分别平滑，再计算上涨部分在两者合计中的比例，以 $0$–$100$ 点表示。这里的“相对强弱”比较该序列自身的涨跌幅度，不是两只股票之间的收益率排名。
<!-- bilingual-en:start -->
The relative strength index (RSI) separately smooths upward and downward changes in one price series, then expresses the upward component as a share of their sum on a $0$–$100$ scale. Relative strength here concerns that series' own up and down changes, not return rankings between securities.
<!-- bilingual-en:end -->

本卡采用 Wilder 平滑。设 $C_t$ 为收盘价，$N\ge2$ 为整数期间；先把相邻收盘价之差拆为非负上涨量 $u_t$ 与非负下跌量 $d_t$，没有该方向变化的期间记零。两种均值都除以同一个 $N$，不是分别除以上涨日数和下跌日数。
<!-- bilingual-en:start -->
This card uses Wilder smoothing. For closing prices $C_t$ and integer period $N\ge2$, split each close-to-close change into nonnegative gain $u_t$ and loss magnitude $d_t$, using zero when that direction does not occur. Both initial averages divide by $N$, not by separate counts of up and down days.
<!-- bilingual-en:end -->

$$
\Delta C_t=C_t-C_{t-1},\qquad
u_t=\max(\Delta C_t,0),\qquad d_t=\max(-\Delta C_t,0).
$$

取得 $N+1$ 个收盘价 $C_0,\ldots,C_N$ 后，以首 $N$ 个价格变化初始化；后续平滑系数为 $1/N$，与通常价格 EMA 的 $2/(N+1)$ 不同。
<!-- bilingual-en:start -->
After observing $N+1$ closes, $C_0,\ldots,C_N$, initialize from their $N$ changes. Subsequent smoothing uses coefficient $1/N$, which differs from the usual price EMA coefficient $2/(N+1)$.
<!-- bilingual-en:end -->

$$
U_N=\frac1N\sum_{i=1}^{N}u_i,\qquad D_N=\frac1N\sum_{i=1}^{N}d_i,
$$
$$
U_t=\frac{(N-1)U_{t-1}+u_t}{N},\qquad
D_t=\frac{(N-1)D_{t-1}+d_t}{N},\quad t>N,
$$
$$
RSI_t=100\frac{U_t}{U_t+D_t},\qquad U_t+D_t>0.
$$

若 $D_t=0<U_t$，RSI 为 $100$；若 $U_t=0<D_t$，RSI 为 $0$。若两者都为零，则比例是 $0/0$：本卡记为未定义（NA），不产生阈值信号。TA-Lib 在全平情况下返回 $0$，这是其实现约定。初始历史不足或价格缺失同样不能填零伪造 RSI。
<!-- bilingual-en:start -->
When $D_t=0<U_t$, RSI is $100$; when $U_t=0<D_t$, it is $0$. If both are zero, the ratio is $0/0$: this card records it as undefined (NA) and generates no threshold signal. TA-Lib returns $0$ in that case as an implementation convention. Insufficient initial history or missing prices must not be filled with zero to manufacture RSI values.
<!-- bilingual-en:end -->

例如 $N=3$，收盘价为 $10,12,11,13,12$。前三个变化是 $+2,-1,+2$，所以 $U_3=4/3$、$D_3=1/3$、$RSI_3=80$。下一期跌 $1$，得到 $U_4=8/9$、$D_4=5/9$、$RSI_4=800/13\approx61.5385$；若重新对最近三个变化取简单均值，会得到 $50$，已经是不同平滑口径。
<!-- bilingual-en:start -->
For $N=3$ and closes of $10,12,11,13,12$, the initial changes are $+2,-1,+2$, giving $U_3=4/3$, $D_3=1/3$, and RSI $80$. After the next loss of $1$, $U_4=8/9$, $D_4=5/9$, and RSI is approximately $61.5385$. Replacing the recursion with simple averages of the latest three changes instead gives $50$, a different smoothing convention.
<!-- bilingual-en:end -->

原课只给出平均上涨占平均涨跌合计的比例，没有指定平滑与初值；本卡将两者补全。$70/30$ 或 $80/20$ 是使用者选定的状态阈值，不改变 RSI 的定义。
<!-- bilingual-en:start -->
The course supplies the ratio of average gains to total average gains and losses without specifying smoothing or initialization; this card completes both. Thresholds such as $70/30$ or $80/20$ classify indicator states and do not change its definition.
<!-- bilingual-en:end -->

## 来源与核验

- [[02_Economy/06_证券投资学/证券投资学.pdf#page=834|《证券投资学》PDF 第834页]]：核对非负涨跌幅度合计中的上涨占比；原课未规定平滑和初值。
- [Fidelity：Relative Strength Index](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/RSI)：核对 Wilder 归属、$0$–$100$ 尺度和一般比例式。
- [TA-Lib：RSI 实现](https://github.com/TA-Lib/ta-lib/blob/972c5cc934fe78ba08fcdf2e805bc1c4318fb6cf/src/ta_func/ta_RSI.c)：核对首 $N$ 个变化的均值、Wilder 的 $(N-1,1)/N$ 递推、等价比值及全平返回零的实现差异。边界与数例独立复算。
