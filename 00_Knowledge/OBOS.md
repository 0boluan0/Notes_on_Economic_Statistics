---
student_os: knowledge-atom
atom_id: FI-TA-027
atom_type: definition
aliases:
  - OBOS是固定窗口内上涨家数与下跌家数的累计差
  - 超买超卖指标OBOS
  - Overbought oversold breadth indicator
status: source-checked
---

# OBOS是固定窗口内上涨家数与下跌家数的累计差
<!-- bilingual-en:start -->
*OBOS is the accumulated difference between advancing and declining issue counts over a fixed window.*
<!-- bilingual-en:end -->

超买超卖指标 OBOS 是市场宽度指标：在固定长度的窗口内，累加每日上涨家数，再减去每日下跌家数的合计。它描述参与上涨与下跌的股票家数差，不直接测量指数收益率或资金量。
<!-- bilingual-en:start -->
OBOS is a market-breadth indicator that sums daily advancing issue counts over a fixed window and subtracts the sum of declining counts. It describes the balance of participation, rather than directly measuring index returns or cash flows.
<!-- bilingual-en:end -->

先指定股票池、交易期及可比前收盘价口径。$A_t$ 为该池当期上涨家数，$D_t$ 为下跌家数，持平不计入两者；拆股等价格调整与数据缺失的处理须一致，缺报价不能悄悄当成平盘。对包含当前期的完整 $N$ 期窗口：
<!-- bilingual-en:start -->
Specify the universe, trading periods, and basis for comparable previous closes. Let $A_t$ and $D_t$ count advances and declines, excluding unchanged issues from both. Apply consistent treatment of price adjustments and missing data; an absent quote must not silently become an unchanged close. For a complete $N$-period window including the present:
<!-- bilingual-en:end -->

$$
OBOS_{N,t}=\sum_{j=0}^{N-1}A_{t-j}-\sum_{j=0}^{N-1}D_{t-j}.
$$

例如连续三期上涨家数为 $6,2,7$，下跌家数为 $3,6,2$，则 $OBOS_3=15-11=4$。同一股票可在不同日期重复贡献；这里不是对窗口内上涨股票去重计数，也不是上涨股票占比。
<!-- bilingual-en:start -->
If advancing counts are $6,2,7$ and declining counts are $3,6,2$, the three-period OBOS is $15-11=4$. The same stock can contribute on multiple dates. This is neither a count of distinct advancing stocks over the window nor a percentage of advancing stocks.
<!-- bilingual-en:end -->

OBOS 没有固定的 $-100$ 至 $100$ 尺度，绝对值随股票池大小与窗口长度变化。首个完整窗口之前不输出；跨市场、跨窗口或股票池覆盖变动时，不能把相同数值当作相同强度。
<!-- bilingual-en:start -->
OBOS has no fixed $-100$ to $100$ scale; its magnitude depends on universe size and window length. It is unavailable before the first complete window. Equal values across markets, windows, or changing coverage do not automatically indicate equal intensity.
<!-- bilingual-en:end -->

若 [[腾落线]] 使用完全相同的每日涨跌家数和连续时间口径，则有限窗口的累计差满足 $OBOS_{N,t}=ADL_t-ADL_{t-N}$。两者在这一条件下是同一数据流的不同累计方式。
<!-- bilingual-en:start -->
If the [[腾落线|advance-decline line]] uses exactly the same daily counts on a continuous time basis, $OBOS_{N,t}=ADL_t-ADL_{t-N}$. Under that condition, the two indicators accumulate the same data stream over different horizons.
<!-- bilingual-en:end -->

## 来源与核验

- [[02_Economy/06_证券投资学/证券投资学.pdf#page=857|《证券投资学》PDF 第857页]]：核对 $N$ 期每日涨跌家数之和的差。
- [RQData：股票技术指标因子](https://www.ricequant.com/doc/rqdata/python/stock-mod)：超买超卖指标中的 OBOS 行核对窗口公式及相对前收的上涨判定；本卡不把该页接口示例当作股票池覆盖证明。数例、尺度依赖及与腾落线的恒等式独立复算。
