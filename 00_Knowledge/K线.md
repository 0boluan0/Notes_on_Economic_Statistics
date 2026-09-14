---
aliases:
  - K线用实体和影线编码同一期间的开盘最高最低与收盘价格
  - Candlestick
student_os: knowledge-atom
atom_id: FI-TA-041
atom_type: definition
status: source-checked
---

# K线用实体和影线编码同一期间的开盘最高最低与收盘价格

<!-- bilingual-en:start -->
A candlestick encodes a period's open, high, low and close through its real body and shadows.
<!-- bilingual-en:end -->

K线是把同一交易期间的开盘价 $O$、最高价 $H$、最低价 $L$ 和收盘价 $C$ 画成实体与影线的价格图示。实体连接 $O$ 与 $C$，上影线从实体上端延伸至 $H$，下影线从实体下端延伸至 $L$。先说明期间是一天、一周还是其他长度，且四个价格必须使用相同的数据口径。

<!-- bilingual-en:start -->
A candlestick represents the open $O$, high $H$, low $L$ and close $C$ of one trading period. The real body joins $O$ and $C$; the upper shadow extends from the body's top to $H$, and the lower shadow from its bottom to $L$. Specify the period and use a consistent data basis for all four prices.
<!-- bilingual-en:end -->

有效的四价满足 $L\leq\min(O,C)\leq\max(O,C)\leq H$。实体、上影线、下影线的长度分别为：

<!-- bilingual-en:start -->
Valid prices satisfy $L\leq\min(O,C)\leq\max(O,C)\leq H$. The lengths of the real body, upper shadow and lower shadow are respectively:
<!-- bilingual-en:end -->

$$
|C-O|,\qquad H-\max(O,C),\qquad \min(O,C)-L.
$$

$C>O$ 是阳线，$C<O$ 是阴线；$C=O$ 时实体收成一条横线。红绿、黑白、空心或实心是软件或教材的绘图约定。阳线只说明本期收盘高于本期开盘，未必高于上期收盘；一根K线也不记录最高价与最低价发生的先后顺序。

<!-- bilingual-en:start -->
$C>O$ gives an up candle and $C<O$ a down candle; when $C=O$, the body collapses to a horizontal line. Colors and hollow or filled bodies are display conventions. An up candle need not close above the previous period's close, and a candle does not show whether its high or low occurred first.
<!-- bilingual-en:end -->

例如 $O=100,H=105,L=98,C=103$，实体长3、上影线长2、下影线长2。若上期收盘是104，这根阳线仍对应收盘对收盘下跌1。K线首先是价格摘要；某种外形是否能预测后续收益是另一个需要检验的问题。

<!-- bilingual-en:start -->
With $O=100,H=105,L=98,C=103$, the body is 3 units long and both shadows are 2. If the previous close was 104, this up candle still represents a close-to-close fall of 1. A candle is a price summary; predictive performance requires a separate test.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
Sources and verification
<!-- bilingual-en:end -->

- [[02_Economy/06_证券投资学/证券投资学.pdf#page=705|《证券投资学》PDF实际705页]]：四价、实体、影线与阴阳线原图；本页两种实体均有填色，支持将填充方式与阴阳定义区分。长度式和数例由四价定义直接复算。

<!-- bilingual-en:start -->
- [[02_Economy/06_证券投资学/证券投资学.pdf#page=705|Securities Investment, PDF page 705]] supplies the OHLC diagram. Both bodies are filled, distinguishing fill style from candle direction. The lengths and example are checked directly from the four prices.
<!-- bilingual-en:end -->
