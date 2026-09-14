---
aliases:
  - 人均GDP是一个经济体同期国内生产总值除以总人口得到的平均产出指标
  - GDP per capita
student_os: knowledge-atom
atom_id: DEV-MEAS-003
atom_type: definition
status: source-checked
---

# 人均GDP是一个经济体同期国内生产总值除以总人口得到的平均产出指标

人均 GDP 把[[国内生产总值]]按人口规模标准化，便于区分“经济体很大”和“平均到每个人的生产规模很大”。分母是同一经济体的人口，不是就业人数；若按就业人数除，问题就变成每名劳动者的产出。
<!-- bilingual-en:start -->
GDP per capita divides [[国内生产总值|gross domestic product]] by population, separating a large economy from a high average level of production per person. The denominator is the economy's population, not employment; dividing by employment instead measures output per worker.
<!-- bilingual-en:end -->

$$
y_t=\frac{Q_t}{N_t}.
$$

$Q_t$ 是期间 GDP，$N_t$ 是与统计期匹配的人口估计。即使每个人都被放进分母，$y_t$ 也不表示每人实际领取了这么多收入，更不表示收入平均分配；测量贫困要回到[[贫困测量的福利口径|家庭福利分布]]。
<!-- bilingual-en:start -->
$Q_t$ is GDP during the period and $N_t$ is the corresponding population estimate. Including everyone in the denominator does not mean everyone received this amount or that income is equally distributed. Poverty measurement requires a [[贫困测量的福利口径|household welfare distribution]].
<!-- bilingual-en:end -->

考察时间变化时，先把 $Q_t$ 取成可比价格的实际 GDP。由比值可以直接得到：
<!-- bilingual-en:start -->
For change over time, use real GDP at comparable prices for $Q_t$. The ratio gives:
<!-- bilingual-en:end -->

$$
1+g_y=\frac{1+g_Q}{1+g_N}.
$$

若[[实际GDP增长]]为 6%，人口增长为 2%，实际人均 GDP 增长是 $1.06/1.02-1\approx3.92\%$，不是 6%。增长率相减得到的 4% 是小变动近似。
<!-- bilingual-en:start -->
If [[实际GDP增长|real GDP growth]] is 6% and population growth is 2%, real GDP per capita rises by $1.06/1.02-1\approx3.92\%$, not 6%. Subtracting growth rates gives the small-change approximation of 4%.
<!-- bilingual-en:end -->

比较不同国家的水平，还要选价格和货币口径：当前美元、固定基年美元、PPP 国际元回答的问题不同，选择方法见[[汇率与购买力比较]]。
<!-- bilingual-en:start -->
Cross-country level comparisons also require a price and currency basis. Current US dollars, constant-base-year US dollars, and PPP international dollars answer different questions; see [[汇率与购买力比较|choosing exchange-rate or purchasing-power comparisons]].
<!-- bilingual-en:end -->

## 来源与核验

- [World Bank, WDI，NY.GDP.PCAP.KD 元数据](https://databank.worldbank.org/metadataglossary/world-development-indicators/series/NY.GDP.PCAP.KD)：总人口分母、实际价格含义与该系列的 2015 年基年；人均增长恒等式由定义推导，数例已复算。

<!-- bilingual-en:start -->
WDI metadata supports the population denominator and the interpretation of constant prices, including the 2015 reference year for this specific series. The growth identity follows algebraically from the definition; its numerical application is checked.
<!-- bilingual-en:end -->
