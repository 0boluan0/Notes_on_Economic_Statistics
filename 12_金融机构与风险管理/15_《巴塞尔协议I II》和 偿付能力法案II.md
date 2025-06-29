
# 1. 对银行资本进行监管的原因

防破产,防止一连串破产

1. **降低破产概率，稳住信心** —— 完全杜绝银行破产并不现实，但足额资本能把破产概率压到极低，从而维持公众与企业对金融体系的信任 。
    
2. **抑制存款保险诱发的道德风险** —— 有了存款保险，银行容易“拿别人的钱去冒更大的险”；强制资本缓冲把“肆意加杠杆”变得代价高昂 。
    
3. **防范系统性风险** —— 一家巨型银行倒下可能连锁拖垮同业；监管部门关注整个体系的“火灾隔断”效果，而资本要求正是防火墙 。

# 2. basel I

> 目标：学会 **① 资本分层（Tier 1 / Tier 2）**、**② 风险加权资产 RWA 的四档权重**、**③ 计算 Cooke Ratio 并判断合规性**。

| **模块**           | **核心内容**                                      | **监管意图**        |
| ---------------- | --------------------------------------------- | --------------- |
| **资本定义**         | Tier 1（核心资本：股本 + 留存收益）；Tier 2（附属资本：次级债、一般准备等） | 确保最能吸收损失的是真金白银  |
| **风险加权资产 (RWA)** | 把资产按 0 % / 20 % / 50 % / 100 % 四档系数加权         | 让“多赚多压本”，降低监管套利 |
| **Cooke Ratio**  | 资本充足率 = 总资本 ÷ RWA ≥ **8 %**，且 Tier 1 ≥ 4 %    | 全球统一尺子，限制过度杠杆   |

|**资产类别**|**示例**|**权重 (wᵢ)**|
|---|---|---|
|**0 %**|OECD 主权债、本国政府债|0.00|
|**20 %**|OECD 银行同业存放、政府机构债|0.20|
|**50 %**|住宅抵押贷款|0.50|
|**100 %**|企业贷款、股权、非常驻主权债|1.00|

_表外项目_ 先乘 **转换系数 (Credit Conversion Factor, CCF)** 再乘风险权重，例如：

- 授信承诺 ≤ 1 年：CCF = 20 %
- OTC 利率互换：CCF = 0.0 %（早期免资本）

[[库克比率]]
$$\text{库克比率}=\frac{\text{资本（一级资本+二级资本）}}{\text{风险加权资产（RWA）}}$$ 
# 3.G30

|**年份**|**事件**|**缺口暴露**|
|---|---|---|
|**1987**|黑色星期一，股指期货与现金市场价差失控|传统头寸限额无法涵盖衍生品联动风险|
|**1991**|Metallgesellschaft 远期滚动套保巨亏|银行没有日盯市 → 损失被拖延发现|
|**1992**|Procter & Gamble 利率互换“爆仓”|VaR/压力测试缺位，董事会不懂产品|

面对这些教训，G30（汇聚交易商、财务主管、律师及学者）用一份 40 页报告列出 **20 项操作守则 + 4 条监管呼吁**，成为业界第一份“衍生品风险管理 ISO 标准” 。

|**主题**|**关键守则（选摘 & 编号）**|**要解决的痛点**|**PPT 章节**|
|---|---|---|---|
|**A. 治理与文化**|(1)董事会批准风险政策；(2)独立风险部门；(3)制定清晰授权矩阵|风险“谁拍板、谁监督”不清||
|**B. 计量与监控**|(4)每日盯市 _(mark-to-market)_；(5)统一 **VaR** 口径；(6)设置头寸限额 _(limits)_；(7)压力测试|账面价格滞后 & 模型口径各异||
|**C. 信用风险管理**|(8)净额结算应计入敞口；(9)设交易对手限额独立于前台；(10)审慎使用抵押品 / 保证金；(11)关注潜在未来敞口 _(PFE)_|衍生品的双向信用风险被低估||
|**D. 人才与系统**|(12)保证交易、风险、后台人员资质；(13)IT 系统需捕获完整交易数据；(14)及时生成对账与管理报告|“人/机”双短板导致操作风险||
|**E. 财务报告与用途**|(15)衍生品收益来源要与被对冲项目配对披露；(16)禁止“纯投机”掩饰为对冲；(17)按公允价值列报表外项目；(18)披露模型假设；(19)与审计充分沟通；(20)持续评估政策有效性|提高透明度，抑制掩饰性风险||

> **口诀**：**“管人、管模、管限额；看市、看信用、看系统。”**


1. **立法支持净额结算的法律效力**——让 ISDA Master Agreement 真正落地；否则银行无权在破产时抵销正负敞口。
    
2. **推动公开市场公平透明**——提升衍生品交易信息披露，防范信息不对称。
    
3. **监管机构应评价银行 VaR 与压力测试质量**，并将其纳入资本要求。
    
4. **跨国监管合作**——避免监管套利，提高对复杂跨境衍生品的监督力度。


|**G30 守则 →**|**1996 市场风险修正案**|**Basel II / Pillar-3**|
|---|---|---|
|日盯市、VaR(99%,10d)|被写进 **内部模型法** 资本公式 Max(VaRt-1, mc × VaRavg)|银行须披露 VaR、压力测试及限额执行情况，供市场监督|
|回溯测试 & 惩罚系数 mc|例外次数 5-9 对应 mc 3.4-3.85 表格直接源自 G30 思想|—|
|净额结算计量|引入 **NRR**（净替换比率）公式显著降低 RWA|—|
|独立风险部 / 人才资质|被 Basel II 第 2 支柱“监管审查过程”吸收，视作合规条件|—|

**结论**：G30 = Basel I 的“流程补丁”，对之后所有巴塞尔迭代具有“原型设计”意义。

# 4.净额结算

## 4.1 什么叫净额结算？为什么能降信用风险

|**名称**|**定义**|**风险削减机理**|
|---|---|---|
|**支付净额 (payment netting)**|到期日把正负现金流先抵销，只结算“净额”|发生违约前就已减少待收/待付金额|
|**清算/关闭净额 (close-out netting)**|若对手违约，双方所有合约同时终止，按**净赔偿额**结算|把“挑好合约赖账、挑坏合约履约”的 **选摘权**（cherry-picking）拔掉，确保你的正价值和负价值“同生共死”|

在 OTC 衍生品里，**ISDA Master Agreement + CSA** 赋予 close-out netting 的法律效力。若获监管认可，银行可在资本计算里使用“净敞口”而不是“逐笔敞口”，信用 RWA 立刻打折。

## 4.2 **CEM → NRR → EAD → RWA**

1. **现期敞口法 (Current Exposure Method, CEM)**
    
    - 每笔合约敞口 =  max(V,0)  (正价值) +  α × L  (未来潜在敞口 _add-on_)
        
2. **净替换比率 NRR**
    
    $\text{NRR}=\frac{\sum_{i=1}^{N}\max(V_i,0)}{\sum_{i=1}^{N}|V_i|}$
    
    ——用**净额正敞口** ÷ **绝对敞口** 量化净额效率（范围 0–1）。
    
3. **等价信用量 (EAD)** — 有净额时
    
    $\text{EAD}= \underbrace{\sum_{i}\max(V_i,0)}{\text{现期}} \;+\;\bigl(0.4 + 0.6 \times \text{NRR}\bigr)\,\times \sum{i}L_i$
    
    没有净额时就是$\sum \max(V_i,0) + \sum \alpha_i L_i$。
    
4. **风险加权资产 (RWA)**
    
    $\text{RWA} = \text{EAD}\, \times \text{对手权重}$

## 4.3 PPT例题

> **互换组合**：
> +24 m、–17 m、+8 m  （货币：USD）
> **Add-on 合计**：110 m（监管表给定）
> **对手评级**：OECD 银行，权重 20 %

> **任务**：某组合 5 笔互换，正价值 +40/+15，负价值 –25/–18/–12（单位 mUSD）；总 add-on 200 m，交易对手权重 50 %。

1. > 计算无净额 EAD、RWA。
    
2. > 计算 NRR、EADnet、RWAnet。
    
3. > 比较资本节省率 (%)。
    

  

可把答案发给我或自行对照脚本检查 —— 练会 NRR 公式才算真正掌握！

---

## **6️⃣ 延伸阅读 & 工具**

1. **ISDA® 2021 Definitions** —— 最新净额条款范本。
    
2. **BIS “Capital Treatment for Bilateral Netting” (1995)** —— NRR 公式官方文件。
    
3. **Python Notebook** —— 用 pandas + numpy 写一个 calc_EAD(netting_set) 函数，循环遍历交易对手，做资本敏感度分析。
    

---

### **⏭️ 下一节预告 ——** 

### **15.6 1996 市场风险修正案**

- 引入 **VaR(10d, 99 %) × 惩罚系数 mc** 的资本公式
    
- 解释市场风险资本与信用资本如何拼成“总 RWA”
    

  

如果对净额、NRR 或案例数字还有疑惑，请直接提问；否则回复“继续”，我们马上上车！

## 3. 1996年《市场风险修正案》的新增资本要求（VaR方法）

1996年巴塞尔委员会发布了市场风险修正案，为交易账簿引入了市场风险资本要求。修正案规定两种计算方法：**标准化方法**（对外汇、商品等头寸按固定比例计提资本）和**内部模型方法**（以VaR为基础）。内部模型方法要求银行计算99%置信度下的10天持有期VaR，并以此确定资本要求。通常使用平方根法将日VaR转换为10天VaR：
$$\text{VaR}_{10\text{天}} \approx \sqrt{10}\times \text{VaR}_{1\text{天}},$$
然后乘以监管乘数（如3倍）以计算最终资本。掌握VaR的计算与持有期调整是重点。

**例题:** 某银行交易组合计算出1天99%VaR为10亿元（假设收益正态且独立）。试计算该组合的10天99%VaR（使用平方根法），并在监管乘数为3的情况下计算所需市场风险资本。

**答案:** 1. 将1天VaR转换为10天：$\text{VaR}_{10}=10\times\sqrt{10}\approx10\times3.162=31.62$（亿元）。2. 乘以监管乘数：资本要求$=3\times31.62\approx94.9$（亿元）。因此，该行需计提约94.9亿元的市场风险资本 。

## 4. 《巴塞尔协议II》的三大支柱结构及与Basel I的比较
巴塞尔II（2004年）采用了“三支柱”框架：**第一支柱**（最低资本要求）、**第二支柱**（监管审查）和**第三支柱**（市场约束/信息披露） [oai_citation:22‡investopedia.com](https://www.investopedia.com/terms/b/basel_accord.asp#:~:text=The%20second%20Basel%20Accord%2C%20called,known%20as%20the%20three%20pillars) [oai_citation:23‡investopedia.com](https://www.investopedia.com/terms/b/baselii.asp#:~:text=Basel%20II%20is%20the%20second,weighted%20assets)。第一支柱仍要求资本充足率不低于8% [oai_citation:24‡investopedia.com](https://www.investopedia.com/terms/b/baselii.asp#:~:text=Building%20on%20Basel%20I%2C%20Basel,weighted%20assets)，但信用风险权重更精细，引入了内部评级（IRB）方法，评级越高的资产风险权重越低 [oai_citation:25‡investopedia.com](https://www.investopedia.com/terms/b/baselii.asp#:~:text=weighting%20is%20intended%20to%20discourage,the%20lower%20the%20risk%20weight)。此外，巴塞尔II首次引入了操作风险的资本要求。第二、三支柱分别强化了监管机构的审查要求和信息披露，以提高监管效力和市场透明度。与Basel I相比，Basel II提高了风险敏感度，并明确了监管和市场约束机制 [oai_citation:26‡investopedia.com](https://www.investopedia.com/terms/b/baselii.asp#:~:text=weighting%20is%20intended%20to%20discourage,the%20lower%20the%20risk%20weight) [oai_citation:27‡investopedia.com](https://www.investopedia.com/terms/b/basel_accord.asp#:~:text=The%20second%20Basel%20Accord%2C%20called,known%20as%20the%20three%20pillars)。

**例题:** 某银行在巴塞尔II框架下风险加权资产为800亿元，一级资本50亿元，二级资本20亿元。计算该银行的资本充足率，并判断是否满足第一支柱8%的要求；同时检查一级资本比例是否满足“至少50%”的要求。

**答案:** 1. 总资本$=50+20=70$（亿元），资本充足率$=70/800=8.75\%\ge8\%$ [oai_citation:28‡investopedia.com](https://www.investopedia.com/terms/b/baselii.asp#:~:text=Building%20on%20Basel%20I%2C%20Basel,weighted%20assets)；2. 一级资本占比$=50/70=71.4\%\ge50\%$ [oai_citation:29‡analystprep.com](https://analystprep.com/study-notes/frm/part-2/operational-and-integrated-risk-management/basel-1-basel-2-and-solvency-2/#:~:text=At%20least%2050,a%20requirement%20by%20the%20accord)。因此，该银行既满足最低资本率要求，也符合一级资本比例要求。

## 5. 偿付能力法案II框架简介，三大支柱与Basel II的对比，保险业资本监管指标（MCR与SCR）

偿付能力II（Solvency II）是欧盟于2016年实施的保险业资本监管框架，也采用“三支柱”结构：**第一支柱**为量化资本要求（资产负债的市场价值评估及资本计量），**第二支柱**为治理与监管（包括风险管理和内部偿付能力评估ORSA），**第三支柱**为市场披露与透明度.第一支柱下的资本要求包括两种阈值：**偿付能力资本要求**（SCR）和**最低资本要求**（MCR）。SCR可通过标准公式或内部模型计算，涵盖承保风险、市场风险等；MCR则为更低的最低门槛，根据公式设定并限制在SCR的25%至45%范围内 [oai_citation:32‡lloyds.com](https://www.lloyds.com/conducting-business/regulatory-information/solvency-ii/about/what-is-solvency-ii#:~:text=%2A%20Two%20thresholds%3A%20,the%20valuation%20of%20assets%20and)。常考知识点包括计算MCR的上下限（25%–45% SCR） [oai_citation:33‡lloyds.com](https://www.lloyds.com/conducting-business/regulatory-information/solvency-ii/about/what-is-solvency-ii#:~:text=%2A%20Two%20thresholds%3A%20,the%20valuation%20of%20assets%20and)。Solvency II与Basel II类似重视风险敏感度，但针对保险业风险（如承保风险、精算假设）制定了不同的资本指标，强调ORSA等内部风险管理要求 [oai_citation:34‡lloyds.com](https://www.lloyds.com/conducting-business/regulatory-information/solvency-ii/about/what-is-solvency-ii#:~:text=%2A%20Two%20thresholds%3A%20,the%20valuation%20of%20assets%20and) [oai_citation:35‡investopedia.com](https://www.investopedia.com/terms/b/baselii.asp#:~:text=Basel%20II%20is%20the%20second,weighted%20assets)。

**例题:** 某保险公司计算出偿付能力资本要求（SCR）为1000万元。根据Solvency II规定，求该公司的最低资本要求（MCR）范围。

**答案:** 根据Solvency II，MCR为SCR的25%~45%。1. MCR最低值$=1000\times25\%=250$万元；2. MCR最高值$=1000\times45\%=450$万元 [oai_citation:36‡lloyds.com](https://www.lloyds.com/conducting-business/regulatory-information/solvency-ii/about/what-is-solvency-ii#:~:text=%2A%20Two%20thresholds%3A%20,the%20valuation%20of%20assets%20and)。因此，该公司MCR范围为250万至450万元。

# 作业

## 15.1



## 15.6

## 15.10

## 15.12

## 15.17

## 15.21

## 15.22

