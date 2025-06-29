
# 1. 对银行资本进行监管的原因

防破产,防止一连串破产

1. **降低破产概率，稳住信心** —— 完全杜绝银行破产并不现实，但足额资本能把破产概率压到极低，从而维持公众与企业对金融体系的信任 。
    
2. **抑制存款保险诱发的道德风险** —— 有了存款保险，银行容易“拿别人的钱去冒更大的险”；强制资本缓冲把“肆意加杠杆”变得代价高昂 。
    
3. **防范系统性风险** —— 一家巨型银行倒下可能连锁拖垮同业；监管部门关注整个体系的“火灾隔断”效果，而资本要求正是防火墙 。

# 2. basel I







## 2. 库克比率的计算公式与应用示例
库克比率（Cooke Ratio）是巴塞尔I提出的资本充足度指标，定义为银行资本与风险加权资产的比率 [oai_citation:12‡analystprep.com](https://analystprep.com/study-notes/frm/part-2/operational-and-integrated-risk-management/basel-1-basel-2-and-solvency-2/#:~:text=The%20first%20attempt%20of%20an,innovation%20in%20the%201988%20accord)。其计算公式为 
$$\text{库克比率}=\frac{\text{资本（一级资本+二级资本）}}{\text{风险加权资产（RWA）}}$$ 
，通常要求不低于8%（即$CAR\ge8\%$） [oai_citation:13‡analystprep.com](https://analystprep.com/study-notes/frm/part-2/operational-and-integrated-risk-management/basel-1-basel-2-and-solvency-2/#:~:text=Capital%20Requirement)。例如，如果一家银行的RWA为100亿元，实际资本为9亿元，则库克比率为$9/100=9\%$，满足8%的要求。常考知识点包括正确识别资本构成（一级资本和二级资本）和风险加权资产的计算方法 [oai_citation:14‡analystprep.com](https://analystprep.com/study-notes/frm/part-2/operational-and-integrated-risk-management/basel-1-basel-2-and-solvency-2/#:~:text=The%20first%20attempt%20of%20an,innovation%20in%20the%201988%20accord) [oai_citation:15‡analystprep.com](https://analystprep.com/study-notes/frm/part-2/operational-and-integrated-risk-management/basel-1-basel-2-and-solvency-2/#:~:text=Capital%20Requirement)。

**例题:** 某银行风险加权资产为1000亿元，一级资本500亿元，二级资本100亿元。计算该银行的库克比率，并判断是否满足8%的要求。

**答案:** 1. 总资本$=500+100=600$（亿元）；2. 库克比率$=600/1000=0.60=60\%$；3. 60%远高于8%，故满足最低资本要求 [oai_citation:16‡analystprep.com](https://analystprep.com/study-notes/frm/part-2/operational-and-integrated-risk-management/basel-1-basel-2-and-solvency-2/#:~:text=Capital%20Requirement) [oai_citation:17‡analystprep.com](https://analystprep.com/study-notes/frm/part-2/operational-and-integrated-risk-management/basel-1-basel-2-and-solvency-2/#:~:text=The%20first%20attempt%20of%20an,innovation%20in%20the%201988%20accord)。

## 3. 1996年《市场风险修正案》的新增资本要求（VaR方法）
1996年巴塞尔委员会发布了市场风险修正案，为交易账簿引入了市场风险资本要求 [oai_citation:18‡bis.org](https://www.bis.org/publ/bcbs23.htm#:~:text=The%20document%20summarises%20the%20Committee%27s,on%20the%20backtesting%20of%20models) [oai_citation:19‡analystprep.com](https://analystprep.com/study-notes/frm/part-2/operational-and-integrated-risk-management/basel-ii-5-basel-iii-and-other-post-crisis-changes/#:~:text=Capital%20for%20banks%20was%20to,in%201996%2C%20the%20computation%20of)。修正案规定两种计算方法：**标准化方法**（对外汇、商品等头寸按固定比例计提资本）和**内部模型方法**（以VaR为基础）。内部模型方法要求银行计算99%置信度下的10天持有期VaR，并以此确定资本要求 [oai_citation:20‡analystprep.com](https://analystprep.com/study-notes/frm/part-2/operational-and-integrated-risk-management/basel-ii-5-basel-iii-and-other-post-crisis-changes/#:~:text=Capital%20for%20banks%20was%20to,in%201996%2C%20the%20computation%20of)。通常使用平方根法将日VaR转换为10天VaR：
$$\text{VaR}_{10\text{天}} \approx \sqrt{10}\times \text{VaR}_{1\text{天}},$$
然后乘以监管乘数（如3倍）以计算最终资本。掌握VaR的计算与持有期调整是重点。

**例题:** 某银行交易组合计算出1天99%VaR为10亿元（假设收益正态且独立）。试计算该组合的10天99%VaR（使用平方根法），并在监管乘数为3的情况下计算所需市场风险资本。

**答案:** 1. 将1天VaR转换为10天：$\text{VaR}_{10}=10\times\sqrt{10}\approx10\times3.162=31.62$（亿元）。2. 乘以监管乘数：资本要求$=3\times31.62\approx94.9$（亿元）。因此，该行需计提约94.9亿元的市场风险资本 [oai_citation:21‡analystprep.com](https://analystprep.com/study-notes/frm/part-2/operational-and-integrated-risk-management/basel-ii-5-basel-iii-and-other-post-crisis-changes/#:~:text=Capital%20for%20banks%20was%20to,in%201996%2C%20the%20computation%20of)。

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

