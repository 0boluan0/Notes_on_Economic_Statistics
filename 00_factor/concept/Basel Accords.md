---
aliases:
- 巴塞尔资本协议
- 巴塞尔协议
- Basel Accords
tags:
- 风险管理
- 银行监管
- concept
---
巴塞尔协议是国际清算银行（BIS）下属的巴塞尔银行监管委员会制定的银行业资本监管框架。

## 巴塞尔协议演进

### Basel I（1988年）

**核心内容**：
- 首次建立统一的国际银行资本充足率标准
- 信用风险加权资本要求
- 资本充足率 ≥ 8%

**资本分级**：
- 一级资本（Tier 1）：核心资本
- 二级资本（Tier 2）：补充资本

**风险分类**：
- 表内项目：0%, 20%, 50%, 100%风险权重
- 表外项目：信用转换系数（CCF）
- 衍生品：当前暴露法（CEM）和净额结算（NRR）

### Basel II（2004年）

**三大支柱**：

#### 第一支柱：最低资本金要求
- 扩展风险覆盖范围：信用风险、市场风险、操作风险
- 改进风险度量方法：标准法、内部评级法（IRB）

#### 第二支柱：监管审查过程
- 银行监管机构对银行资本充足率进行评估
- 要求银行内部资本充足评估程序（ICAAP）

#### 第三支柱：市场纪律
- 提高银行信息披露透明度
- 强化市场约束

### Basel III（2010年，后金融危机改革）

**核心改进**：
1. **提高资本质量和数量**
   - 普通股权益一级资本比率 ≥ 4.5%
   - 一级资本充足率 ≥ 6%
   - 总资本充足率 ≥ 8%

2. **引入杠杆率**
   - 杠杆率 = 一级资本 / 总暴露（含表外）≥ 3%

3. **引入流动性监管**
   - 流动性覆盖率（LCR）≥ 100%
   - 净稳定资金比率（NSFR）≥ 100%

4. **引入资本缓冲**
   - 资本留存缓冲（CCyB）：2.5%
   - 逆周期资本缓冲（0-2.5%）
   - 系统重要性银行附加资本（1-3.5%）

5. **提高风险捕获能力**
   - 交易簿：更严格的VaR要求
   - 引入压力VaR

## 风险加权资产（RWA）

$RWA = \sum_{i} \text{资产}_i \times \text{风险权重}_i$

## 资本充足率

$\text{资本充足率} = \frac{\text{资本}}{\text{RWA}} \times 100\%$

## 监管资本要求

$\text{监管资本} = \text{RWA} \times 8\%$

## 中国实施

- 中国引入巴塞尔协议：2004年（Basel II）
- 中国版巴塞尔协议III：2013年发布
- 核心监管框架：银保会《商业银行资本管理办法》

相关链接: [[Credit Risk|信用风险]], [[Market Risk|市场风险]], [[Operational Risk|操作风险]], [[VaR]], [[Capital Holding Ratio|资本金持有率]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
