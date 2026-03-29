---
aliases:
- 基础货币
- High-Powered Money
- Monetary Base
tags:
- concept
- money and banking
---
# 基础货币

>[!note] 定义
>
> 基础货币（Monetary Base）又称高能货币（High-Powered Money），是中央银行的负债。
>
> $ MB = C + R $
>
> 其中：
> - C：流通中的现金（Currency in Circulation）
> - R：准备金（Reserves）
>
## 组成部分

### 1. 流通中的现金（C）

公众手中持有的货币。

**注意**：
1. 存款机构持有的货币属于美联储负债的一部分，但属于准备金。
2. 铸币局印出来的货币只要没有流通就不属于负债。

### 2. 准备金（R）

包括在美联储的存款和银行实际持有的现金。

**分类**：
- **法定准备金（RR）**：美联储要求的数额
- **超额准备金（ER）**：美联储没要求的数额，银行自己想持有的

## 三位参与者

### 1. 中央银行
监管银行体系的政府机构，负责实施货币政策。

### 2. 银行（存款机构）
从个人和机构中吸收存款并发放贷款的金融中介机构。

### 3. 储户
持有银行存款的个人和机构。

## 联储资产负债表

### 负债（货币负债）

通常被称为货币负债（Monetary Liabilities），由流通中的现金 C 与准备金 R 之和构成。

美联储的货币负债 + 美国财政部的货币负债（主要是铸币）之和被称为基础货币或高能货币。

### 资产

#### 证券（Government Securities）
美联储持有的美国财政部发行的国债。

#### 向金融机构发放的贷款（Loans to Financial Institutions）
向银行以及其他金融机构发放贷款，为银行体系提供准备金。

只要是央行流到商业银行的钱都是准备金（借入准备金），利息为贴现率。

## 基础货币的控制

### 1. 可控部分（非借入准备金）

$ MB_n = MB - BR $

这部分美联储能够完全控制（公开市场操作）。

### 2. 总控制能力

美联储控制公开市场操作，但是不能单方面决定和准确预测银行。

基础货币分为两部分：
- 一部分美联储能够完全控制（公开市场操作）
- 对一部分管控能力较弱（银行的借款），只能通过利率间接控制。

## 影响因素

1. **公开市场操作**：主要控制手段
2. **银行借款**：难以预测和控制
3. **财政存款**：影响银行体系准备金
4. **支票浮存**：暂时性影响
5. **现金偏好**：影响 C/R 比例

## 货币乘数关系

$ M = m \times MB $

货币供应量 M 等于货币乘数 m 乘以基础货币 MB。

## 相关概念

- [[Money Multiplier|货币乘数]]
- [[Required Reserves|准备金]]
- [[Open Market Operations|公开市场操作]]

## 重要性

基础货币是货币供给的源头：
- 央行通过控制基础货币来控制货币供给
- 货币乘数放大基础货币的影响
- 是货币政策传导机制的核心环节

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
