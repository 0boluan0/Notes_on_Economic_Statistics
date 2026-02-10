---
aliases:
- 流动性覆盖率
- Liquidity Coverage Ratio
- LCR
tags:
- system
- 金融机构与风险管理
---
# LCR（流动性覆盖率）

## 诊断目的

衡量银行在短期（30天）压力情景下，持有的高质量流动资产（HQLA）能否覆盖净现金流出，确保银行在危机初期能够应对流动性冲击。

## 计算方法

$\text{LCR} = \frac{\text{高质量流动资产（HQLA）}}{\text{30天净现金流出}} \geq 100\%$

### 高质量流动资产（HQLA）

| 类别 | 折扣系数 | 典型资产 |
|------|----------|----------|
| 1级资产（HQLA1） | 0% | 现金、央行准备金、主权债 |
| 2A级资产（HQLA2A） | 15% | 评级AA-至A+的政府支持机构债 |
| 2B级资产（HQLA2B） | 50% | 评级BBB+至BBB-的合格公司债 |

### 净现金流出

$\text{净现金流出} = \sum \text{现金流出} \times \text{流失率} - \sum \text{现金流入} \times \text{流入系数}$

## 判断标准

| LCR水平 | 评价 | 管理行动 |
|----------|------|----------|
| > 150% | 优秀 | 可适度释放流动性 |
| 100.1-150% | 健康 | 维持当前水平 |
| 90-100% | 关注 | 准备补充流动性 |
| < 90% | 危险 | 立即补充流动性，限制业务 |

## 常见问题与对策

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| LCR接近100% | HQLA不足或现金流波动大 | 增持HQLA，优化资产负债期限结构 |
| HQLA变现困难 | 2B级资产比例过高、市场深度不足 | 增加流动性最好的资产储备 |
| 现金流失被低估 | 压力情景不够严重 | 使用更保守的流失率假设 |
| 分行LCR不均 | 流动性集中在大行 | 建立内部流动性转移机制 |

## 相关概念
[[NSFR]]
[[Stress Testing|压力测试]]
[[Basel Capital Adequacy Ratio|巴塞尔资本充足率]]

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
