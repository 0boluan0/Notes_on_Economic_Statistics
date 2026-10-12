<!-- visual-lab:generated:start -->
# 准确率之外的概率校准

<!-- bilingual-en:start -->
*Reliability, bin counts & probability calibration*
<!-- bilingual-en:end -->

保持真实风险和标签不变，改变报告概率的自信程度，比较可靠性图、分箱样本数、ECE 和 Brier score。

> [!question] 先预测，再操作
> 同一批样本的准确率完全没变，为什么概率预测仍可能明显变差？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/calibration.mp4]]

约 31 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 31 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 说“65%”，长期意味着什么？
- **0:04** 在这一组内，数真实发生了多少次
- **0:10** 每个组变成一个可靠性图上的点
- **0:22** 频率估计也有抽样波动

<!-- bilingual-en:start -->
- **0:00** What should the reported probability mean over repeated cases?
- **0:04** Count actual outcomes within that probability group.
- **0:10** Each bin becomes one reliability-diagram point.
- **0:22** Reliability estimates themselves are noisy.
<!-- bilingual-en:end -->

## 亲手实验

<!-- bilingual-en:start -->
*Try it yourself*
<!-- bilingual-en:end -->

在下方展开“影片参数与延伸对照”，选择“载入本片的模型与数据”，先核对影片中的一个对象或运算，再改变一个条件。其他对照用于继续探索。

<!-- bilingual-en:start -->
Expand the film inputs below and load the model and data used in this film. Check one object or calculation, then change one condition. The other comparisons provide further exploration.
<!-- bilingual-en:end -->

<!-- visual-lab:film:end -->
```visual-lab
id: calibration
height: 980
```

## 观察路线

1. 从 q=p 开始，比较样本频率折线与已知风险箱均值，区分抽样噪声与真正失准。
2. 保持 b=0，只改变 a，检查准确率不变但概率图和 Brier/ECE 改变。
3. 观察箱计数，解释极小箱的频率为何不稳定；改变 bins 检查 ECE 依赖分箱。
4. 增加 n 而保持参数，检验经验校准曲线是否更稳定；不要把单次 ECE 作为无噪声真值。

## 公式与适用条件

$$
q=\sigma(a\operatorname{logit}p+b),\quad ECE=\sum_m\frac{n_m}{n}|\bar y_m-\bar q_m|,\quad E[(Y-q)^2\mid p]=p(1-p)+(p-q)^2
$$

- IID 生成 p∈[0.1,0.9] 后抽 Bernoulli(p) 标签；改变置信参数不更换样本。
- 采用二分类正类概率的等宽分箱 ECE，区别于按预测类别置信度计算的 top-label ECE。
- 这里人工设定置信映射，没有用同一评估数据拟合校准器。

## 回到知识点

- [[00_Knowledge/分类概率校准|分类概率校准]]：可靠性图、分箱计数、有限样本频率误差、Brier 混合含义与准确率/概率质量分离均直接实现。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/metrics|阈值、混淆矩阵与 ROC/PR]] · [[07_Programme/可视化实验室/实验/softmax|Softmax 温度与交叉熵梯度]] · [[07_Programme/可视化实验室/实验/logistic|概率、阈值与交叉熵]] · [[07_Programme/可视化实验室/实验/modelselection|验证集怎样被选择过程过拟合]]

## 来源与核验

- [scikit-learn · Probability calibration](https://scikit-learn.org/stable/modules/calibration.html) — §1.16; §1.16.1 Calibration curves; §1.16.2。支持：核对可靠性图、分箱、概率评分与独立校准拟合。
- [Guo et al. · On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) — §2 Reliability diagrams / Expected Calibration Error; §4.2 Temperature scaling。支持：核对有限分箱 ECE 与置信变换动机；本页明确采用正类频率版本，不冒充论文的 top-label ECE。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
