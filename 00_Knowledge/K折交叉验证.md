---
aliases:
  - "K 折交叉验证轮流用一折评价、其余折重走训练流程，并平均各折的未见数据损失"
  - "K-fold cross-validation rotates one fold for evaluation, refits the procedure on the others, and averages unseen-data loss across folds"
student_os: knowledge-atom
atom_id: ECON-SEL-008
atom_set: regression-model-selection
atom_type: definition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[回归模型比较与选择.canvas|回归模型比较与选择]]"
requires:
  - "[[训练验证测试划分]]"
  - "[[模型比较可比性]]"
related:
  - "[[验证管线隔离]]"
  - "[[验证结构匹配]]"
leads_to:
  - "[[测试集自适应复用]]"
---

# K 折交叉验证轮流用一折评价、其余折重走训练流程，并平均各折的未见数据损失
<!-- bilingual-en:start -->
*K-fold cross-validation rotates one fold for evaluation, refits the procedure on the others, and averages unseen-data loss across folds*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 把可用于开发的数据分成互不重叠的 $K$ 折 $I_1,\ldots,I_K$。第 $j$ 轮只用 $I_{-j}$ 拟合整条流程，在未参与该轮拟合的 $I_j$ 上计算预先确定的损失；最后按观测数或既定权重汇总：
> $$
> \widehat R_{CV}=\frac{1}{n}\sum_{j=1}^K\sum_{i\in I_j}L\{y_i,\hat f^{(-j)}(x_i)\}.
> $$
> 每个观测都得到一次来自“不含自身训练集”的预测，从而比 resubstitution error 更接近未见数据问题。
> <!-- bilingual-en:start -->
> Partition the development data into disjoint folds $I_1,\ldots,I_K$. In round $j$, fit the entire procedure only on $I_{-j}$ and evaluate a prespecified loss on the unseen fold $I_j$. Aggregate with the intended observation weights. Every observation then receives one prediction from a fit that excluded it, making the estimate closer to an unseen-data problem than resubstitution error.
> <!-- bilingual-en:end -->

## 交叉验证评估的是流程
<!-- bilingual-en:start -->
*Cross-validation evaluates a procedure*
<!-- bilingual-en:end -->

用 K 折比较固定候选时，每个候选的插补、标准化、变量筛选、PCA 和模型系数都必须在每轮训练折内重新拟合，见[[验证管线隔离]]；各折 validation loss 可以汇总后选择正则强度、成分数或模型。若 K 折本身要评价“连同调参在内的选择流程”，则每个外层训练折还需在内部完成选择。先用全体数据选择特征再做 K 折，只轮换最后一个回归器，并没有验证真实部署流程。

同一组 folds 和同一损失应供所有候选使用，使差值主要来自模型而非切分噪声。选择 K、重复次数和损失函数也会影响估计的偏差、方差与计算成本；不能在看到许多 CV 结果后把最有利的一次当作未选择过的证据。

<!-- bilingual-en:start -->
When K-fold compares fixed candidates, each candidate's imputation, scaling, feature selection, PCA, and model coefficients must be refitted inside every training fold; see [[验证管线隔离|validation-pipeline isolation]]. Validation losses may then select a regularisation strength, component count, or model. If the folds are meant to evaluate the whole tuning procedure, selection must instead occur inside each outer-training split. All candidates should use the same folds and loss so that score differences chiefly reflect the procedures rather than split noise; searching many CV configurations can itself overfit the selection criterion.
<!-- bilingual-en:end -->

## 适用边界
<!-- bilingual-en:start -->
*Use boundary*
<!-- bilingual-en:end -->

普通随机 K 折把观测近似视为按该方式可交换；时间、群组或空间依赖需要[[验证结构匹配|结构匹配的切分]]。若 CV 被用来挑选模型或超参数，它给的是选择证据，不再是所选模型完全独立的最终性能估计；严格外部评价可保留 test set 或使用 outer folds 的 nested CV。

<!-- bilingual-en:start -->
Ordinary random K-fold treats observations as approximately exchangeable under that split. Temporal, grouped, or spatial dependence requires a [[验证结构匹配|structure-matched design]]. When CV selects a model or hyperparameters, it supplies selection evidence rather than a fully independent final estimate of the winner; an untouched test set or outer folds in nested CV can provide stricter evaluation.
<!-- bilingual-en:end -->

> [!question]- 自检
> 五折验证前先在全部数据上选择与 $y$ 最相关的 30 个变量，之后每折只重拟合回归系数。问题在哪里？
>
> **答案：** 每个验证折的结果已经影响了 30 个变量的选择。必须在每个训练折内重新选择变量，再把固定后的选择应用到对应验证折。

## 来源与核验

- scikit-learn, [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html)：核验 K 折训练/验证轮换、平均分数、最终 test 与 group/time-aware split 的边界。
- Cawley and Talbot（2010），[On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf)：核验 CV 选择准则也有有限样本方差、可被过拟合，以及 nested evaluation 的必要性。
