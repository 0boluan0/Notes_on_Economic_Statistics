---
aliases:
  - "Cluster Analysis"
  - "Hierarchical Clustering and K-means"
  - "聚类分析"
status: source-checked
---

# 聚类：层次聚类与 K-means
<!-- bilingual-en:start -->
*Clustering: hierarchical clustering and K-means*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 在没有已知标签时，依据明确的表示、距离与算法探索哪些观察对象形成相对相似的组。
> **具体锚点：** 只有客户消费特征而没有类型标签时，可探索是否存在稳定群体；结果是分析定义下的分组，不是自动发现自然种类。
> **核心难点：** 尺度、距离、链接、初始值和簇数都会改变结果；可视化漂亮不等于稳定或有用。
> **为什么重要：** 市场细分、生态群落、基因表达和异常探索都使用聚类，但误读也极常见。
> **继续：** 先写对象与变量含义并标准化，再比较层次法与 K-means；已有标签时应转到 [[判别分析：Bayes、LDA 与 QDA]]。
<!-- bilingual-en:start -->
> [!summary] Quick recovery
> **What it solves:** Without known labels, it explores which observations form relatively similar groups under an explicitly chosen representation, distance, and algorithm.
> **Concrete anchor:** With customer spending features but no type labels, one may explore whether stable groups exist. The result is a partition under the analysis definition, not automatic discovery of natural kinds.
> **Central difficulty:** Scaling, distance, linkage, initialisation, and the number of clusters can all change the result. An attractive visualisation does not establish stability or usefulness.
> **Why it matters:** Market segmentation, ecological communities, gene expression, and anomaly exploration all use clustering, and all are vulnerable to overinterpretation.
> **Continue with:** Define observations and variables and choose scaling before comparing hierarchical methods with K-means. If labels already exist, go to [[判别分析：Bayes、LDA 与 QDA|discriminant analysis]].
<!-- bilingual-en:end -->

> [!source] 本节依据
> - [Penn State STAT 505, Lesson 14](https://online.stat.psu.edu/stat505/Lesson14)：核验距离、凝聚层次聚类、Ward 方法、K-means 与后续描述。
> - Johnson & Wichern, *Applied Multivariate Statistical Analysis*, 6th ed.：核验目标函数与聚类诊断。
<!-- bilingual-en:start -->
> [!source] Sources for this section
> - [Penn State STAT 505, Lesson 14](https://online.stat.psu.edu/stat505/Lesson14) was used to verify distances, agglomerative hierarchical clustering, Ward's method, K-means, and post-cluster description.
> - Johnson and Wichern, *Applied Multivariate Statistical Analysis*, 6th ed., was used to verify objectives and diagnostics for clustering.
<!-- bilingual-en:end -->

## 表示、尺度与距离
<!-- bilingual-en:start -->
*Representation, scaling, and distance*
<!-- bilingual-en:end -->

Euclidean 距离适合连续且各维具有可比尺度的特征；标准化改变每个变量权重。相关距离强调形状而忽略水平，二元变量可用 Jaccard 等相似度。距离选择应回答“何种差异在应用中算大”，而不是由软件默认决定。
<!-- bilingual-en:start -->
Euclidean distance suits continuous features on comparable scales; standardisation changes the weight of every variable. Correlation distance emphasises shape while ignoring level, and binary variables may call for measures such as Jaccard similarity. Distance should answer which differences matter substantively rather than being inherited from software defaults.
<!-- bilingual-en:end -->

## 层次聚类
<!-- bilingual-en:start -->
*Hierarchical clustering*
<!-- bilingual-en:end -->

凝聚层次法从单点开始，按 single、complete、average 或 Ward 等链接规则合并。树状图显示合并历史，但切在哪里仍是分析选择。single linkage 易链化，Ward 更偏好紧凑近球形簇。
<!-- bilingual-en:start -->
Agglomerative hierarchical clustering begins with individual observations and merges them according to single, complete, average, Ward, or another linkage rule. A dendrogram displays the merger history, but the cut level remains an analytical choice. Single linkage is prone to chaining, while Ward's method favours compact, roughly spherical groups.
<!-- bilingual-en:end -->

链接规则定义“两个簇之间的距离”，因此不是同一算法的无关设置。single 看最近点，complete 看最远点，average 看平均跨簇距离，Ward 选择使组内平方和增加最小的合并。它们在有桥点、异常点或非球形结构时可给完全不同的树。
<!-- bilingual-en:start -->
Linkage defines the distance between clusters and is not an incidental setting. Single linkage uses the nearest pair, complete linkage the farthest pair, average linkage the mean cross-cluster distance, and Ward's method the merge with the smallest increase in within-cluster sum of squares. Bridges, outliers, and non-spherical structure can make their dendrograms entirely different.
<!-- bilingual-en:end -->

## K-means
<!-- bilingual-en:start -->
*K-means*
<!-- bilingual-en:end -->

K-means 最小化点到簇中心的组内平方距离，适合 Euclidean 空间中的近球形、尺度相近簇。它依赖初始值和 K，应多次初始化；异常值和未标准化变量会显著影响结果。
<!-- bilingual-en:start -->
K-means minimises within-cluster squared Euclidean distance to cluster centroids and is best suited to roughly spherical clusters of comparable scale. It depends on initialisation and the chosen $K$, so multiple starts are needed. Outliers and unstandardised variables can strongly affect the result.
<!-- bilingual-en:end -->

算法交替进行分配与更新，每步都不增加目标函数，但只保证到达局部最优。空簇、不同初始化和极不平衡簇大小是常见问题。对于类别变量、任意距离或非凸簇，K-means 的均值中心与平方 Euclidean 目标可能没有意义。
<!-- bilingual-en:start -->
The algorithm alternates assignment and centroid updates, never increasing its objective at either step, but it guarantees only a local optimum. Empty clusters, different initialisations, and highly unequal cluster sizes are common problems. For categorical variables, arbitrary distances, or non-convex groups, the mean-centroid and squared-Euclidean objective of K-means may be meaningless.
<!-- bilingual-en:end -->

## Worked example：尺度改变分组
<!-- bilingual-en:start -->
*Worked example: scaling changes the clusters*
<!-- bilingual-en:end -->

客户数据含年收入（数万单位）与购买次数（个位数）。直接 Euclidean 聚类几乎只按收入分组；将两列标准化后，购买频率获得相近权重，分组可能完全改变。两者没有哪个自动正确，关键是业务上收入差一万元与购买次数差一次应如何比较。
<!-- bilingual-en:start -->
Suppose customer data contain annual income measured in tens of thousands and purchase count measured in single digits. Raw Euclidean clustering will be driven almost entirely by income. After standardisation, purchase frequency receives comparable weight and the groups may change completely. Neither answer is automatically correct; the substantive question is how an income difference of ten thousand should compare with one additional purchase.
<!-- bilingual-en:end -->

## 聚类的验证
<!-- bilingual-en:start -->
*Validating clusters*
<!-- bilingual-en:end -->

结合内部指标、重采样稳定性、领域可解释性和外部结果，而非只追求漂亮图形。若不同合理预处理产生完全不同簇，应把不稳定性作为结果报告。
<!-- bilingual-en:start -->
Combine internal indices, resampling stability, domain interpretability, and external outcomes rather than pursuing an attractive plot alone. If reasonable preprocessing choices yield entirely different clusters, that instability is itself a result that should be reported.
<!-- bilingual-en:end -->

轮廓系数等内部指标只评价所选距离下的紧密与分离，不能证明类别有现实本体。聚类后再对同一变量做显著性检验会有选择偏差，因为这些变量已经用于制造组；描述应标注探索性，并尽可能在独立数据上复现。
<!-- bilingual-en:start -->
Internal indices such as silhouette width assess compactness and separation under the chosen distance; they do not prove that the groups have an objective real-world existence. Significance tests on the same variables after clustering are selection-biased because those variables created the groups. Such descriptions should be labelled exploratory and replicated where possible.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### K-means 前为什么常要标准化？
<!-- bilingual-en:start -->
*Why are variables often standardised before K-means?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 平方 Euclidean 距离会让数值尺度大的变量主导簇划分；标准化使各维权重更可控。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Squared Euclidean distance lets numerically large-scale variables dominate the partition. Standardisation makes the relative weight of each dimension more controllable.
<!-- bilingual-en:end -->

### dendrogram 的一次明显断层是否自动给出真实簇数？
<!-- bilingual-en:start -->
*Does a visible gap in a dendrogram automatically reveal the true number of clusters?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 不自动。断层依赖距离、链接、尺度和样本；还需稳定性、领域意义与外部验证。
<!-- bilingual-en:start -->
> [!answer]- Answer
> No. The gap depends on distance, linkage, scaling, and the sample. Stability, domain meaning, and external validation are still required.
<!-- bilingual-en:end -->

### 不同合理预处理产生完全不同聚类时应怎样报告？
<!-- bilingual-en:start -->
*How should results be reported when reasonable preprocessing choices produce entirely different clusters?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 把敏感性作为主要结论，展示哪些选择改变结果，不应只挑最容易解释的一次分组。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Treat sensitivity as a central result, showing which choices change the partition rather than selecting only the easiest grouping to explain.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Penn State STAT 505, Lesson 14](https://online.stat.psu.edu/stat505/Lesson14)：逐项核验距离、层次聚类、Ward 方法、K-means 与 post hoc 描述。
<!-- bilingual-en:start -->
- [Penn State STAT 505, Lesson 14](https://online.stat.psu.edu/stat505/Lesson14) was checked section by section for distances, hierarchical clustering, Ward's method, K-means, and post hoc description.
<!-- bilingual-en:end -->
- Johnson & Wichern, *Applied Multivariate Statistical Analysis*, 6th ed.：交叉核验目标函数、链接规则和验证边界。
<!-- bilingual-en:start -->
- Johnson and Wichern, *Applied Multivariate Statistical Analysis*, 6th ed., was used to cross-check objective functions, linkage rules, and limits of validation.
<!-- bilingual-en:end -->
