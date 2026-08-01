# 1. 多元统计分析导论（Introduction）
<!-- bilingual-en:start -->
*1. Introduction to Multivariate Statistical Analysis*
<!-- bilingual-en:end -->

>[!note] 本章主线
> 多元统计分析的对象不是“一个变量”，而是一个由多个变量组成的观测向量。第 1 章先建立三个入口：数据矩阵、描述统计矩阵、图形与距离。
> <!-- bilingual-en:start -->
> Multivariate statistical analysis studies an observation vector made up of several variables, not a single variable. Chapter 1 establishes three entry points: the data matrix, descriptive-statistics matrices, and graphical and distance-based views.
> <!-- bilingual-en:end -->

## 1.1. 多元方法的目标
<!-- bilingual-en:start -->
*1.1. Objectives of Multivariate Methods*
<!-- bilingual-en:end -->

多元方法通常服务于五类任务：
<!-- bilingual-en:start -->
Multivariate methods commonly serve five types of task:
<!-- bilingual-en:end -->

1. 数据压缩或结构简化。
2. 排序或分组。
3. 考察变量之间的相关结构。
4. 预测。
5. 构建和检验关于多个变量的统计假设。
<!-- bilingual-en:start -->

&nbsp;
**1.** Compress data or simplify structure.<br>
**2.** Order or group observations.<br>
**3.** Examine dependence among variables.<br>
**4.** Make predictions.<br>
**5.** Construct and test statistical hypotheses involving several variables.<br>
<!-- bilingual-en:end -->

>[!note] 复习抓手
> 一元统计只问“某个变量怎样”；多元统计多问一步：“多个变量如何一起变化”。
> <!-- bilingual-en:start -->
> Univariate statistics asks what happens to one variable; multivariate statistics adds the question of how several variables move together.
> <!-- bilingual-en:end -->

## 1.2. 数据的组织
<!-- bilingual-en:start -->
*1.2. Organising the Data*
<!-- bilingual-en:end -->

### 1.2.1. data（数据）
<!-- bilingual-en:start -->
*1.2.1. Data*
<!-- bilingual-en:end -->

数据是对多个变量（variables）或特征（features）进行测量后得到的记录。
<!-- bilingual-en:start -->
Data are records obtained by measuring several variables or features.
<!-- bilingual-en:end -->

### 1.2.2. array（数组）
<!-- bilingual-en:start -->
*1.2.2. Arrays*
<!-- bilingual-en:end -->

当研究者为了理解某个社会或物理现象，选择 $p\geq 1$ 个变量进行记录时，就会得到多变量数组。
<!-- bilingual-en:start -->
When a researcher records $p\geq1$ variables to understand a social or physical phenomenon, the result is a multivariate array.
<!-- bilingual-en:end -->

常见数据矩阵写作
<!-- bilingual-en:start -->
A common data-matrix representation is
<!-- bilingual-en:end -->
$$
X_{n\times p}=
\begin{bmatrix}
x_{11}&x_{12}&\cdots&x_{1p}\\
x_{21}&x_{22}&\cdots&x_{2p}\\
\vdots&\vdots&\ddots&\vdots\\
x_{n1}&x_{n2}&\cdots&x_{np}
\end{bmatrix},
$$
其中每一行是一个观测对象，每一列是一个变量。
<!-- bilingual-en:start -->
Each row is an observational unit, and each column is a variable.
<!-- bilingual-en:end -->

## 1.3. 描述性统计
<!-- bilingual-en:start -->
*1.3. Descriptive Statistics*
<!-- bilingual-en:end -->

### 1.3.1. 样本均值（Sample Mean）
<!-- bilingual-en:start -->
*1.3.1. Sample Mean*
<!-- bilingual-en:end -->

对单变量样本 $x_1,\ldots,x_n$，
<!-- bilingual-en:start -->
For the univariate sample $x_1,\ldots,x_n$,
<!-- bilingual-en:end -->
$$
\bar x=\frac1n\sum_{i=1}^n x_i.
$$

### 1.3.2. 样本方差（Sample Variance）
<!-- bilingual-en:start -->
*1.3.2. Sample Variance*
<!-- bilingual-en:end -->

$$
s_x^2=\frac{1}{n-1}\sum_{i=1}^n(x_i-\bar x)^2.
$$

这里分母用 $n-1$ 是为了让样本方差成为总体方差的无偏估计。
<!-- bilingual-en:start -->
The denominator is $n-1$ so that the sample variance is an unbiased estimator of the population variance.
<!-- bilingual-en:end -->

### 1.3.3. 样本协方差（Sample Covariance）
<!-- bilingual-en:start -->
*1.3.3. Sample Covariance*
<!-- bilingual-en:end -->

对成对样本 $(x_i,y_i)$，
<!-- bilingual-en:start -->
For paired observations $(x_i,y_i)$,
<!-- bilingual-en:end -->
$$
s_{xy}=\frac{1}{n-1}\sum_{i=1}^n(x_i-\bar x)(y_i-\bar y).
$$

协方差为正，说明两个变量倾向于同向变化；协方差为负，说明倾向于反向变化。
<!-- bilingual-en:start -->
A positive covariance means that the variables tend to move in the same direction; a negative covariance means that they tend to move in opposite directions.
<!-- bilingual-en:end -->

### 1.3.4. 样本相关系数（Sample Correlation）
<!-- bilingual-en:start -->
*1.3.4. Sample Correlation*
<!-- bilingual-en:end -->

$$
r_{xy}=\frac{s_{xy}}{s_xs_y}.
$$

相关系数把协方差标准化到 $[-1,1]$，便于比较不同量纲变量之间的线性关系。
<!-- bilingual-en:start -->
Correlation standardises covariance to $[-1,1]$, making linear relationships comparable across variables measured on different scales.
<!-- bilingual-en:end -->

## 1.4. 基本描述统计矩阵
<!-- bilingual-en:start -->
*1.4. Basic Descriptive-Statistics Matrices*
<!-- bilingual-en:end -->

### 1.4.1. 样本均值向量
<!-- bilingual-en:start -->
*1.4.1. Sample Mean Vector*
<!-- bilingual-en:end -->

对 $p$ 个变量，样本均值向量为
<!-- bilingual-en:start -->
For $p$ variables, the sample mean vector is
<!-- bilingual-en:end -->
$$
\bar x=
\begin{bmatrix}
\bar x_1\\
\bar x_2\\
\vdots\\
\bar x_p
\end{bmatrix}.
$$

### 1.4.2. 样本协方差矩阵
<!-- bilingual-en:start -->
*1.4.2. Sample Covariance Matrix*
<!-- bilingual-en:end -->

令 $D$ 为中心化后的偏差矩阵：
<!-- bilingual-en:start -->
Let $D$ be the centred deviation matrix:
<!-- bilingual-en:end -->
$$
D=X-\mathbf 1\bar x'.
$$

样本协方差矩阵为
<!-- bilingual-en:start -->
The sample covariance matrix is
<!-- bilingual-en:end -->
$$
S=\frac{1}{n-1}D'D.
$$

展开后：
<!-- bilingual-en:start -->
Expanded, it is:
<!-- bilingual-en:end -->
$$
S=
\begin{bmatrix}
s_{11}&s_{12}&\cdots&s_{1p}\\
s_{21}&s_{22}&\cdots&s_{2p}\\
\vdots&\vdots&\ddots&\vdots\\
s_{p1}&s_{p2}&\cdots&s_{pp}
\end{bmatrix}.
$$

>[!attention] 容易错
> 协方差矩阵必须用中心化后的偏差矩阵 $D$。直接写成 $\frac1{n-1}X'X$ 只有在每一列已经中心化时才成立。
> <!-- bilingual-en:start -->
> The covariance matrix must use the centred deviation matrix $D$. Writing it directly as $\frac1{n-1}X'X$ is valid only when every column of $X$ has already been centred.
> <!-- bilingual-en:end -->

### 1.4.3. 样本相关矩阵
<!-- bilingual-en:start -->
*1.4.3. Sample Correlation Matrix*
<!-- bilingual-en:end -->

样本相关矩阵 $R$ 的元素为
<!-- bilingual-en:start -->
The entries of the sample correlation matrix $R$ are
<!-- bilingual-en:end -->
$$
r_{ij}=\frac{s_{ij}}{\sqrt{s_{ii}s_{jj}}}.
$$

矩阵形式为
<!-- bilingual-en:start -->
Its matrix form is
<!-- bilingual-en:end -->
$$
R=
\begin{bmatrix}
1&r_{12}&\cdots&r_{1p}\\
r_{21}&1&\cdots&r_{2p}\\
\vdots&\vdots&\ddots&\vdots\\
r_{p1}&r_{p2}&\cdots&1
\end{bmatrix}.
$$

## 1.5. 图形技术
<!-- bilingual-en:start -->
*1.5. Graphical Techniques*
<!-- bilingual-en:end -->

### 1.5.1. 散点图
<!-- bilingual-en:start -->
*1.5.1. Scatterplots*
<!-- bilingual-en:end -->

散点图用于观察两个变量的关系，包括线性关系、非线性关系、异常点和分组结构。
<!-- bilingual-en:start -->
A scatterplot reveals relationships between two variables, including linear and nonlinear patterns, outliers, and group structure.
<!-- bilingual-en:end -->

### 1.5.2. 箱线图
<!-- bilingual-en:start -->
*1.5.2. Boxplots*
<!-- bilingual-en:end -->

箱线图用于快速查看单个变量的分布位置、离散程度和异常值。
<!-- bilingual-en:start -->
A boxplot gives a quick view of a variable's location, dispersion, and outliers.
<!-- bilingual-en:end -->

#### (1) 组成部分
<!-- bilingual-en:start -->
*(1) Components*
<!-- bilingual-en:end -->

| 部分 | 含义 |
|---|---|
| 中位数 | 箱体中的线 |
| Q1 与 Q3 | 箱体上下边界 |
| IQR | $Q3-Q1$ |
| 须 | 通常延伸到 $Q1-1.5IQR$ 与 $Q3+1.5IQR$ 范围内 |
| 异常值 | 须之外的点 |
<!-- bilingual-en:start -->
| Component | Meaning |
|---|---|
| Median | The line inside the box |
| Q1 and Q3 | The lower and upper edges of the box |
| IQR | $Q3-Q1$ |
| Whiskers | Usually extend to the most extreme points within $Q1-1.5IQR$ and $Q3+1.5IQR$ |
| Outliers | Points beyond the whiskers |
<!-- bilingual-en:end -->

#### (2) 阅读顺序
<!-- bilingual-en:start -->
*(2) Reading Order*
<!-- bilingual-en:end -->

1. 先看中位数位置，判断集中趋势。
2. 再看箱体高度，判断离散程度。
3. 再看须和异常点，判断尾部和离群点。
4. 最后看左右是否对称，判断偏态。
<!-- bilingual-en:start -->

&nbsp;
**1.** Start with the median to assess central location.<br>
**2.** Examine the height of the box to assess dispersion.<br>
**3.** Inspect the whiskers and outliers to assess tails and unusual observations.<br>
**4.** Finally, assess symmetry to identify skewness.<br>
<!-- bilingual-en:end -->

### 1.5.3. 直方图
<!-- bilingual-en:start -->
*1.5.3. Histograms*
<!-- bilingual-en:end -->

直方图用于观察单变量分布形状，例如偏态、厚尾、多峰和近似正态性。
<!-- bilingual-en:start -->
A histogram displays the shape of a univariate distribution, including skewness, heavy tails, multiple modes, and approximate normality.
<!-- bilingual-en:end -->

### 1.5.4. 折线图
<!-- bilingual-en:start -->
*1.5.4. Line Charts*
<!-- bilingual-en:end -->

折线图适合按时间或自然顺序排列的数据，主要观察趋势和阶段变化。
<!-- bilingual-en:start -->
A line chart suits data ordered by time or another natural sequence and is mainly used to inspect trends and changes across phases.
<!-- bilingual-en:end -->

## 1.6. 距离（Distance）
<!-- bilingual-en:start -->
*1.6. Distance*
<!-- bilingual-en:end -->

一个距离函数通常需要满足：
<!-- bilingual-en:start -->
A distance function normally satisfies:
<!-- bilingual-en:end -->

1. 非负性：$d(p,q)\geq0$。
2. 同一性：$d(p,q)=0$ 当且仅当 $p=q$。
3. 对称性：$d(p,q)=d(q,p)$。
4. 三角不等式：$d(p,r)\leq d(p,q)+d(q,r)$。
<!-- bilingual-en:start -->

&nbsp;
**1.** Non-negativity: $d(p,q)\geq0$.<br>
**2.** Identity: $d(p,q)=0$ if and only if $p=q$.<br>
**3.** Symmetry: $d(p,q)=d(q,p)$.<br>
**4.** The triangle inequality: $d(p,r)\leq d(p,q)+d(q,r)$.<br>
<!-- bilingual-en:end -->

>[!item] 后续连接
> 多元正态中的 [[多元数据、随机向量与样本协方差#距离与几何|Mahalanobis Distance]] 是普通距离的协方差调整版；聚类中的距离会直接决定分组结果。
> <!-- bilingual-en:start -->
> [[多元数据、随机向量与样本协方差#距离与几何|Mahalanobis Distance]] is a covariance-adjusted version of ordinary distance for multivariate normal data; in clustering, the chosen distance directly determines the grouping result.
> <!-- bilingual-en:end -->

## 1.7. 关联卡片
<!-- bilingual-en:start -->
*1.7. Related Cards*
<!-- bilingual-en:end -->

- [[多元统计分析 Course Atlas|Multivariate Statistics-hub]]
- [[多元数据、随机向量与样本协方差#数据矩阵与随机向量|Random Vector]]
- [[多元数据、随机向量与样本协方差#样本协方差矩阵|Sample Mean Vector]]
- [[多元数据、随机向量与样本协方差#样本协方差矩阵|Sample Covariance Matrix]]
- [[多元数据、随机向量与样本协方差#均值、协方差与相关|Correlation Matrix]]
- [[多元数据、随机向量与样本协方差#距离与几何|Mahalanobis Distance]]
