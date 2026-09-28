# Revision Mathematics 资料索引

> [!info] 格式说明
> 讲义含大量矩阵、公式和图形。Markdown 全文转写会破坏符号与排版，因此这里保留原始 PDF，并用 Markdown 提供章节导航。

## 使用顺序

LSE 的建议是先做 Background Material quizzes A–D，再做 Core Topics quizzes 1–5；只在薄弱主题回看对应讲义。目前 quiz 文件夹要求 Dropbox 链接密码，以下 notes 已完整保留。

## 合订本

- [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/01_Revision_Mathematics/Notes/Complete/Revision Maths Notes all.pdf|Revision Maths Notes all（95 页）]]

![[01_Math/08_EC400_MathsCamp/2026_Course_Materials/01_Revision_Mathematics/Notes/Complete/Revision Maths Notes all.pdf#height=620]]

## 分章讲义

1. [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/01_Revision_Mathematics/Notes/By_Topic/Revision Maths Notes 1, Content and Introduction.pdf|Content and Introduction]]
2. [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/01_Revision_Mathematics/Notes/By_Topic/Revision Maths Notes 2, Vectors.pdf|Vectors]]
3. [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/01_Revision_Mathematics/Notes/By_Topic/Revision Maths Notes 3, Matrices.pdf|Matrices]]
4. [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/01_Revision_Mathematics/Notes/By_Topic/Revision Maths Notes 4, Determinants.pdf|Determinants]]
5. [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/01_Revision_Mathematics/Notes/By_Topic/Revision Maths Notes 5, Eigenvalues and Eigenvectors.pdf|Eigenvalues and Eigenvectors]]
6. [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/01_Revision_Mathematics/Notes/By_Topic/Revision Maths Notes 6, Introduction to Multivariate Caclulus.pdf|Introduction to Multivariate Calculus]]
7. [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/01_Revision_Mathematics/Notes/By_Topic/Revision Maths Notes 7, Working with Multivariate Calculus.pdf|Working with Multivariate Calculus]]
8. [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/01_Revision_Mathematics/Notes/By_Topic/Revision Maths Notes 8, Introduction to Topology.pdf|Introduction to Topology]]

## 拓扑基础学习主线
<!-- bilingual-en:start -->
*Topology Foundations Learning Path*
<!-- bilingual-en:end -->

![[拓扑基础：开集、闭集与连续.canvas]]

这条主线先区分三层环境：度量空间用距离产生开球；开集抽象出拓扑；子空间拓扑说明同一集合换了环境后，开闭、内部、闭包和边界可能改变。随后才用序列处理度量空间中的可操作判别，并从连续性进入紧致与连通两条应用链。
<!-- bilingual-en:start -->
This path first separates three levels of structure: a metric produces open balls, open sets abstract to a topology, and the subspace topology explains why openness, closedness, interior, closure, and boundary depend on the ambient space. Sequences then provide operational tests in metric spaces, before continuity branches into compactness and connectedness.
<!-- bilingual-en:end -->

### 从距离到开闭与局部结构
<!-- bilingual-en:start -->
*From Distance to Open, Closed, and Local Set Structure*
<!-- bilingual-en:end -->

![[度量空间]]

![[开集]]

![[拓扑]]

![[闭集]]

![[子空间拓扑]]

![[集合内部]]

![[集合闭包]]

![[集合边界]]

### 从序列到连续
<!-- bilingual-en:start -->
*From Sequences to Continuity*
<!-- bilingual-en:end -->

![[序列收敛]]

![[连续性]]

需要把定义变成计算或证明工具时，进入 [[序列极限唯一]]、[[闭包的序列判别]]、[[闭集的序列判别]] 与 [[连续性的等价判别]]；[[聚点不等于收敛]] 单独保留最容易混淆的边界。
<!-- bilingual-en:start -->
For operational tests, continue to uniqueness of limits, the sequential tests for closure and closedness, and the equivalent criteria for continuity. The distinction between a cluster point and convergence remains a separate boundary card.
<!-- bilingual-en:end -->

### 从紧致与连通到应用
<!-- bilingual-en:start -->
*From Compactness and Connectedness to Applications*
<!-- bilingual-en:end -->

![[紧致性]]

紧致链是 [[紧致性的序列判别]] → [[Heine–Borel 定理的边界]] 或 [[连续像保持紧致]] → [[极值定理]]。不要在一般度量空间中把“闭且有界”直接当成紧致。
<!-- bilingual-en:start -->
The compactness branch proceeds through sequential compactness, the scope of Heine–Borel, and continuous images to the extreme-value theorem. Closed and bounded is not a general definition of compactness.
<!-- bilingual-en:end -->

![[连通性]]

连通链是 [[连续像保持连通]] + [[实线连通集是区间]] → [[介值定理]]。两张中间定理分别回答“连续映射保留什么”和“实线中的连通集合长什么样”。
<!-- bilingual-en:start -->
The connectedness branch combines preservation under continuous maps with the interval characterisation of connected subsets of the real line, yielding the structural route to the Intermediate Value Theorem.
<!-- bilingual-en:end -->

## 多元微分学习主线
<!-- bilingual-en:start -->
*Multivariable Differentiation Learning Path*
<!-- bilingual-en:end -->

![[多元微分.canvas]]

这条主线先分别回答“偏导数是什么”和“为什么偏导存在仍不推出可微”，再进入四个问题：导数怎样成为统一的线性映射；梯度与 Hessian 怎样描述一阶方向和二阶曲率；方阵 Jacobian 怎样连接局部体积、换元和局部逆；方程组何时能在局部解出一条可微分支。
<!-- bilingual-en:start -->
This path first separates the definition of a partial derivative from the proposition that partial derivatives do not imply differentiability. It then asks how a derivative becomes one linear map for every small perturbation, how the gradient and Hessian describe first- and second-order behaviour, how a square Jacobian connects local volume, change of variables, and local inversion, and when a system of equations locally defines a differentiable solution branch.
<!-- bilingual-en:end -->

### 从偏导到复合映射
<!-- bilingual-en:start -->
*From Partial Derivatives to Composite Maps*
<!-- bilingual-en:end -->

![[偏导数]]

![[偏导不可推可微]]

![[全微分]]

![[Jacobian矩阵]]

![[多元链式法则]]

### 从方阵 Jacobian 到体积、换元与局部逆
<!-- bilingual-en:start -->
*From a Square Jacobian to Volume, Change of Variables, and Local Inversion*
<!-- bilingual-en:end -->

![[Jacobian行列式]]

![[多元换元公式]]

![[逆函数定理]]

![[Jacobian奇异不推局部非单射]]

### 从方向变化到水平集几何
<!-- bilingual-en:start -->
*From Directional Change to Level-Set Geometry*
<!-- bilingual-en:end -->

![[方向导数]]

![[方向导数不可推可微]]

![[方向导数梯度公式]]

![[方向导数尺度]]

![[梯度最陡方向]]

![[最陡方向依赖度量]]

![[梯度与水平集]]

![[零梯度不定切空间]]

### 从 Hessian 到局部分类
<!-- bilingual-en:start -->
*From the Hessian to Local Classification*
<!-- bilingual-en:end -->

二阶检验从 Taylor 展开取得依据。正定、负定和不定分别给出严格局部极小、严格局部极大和鞍点的充分判断；半定时二阶项留有平坦方向，必须查看高阶项或其他结构。
<!-- bilingual-en:start -->
Second-order tests derive their force from the Taylor expansion. Positive definiteness, negative definiteness, and indefiniteness give sufficient diagnoses of a strict local minimum, a strict local maximum, and a saddle, respectively. A semidefinite quadratic term leaves flat directions and therefore requires higher-order terms or other structure.
<!-- bilingual-en:end -->

![[Hessian矩阵]]

![[Hessian对称条件]]

![[Hessian方向二阶导数]]

![[多元Taylor近似]]

![[Hessian 局部极小判据]]

![[Hessian 局部极大判据]]

![[Hessian 鞍点判据]]

![[半定Hessian无结论]]

### 从方程组到隐式解支
<!-- bilingual-en:start -->
*From an Equation System to an Implicit Solution Branch*
<!-- bilingual-en:end -->

![[隐函数定理]]

![[隐函数导数公式]]

![[隐函数奇异块不推无解支]]

接下来若研究“最优选择怎样随参数变化”，进入 [[隐函数比较静态]]；若开始寻找无约束或有约束最优，进入 [[多元优化.canvas|多元优化]]。
<!-- bilingual-en:start -->
To study how an optimal choice changes with a parameter, continue to [[隐函数比较静态|implicit-function comparative statics]]. To find unconstrained or constrained optima, enter the [[多元优化.canvas|multivariable optimisation map]].
<!-- bilingual-en:end -->

> [!note] 来源
> 这些 2026 notes 来自仓库已有缓存。当前官方 Dropbox 共享文件夹要求链接密码，所以没有把旧路径恢复回来，而是复制到本次新建的资料目录。
