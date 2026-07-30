---
aliases:
  - "Multivariable Differentiation"
  - "Gradient and Hessian"
  - "Total Differential"
  - "多元微积分"
status: source-checked
---

# 多元微分：偏导、梯度、Hessian 与隐函数
<!-- bilingual-en:start -->
*Multivariable differentiation: partial derivatives, gradients, Hessians, and implicit functions*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 描述多个输入同时变化时函数的局部响应，并在资源或等式约束下寻找最优选择。
> **具体锚点：** 效用取决于两种商品，预算线限制组合；梯度给最快上升方向，最优内点处效用等高线与预算线相切。
> **核心难点：** 偏导只改变一个变量，全微分才组合共同变化；一阶条件是候选，Hessian、约束资格与边界决定结论。
> **为什么重要：** 微观优化、计量比较静态、机器学习和动态模型都使用这套语言。
> **继续：** 无约束先读梯度/Hessian，等式约束再用 Lagrange；局部解随参数变化见隐函数与包络。
<!-- bilingual-en:start -->
> [!summary] Quick recovery
> **What it solves:** Multivariable differentiation combines local responses to simultaneous changes in several inputs.
> **Concrete anchor:** For a utility function of two goods, partial derivatives measure one-good changes, the gradient collects both marginal effects, and the total differential approximates a joint change.
> **Central difficulty:** A partial derivative freezes other coordinates, whereas a total differential follows a specified joint movement. Hessian conclusions depend on definiteness, not isolated diagonal entries.
> **Why it matters:** Comparative statics, statistical approximations, machine learning, and multivariable optimisation all use these objects.
> **Continue with:** Learn the local linear and quadratic structures here, then use them in [[多元优化：无约束、Lagrange、KKT 与包络|multivariable optimisation]].
<!-- bilingual-en:end -->

> [!source] 本节依据
> - [[01_Math/08_MathsCamp-EC400/01_Revision_Mathematics/Notes/Complete/Revision Maths Notes all.pdf]]：支持 EC400 课程范围、记号与例题。
<!-- bilingual-en:start -->
> [!source] Sources for this section
> - [[01_Math/08_MathsCamp-EC400/01_Revision_Mathematics/Notes/Complete/Revision Maths Notes all.pdf|EC400 revision mathematics notes]] were checked for partial derivatives, gradients, total differentials, Hessians, definiteness, and implicit differentiation.
<!-- bilingual-en:end -->

## 偏导、梯度与全微分
<!-- bilingual-en:start -->
*Partial derivatives, the gradient, and the total differential*
<!-- bilingual-en:end -->

$\partial f/\partial x_i$ 固定其他变量看单方向变化。梯度 $\nabla f$ 汇总偏导并指向 Euclidean 几何下最快上升方向。全微分 $df\approx\nabla f^Tdx$ 给所有输入小变化的总一阶效应；变量尺度不同会影响“最快”方向的解释。
<!-- bilingual-en:start -->
$\partial f/\partial x_i$ changes one coordinate while holding the others fixed. The gradient $\nabla f$ collects partial derivatives and points in the direction of steepest increase under Euclidean geometry. The total differential $df\approx\nabla f^Tdx$ combines all small input changes into a first-order response. Variable scaling affects what “steepest” means.
<!-- bilingual-en:end -->

方向导数沿单位向量 $v$ 为 $D_vf=\nabla f^Tv$。Cauchy–Schwarz 不等式给 $D_vf\le\|\nabla f\|$，等号在 $v$ 与梯度同向时成立，这才是“梯度指向最快上升”的数学内容。
<!-- bilingual-en:start -->
The directional derivative along a unit vector $v$ is $D_vf=\nabla f^Tv$. Cauchy–Schwarz gives $D_vf\le\|\nabla f\|$, with equality when $v$ points along the gradient. This is the precise content of “the gradient points uphill most steeply.”
<!-- bilingual-en:end -->

## Hessian 与局部曲率
<!-- bilingual-en:start -->
*The Hessian and local curvature*
<!-- bilingual-en:end -->

Hessian 收集二阶偏导。驻点处正定对应严格局部最小、负定对应严格局部最大、不定对应鞍点；半正定情形二阶检验可能无结论。全局最优还需凸性/凹性或直接比较。
<!-- bilingual-en:start -->
The Hessian collects second partial derivatives. At a stationary point, positive definiteness gives a strict local minimum, negative definiteness a strict local maximum, and indefiniteness a saddle point; semidefiniteness may leave the second-order test inconclusive. Global optimality additionally requires convexity or concavity, or a direct comparison.
<!-- bilingual-en:end -->

## 隐函数与比较静态
<!-- bilingual-en:start -->
*Implicit functions and comparative statics*
<!-- bilingual-en:end -->

隐函数定理在 Jacobian 非奇异时给最优解对参数的局部变化。包络定理说明最优值对参数的一阶变化可忽略最优选择的间接变化，但需满足最优性与光滑条件。
<!-- bilingual-en:start -->
The implicit function theorem gives local changes in an endogenous solution with respect to parameters when the relevant Jacobian is nonsingular. The envelope theorem says that the first-order response of the optimised value can ignore the indirect movement of the optimiser, provided the required smoothness and optimality conditions hold.
<!-- bilingual-en:end -->

更基本地，若 $F(x,y)=0$ 且 $F_y\ne0$，则局部有 $y(x)$ 并且 $dy/dx=-F_x/F_y$。多方程系统中，需要对要解出的内生变量块的 Jacobian 求逆；行列式为零时，局部唯一可解性可能失败。
<!-- bilingual-en:start -->
More basically, if $F(x,y)=0$ and $F_y\ne0$, a local function $y(x)$ exists with $dy/dx=-F_x/F_y$. For systems, the Jacobian block with respect to the endogenous variables must be invertible; a zero determinant may destroy local uniqueness and solvability.
<!-- bilingual-en:end -->

## Worked example：全微分与交叉效应
<!-- bilingual-en:start -->
*Worked example: a total differential and an interaction effect*
<!-- bilingual-en:end -->

设 $f(x,y)=x^2y$，则 $\nabla f=(2xy,x^2)^T$，Hessian 为
$$
H=\begin{pmatrix}2y&2x\\2x&0\end{pmatrix}.
$$
在 $(1,2)$ 处，当 $dx=0.01,dy=-0.02$ 时，一阶近似给 $df\approx4(0.01)+1(-0.02)=0.02$。这不是单独一个偏导的答案，而是按实际共同移动对各边际效应加权。
<!-- bilingual-en:start -->
Let $f(x,y)=x^2y$. Then $\nabla f=(2xy,x^2)^T$ and
$$
H=\begin{pmatrix}2y&2x\\2x&0\end{pmatrix}.
$$
At $(1,2)$, a move $dx=0.01,dy=-0.02$ gives the first-order approximation $df\approx4(0.01)+1(-0.02)=0.02$. This is not one partial derivative; it weights all marginal effects by the actual joint movement.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### 偏导和全微分回答的问题有何不同？
<!-- bilingual-en:start -->
*How do the questions answered by a partial derivative and a total differential differ?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 偏导看单一坐标方向；全微分把所有输入的共同小变化按梯度加权汇总。
<!-- bilingual-en:start -->
> [!answer]- Answer
> A partial derivative moves along one coordinate while holding the others fixed; a total differential weights and combines all simultaneous small input changes.
<!-- bilingual-en:end -->

### Hessian 半正定能否证明严格最小？
<!-- bilingual-en:start -->
*Can a positive-semidefinite Hessian prove a strict local minimum?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 一般不能，二阶检验可能无结论；需更高阶、结构凸性或直接比较。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Not in general. The second-order test may be inconclusive; higher-order terms, structural convexity, or direct comparison may be needed.
<!-- bilingual-en:end -->

### 梯度为什么会指向最快上升方向？
<!-- bilingual-en:start -->
*Why does the gradient point in the direction of steepest ascent?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 单位方向 $v$ 上的变化率是 $\nabla f^Tv$，Cauchy–Schwarz 表明其最大值为 $\|\nabla f\|$，在 $v$ 与梯度同向时取到。
<!-- bilingual-en:start -->
> [!answer]- Answer
> The directional derivative is $\nabla f^Tv$. Cauchy–Schwarz bounds it by $\|\nabla f\|$, with equality when $v$ points along the gradient.
<!-- bilingual-en:end -->

### 隐函数定理中 Jacobian 非奇异为什么重要？
<!-- bilingual-en:start -->
*Why is a nonsingular Jacobian important in the implicit function theorem?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 它允许局部解出内生变量对参数的唯一光滑反应；奇异时可能出现多解、分支或竖直关系。
<!-- bilingual-en:start -->
> [!answer]- Answer
> It permits a locally unique, smooth solution for endogenous variables as functions of parameters. Singularity may produce multiple branches or vertical relations.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[01_Math/08_MathsCamp-EC400/01_Revision_Mathematics/Notes/Complete/Revision Maths Notes all.pdf]]：支持 EC400 课程范围、记号与例题。
<!-- bilingual-en:start -->
- [[01_Math/08_MathsCamp-EC400/01_Revision_Mathematics/Notes/Complete/Revision Maths Notes all.pdf|EC400 revision mathematics notes]] were reopened and checked for partial derivatives, gradients, total differentials, Hessians, definiteness tests, implicit functions, and course notation.
<!-- bilingual-en:end -->
