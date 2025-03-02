==主要讲低阶差分方程,考试最多考到二阶==

# 1.介绍

## 1.1 术语:算子operator

从一个拓扑空间到另一个拓扑空间的映射.

## 1.2术语:差分算子Difference operator


>[!NOTE] 差分算子 $\Delta$的定义
> 定义一阶差分为 $\Delta y_t \equiv y_t - y_{t-1}$，表示变量在相邻两个时期的变化量。例如，如果 $y_t$ 是某期的值，那么 $\Delta y_t$ 就是 $y_t$ 与前一期 $y_{t-1}$ 之差。 
> 
> 同理，**二阶差分**定义为 $\Delta^2 y_t \equiv \Delta(\Delta y_t) = (y_t - y_{t-1}) - (y_{t-1} - y_{t-2}) = y_t - 2y_{t-1} + y_{t-2}$ 。差分算子在时间序列分析中类似于微积分中的导数运算，用于考察序列的变化。

**直观理解**：$\Delta f(x)$ 计算的是 $f(x)$ 在 $x$ 处的增量，相当于离散版本的**一阶导数**。

## 1.3术语:滞后算子Lag operator

>[!NOTE] 滞后算子 $L$的定义
> $L^i y_t = y_{t-i}$，即将序列“往后推”$i$期（滞后 $i$ 期） 。例如，$L y_t = y_{t-1}$，$L^2 y_t = y_{t-2}$ 等。
> 
> 滞后算子有一些类似指数的运算性质，如 $L^i \cdot L^j = L^{i+j}$，$L^0$ 为恒等算子等/利用滞后算子可以简洁地表示差分，例如 $\Delta y_t = y_t - y_{t-1} = y_t - L y_t = (1 - L) y_t$ 。

## 1.4 为什么要进行差分方程

考虑经典的分解方法：
$$y_t = T_t + S_t + I_t$$
并设：
$$T_t = 1 + 0.1 t, \quad S_t = 1.6 \sin\left(\frac{t\pi}{6}\right), \quad I_t = 0.7 I_{t-1} + \varepsilon_t$$

1. **趋势成分（Trend Component）$T_t$**  
   这里的趋势项 $T_t = 1 + 0.1 t$ 表示 $y_t$ 随着时间 $t$ 线性增长，每个时间点 $t$ 增加 $0.1$。
2. **季节成分（Seasonal Component）$S_t$**  
   季节项 $S_t = 1.6 \sin(t\pi/6)$ 采用正弦函数，表示 $y_t$ 随着时间 $t$ 具有周期性变化。周期为 $12$（因为 $\sin$ 函数的周期是 $2\pi$，即 $t\pi/6 = 2\pi$ 对应 $t=12$），说明该数据可能有**年度季节性**（比如按月变化的经济指标）。
3. **不规则成分（Irregular Component）$I_t$**  
   不规则项 $I_t$ 由一阶自回归模型（AR(1) 过程）给出：
   $$I_t = 0.7 I_{t-1} + \varepsilon_t$$
   其中 $\varepsilon_t$ 是随机扰动，表示无法预测的随机变化。这个方程表示 $I_t$ 依赖于其前一期值 $I_{t-1}$，并受到随机冲击的影响。这是一种**平稳时间序列模型**，如果 $|\phi| < 1$（这里 $\phi = 0.7$），则该过程是均值回归的，不会发散。
4. **差分方程（Difference Equation）**  
   差分方程本质上是描述变量如何通过**过去的滞后值**和**当前时间点的信息**来决定当前值的方程。在这里：
   - **$T_t$ 作为趋势项是一个** **线性方程**，它随时间变化但不依赖于滞后值。
   - **$S_t$ 作为季节项是一个** **周期性方程**，它依赖于时间 $t$，但没有滞后依赖。
   - **$I_t$ 是一个一阶差分方程（AR(1) 模型）**，表示它的当前值由上一期的值和一个随机误差项决定。

# 2.线性差分方程和其解Linear Difference Equation and Its Solution

## 2.1线性差分方程

<span style="color: yellow;">线性差分方程的“线性”指的是在方程中，变量及其滞后项只以一次幂出现，并且不会相互乘积或出现在非线性函数中。</span>

具有常系数的$n$ 阶线性差分方程的一般形式如下所示：
$$y_t = a_0 + \sum_{i=1}^n a_i \, y_{t-i} + x_t$$

1. **$n$ 阶线性差分方程**  
   这里的方程：
   $$y_t = a_0 + \sum_{i=1}^n a_i \, y_{t-i} + x_t$$
   表示当前时刻 $t$ 的变量 $y_t$ 可以由其过去 $n$ 期的值 $y_{t-1}, y_{t-2}, \dots, y_{t-n}$ 以及一个额外的外生项 $x_t$ 决定。系数 $a_0, a_1, \dots, a_n$ 都是常数。
2. **外生过程（Forcing Process）$x_t$**  
   - $x_t$ 可以是时间 $t$ 的确定性函数，比如趋势或季节性项。
   - 也可以是其他经济变量（或随机过程）的滞后值或当前值。
   - 或者是随机扰动，如白噪声的加权和，$x_t = \sum_{i=0}^\infty \beta_i \, \varepsilon_{t-i}$，这在时间序列和信号处理中都很常见。
3. **差分算子表示**  
   差分算子 $\Delta$ 定义为 $\Delta y_t = y_t - y_{t-1}$。  
   将方程用 $\Delta y_t$ 表示时：
   $$
   \Delta y_t = a_0 + \gamma \, y_{t-1} + \sum_{i=2}^n a_i \, y_{t-i} + x_t
   $$
   其中 $\gamma = a_1 - 1$。这是把一阶差分后的 $y_t$ 与其滞后值联系起来的一种写法，有时在研究单位根、平稳性等问题时更方便。

## 2.2 差分方程的解

一个差分方程的解，给出了 $y_t$ 如何作为 $\{x_t\}$ 的元素与时间 $t$（以及可能给定的 $y_t$ 序列初始条件）来确定其数值的函数形式。这可以类比微积分中的解微分方程。

- 在微分方程中，我们通常说“解”是一个函数，它告诉我们如何根据自变量（如时间 $t$）以及必要的初始条件来确定未知函数的值。  
- 对差分方程而言，情况类似：一个差分方程的解就是告诉我们 **$y_t$** 如何由过去或当前的外生序列 **$\{x_t\}$**、时间 **$t$** 以及可能给定的初始条件（如 **$y_0, y_1, \dots$**）来确定。  
  - **微分方程**：需要初始值（或边界值）才能唯一确定解；  
  - **差分方程**：也需要初始条件（如早期的几个 $y_t$ 的值），然后才能向前（或向后）迭代出整个序列的解。  

因此，这句话的意思是：**差分方程的解，就像微分方程的解一样，能明确表示“在给定初始条件和外生变量的情况下，$y_t$ 的具体取值是怎样随时间演化的”。**  

>[!example] **示例 1**：
>  $$\Delta y_t = 2$$  
>  其解为  
>  $$y_t = 2t + c$$  
>  其中 $c$ 是任意常数。我们可以通过代入来验证。

>[!example] **示例 2**：  
>  $$I_t = 0.7 I_{t-1} + \varepsilon_t$$  （这是不规则成分 $I_t$ 的模型），其解为  
>  $$I_t = \sum_{i=0}^\infty 0.7^i \,\varepsilon_{t-i}$$  
>  同样可以通过代入来验证。

上面这些解只是直接“假设”出来的吗？有没有什么方法可以系统地求解这些差分方程呢？

1. **迭代展开（Forward Iteration）**  
   - 对于形如  
     $$I_t = 0.7 I_{t-1} + \varepsilon_t,$$  
     可以反复向前展开：  
     $$
     I_t 
     = 0.7 I_{t-1} + \varepsilon_t 
     = 0.7 \bigl(0.7 I_{t-2} + \varepsilon_{t-1}\bigr) + \varepsilon_t 
     = 0.7^2 I_{t-2} + 0.7 \varepsilon_{t-1} + \varepsilon_t
     $$
     继续迭代，就能得到  
     $$
     I_t = 0.7^k I_{t-k} + \sum_{i=0}^{k-1} 0.7^i \,\varepsilon_{t-i}.
     $$
     当 $|0.7|<1$ 且 $k \to \infty$，$0.7^k I_{t-k}$ 收敛为 0，最终得到  
     $$
     I_t = \sum_{i=0}^{\infty} 0.7^i \,\varepsilon_{t-i},
     $$
     这与示例中的解相符。

2. **特征方程法（Characteristic Equation）**  
   - 对于线性齐次差分方程，如  
     $$\Delta y_t = 2 \quad \Longleftrightarrow \quad y_t - y_{t-1} = 2,$$  
     可以先求其齐次解和 particular 解。  
     - 齐次方程：  
       $$y_t - y_{t-1} = 0,$$  
       其解为常数或等比数列。  
     - 特殊解（particular solution）：  
       由于右边是常数 2，假设 $y_t = A t + B$ 形式，代入方程解出 $A=2$，$B$ 可并入常数项。  
     - 因此总解为  
       $$y_t = 2t + c,$$  
       其中 $c$ 是由初始条件决定的常数。

# 3.蛛网模型 Cobweb Model

# 4. 解齐次微分方程Solving Homogeneous Difference Equations

# 5. 找特解Finding Particular Solution

# 6.作业



  

**为什么要研究差分方程？** 在时间序列中，许多现象可以用差分方程来描述。例如经典的分解方法将时间序列分为趋势 ($T_t$)、季节 ($S_t$) 和不规则 ($I_t$) 三部分，即 $y_t = T_t + S_t + I_t$ 。给定具体形式，如 $T_t = 1 + 0.1t$（线性趋势），$S_t = 1.6\sin(t\pi/6)$（周期为12的正弦季节波动），以及 $I_t = 0.7 I_{t-1} + \varepsilon_t$（AR(1)随机扰动，其中 $\varepsilon_t$ 是零均值随机噪声） 。可以发现，每个部分都满足一定形式的差分方程：

• 趋势部分 $T_t$ 满足 $\Delta T_t = 0.1$（一阶差分为常数）；

• 季节部分 $S_t$ 满足一个二阶差分方程（正弦可以视为二阶差分方程的解，因其表现出周期性振荡）；

• 不规则部分 $I_t$ 明确地由差分方程 $I_t = 0.7 I_{t-1} + \varepsilon_t$ 给出。

  

因此，差分方程的一般形式就是**当前变量的值由其滞后值、时间和其他变量（包括随机扰动）共同决定的关系** 。时间序列经济计量的核心之一，就是对含随机成分的差分方程进行建模和估计 。

  

**2. 线性差分方程及其解 (Linear Difference Equation and Its Solution)**

  

**线性差分方程的一般形式：** 一个$n$阶线性差分方程（常系数）的标准形式可表示为：

  

$$

y_t = a_0 + \sum_{i=1}^n a_i,y_{t-i} + x_t, \tag{1}

$$

  

其中 $a_0, a_1, \dots, a_n$ 为常数系数，$x_t$ 被称为**外生驱动过程**（forcing process） 。$x_t$ 可以是时间$t$的函数、其他变量当前或滞后的值，或者是随机扰动项（例如 $x_t$ 可能表示一个无限长的移动平均过程，如 $x_t = \sum_{i=0}^{\infty} \beta_i \varepsilon_{t-i}$） 。上述差分方程表明：当前值 $y_t$ 由一个常数项 $a_0$、过去 $n$ 期的自身值 $y_{t-1},\dots,y_{t-n}$ 经过系数加权和、以及当前的外生影响 $x_t$ 所共同决定。

  

利用前述算子，可以将(1)式改写为：

  

$$

\Delta y_t = y_t - y_{t-1} = a_0 + (a_1 - 1)y_{t-1} + a_2 y_{t-2} + \cdots + a_n y_{t-n} + x_t,

$$

  

其中令 $\gamma = a_1 - 1$ 。这个形式揭示了如果 $a_1=1$（即$\gamma=0$），则方程两边的差分$\Delta y_t$与过去更久之前的值相关；这种情况对应于**单位根**情形，需要特别处理（后文讨论）。总之，差分算子形式有助于我们分析方程的性质和求解。

  

**差分方程的解：** 所谓**差分方程的解**，指的是找出一个表达式，使其表示 $y_t$ **完全由已知的信息（包括驱动过程 ${x_t}$、时间$t$以及初始条件）表示出来** 。这类似于微分方程的解，它将 $y_t$ 表示为输入序列和初始值的函数。在求解时，我们通常需要给定一定数量的**初始条件**（initial conditions），如 $y_0, y_1, \dots, y_{n-1}$ 等，用于确定解中出现的任意常数。

  

_举例1：_ 考虑简单的一阶差分方程 $\Delta y_t = 2$，即 $y_t - y_{t-1} = 2$。直观上这表示 $y_t$ 每期以常数2增长。它的通解为 $y_t = 2t + C$，其中 $C$ 是任意常数 。我们可以验证：$y_t - y_{t-1} = (2t + C) - [2(t-1) + C] = 2$，满足原差分方程。因此 $y_t = 2t + C$ 确实是解。不同的初始条件对应于不同的$C$值。例如，若已知$y_0$，则$C$应取为$y_0$（因为$y_0 = 2\cdot 0 + C = C$）。

  

_举例2：_ 考虑随机扰动驱动的一阶差分方程 $I_t = 0.7,I_{t-1} + \varepsilon_t$ 。直观理解，这是一个**AR(1)**（一阶自回归）模型，$\varepsilon_t$是噪声。可以猜测其解为**无限长的加权和**形式：

  

$$

I_t = \sum_{i=0}^{\infty} 0.7^i, \varepsilon_{t-i},.

$$

  

也就是说，当前值 $I_t$ 由当前及过去各期的随机冲击累积而成，每过去一时期，影响按系数0.7递减 。验证这一解很容易：将上述 $I_t$ 表达式代入原方程右侧：

• 右侧：$0.7 I_{t-1} + \varepsilon_t = 0.7 \sum_{i=0}^{\infty} 0.7^i \varepsilon_{t-1-i} + \varepsilon_t = \sum_{i=0}^{\infty} 0.7^{i+1} \varepsilon_{t-1-i} + \varepsilon_t = \sum_{j=1}^{\infty} 0.7^{j} \varepsilon_{t-j} + \varepsilon_t$ (令$j=i+1$)。

• 左侧：$I_t = \sum_{i=0}^{\infty} 0.7^i \varepsilon_{t-i} = \varepsilon_t + \sum_{j=1}^{\infty} 0.7^j \varepsilon_{t-j}$。

  

可见左右两侧相等，因此该表达式是原方程的一个解。这个解表明，如果我们假设过程在远过去($t \to -\infty$)趋于0且收敛，那么$0.7^i$阶降权确保远期冲击影响可以忽略。在这种假设下，我们无需指定具体初始值，解就已被完全确定。

  

上述例子给出了特定差分方程的解，但**这些解是凭猜测给出的吗？是否有系统的方法求解一般的差分方程？** 下面我们将介绍两种思路：**迭代法**和**通解法**，并推广到更高阶的情形。

  

**2.1 一阶差分方程的迭代解法 (Solution by Iteration for First-order Equation)**

  

首先考虑一般的一阶方程：

  

$$

y_t = a_0 + a_1,y_{t-1} + \varepsilon_t, \tag{2}

$$

  

假设已知初始值 $y_0$ 。我们可以通过**前向迭代**（forward iteration）逐步求出后续$y_t$：

• 当 $t=1$ 时：$y_1 = a_0 + a_1 y_0 + \varepsilon_1$ 。

• 当 $t=2$ 时：$y_2 = a_0 + a_1 y_1 + \varepsilon_2 = a_0 + a_1(a_0 + a_1 y_0 + \varepsilon_1) + \varepsilon_2 = a_0(1 + a_1) + a_1^2 y_0 + a_1 \varepsilon_1 + \varepsilon_2$ 。

• 当 $t=3$ 时：$y_3 = a_0 + a_1 y_2 + \varepsilon_3 = a_0 + a_1[a_0(1+a_1) + a_1^2 y_0 + a_1 \varepsilon_1 + \varepsilon_2] + \varepsilon_3$。继续展开并归纳，可得到一般形式。

  

**归纳结果：** 对任意 $t \ge 1$，通过重复代入，可以得到：

  

$$

y_t ;=; a_0 \sum_{i=0}^{t-1} a_1^i ;+; a_1^t,y_0 ;+; \sum_{i=0}^{t-1} a_1^i,\varepsilon_{t-i},. \tag{3}

$$

  

上述三个部分分别代表：累积的常数项贡献、初始值经过$t$期后的贡献、以及过去冲击的累积贡献 。我们可以验证 (3) 式确实满足原方程 (2)。

  

如果初始值不仅包括 $y_0$，还可以考虑在更早期（如负时间）有所谓**不定初始条件**的情形。对于 (2) 式，若我们并未给定$y_0$，可以尝试通过**向后迭代**（backward iteration）将公式向过去扩展一 period：

• 对于 $t$ 时刻，利用 $t-1$ 时刻的方程：$y_{t-1} = a_0 + a_1 y_{t-2} + \varepsilon_{t-1}$，代入 $y_t$ 式子中：

$y_t = a_0 + a_1 [a_0 + a_1 y_{t-2} + \varepsilon_{t-1}] + \varepsilon_t = a_0(1 + a_1) + a_1^2 y_{t-2} + a_1 \varepsilon_{t-1} + \varepsilon_t$。

• 重复向前代入$ m $次，可将$y_t$表示为从$t-m$时刻开始的迭代结果 ：

  

$$

y_t = a_0 \sum_{i=0}^{t+m-1} a_1^i ;+; a_1^{t+m} y_{t-m} ;+; \sum_{i=0}^{t+m-1} a_1^i,\varepsilon_{t-i},. \tag{4}

$$

  

当 $m$ 趋于无穷大（即将初始时刻推至远过去）时，结果取决于 $a_1$ 的大小：如果 $|a_1|<1$，则 $a_1^{t+m} y_{t-m}$ 项随着 $m \to \infty$ 衰减为0 。此时我们得到**稳定的特解**（particular solution）：

  

$$

y_t = \frac{a_0}{,1 - a_1,} + \sum_{i=0}^{\infty} a_1^i,\varepsilon_{t-i},, \quad \text{(当 } |a_1|<1\text{ 时)} \tag{6}

$$

  

其中第一项 $\frac{a_0}{1-a_1}$ 是系统在没有随机波动时的稳态水平（因为设置 $\varepsilon_t = 0$ 时，该值使 $y_t = y_{t-1}$），第二项是所有过去随机冲击经过衰减后的累计影响 。可以验证 (6) 式满足原方程 (2) 。由于此解不包含任意常数，它是特定于假设“远过去无影响”的_特解_。

  

当然，(2) 式的一般**通解**应该允许包含任意初始影响。实际上，即使 $|a_1|<1$，在 (6) 的基础上我们仍可加入一个形如 $A a_1^t$ 的齐次通解项：

  

$$

y_t = A,a_1^t ;+; \frac{a_0}{,1 - a_1,} + \sum_{i=0}^{\infty} a_1^i,\varepsilon_{t-i},, \tag{7}

$$

  

其中 $A$ 是任意常数 。容易验证 (7) 也满足原方程。此时，如果给定初始条件 $y_0$，就可以确定 $A$ 的取值：令 $t=0$ 代入 (7)，得到 $y_0 = A + \frac{a_0}{1-a_1} + \sum_{i=0}^{\infty} a_1^i \varepsilon_{-i}$ 。在假设$t<0$时无冲击（或冲击和有限）以简化计算的情况下，可认为 $\sum_{i=0}^{\infty} a_1^i \varepsilon_{-i} \approx 0$，于是得到 $A = y_0 - \frac{a_0}{1-a_1}$ 。将 $A$ 代回 (7)，我们就得到了满足初始条件的特定解形式：

  

$$

y_t = \Big(y_0 - \frac{a_0}{,1 - a_1,}\Big) a_1^t ;+; \frac{a_0}{,1 - a_1,} + \sum_{i=0}^{t-1} a_1^i,\varepsilon_{t-i},. \tag{8}

$$

  

可以验证，这一表达式与之前通过前向迭代得到的有限和公式 (3) 是等价的 。换言之，我们通过允许 $t$ 趋向 $-\infty$ 得到了一个特解 (6)，再加上任意齐次解并利用初始条件确定常数，最终得到了一般解 (8)。

  

**注意：** 当 $|a_1| \ge 1$ 时，(6) 式的推导不再成立，因为 $a_1^{t+m}$ 项不会衰减为0 。对于这种情况，必须使用 (3) 式所示的有限和通解；其中如果 $|a_1|>1$，齐次项 $a_1^t$ 会随着时间_发散_或者_振荡扩大_，过程不稳定；若 $a_1=1$，则方程 (2) 变为 $y_t = a_0 + y_{t-1} + \varepsilon_t$，这是单位根情形，其解可以写为 $y_t = a_0 t + \sum_{i=1}^t \varepsilon_i + y_0$（呈现线性趋势加累积随机游走） 。总之，$|a_1|<1$ 时解可收敛于稳态水平，而 $|a_1|\ge 1$ 时解会不收敛或无固定均值，需要特别对待。

  

**2.2 高阶差分方程的通解方法 (General Solution Method for Higher-order Equations)**

  

上述迭代法对一阶方程非常直观，但对于高阶（$n>1$）差分方程，用类似方法代入展开会遇到复杂的代数推导。因此，我们寻求一种更系统的通解方法。这个方法可以分为**四步** ：

1. **写出齐次方程并求其全部齐次解。**

将原差分方程 (1) 中外生项（包括 $a_0$ 常数项和 $x_t$ 驱动项）设为0，得到对应的**齐次方程**。求解该齐次方程，找出$n$个线性独立的齐次解。

1. **求一个特解（特定解）。**

为原始非齐次方程（即带有 $a_0$ 和 $x_t$）找到任意一个解$y_t^p$。这个解不需要包含所有任意常数，因为任意常数组合会在下一步加入。

1. **写出方程的一般解。**

一般解 = 特解 $+$ 齐次解的线性组合。也就是说，将求得的一个特解与所有齐次解的线性组合相加，构成原方程的一般形式。由于齐次解包含任意常数，这一组合会带有$n$个待定常数。

1. **利用初始条件确定常数。**

根据给定的初始条件（一般需要给定 $n$ 个初始值，例如 $y_0, y_1, \dots, y_{n-1}$），构造方程并解出上述线性组合中的$n$个常数，从而得到满足初值条件的唯一特定解。

  

下面通过一个具体**例子**来说明四步法的应用。

  

**例子：二阶线性差分方程的求解**

  

考虑如下二阶差分方程：

$$

y_t = 0.9,y_{t-1} ;-; 0.2,y_{t-2} ;+; 3,. \tag{10}

$$

这是形如 $y_t = a_0 + a_1 y_{t-1} + a_2 y_{t-2}$ 的情况，其中 $a_0=3, a_1=0.9, a_2=-0.2$（注意这里负号属于系数的一部分）。假设初始条件 $y_0=13,;y_1=11.3$。我们按四步求解：

  

**第1步（齐次方程）：** 将常数和外生项设为0，得到齐次方程

$$y_t - 0.9,y_{t-1} + 0.2,y_{t-2} = 0,.$$

求该齐次方程的通解。我们**猜测**齐次解形式为 $y_t^h = r^t$（这是线性齐次方程常用的假设形式），代入齐次方程得到**特征方程** ：

$$

r^2 - 0.9,r + 0.2 = 0,.

$$

求解此二次特征方程：$r^2 - 0.9r + 0.2 = 0$。使用求根公式，

$$

r = \frac{0.9 \pm \sqrt{(0.9)^2 - 4(0.2)}}{2} = \frac{0.9 \pm \sqrt{0.81 - 0.8}}{2} = \frac{0.9 \pm \sqrt{0.01}}{2} = \frac{0.9 \pm 0.1}{2},.

$$

因此得到两个实特征根：$r_1 = \frac{0.9+0.1}{2} = 0.5$，$r_2 = \frac{0.9-0.1}{2} = 0.4$，互不相等。对应的两个线性独立齐次解为 $y_{t}^{h(1)} = (0.5)^t$ 和 $y_{t}^{h(2)} = (0.4)^t$ 。因此齐次方程的一般解为这两个解的线性组合：

$$

y_t^h = A_1 \cdot (0.5)^t + A_2 \cdot (0.4)^t,,

$$

其中 $A_1, A_2$ 为任意常数。

  

**第2步（特解）：** 寻找原非齐次方程 $y_t = 0.9y_{t-1} - 0.2y_{t-2} + 3$ 的一个特解。观察驱动项为常数3，我们可以尝试令解为常数形式 $y_t^p = C$（即假设解不随时间变化）。将其代入方程左侧：$y_t^p = C,;y_{t-1}^p = C,;y_{t-2}^p = C$，代入原方程得到

$$

C \stackrel{!}{=} 0.9C - 0.2C + 3,,

$$

即 $C = 0.7C + 3$。解得 $0.3C = 3$，于是 $C = 10$。因此 $y_t^p = 10$ 是原方程的一个特解 。_(验证：$0.9\cdot 10 - 0.2\cdot 10 + 3 = 9 - 2 + 3 = 10$，左等于右，成立。)_

  

**第3步（一般解）：** 将特解和齐次解相加，得到原方程的一般解形式：

$$

y_t = A_1 \cdot (0.5)^t + A_2 \cdot (0.4)^t + 10,.

$$

此时 $A_1, A_2$ 仍是待定常数。

  

**第4步（套用初始条件）：** 利用给定的初始值求 $A_1, A_2$。已知 $y_0 = 13,; y_1 = 11.3$。代入一般解：

• 当 $t=0$ 时：$y_0 = A_1(0.5^0) + A_2(0.4^0) + 10 = A_1 + A_2 + 10 = 13$，得到方程 (i)：$A_1 + A_2 = 3$。

• 当 $t=1$ 时：$y_1 = A_1(0.5^1) + A_2(0.4^1) + 10 = 0.5A_1 + 0.4A_2 + 10 = 11.3$，得到方程 (ii)：$0.5A_1 + 0.4A_2 = 1.3$（将10移项：$11.3-10=1.3$）。

  

求解联立方程 (i) 和 (ii)。由 (i) 可得 $A_2 = 3 - A_1$。代入 (ii)：

$$

0.5A_1 + 0.4(3 - A_1) = 1.3 ;\implies; 0.5A_1 + 1.2 - 0.4A_1 = 1.3 ;\implies; 0.1A_1 = 0.1 ;\implies; A_1 = 1,.

$$

将 $A_1=1$ 带回 $A_2 = 3 - A_1$，得到 $A_2 = 2$。

  

至此确定常数，得到满足初值的**特定解**：

$$

\boxed{,y_t = 1 \cdot (0.5)^t + 2 \cdot (0.4)^t + 10,.}

$$

可以检查 $t=0$ 和 $t=1$ 时上述解确实给出 $y_0=13, y_1=11.3$。并且当 $t \to \infty$ 时，由于 $0<0.5,0.4<1$，齐次项$(0.5)^t,(0.4)^t$将衰减为0，$y_t$ 趋于稳定在10，这正是特解对应的稳态水平。

  

通过该例子我们验证了四步求解法的可行性和有效性。概括而言，对于线性差分方程，我们通过将问题拆解为“**齐次部分**”和“**特解部分**”，分别求解后再组合，可以系统地得到通解并施加初值条件求出唯一解。

  

**3. 蜘蛛网模型 (The Cobweb Model)**

  

蜘蛛网模型是差分方程在经济学中的经典应用之一。它描述了某些市场中**价格与供给-需求**的动态调整过程。当供给对过去价格的反应存在滞后时，市场价格会呈现振荡收敛或发散的行为，形成类似蜘蛛网的轨迹。

  

我们考虑**随机冲击版本**的蜘蛛网模型 ：

• **需求方程：** $d_t = a - \gamma, p_t$，表示在时期$t$的商品（例如小麦）需求量 $d_t$ 随价格 $p_t$ 升高而线性降低。参数 $a, \gamma > 0$。

• **供给方程：** $s_t = b + \beta, p^__t + \varepsilon_t$，表示时期$t$的供给量 $s_t$ 取决于生产者对当期价格的预期 $p^__t$（价格预期上升会增加供给），以及随机供给冲击 $\varepsilon_t$。参数 $b, \beta > 0$，$\varepsilon_t$ 为均值为零的随机扰动项 。

• **预期形成：** 假定生产者采用适应性预期，即 _采用上期价格作为本期价格预期_：$p^*_t = p_{t-1}$ 。

• **市场均衡条件：** 每期供给等于需求：$s_t = d_t$ 。

  

将上述关系综合，可推导当期价格的差分方程。由 $s_t = d_t$ 得：

  

$$

b + \beta, p^*_t + \varepsilon_t = a - \gamma, p_t,,

$$

  

代入 $p^*_t = p_{t-1}$，并整理得到：

  

$$

\gamma, p_t + \beta, p_{t-1} = a - b - \varepsilon_t,,

$$

  

即

$$

p_t = -\frac{\beta}{\gamma}, p_{t-1} + \frac{a - b}{,\gamma,} - \frac{1}{\gamma}, \varepsilon_t,. \tag{14}

$$

  

这是价格序列${p_t}$的一个一阶线性差分方程。其中**系数** $\phi = -\beta/\gamma$，**常数项** $(a-b)/\gamma$，**随机项** $-(1/\gamma)\varepsilon_t$。下面按照前述四步法求解 (14) 式，分析价格的动态行为：

  

**第1步（齐次方程）：** 忽略常数和随机冲击，得到齐次方程 $p_t = -\frac{\beta}{\gamma} p_{t-1}$ 或等价写为

$$p_t + \frac{\beta}{\gamma} p_{t-1} = 0,.$$

其特征方程为 $r -(-\frac{\beta}{\gamma}) = r + \frac{\beta}{\gamma} = 0$，解得特征根 $r = -\frac{\beta}{\gamma}$。因此齐次通解为

$$p_t^h = A \Big(-\frac{\beta}{\gamma}\Big)^t,,$$

其中 $A$ 为任意常数 。这个齐次解对应价格在没有外生冲击情况下的自由演化：初值为$A$，每期按比例$-\beta/\gamma$变化。注意该根可能为负值，意味着价格的齐次响应会**在正负之间震荡**（价格高了一期则下一期降低，反之亦然）。

  

**第2步（特解）：** 原方程包含常数项和随机项，我们需要找一个特解 $p_t^p$ 使其满足 (14)。类似于之前AR(1)模型的解，我们可以通过**迭代法**或**待定系数法**来找稳态特解。直观地，如果没有随机冲击（$\varepsilon_t=0$）且价格最终稳定在均衡，不再变化，则在稳态有 $p_t = p_{t-1} = p^_$。将稳态值 $p^_$ 代入 (14) 的无冲击部分：

$$

p^* = -\frac{\beta}{\gamma} p^* + \frac{a - b}{\gamma},,

$$

移项整理得

$$p^* + \frac{\beta}{\gamma} p^* = \frac{a - b}{\gamma},,$$

即

$$p^* \Big(1 + \frac{\beta}{\gamma}\Big) = \frac{a - b}{\gamma},,$$

从而得到**长期均衡价格**

$$p^* = \frac{a - b}{,\gamma + \beta,},.$$

  

当存在随机冲击$\varepsilon_t$时，价格在均衡附近波动。若$\Big|\frac{\beta}{\gamma}\Big| < 1$（稍后讨论此稳定性条件），则我们可以采用类似第2节一阶方程的结果，写出 $p_t$ 关于过去冲击的无限和特解 ：

  

$$

p_t^p = \frac{a - b}{,\gamma + \beta,} ;-; \frac{1}{\gamma} \sum_{i=0}^{\infty} \Big(-\frac{\beta}{\gamma}\Big)^i, \varepsilon_{t-i},. \tag{*}

$$

  

该式由两部分组成：第一项是前述**长期均衡价格**，第二项是过去各期供给冲击对价格的累计影响。由于冲击通过供给影响价格的方向是反向的（供给正冲击导致价格下降），公式中累加项带有负号$-\frac{1}{\gamma}$。指数$\left(-\frac{\beta}{\gamma}\right)^i$表示冲击的滞后影响随着时间推移按比例$(\beta/\gamma)^i$缩小，同时正负号每滞后一期交替（因为$-$号的$i$次方体现出震荡影响）。如果 $\big|\frac{\beta}{\gamma}\big|<1$，该级数收敛，说明远期的陈旧冲击影响很小，可以用此无穷和表示特解 。

  

反之，如果 $\big|\frac{\beta}{\gamma}\big| \ge 1$，则上式发散，我们无法找到这样的稳态特解。这种情况下，价格过程不收敛于固定均值，需要将初始条件纳入考虑，用有限期的迭代表达式描述价格（即类似于(3)式的有限和，而非无穷和)。

  

**第3步（一般解）：** 将齐次解和特解相加，得到 (14) 的一般解：

$$

p_t = \frac{a - b}{,\gamma + \beta,} ;-; \frac{1}{\gamma} \sum_{i=0}^{\infty} \Big(-\frac{\beta}{\gamma}\Big)^i, \varepsilon_{t-i} ;+; A \Big(-\frac{\beta}{\gamma}\Big)^t,.

$$

对于稳定情形（$\big|\frac{\beta}{\gamma}\big|<1$），上述无穷求和有效，此时一般解由“稳态部分 + 冲击累积 + 初始响应”组成 。对于非稳定情形（$\big|\frac{\beta}{\gamma}\big|\ge1$），无穷和不收敛，我们只能将求和截断在初始时刻，并通过初始条件来刻画解，此时$A$和冲击项共同吸收了初始状态的信息。

  

**第4步（应用初始条件）：** 若给定初始价格 $p_0$，可以利用 $t=0$ 时的一般解确定常数 $A$ 。对稳定情形，通常假设系统起始于稳态（或远过去冲击累积均值有限），我们常取 **$A=0$**，即假设无额外的任意初始偏差，使价格过程围绕稳态波动（这相当于选择了特解解答）。对非稳定情形，如果 $|\beta/\gamma| \ge 1$，则必须依赖初始条件确定$A$才能得到合理解。例如，当 $\beta/\gamma = 1$（即 $\phi=-1$）时，需要一个初始冲击才能决定价格将围绕均衡价格上下交替振荡的幅度；当 $\beta/\gamma > 1$，价格将越振越远，$A$的取值决定最初偏离均衡的规模。

  

现在我们分析**蜘蛛网模型的稳定性**：模型参数决定了供需曲线的斜率比 $\beta/\gamma$，进而决定 $\phi = -\beta/\gamma$ 的绝对值大小：

• 如果 $\big|\frac{\beta}{\gamma}\big| < 1$（供给曲线相对需求曲线**较平缓**），则特征根 $r = -\frac{\beta}{\gamma}$ 满足 $|r|<1$，价格齐次项 $A r^t$ 会指数衰减至0。此时价格的波动将**收敛**于长期均衡 $p^_$_ _。具体表现为价格围绕 $p^_$ 震荡且幅度越来越小，最终趋于稳定。此情形被称为**收敛的蜘蛛网**。

• 如果 $\big|\frac{\beta}{\gamma}\big| > 1$（供给曲线**更陡峭**），则 $|r|>1$，齐次项的影响会随时间放大，导致价格波动**发散** 。也就是说，价格将越来越偏离均衡，呈现爆炸性的振荡。此为**发散的蜘蛛网**情况，现实中意味着市场不稳定，价格可能越来越极端。

• 如果 $\big|\frac{\beta}{\gamma}\big| = 1$，则 $|r|=1$，价格会**持续震荡**且幅度保持不变（$r=-1$ 会导致价格在两个值之间来回跳动，称为边际稳定或周期2振荡） 。这种情况下系统处于临界状态，轻微扰动将导致持续波动。

  

_如上图所示，一个_**_收敛的蜘蛛网_**_模型：初始价格偏离均衡，引发价格沿供需曲线交替调整，红色折线轨迹逐渐逼近均衡点（绿色）。当需求相对弹性更大（$\beta/\gamma < 1$）时，振荡幅度递减并收敛。反之，若供给更具弹性导致 $\beta/\gamma \ge 1$，价格轨迹将不会收敛。_

  

通过蜘蛛网模型，我们直观演示了差分方程求解方法在经济动力学中的应用。模型表明：**稳定性取决于差分方程特征根的模是否小于1**。在本例中，特征根为$-\beta/\gamma$。因此，当 $\beta/\gamma < 1$ 时，$|-\beta/\gamma|<1$，价格收敛；当 $\beta/\gamma \ge 1$ 时，$|-\beta/\gamma|\ge 1$，价格不收敛。

  

**4. 齐次差分方程的求解 (Solving Homogeneous Difference Equations)**

  

在上一节，我们已经多次用到**特征根法**求解齐次差分方程。现在对这一方法进行总结和推广。

  

一般地，考虑 $n$阶齐次线性差分方程：

$$

y_t - a_1 y_{t-1} - a_2 y_{t-2} - \cdots - a_n y_{t-n} = 0,.

$$

其对应的**特征方程**为：

$$

r^n - a_1 r^{,n-1} - a_2 r^{,n-2} - \cdots - a_{n-1} r - a_n = 0,.

$$

解这个$n$次代数方程，将得到$n$个特征根（记为 $r_1, r_2, \dots, r_n$）。这些根可能是**相异实根**、**重根**或者**成对共轭的复根**。根据线性差分方程理论：

• 如果得到 $n$ 个**互不相同的实根** $r_1, r_2, \dots, r_n$，则齐次通解为这些根对应指数序列的线性组合：

$$y_t^h = A_1 r_1^t + A_2 r_2^t + \cdots + A_n r_n^t,,$$

其中 $A_1, \dots, A_n$ 为任意常数 。例如前述例子中 $r_1=0.5,r_2=0.4$，通解是 $A_1 0.5^t + A_2 0.4^t$。

• 如果存在**重根**情况：假设某特征根 $r$ 的重数为 $m$（即特征方程中 $(r - \lambda)^m$ 是因子），那么除了 $r^t$ 之外，还需要乘上逐次递增的多项式因子，引入 $t$ 来获得 $m$ 个线性独立解 。具体来说，对于重根 $\lambda$（重数$m$），齐次解中应包含：

$$\lambda^t,; t,\lambda^t,; t^2 \lambda^t,; \dots,; t^{m-1} \lambda^t$$

这 $m$ 个解彼此独立，确保构成完备解空间。例如，若二阶方程特征根重复（判别式$d=0$），设 $\alpha_1 = \alpha_2 = \lambda$，则两个齐次解可取 $y_t^{h(1)} = \lambda^t$ 和 $y_t^{h(2)} = t,\lambda^t$ 。因此通解为 $y_t^h = A_1 \lambda^t + A_2,t,\lambda^t$。

• 如果存在**共轭复根**：设一对复根为 $\lambda_{1,2} = \mu \pm i \nu$（其中 $\mu,\nu \in \mathbb{R}$，且 $\nu \neq 0$）。可以将它们表示为极坐标形式 $\lambda_{1,2} = r e^{\pm i \theta}$，其中 $r = \sqrt{\mu^2 + \nu^2}$，$\cos \theta = \mu/r$，$\sin \theta = \nu/r$。对应的齐次解可以用实数函数表示为：

$$y_t^h = B_1 , r^t \cos(\theta t) + B_2 , r^t \sin(\theta t),,$$

或等价地写成 $y_t^h = C_1, r^t \cos(\theta t + \phi)$ 的形式（$C_1,\phi$为常数），这与前述 $\beta_1 r^t \cos(\theta t + \beta_2)$ 表达一致 。这种形式保证了解为实数。从物理意义看，$r^t$部分决定振幅的膨胀或收缩，$\cos/\sin$部分决定震荡。比如若$r<1$，则振荡幅度随时间递减。

  

**稳定性条件：** 对于齐次解 $y_t^h = \sum_{i} A_i r_i^t$ 而言，其长期行为取决于各特征根 $r_i$ 的大小。**稳定**（或平稳）要求所有特征根的绝对值均小于1 。此时 $r_i^t \to 0$ 随 $t \to \infty$，齐次项对长期影响消失，系统才会收敛于稳态。如果存在任何一个特征根 $|r_i| \ge 1$，则解不会收敛：$|r_i|=1$ 会导致持续振荡或线性增长（如 $r_i=1$ 或 $r_i=-1$ 的情形前述备注过），$|r_i|>1$ 则导致解发散。总结来说，**特征根全部位于单位圆内**是差分方程稳定性的充要条件之一 。

  

对于具体参数判别，以下是一些有用的判据（针对 $y_t = a_1 y_{t-1} + \cdots + a_n y_{t-n}$ 这样的AR($n$)方程）：

• **必要条件：** 所有系数和满足 $|a_1 + a_2 + \cdots + a_n| < 1$ 则没有特征根等于1（因为若有根$r=1$，则 $1 - (a_1 + \cdots + a_n) = 0$）。更弱地，一般要求 $a_1 + \cdots + a_n < 1$ 作为稳定的必要条件 。

• **充分条件：** 若 $|a_1| + |a_2| + \cdots + |a_n| < 1$，则根据广义边界圆判据，可以确保所有根都在单位圆内 。因为系数绝对值和<1意味着整体反馈小于1。

• **单位根检测：** 若 $a_1 + a_2 + \cdots + a_n = 1$，则 $r=1$ 是特征方程的一个根（即存在单位根），这时差分方程**非稳定**或称存在**随机游走**成分，需要差分后才能稳定 。例如，对于一阶情形$a_1=1$，我们见过其解包含线性趋势项，非平稳。

  

**小结：** 齐次方程求解关键在于求特征多项式的根，根据根的种类构造齐次通解。稳定性要求根的模小于1，判据可以辅助判断模型是否满足平稳条件。

  

**5. 特解的求法 (Finding Particular Solutions)**

  

对于非齐次差分方程（即包含驱动项 $a_0$ 或 $x_t$），在求得齐次通解后，我们还需找到一个**特解**。**寻找特解往往没有统一公式，需要根据驱动过程 ${x_t}$ 的形式运用技巧和猜测** 。常用的方法包括**裂项讨论**和**待定系数法**等，下面分情形讨论：

  

**5.1 确定性过程的特解 (Particular Solution for Deterministic Process)**

  

当驱动项 $x_t$ 是确定性（非随机）的函数时，可依据其形式猜测相应形状的特解。考虑一般的$n$阶方程：

$$

y_t = a_0 + a_1 y_{t-1} + \cdots + a_n y_{t-n} + x_t,,

$$

根据 $x_t$ 类型不同，常见情形包括：

• **情况1： $x_t$ 为常数或零**（即仅有常数项 $a_0$，没有随$t$变化的驱动项）。

如果 $1 - a_1 - \cdots - a_n \neq 0$，可以尝试**常数特解** $y_t^p = C$。将其代入，要求

$$C = a_0 + a_1 C + a_2 C + \cdots + a_n C = a_0 + (a_1 + \cdots + a_n) C,.$$

解得

$$C = \frac{a_0}{,1 - (a_1+\cdots+a_n),},,$$

只要分母不为0即可 。如果 $1 - (a_1+\cdots+a_n) = 0$，意味着特征方程有根$r=1$（存在单位根），**常数解无法找到**，此时需要尝试**线性特解** $y_t^p = C t$ 。将 $y_t^p = Ct$ 代入方程，如果成功，可以求得 $C$；若仍失败（说明存在更高重的单位根，如 $r=1$ 为重根导致 $t$ 也在齐次解空间内），则可以进一步尝试 $y_t^p = C t^2$ 甚至 $C t^k$ 直到成功为止 。例如，对于二阶方程若 $r=1$ 是双重根，则需要尝试二次多项式特解。

• **情况2： $x_t$ 为指数序列**，形如 $x_t = B \cdot \lambda^t$（这里$\lambda$可能是某常数的幂，表示指数增长/衰减）。

我们可以猜测特解也具有相同的指数形式。如尝试 $y_t^p = K \lambda^t$。将其代入方程，需要满足：

$$K \lambda^t = a_0 + a_1 K \lambda^{t-1} + a_2 K \lambda^{t-2} + \cdots + a_n K \lambda^{t-n} + B \lambda^t,.$$

将两边按 $\lambda^t$ 因子整理，可解出 $K$ 表达式。如果 $\lambda$ 不是齐次方程的根，则一般可以解得唯一的 $K$。但若**$\lambda$ 恰好是齐次特征根**，则$K\lambda^t$项会在代入后与齐次部分抵消，导致方程不成立。这种情况下，需要在猜测形式上乘以 $t$ 因子：尝试 $y_t^p = K t \lambda^t$ 。若$\lambda$是重根，可能需要乘以更高次的$t$。此外，(2)式中如果$a_1=1$且$x_t$中存在增长项，也可能需要特殊处理线性趋势 。总的原则是：当驱动项形式与齐次解形式“冲突”时，用乘以$t$的方式使其成为新的独立解候选。

• **情况3： $x_t$ 为多项式（时间趋势）**，形如 $x_t = B t^d$，其中$d$为非负整数。

我们可尝试一个$d$次的多项式特解：

$$y_t^p = C_0 + C_1 t + C_2 t^2 + \cdots + C_d t^d,.$$

将其代入原方程，并将两边按 $t$ 的幂次展开，比较同次幂系数，可以解出 $C_0, \dots, C_d$ 。同样地，如果某种低次多项式解不成功，通常是由于存在单位根或低阶多项式解被齐次项吸收，则尝试提高多项式阶数直到找到解为止。例如，$r=1$ 且为单根的情况，常数解失败需试一次项；若$r=1$为双重根，则一次项也在齐次解中，需要试二次项，等等。

  

上述方法都是利用驱动项的结构“猜”出一个特解形式，再**待定系数**求解。这需要一定技巧，但有经验可循。

  

为了更清晰，总结常见驱动类型及特解猜测：

|**驱动 $x_t$ 类型**|**特解猜测形式 $y_t^p$**|**注意事项**|
|---|---|---|
|常数 $C$ 或 0|常数 $y_t^p = C$|若 $\sum a_i = 1$（单位根），改试 $C t$ 等|
|指数 $B \lambda^t$|同形 $y_t^p = K \lambda^t$|若 $\lambda$ 是特征根，则改试 $K t \lambda^t$|
|$d$次多项式 $B t^d$|同次多项式 $y_t^p = C_0 + \cdots + C_d t^d$|若 $r=1$ 为特征根（单位根），需升高次数（乘$t$）|

**5.2 随机过程的特解 (Particular Solution for Stochastic Process)**

  

当 $x_t$ 或方程包含**随机成分**（如噪声$\varepsilon_t$）时，求解特解往往需要处理无限阶的滞后项。这里介绍两种常用方法：

• **未定系数法（方法一）：** 这是将确定性情形的待定系数思想拓展到含随机项的情况。核心思想是：**假设解为线性的、并且可以表示为时间多项式 + 随机冲击的滞后加权和** 。然后代入原方程平衡各期系数，求解未知系数。

以一阶随机方程 $y_t = a_0 + a_1 y_{t-1} + \varepsilon_t$ 为例，我们假设特解形式为：

$$y_t^p = b_0 + b_1 t + \sum_{i=0}^{\infty} \alpha_i \varepsilon_{t-i},,$$

其中 $b_0, b_1, \alpha_0, \alpha_1, \dots$ 为待定系数 。将其代入：

$$b_0 + b_1 t + \alpha_0 \varepsilon_t + \alpha_1 \varepsilon_{t-1} + \cdots = a_0 + a_1 [b_0 + b_1(t-1) + \alpha_0 \varepsilon_{t-1} + \alpha_1 \varepsilon_{t-2} + \cdots] + \varepsilon_t,.$$

将等式两边按照$\varepsilon_t, \varepsilon_{t-1}$以及纯函数项分组，并令各对应系数相等 ：

• 比较 $\varepsilon_t$ 系数：$\alpha_0$（左边）应等于 $1$（右边有 $+ \varepsilon_t$），所以 $\alpha_0 = 1$ 。

• 比较 $\varepsilon_{t-1}$ 系数：$\alpha_1$（左边）应等于 $a_1 \alpha_0$（右边$\varepsilon_{t-1}$系数来自 $a_1 \alpha_0 \varepsilon_{t-1}$），所以 $\alpha_1 = a_1 \alpha_0 = a_1$。

• 继续递推可得：$\alpha_2 = a_1 \alpha_1 = a_1^2$，一般地 $\alpha_i = a_1^i$ 对所有 $i \ge 0$。这重现了我们先前在 AR(1) 模型中得到的无限和系数序列。

• 再比较无噪声项：常数项和$t$项。常数项：$b_0$（左）须满足 $b_0 - a_0 - a_1 b_0 + a_1 b_1 = 0$（右边常数项为 $a_0 + a_1 b_0$，注意 $a_1 b_1 (t-1)$ 产生一个常数$-a_1 b_1$） 。整理得 $(1 - a_1) b_0 + a_1 b_1 = a_0$。$t$项：左边$b_1 t$，右边 $a_1 b_1 (t-1)$，比较系数得 $b_1 - a_1 b_1 = 0$，即 $(1 - a_1) b_1 = 0$ 。如果 $a_1 \neq 1$，则 $1 - a_1 \neq 0$，从而 $b_1 = 0$；把 $b_1=0$ 带入前一个常数项方程，得到 $b_0(1-a_1) = a_0$，即 $b_0 = \frac{a_0}{,1-a_1,}$ 。这样我们求得

$$y_t^p = \frac{a_0}{,1-a_1,} + \sum_{i=0}^{\infty} a_1^i \varepsilon_{t-i},,$$

这正是之前通过迭代法得到的特解 (6) 。

如果 $a_1 = 1$，上述过程中 $(1-a_1)b_1=0$不再限制$b_1$，表明线性项可以存在——这与我们知道单位根时解会出现线性漂移一致。这时需要增加线性项让方程均衡，例如 $b_1$ 可通过常数项条件确定或者保持符号表示。总之，未定系数法系统地再现了差分方程的解。

• **滞后算子法（方法二）：** 这种方法利用滞后算子把差分方程写成“代数形式”并进行代数求逆，适用于线性稳态系统。还是以 $y_t = a_0 + a_1 y_{t-1} + \varepsilon_t$ 为例，我们用滞后算子$L$表示为：

$$(1 - a_1 L) y_t = a_0 + \varepsilon_t,.$$

在$|a_1|<1$的条件下，$(1 - a_1 L)$是可逆的，我们可以将其形式上展开为几何级数的和：

$$\frac{1}{,1 - a_1 L,} = 1 + a_1 L + a_1^2 L^2 + a_1^3 L^3 + \cdots,.$$

将两边作用于 $a_0 + \varepsilon_t$，就得到：

$$y_t = (1 + a_1 L + a_1^2 L^2 + \cdots)(a_0 + \varepsilon_t),.$$

展开后：

$$y_t = a_0 (1 + a_1 + a_1^2 + \cdots) + \varepsilon_t + a_1 \varepsilon_{t-1} + a_1^2 \varepsilon_{t-2} + \cdots,.$$

由于 $1 + a_1 + a_1^2 + \cdots = \frac{1}{1-a_1}$（$|a_1|<1$保证收敛），上式正好化简为：

$$y_t = \frac{a_0}{,1-a_1,} + \sum_{i=0}^{\infty} a_1^i \varepsilon_{t-i},,$$

再次得到我们熟悉的结果。这种方法对于更高阶方程同样适用：将方程写为 $P(L) y_t = Q(L) \varepsilon_t$（$P, Q$ 为多项式），然后形式求逆 $y_t = P(L)^{-1} Q(L) \varepsilon_t$，通过**长除或级数展开**得到MA(∞)形式的解。滞后算子法在处理ARMA模型（自回归移动平均模型）的理论中非常常用，能够快速推导平稳过程的无限阶表达。

  

综合而言，未定系数法和滞后算子法是求解含随机项差分方程的两种主要技巧。两者本质等价：都是寻找满足方程的冲击响应序列，只是表达形式不同。运用这些方法，我们可以将线性差分方程转化为对冲击的加权和表示，从而更方便地分析其统计性质。

  

**6. 习题 (Homework)**

  

**练习：** 请完成教材相关章节的习题1、3、7，以巩固本讲所学概念和解题方法（下周课堂提交） 。

  

_提示：练习中可能包括将连续模型离散化为差分方程的应用，例如著名的_**_SIR模型_**_在离散时间下可视为差分方程求解的案例，请尝试将其微分方程形式转换为差分方程并分析其性质_ _。通过练习，将帮助理解差分方程在实际问题（如传染病模型、经济周期等）中的建模过程。_