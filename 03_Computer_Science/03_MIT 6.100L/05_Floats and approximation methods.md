---
aliases:
  - MIT 6.100L Lecture 05
  - 6.100L L05
  - Floats and approximation methods
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 05
---

# Lecture 05: Floats and approximation methods

> [!tip] Hint
> - 这节课的出发点是上节那个看似离谱的事实：`0.1` 加十次不等于 1。
> - 老师先回顾十进制整数转二进制，再问“那小数怎么办”，整个 lecture 的推进是从表示问题走到算法问题。
> - `3/8` 能精确写成有限二进制，而 `1/10` 不行；这解释了为什么 float 有时不能精确代表程序想表达的实数。
> - float 不是“近似实数”这个哲学句子而已，老师真正在讲的是：机器只能用有限 bits 近似无限展开。
> - approximation method 不是猜-and-check 的重复，而是把 candidate 从整数枚举换成了小步长浮点枚举。
> - epsilon 决定什么叫 close enough，increment 决定你走得多快，两者不是一回事。
> - 54321 的平方根例子故意让程序很慢，是为了让你真正感到 fixed increment 的代价。
> - 近似算法的失败不是 bug，而是算法结构本身必须显式处理的一个出口。
> - 本讲的真正主线是：`==` 会确定地比较两个已存值；当表示与运算舍入使它们偏离想比较的数学量时，程序需要与任务相匹配的“足够接近”标准，并保留明确失败分支。
> - 这节课其实在为 bisection search 做铺垫：我们已经知道小步子能逼近，但也已经看到它太慢。
> <!-- bilingual-en:start -->
> - The lecture begins from the previous lecture's unsettling observation that adding `0.1` ten times does not produce a value exactly equal to 1.
> - The instructor reviews decimal-to-binary conversion for integers and then asks how fractions are represented, moving from a representation problem to an algorithmic one.
> - `3/8` has a finite exact binary expansion, whereas `1/10` does not. That is the key divide behind floating-point approximation.
> - Saying that a float “approximates a real number” is not merely philosophical: finite bits must approximate a potentially infinite expansion.
> - The approximation method modifies guess-and-check by enumerating floating-point candidates at a small fixed increment instead of testing integers.
> - `epsilon` defines what counts as close enough; `increment` defines the step size and therefore the search speed. They are not interchangeable.
> - The deliberately slow search for the square root of 54321 makes the cost of a fixed increment tangible.
> - Failure of the approximation method is not an implementation bug, but an outcome the algorithm must represent explicitly.
> - `==` deterministically compares two stored values. When representation and arithmetic rounding make those values differ from the mathematical quantities of interest, the program needs a task-appropriate closeness criterion and an explicit failure branch.
> - The lecture prepares bisection: small steps can approximate the answer, but they may do so far too slowly.
> <!-- bilingual-en:end -->

> [!links] 本讲知识入口
> [[枚举验证]] → [[固定步长求根]] → [[固定步长漏根]] → [[数值迭代停止条件]]；整条关系见 [[数值求根.canvas|数值求根总图]]。二分、Newton 与混合方法属于后续讲次或理论扩展。
> <!-- bilingual-en:start -->
> [[枚举验证|Exhaustive enumeration]] → [[固定步长求根|fixed-step root search]] → [[固定步长漏根|fixed-step grid miss]] → [[数值迭代停止条件|numerical stopping criteria]]. See [[数值求根.canvas|Numerical Root-Finding]] for the complete relationship; bisection, Newton, and hybrid methods belong to later lectures or theoretical extensions.
> <!-- bilingual-en:end -->

## Lecture flow

### 1. 开场先回到那个令人不安的例子：`0.1` 累加十次不等于 1
<!-- bilingual-en:start -->
*1. Returning to the Unsettling Example: Ten Additions of `0.1` Do Not Equal 1 Exactly*
<!-- bilingual-en:end -->
Lecture 5 一开始就回到上一讲埋下的问题：
<!-- bilingual-en:start -->
Lecture 5 opens by returning to a question left by the previous lecture:
<!-- bilingual-en:end -->

```python
x = 0
for i in range(10):
    x += 0.1
print(x == 1)
print(x, '==', 10*0.1)
```

程序给出的不是你以为的 `True`，而是 `False`，因为最后的 `x` 实际上会变成一串非常接近 1、但不完全等于 1 的数。
<!-- bilingual-en:start -->
The comparison is `False`, because the final value of `x` is extremely close to 1 without being exactly 1.
<!-- bilingual-en:end -->

老师把这当作整讲的动机：

- `==` 对两个已存浮点值的比较是确定的
- 但表示与运算舍入后，这两个值未必仍对应数学上想比较的量
- 那么数值程序应该用什么标准判断结果？
<!-- bilingual-en:start -->
This motivates the lecture without treating equality as random or unstable: `==` compares the stored values deterministically, but representation and arithmetic rounding may leave those values different from the mathematical quantities the programmer intended to compare. Numerical code therefore needs an appropriate success criterion.
<!-- bilingual-en:end -->

所以这一讲并不是随便讲讲机器底层，而是在回答一个非常直接的问题：  
为什么这种事会发生？
<!-- bilingual-en:start -->
The discussion of machine representation answers that immediate question rather than serving as unrelated low-level detail.
<!-- bilingual-en:end -->

### 2. 先回顾整数转二进制算法，因为接下来要谈“小数怎么存”
<!-- bilingual-en:start -->
*2. Reviewing Integer-to-Binary Conversion Before Asking How Fractions Are Stored*
<!-- bilingual-en:end -->
老师先快速回顾上一讲的整数转 binary algorithm。
<!-- bilingual-en:start -->
The instructor briefly reviews the preceding integer-to-binary algorithm.
<!-- bilingual-en:end -->

回顾它的目的不是重复内容，而是建立对照：

- 整数部分有一套很干净的表示方法
- 那么 fraction 部分能不能也找到同样干净的 recipe？
<!-- bilingual-en:start -->
The purpose is comparison: integers have a clean representation procedure, so can fractions be given an equally clean one?
<!-- bilingual-en:end -->

这一转场很自然。因为一旦你接受“计算机最终都用 bits 表示数”，那么接下来一定会问：

- 整数怎么用 bits 表示？
- 小数怎么用 bits 表示？
<!-- bilingual-en:start -->
Once all numbers are understood to be stored as bits, the natural questions are how bits represent integers and how they represent fractions.
<!-- bilingual-en:end -->

### 3. 十进制 fraction 和二进制 fraction 的类比
<!-- bilingual-en:start -->
*3. The Analogy Between Decimal and Binary Fractions*
<!-- bilingual-en:end -->
老师随后写出十进制小数的意义：
<!-- bilingual-en:start -->
The instructor first expands a decimal fraction:
<!-- bilingual-en:end -->

- `0.abc` 在十进制里表示  
  `a*10^-1 + b*10^-2 + c*10^-3 + ...`

然后把这个思路平移到二进制：

- `0.abc` 在二进制里表示  
  `a*2^-1 + b*2^-2 + c*2^-3 + ...`
<!-- bilingual-en:start -->
The same positional idea transfers to base two: decimal places carry powers of 10, while binary places carry powers of 2.
<!-- bilingual-en:end -->

所以如果你想把一个十进制 fraction 变成二进制 fraction，本质是在问：

- 能否找到一串 `0` / `1`
- 让这些 `2` 的负幂加起来等于原来的值
<!-- bilingual-en:start -->
Converting a decimal fraction therefore means finding a sequence of `0` and `1` coefficients whose negative powers of 2 sum to the original value.
<!-- bilingual-en:end -->

### 4. `3/8` 为什么可以精确表示，而 `1/10` 不行
<!-- bilingual-en:start -->
*4. Why `3/8` Has an Exact Binary Representation but `1/10` Does Not*
<!-- bilingual-en:end -->
接着老师给了一个关键例子：`3/8 = 0.375`。
<!-- bilingual-en:start -->
The key example is `3/8 = 0.375`.
<!-- bilingual-en:end -->

思路是：

- 如果能乘上某个 `2**p` 变成整数
- 就先把它转成整数的二进制表示
- 再把小数点挪回去
<!-- bilingual-en:start -->
If multiplying the fraction by some `2**p` produces an integer, convert that integer to binary and then restore the binary point.
<!-- bilingual-en:end -->

例如：

- `0.375 * 2**3 = 3`
- `3` 的二进制是 `11`
- 所以 `0.375` 的二进制就是 `0.011`
<!-- bilingual-en:start -->
Here, `0.375 * 2**3 = 3`, the binary form of `3` is `11`, and the binary fraction is consequently `0.011`.
<!-- bilingual-en:end -->

这说明精确有理数 $3/8$ 是一个 “power-of-two friendly” 的 fraction。对数学上的精确 $1/10$，不存在有限的 $p$ 使 $2^p/10$ 成为整数。

这里必须先固定讨论对象：数学上的 $1/10$ 没有有限二进制展开；Python 读入字面量 `0.1` 后，存下的却是附近一个分母为 $2$ 的幂的有理数。下一节专门区分这两件事。
<!-- bilingual-en:start -->
The exact rational number $3/8$ is friendly to powers of two. For the exact mathematical number $1/10$, no finite $p$ makes $2^p/10$ an integer. The object must nevertheless be stated carefully: mathematical $1/10$ has no finite binary expansion, whereas the Python literal `0.1` has already been rounded to a nearby rational number whose denominator is a power of two. The next section separates these two facts.
<!-- bilingual-en:end -->

> [!note]
> 这就是本讲最重要的分界线：  
> 有些十进制 fraction 在二进制里是有限展开；  
> 有些则必须无限展开。
> <!-- bilingual-en:start -->
> This is the lecture's decisive distinction: some decimal fractions have finite binary expansions, while others require infinitely many binary digits.
> <!-- bilingual-en:end -->

一旦需要无限展开，机器就只能截断或舍入，于是误差就出现了。
<!-- bilingual-en:start -->
Because a machine cannot store an infinite expansion, it must truncate or round it, introducing representation error.
<!-- bilingual-en:end -->

### 5. float 的本质：用有限 bits 近似潜在无限展开
<!-- bilingual-en:start -->
*5. The Nature of a Float: Finite Bits Approximate a Potentially Infinite Expansion*
<!-- bilingual-en:end -->
老师把这个结论收成一句更一般的话：
<!-- bilingual-en:start -->
The instructor generalizes the conclusion:
<!-- bilingual-en:end -->

- integers 在 binary 里比较直接
- real numbers/fractions 则可能需要无限多位
- 但机器只能存有限 bits
<!-- bilingual-en:start -->
Integers have relatively direct binary representations; real-number fractions may require infinitely many places, whereas a machine stores only finitely many bits.
<!-- bilingual-en:end -->

所以每个有限 float 都精确代表一个二进有理数；当程序想表达的数学实数不在这组有限可表示值中时，存下来的 float 才是那个目标实数的舍入近似。
<!-- bilingual-en:start -->
Every finite float therefore represents one dyadic rational exactly. When the intended mathematical real number is outside the finite set of representable values, the stored float is a rounded approximation to that target real number.
<!-- bilingual-en:end -->

接着她介绍了 floating point number 的一个简单抽象：

- significand / significant digits
- exponent
<!-- bilingual-en:start -->
The lecture abstracts a floating-point number into a significand and an exponent—a finite set of significant digits multiplied by a power of two.
<!-- bilingual-en:end -->

也就是把数表示成某种 “有效数字 × 2 的幂” 的形式。  
这样做的目的不是让你去背 IEEE 标准，而是让你理解：

- float 其实是一个有限位宽的工程折中
- 位数有限意味着有些目标值必须舍入，不意味着每个已存 float 都“不精确”
<!-- bilingual-en:start -->
The goal is not to memorize an IEEE standard, but to understand the engineering compromise: a fixed width necessarily entails rounding for some target values, not that every stored float is inexact.
<!-- bilingual-en:end -->

### 6. 用 fraction-to-binary 代码时，先分清数学上的 $1/10$ 与已存的 `float(0.1)`
<!-- bilingual-en:start -->
*6. Fraction-to-Binary Code Must Distinguish Mathematical $1/10$ from the Stored `float(0.1)`*
<!-- bilingual-en:end -->
老师展示了一段尝试把输入 fraction 写成有限二进制小数的代码。
<!-- bilingual-en:start -->
The instructor shows code that attempts to express an input fraction as a finite binary fraction.
<!-- bilingual-en:end -->

核心结构大致是：

```python
p = 0
while ((2**p) * x) % 1 != 0:
    p += 1
```

这段代码在做的事情是：

- 不断尝试把 `x` 乘上更大的 `2**p`
- 看看能不能变成整数
<!-- bilingual-en:start -->
It repeatedly multiplies `x` by larger powers `2**p` and checks whether the result has become an integer.
<!-- bilingual-en:end -->

对能被精确存储的 `0.625=5/8`，循环在 $p=3$ 时停止。数学上的精确 $1/10$ 确实没有这样的有限 $p$；但把 Python 字面量 `0.1` 直接代入这段循环时，情况不同：输入阶段已经先把 $1/10$ 舍入成

```python
0.1.as_integer_ratio()
# (3602879701896397, 36028797018963968)
```

分母 $36028797018963968=2^{55}$，所以循环会在 $p=55$ 停止。它此时求出的，是**已存浮点近似值**的精确有限二进制表示，不是数学上精确 $1/10$ 的有限表示。
<!-- bilingual-en:start -->
For the exactly stored value `0.625 = 5/8`, the loop stops at $p=3$. The exact mathematical number $1/10$ has no such finite $p$. Passing the Python literal `0.1` to the loop is different, however, because input conversion has already rounded $1/10$ to `3602879701896397 / 2**55`. The loop therefore stops at $p=55$ and recovers the finite binary representation of the stored approximation, not a finite representation of exact mathematical $1/10$.
<!-- bilingual-en:end -->

课件在这里还特意提醒：推导“某个十进制 fraction 能否有限转成二进制”时，曾暂时假定输入值本身完全准确，而不是已经被 Python 近似。代码一旦运行在 float 上，就必须把这层输入舍入算进去。因此真正应记住的是：

- 先说明讨论的是数学实数，还是机器已经存下的浮点数；
- 精确 $1/10$ 需要无限二进制展开；
- 已存的 Python `0.1` 是有限 dyadic rational（二进有理数），但它不等于精确 $1/10$。
<!-- bilingual-en:start -->
The slides explicitly warn that the mathematical argument temporarily assumes the decimal input itself is exact rather than already approximated on entry. Code running on a float must account for that first rounding step. The lasting distinction is therefore between the mathematical real number and the machine value already stored: exact $1/10$ needs infinitely many binary digits, whereas Python's stored `0.1` is a finite dyadic rational that is not exactly $1/10$.
<!-- bilingual-en:end -->

### 7. 目标实数未必能由 float 精确表示，数值算法不能盲目依赖 `==`
<!-- bilingual-en:start -->
*7. When the Intended Real Is Not Exactly Representable as a Float, Numerical Algorithms Cannot Rely Blindly on `==`*
<!-- bilingual-en:end -->
讲完表示问题后，整讲进入第二部分：approximation method。
<!-- bilingual-en:start -->
After representation, the lecture turns to an approximation method.
<!-- bilingual-en:end -->

这里的出发点很清楚：

- 对 perfect square，我们可以用整数 guess-and-check
- 但大多数平方根不是整数
- 而且反复浮点运算得到的已存值，未必与数学上预期的目标值完全相同
<!-- bilingual-en:start -->
Integer guess-and-check works for perfect squares, but most square roots are nonintegers, and stored results after floating-point operations need not equal the ideal mathematical target exactly.
<!-- bilingual-en:end -->

所以新的问题变成：

> 我们不再追求 “guess**2 恰好等于 x”，  
> 而是追求 “guess**2 离 x 足够近”。
> <!-- bilingual-en:start -->
> The goal is no longer `guess**2` exactly equal to `x`, but `guess**2` sufficiently close to `x`.
> <!-- bilingual-en:end -->

这就是 `epsilon` 出场的原因。
<!-- bilingual-en:start -->
That change in success criterion introduces `epsilon`.
<!-- bilingual-en:end -->

### 8. approximation method：把整数枚举换成小步长浮点枚举
<!-- bilingual-en:start -->
*8. The Approximation Method: Replacing Integer Enumeration with Small Floating-Point Steps*
<!-- bilingual-en:end -->
老师给出的基础版本大致是：
<!-- bilingual-en:start -->
The basic version is approximately:
<!-- bilingual-en:end -->

```python
x = 36
epsilon = 0.01
guess = 0.0
increment = 0.0001
while abs(guess**2 - x) >= epsilon:
    guess += increment
```

这个算法和上一讲 guess-and-check 非常像，只是候选空间变了：

- 以前是 `0, 1, 2, 3, ...`
- 现在是 `0.0, 0.0001, 0.0002, 0.0003, ...`
<!-- bilingual-en:start -->
The structure still resembles guess-and-check, but the candidates change from integers `0, 1, 2, 3, ...` to finely spaced values `0.0, 0.0001, 0.0002, 0.0003, ...`.
<!-- bilingual-en:end -->

仍然是系统枚举，只不过步长更细。
<!-- bilingual-en:start -->
It remains systematic enumeration, now with a much smaller step.
<!-- bilingual-en:end -->

> [!example]
> 这个算法的四个核心量要分清：
> - `x`：目标值
> - `guess`：当前候选
> - `increment`：每次往前走多大一步
> - `epsilon`：多近才算接受
> <!-- bilingual-en:start -->
> Keep four quantities distinct:
> - `x` is the target.
> - `guess` is the current candidate.
> - `increment` is the distance advanced on each step.
> - `epsilon` defines how close the candidate must be to count as acceptable.
> <!-- bilingual-en:end -->

### 9. 第一次真正感到算法“太慢”：54321 的平方根
<!-- bilingual-en:start -->
*9. Feeling Algorithmic Slowness Directly: The Square Root of 54321*
<!-- bilingual-en:end -->
老师没有停在小数字上，而是故意让大家看一个大输入，比如 `x = 54321`。
<!-- bilingual-en:start -->
The instructor deliberately moves from small examples to a large input such as `x = 54321`.
<!-- bilingual-en:end -->

这时即便 `increment = 0.0001`，程序也会变得非常慢。  
她甚至在代码里加了周期性打印当前 guess 的语句，让你看到程序还在非常机械地一点点往前挪。
<!-- bilingual-en:start -->
Even with `increment = 0.0001`, the program is extremely slow. Periodic printing makes its mechanical, tiny advances visible.
<!-- bilingual-en:end -->

这一段在课堂上的作用不是“让程序跑出来”，而是让你切身体会到：

- 参数合适且网格命中接受区域时，fixed increment 可以给出近似
- 但它可能极慢，也可能完全跨过接受区域
- 目标值大、步长细或容差严时，这两项代价尤其明显
<!-- bilingual-en:start -->
The point is experiential: when the chosen grid intersects the acceptance region, fixed increments can approximate the answer but may require an enormous number of steps. A coarse grid can also miss that region entirely.
<!-- bilingual-en:end -->

这也是 Lecture 6 要讲 bisection search 的直接铺垫。
<!-- bilingual-en:start -->
That limitation directly motivates bisection in Lecture 6.
<!-- bilingual-en:end -->

### 10. approximation method 的失败不是偶发，而是结构上必须承认的结果
<!-- bilingual-en:start -->
*10. Approximation Can Fail, and the Algorithm Must Represent That Outcome*
<!-- bilingual-en:end -->
老师随后又指出一个更 subtle 的问题：
<!-- bilingual-en:start -->
The instructor then identifies a subtler issue:
<!-- bilingual-en:end -->

- 不是每次 increment 都能刚好踩进 epsilon neighborhood
- 也可能你不断往前走，结果 `guess**2` 已经超过 `x`，却仍然没有达到要求精度
<!-- bilingual-en:start -->
A chosen increment need not land inside the `epsilon` neighborhood. The search can pass beyond `x` before ever meeting the accuracy requirement.
<!-- bilingual-en:end -->

因此一个更完整的版本会写成：

```python
while abs(guess**2 - x) >= epsilon and guess**2 <= x:
    guess += increment
    num_guesses += 1

if abs(guess**2 - x) >= epsilon:
    print(f"Failed on square root of {x}")
else:
    print(f"{guess} is close to square root of {x}")
```

这段极其重要，因为它把 “失败分支” 正式引进了数值算法：
<!-- bilingual-en:start -->
The completed version therefore introduces an explicit failure branch into the numerical algorithm:
<!-- bilingual-en:end -->

- 近似法不是永远成功
- 如果你没有写失败分支，程序就会把一个不可信的 guess 冒充成答案
<!-- bilingual-en:start -->
- The approximation method is not guaranteed to succeed for every parameter choice.
- Without a failure branch, an untrustworthy guess may be presented as though it were an answer.
<!-- bilingual-en:end -->

> [!warning]
> 很多初学者会把 “程序给出了一个数” 和 “程序给出了可信答案” 混为一谈。  
> 这一讲就是在帮你建立这个区别。
> <!-- bilingual-en:start -->
> Beginners often confuse “the program printed a number” with “the program produced a trustworthy answer.” This lecture establishes the distinction.
> <!-- bilingual-en:end -->

### 11. epsilon 与 increment 控制的是不同维度
<!-- bilingual-en:start -->
*11. `epsilon` and `increment` Control Different Dimensions*
<!-- bilingual-en:end -->
课堂里虽然没有把这两个量做成单独术语表，但你应该自己在笔记里分开记：
<!-- bilingual-en:start -->
Even without a formal terminology table, the two parameters should remain separate:
<!-- bilingual-en:end -->

- `epsilon` 控制接受标准
- `increment` 控制搜索粒度和速度
<!-- bilingual-en:start -->
- `epsilon` controls the acceptance criterion.
- `increment` controls search granularity and speed.
<!-- bilingual-en:end -->

两者之间不是替代关系。
<!-- bilingual-en:start -->
They are not substitutes for one another.
<!-- bilingual-en:end -->

如果：

- `epsilon` 很小，但 `increment` 很粗

你可能根本踩不到可接受区间。  
如果：

- `increment` 很细

你可能终究能找到更好的近似，但速度会非常慢。
<!-- bilingual-en:start -->
With a small `epsilon` and a coarse `increment`, the search may never land in the acceptable region. With a fine increment, it may find a better approximation eventually, but only after a very long run.
<!-- bilingual-en:end -->

所以算法设计不是只盯一个参数，而是平衡：

- 精度
- 运行时间
- 失败处理
<!-- bilingual-en:start -->
Algorithm design must balance accuracy, running time, and failure handling rather than tune only one parameter.
<!-- bilingual-en:end -->

### 12. 这节课的真正结论：一旦不再追求精确相等，程序结构就要一起变化
<!-- bilingual-en:start -->
*12. Final Lesson: Abandoning Exact Equality Changes the Program's Structure*
<!-- bilingual-en:end -->
Lecture 5 到最后，真正发生变化的不是某一段代码，而是你的判断标准。
<!-- bilingual-en:start -->
The deepest change in Lecture 5 is not a particular code fragment but the standard by which a numerical result is judged.
<!-- bilingual-en:end -->

从现在起，数值程序里经常要问：

- 我能不能精确表示目标值？
- 当前任务要的是已存值精确相等，还是数学量在容差内足够接近？
- 我是不是应该改用 closeness test？
- 如果近似法失败了，程序如何显式说明？
<!-- bilingual-en:start -->
Numerical code must ask whether the target is exactly representable, whether the task requires equality of stored values or closeness of mathematical quantities, and how an approximation failure will be reported explicitly.
<!-- bilingual-en:end -->

所以这一讲并不是单纯讲 float 或近似算法，而是在重写你对“数值正确性”的直觉。
<!-- bilingual-en:start -->
The lecture therefore reshapes the notion of numerical correctness rather than merely introducing floats and one approximation algorithm.
<!-- bilingual-en:end -->

## Exercise log
> [!example] Finger exercise 05
> 官方题目要求：给定字符串 `my_str`，打印出其中偶数 index 的字符。
> <!-- bilingual-en:start -->
> Given the string `my_str`, the official exercise prints the characters at even indexes.
> <!-- bilingual-en:end -->
>
> ```python
> s = ''
> for i in range(0, len(my_str), 2):
>     s += my_str[i]
> print(s)
> ```
>
> 这题和 lecture 标题里的 `float` 看起来不完全一致，但它实际对应本讲前半段仍在使用的一个基础模式：
> - 你必须清楚 index 的步长含义
> - 你必须会构造一个新字符串来累计结果
> <!-- bilingual-en:start -->
> Although it appears separate from the lecture's floating-point title, it practices a foundational pattern still used in the first half:
> - Understand the step size in an index range.
> - Construct a new string by accumulating characters.
> <!-- bilingual-en:end -->
>
> 它也顺手提醒你：即使课程主线已经转向数值方法，字符级循环和序列处理仍然是需要持续熟练的底层动作。
> <!-- bilingual-en:start -->
> It also reminds you that character-level loops and sequence processing remain essential low-level skills even as the course turns toward numerical methods.
> <!-- bilingual-en:end -->

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec05.pdf|Lecture 05 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec05_code.py|Lecture 05 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex05_sol.pdf|Lecture 05 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec05_transcript.pdf|Lecture 05 transcript]]
- Recitation: none attached to this lecture week
- PS 1 halfway hand-in due: [[MIT 6.100L-problem-sets/mit6_100l_ps1.pdf|PS1 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps1_code.zip|PS1 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 3.2-3.3)

## Review checklist
- [ ] 我能解释为什么 `0.1` 连加十次可能不等于 1。
- [ ] 我能说明 `3/8` 和 `1/10` 在二进制表示上的关键差别。
- [ ] 我能用自己的话解释 float 为什么只能近似表示某些实数。
- [ ] 我能说出 approximation method 和整数 guess-and-check 的结构相同点与不同点。
- [ ] 我能区分 `epsilon` 和 `increment` 分别控制什么。
- [ ] 我能解释为什么 fixed increment 方法在大输入上会非常慢。
- [ ] 我能说明为什么近似算法必须有失败分支。
- [ ] 我能判断什么时候应该用 `abs(guess**2 - x) < epsilon` 这类 close-enough 测试。
- [ ] 我能解释为什么“程序输出了一个数”不等于“程序找到了可信答案”。
- [ ] 我能把本讲和下一讲连起来：为什么看完 approximation method 之后，自然会想找更快的搜索方式。
<!-- bilingual-en:start -->
- [ ] I can explain why adding `0.1` ten times may not produce a value equal to 1.
- [ ] I can state the key difference between the binary representations of `3/8` and `1/10`.
- [ ] I can explain in my own words why floats only approximate some real numbers.
- [ ] I can compare the structure of fixed-increment approximation with integer guess-and-check.
- [ ] I can distinguish what `epsilon` and `increment` control.
- [ ] I can explain why fixed-increment search becomes extremely slow on large inputs.
- [ ] I can explain why an approximation algorithm needs a failure branch.
- [ ] I can decide when a close-enough test such as `abs(guess**2 - x) < epsilon` is appropriate.
- [ ] I can explain why numerical output is not automatically a trustworthy answer.
- [ ] I can connect this lecture to the next by explaining why fixed-increment approximation motivates a faster search strategy.
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 没有先区分“已存的二进有理数”和“想表达的数学实数”，也没有判断任务是否真的要求已存值精确相等，就大量使用 `==`；具体边界见 [[浮点精确等号]]。
> - 只关心 epsilon，不关心步长和边界条件，导致程序极慢或失败。
> - 近似算法没有失败分支，最后得到一个看起来像答案但其实不可信的结果。
> <!-- bilingual-en:start -->
> - Using `==` heavily without distinguishing stored dyadic values from the intended mathematical quantities or asking whether equality of stored values is actually required; see [[浮点精确等号]].
> - Tuning only `epsilon` while ignoring the increment and boundary conditions, making the program excessively slow or unsuccessful.
> - Omitting a failure branch and presenting a plausible-looking but untrustworthy approximation as an answer.
> <!-- bilingual-en:end -->
