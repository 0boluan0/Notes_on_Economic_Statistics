---
aliases:
  - MIT 6.100L Lecture 23
  - 6.100L L23
  - Complexity Classes Examples
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 23
---

# Lecture 23: Complexity Classes Examples

连续阅读：[[21_算法成本与渐近分析]] · 关系图：[[渐近记号与算法复杂度.canvas]]

本讲的分类先采用[[算法成本模型|单位成本模型]]：指定一次算术、比较和访问为常数成本，并逐题定义[[输入规模]]。若改按任意精度整数的位运算计费，下面“线性、对数”等操作计数不能原样当成 Python 整个函数的实际时间界。
<!-- bilingual-en:start -->
The classifications below first use a [[算法成本模型|unit-cost model]]: specified arithmetic, comparison, and access operations have constant cost, and [[输入规模|input size]] is defined for each problem. Under an arbitrary-precision bit-cost model, these linear or logarithmic operation counts are not automatically bounds on the full Python runtime.
<!-- bilingual-en:end -->

> [!tip] Hint
> - 这节课一开始先把上节 Theta 的几个原则重说一遍：定义输入规模、抓 dominant term、忽略常数。
> - 然后课程用大量代码例子把常见复杂度类排开，而不是只给公式表。
> - constant、linear、quadratic、exponential、logarithmic 各自都配了具体函数。
> - 真正难点不是会背 `Theta(n^2)`，而是能从代码里找出哪些部分依赖输入规模。
> - 同样是 linear，`Theta(a)`、`Theta(len(s))`、`Theta(n_months)` 里的 `n` 各不一样。
> - 这讲还会把 searching algorithms 拉回来，比较 unsorted/sorted/bisect 的不同代价。
> - `bisect_search1` 和 `bisect_search2` 之所以都重要，是因为一个暴露切片复制成本，一个暴露索引递归思路。
> - 老师一直在提醒：复杂度不是只看循环层数，还要看循环边界、递归树和辅助操作。
> - 这节课本质上是在做“从代码到 Theta”的翻译训练。
> - 听完这节课，你应该能独立给很多小函数判复杂度，而不只是认出几个模板。
> <!-- bilingual-en:start -->
> - The lecture begins by reviewing the main Theta principles from the previous class: define the input size, identify the dominant term, and ignore constants.
> - It then uses many code examples to lay out the standard complexity classes instead of merely presenting a table of formulas.
> - Each category—constant, linear, quadratic, exponential, logarithmic—is paired with specific functions.
> - The real challenge is not memorizing `Theta(n^2)`, but identifying which parts of the code depend on input size.
> - Even when the growth is linear, `Theta(a)`, `Theta(len(s))`, and `Theta(n_months)` refer to different measures of input size.
> - The lecture returns to search algorithms and compares unsorted linear search, sorted linear search, and binary search.
> - Both `bisect_search1` and `bisect_search2` matter: the first exposes the cost of copying slices, while the second shows an index-based recursive approach.
> - The instructor repeatedly emphasizes that complexity analysis requires more than counting loop levels: loop bounds, recursion trees, and auxiliary operations also matter.
> - In essence, this lecture trains you to translate code into Theta notation.
> - After this lecture, you should be able to independently determine the complexity of many small functions, not just recognize a few templates.
> <!-- bilingual-en:end -->

## Lecture flow

### 1. 先把上节规则重新说一遍
<!-- bilingual-en:start -->
*1. Reviewing the Previous Lecture's Rules*
<!-- bilingual-en:end -->
Lecture 23 开场先回顾上节最后几条最重要的分析原则：
<!-- bilingual-en:start -->
Lecture 23 begins with a review of the most important analysis principles from last lecture:
<!-- bilingual-en:end -->

- 先定义输入规模
- 只关心增长趋势
- 抓 dominant term
- 丢掉加法常数和乘法常数
<!-- bilingual-en:start -->
- Define the input size first.
- Focus on the growth rate.
- Identify the dominant term.
- Ignore additive and multiplicative constants.
<!-- bilingual-en:end -->

老师这样做很合理，因为本讲几乎全是在这些规则上做实战。
<!-- bilingual-en:start -->
The review is useful because almost the entire lecture consists of hands-on practice with these rules.
<!-- bilingual-en:end -->

### 2. Constant class：最快速的是“不随输入规模增长”
<!-- bilingual-en:start -->
*2. Constant Time: Work Does Not Grow with Input Size*
<!-- bilingual-en:end -->
老师先从最简单的常数类开始。
<!-- bilingual-en:start -->
The instructor starts with the simplest constant-time class.
<!-- bilingual-en:end -->

典型例子：
<!-- bilingual-en:start -->
Typical examples:
<!-- bilingual-en:end -->

```python
def add(x, y):
    return x + y

def convert_to_km(m):
    return m * 1.609
```

关键点在于：
<!-- bilingual-en:start -->
The key point is:
<!-- bilingual-en:end -->

- 无论输入值本身是大是小
- 执行步骤数大致不变
<!-- bilingual-en:start -->
- Whether the input value itself is large or small, the number of execution steps remains roughly the same.
<!-- bilingual-en:end -->

所以它们属于 `Theta(1)`。
<!-- bilingual-en:start -->
They therefore run in `Theta(1)` time.
<!-- bilingual-en:end -->

这里固定的是基本操作次数。若 `x,y` 是位数不断增长的 Python 整数，`x+y` 并非无条件常数时间；若按固定宽度数值建模，则还需把溢出或数值范围作为实现约束。
<!-- bilingual-en:start -->
The primitive-operation count is constant. If `x,y` are Python integers of increasing bit length, `x+y` is not unconditionally constant-time. A fixed-width model instead needs its numerical range and overflow behavior stated as implementation constraints.
<!-- bilingual-en:end -->

### 3. Linear class：输入规模增大一倍，工作量也大致跟着增一倍
<!-- bilingual-en:start -->
*3. Linear Time: Doubling Input Size Roughly Doubles the Work*
<!-- bilingual-en:end -->
接下来老师列出一组线性例子：
<!-- bilingual-en:start -->
Next, the instructor lists a group of linear examples:
<!-- bilingual-en:end -->

- `mul(x, y)` 对 `y` 来说是线性的
- `add_digits(s)` 对字符串长度线性
- `fact_iter(a)` 对 `a` 线性
- `fact_recur(x)` 对 `x` 线性
- `compound(..., n_months)` 对月份数线性
- `fib_iter(n)` 对 `n` 线性
<!-- bilingual-en:start -->
- `mul(x, y)` is linear in `y`.
- `add_digits(s)` is linear in the length of `s`.
- `fact_iter(a)` is linear in `a`.
- `fact_recur(x)` is linear in `x`.
- `compound(..., n_months)` is linear in the number of months.
- `fib_iter(n)` is linear in `n`.
<!-- bilingual-en:end -->

这组例子特别重要，因为它强调：
<!-- bilingual-en:start -->
This set of examples is particularly important because it emphasizes:
<!-- bilingual-en:end -->

- “线性”不是指所有参数都线性
- 而是指相对某个输入规模维度线性
<!-- bilingual-en:start -->
- Calling a function linear does not mean that its runtime is linear in every parameter.
- It means that runtime grows linearly with a specified measure of input size.
<!-- bilingual-en:end -->

> [!note]
> 复杂度符号里的变量不是固定叫 `n` 就完事，必须和具体输入含义对应起来。
> <!-- bilingual-en:start -->
> The variable in a complexity expression is not automatically `n`; it must correspond to the input-size measure being analyzed.
> <!-- bilingual-en:end -->

### 4. 同样是线性，问题规模定义却可以完全不同
<!-- bilingual-en:start -->
*4. Linear Time Can Refer to Different Measures of Input Size*
<!-- bilingual-en:end -->
老师在这一部分一直逼你说清：
<!-- bilingual-en:start -->
The instructor repeatedly asks you to clarify:
<!-- bilingual-en:end -->

- `Theta(y)`
- `Theta(len(s))`
- `Theta(n_months)`

为什么这些看起来都像 linear，但不能混写？
<!-- bilingual-en:start -->
Why are all these examples linear, yet not interchangeable?
<!-- bilingual-en:end -->

因为：
<!-- bilingual-en:start -->
Because:
<!-- bilingual-en:end -->

- 输入对象不同
- 增长维度不同
- 所以分析变量必须明说
<!-- bilingual-en:start -->
- The input objects differ.
- The growing dimensions differ.
- The variable used in the analysis must therefore be stated explicitly.
<!-- bilingual-en:end -->

这一步是本讲最重要的习惯训练之一。
<!-- bilingual-en:start -->
This is one of the most important analytical habits developed in this lecture.
<!-- bilingual-en:end -->

### 5. Polynomial / Quadratic：嵌套扫描开始出现
<!-- bilingual-en:start -->
*5. Polynomial and Quadratic Time: Nested Scans Appear*
<!-- bilingual-en:end -->
接着课堂切到二次复杂度。
<!-- bilingual-en:start -->
The lecture then moves to quadratic complexity.
<!-- bilingual-en:end -->

典型函数有：
<!-- bilingual-en:start -->
Typical functions include:
<!-- bilingual-en:end -->

- `g(n)`：双重循环
- `is_subset(L1, L2)`
- `intersect(L1, L2)`
- `diameter(L)`
<!-- bilingual-en:start -->
- `g(n)` with double loops
- `is_subset(L1, L2)`
- `intersect(L1, L2)`
- `diameter(L)`
<!-- bilingual-en:end -->

它们共同特征是：
<!-- bilingual-en:start -->
Their common feature is:
<!-- bilingual-en:end -->

- 某一层工作里又包含一层与输入规模相关的完整扫描
<!-- bilingual-en:start -->
- One layer of work contains another full scan whose length depends on the input size.
<!-- bilingual-en:end -->

尤其像 `is_subset(L1, L2)` 这类题，老师在强调：
<!-- bilingual-en:start -->
For functions such as `is_subset(L1, L2)`, the instructor emphasizes:
<!-- bilingual-en:end -->

- 不能只看有几个循环
- 还要看每层循环跑多长
<!-- bilingual-en:start -->
- Do not look only at the number of loops.
- Check how long each loop runs as well.
<!-- bilingual-en:end -->

令 $n=\operatorname{len}(L1)$、$m=\operatorname{len}(L2)$，比较一次元素按常数成本计。`is_subset` 的最坏时间为 $\Theta(nm)$，例如 `L1` 的每个元素都等于 `L2` 的最后一个元素、而此前均不匹配。[[MIT 6.100L-slides/mit6_100l_lec23.pdf#page=25|讲义第 25 页]]举出的“两表完全不相交”反而会在检查完第一个 `e1` 后返回，只需 $\Theta(m)$；不能用它证明平方下界。
<!-- bilingual-en:start -->
Let $n=\operatorname{len}(L1)$ and $m=\operatorname{len}(L2)$, with constant-cost element comparisons. `is_subset` has worst-case time $\Theta(nm)$: every `L1` element can match only the last element of `L2`. The disjoint-list example on [[MIT 6.100L-slides/mit6_100l_lec23.pdf#page=25|slide 25]] instead returns after scanning `L2` for the first `e1`, taking $\Theta(m)$; it does not prove a quadratic lower bound.
<!-- bilingual-en:end -->

`intersect` 还要把两阶段分别算清。[[MIT 6.100L-slides/mit6_100l_lec23.pdf#page=26|第 26–27 页及官方代码]]允许重复元素；第一阶段比较全部 $nm$ 对，第二阶段对 `tmp` 每项执行线性的 `e in unique`。等长 $n=2k$ 时，令两表都是 `[0,1,...,k-1] + [k]*k`：`tmp` 先出现 $k$ 个不同值，再出现 $k^2$ 个 `k`；后者每次都扫描约 $k$ 个已有值，故去重成本为 $\Theta(k^3)=\Theta(n^3)$。上界也成立，因为 `tmp` 至多长 $n^2$、`unique` 至多长 $n$。因此允许重复的这份具体实现最坏为立方阶，不是讲义笼统列出的平方阶；若两表都无重复，第二阶段最多处理 $\min(n,m)$ 个值，整体才保持 $\Theta(nm)$。
<!-- bilingual-en:start -->
Analyze both phases of `intersect`. [[MIT 6.100L-slides/mit6_100l_lec23.pdf#page=26|Slides 26–27 and the official code]] allow duplicates. The first phase compares all $nm$ pairs; the second uses linear membership tests in `unique`. For equal lengths $n=2k$, let both lists be `[0,1,...,k-1] + [k]*k`. Then `tmp` contains $k$ distinct values followed by $k^2$ copies of `k`; each later membership test scans about $k$ earlier values. Deduplication therefore costs $\Theta(k^3)=\Theta(n^3)$. This is also an upper bound because `tmp` has at most $n^2$ entries and `unique` at most $n$. Thus this implementation has cubic worst-case cost with duplicates, despite the slide's quadratic summary. If both inputs have no duplicates, the second phase handles at most $\min(n,m)$ values and the full cost stays $\Theta(nm)$.
<!-- bilingual-en:end -->

### 6. Exponential：最容易失控的一类
<!-- bilingual-en:start -->
*6. Exponential Time: Rapidly Expanding Cost*
<!-- bilingual-en:end -->
老师随后用两类经典函数展示指数复杂度：
<!-- bilingual-en:start -->
The instructor then demonstrates exponential complexity with two classic functions:
<!-- bilingual-en:end -->

- `gen_subsets(L)`
- `fib_recur(x)`

它们都产生指数级工作，但原因不同：
<!-- bilingual-en:start -->
Both require exponential work, but for different reasons:
<!-- bilingual-en:end -->

- `fib_recur` 在一层中调用两个子问题，许多子问题在不同分支中重复出现
- `gen_subsets` 每层只有一个递归调用；返回的子集数量不断翻倍，并且要复制列表内容
<!-- bilingual-en:start -->
- `fib_recur` calls two subproblems, repeatedly solving the same subproblems on different branches.
- `gen_subsets` has only one recursive call per level; its returned subset collection doubles and its list contents must be copied.
<!-- bilingual-en:end -->

对 `fib_recur`，令调用总数为 $C_n$，则 $C_0=C_1=1$、$C_n=1+C_{n-1}+C_{n-2}$。代入即可验证 $C_n=2F_{n+1}-1$，其中 $F_0=0,F_1=1$。由特征方程 $r^2=r+1$，正根 $\phi=(1+\sqrt5)/2$ 主导增长，故单位成本模型下紧界为 $\Theta(\phi^n)$。[[MIT 6.100L-slides/mit6_100l_lec23.pdf#page=33|讲义第 33–34 页]]用 $2^n$ 作粗略比较；它是 $O(2^n)$ 上界，不是准确的 $\Theta(2^n)$。详见[[递归调用树成本]]中的按节点计费原则。
<!-- bilingual-en:start -->
For `fib_recur`, total calls satisfy $C_0=C_1=1$ and $C_n=1+C_{n-1}+C_{n-2}$. Substitution verifies $C_n=2F_{n+1}-1$, where $F_0=0,F_1=1$. The characteristic equation $r^2=r+1$ has dominant root $\phi=(1+\sqrt5)/2$, giving tight unit-cost time $\Theta(\phi^n)$. [[MIT 6.100L-slides/mit6_100l_lec23.pdf#page=33|Slides 33–34]] use $2^n$ as a coarse comparison; it is an $O(2^n)$ upper bound, not the tight $\Theta(2^n)$ bound. This follows the per-node accounting in [[递归调用树成本|call-tree cost analysis]].
<!-- bilingual-en:end -->

对 `gen_subsets`，若 $n$ 个输入元素互异，所有子集列表中共有 $n2^{n-1}$ 个元素引用。[[显式输出下界]]因此已经是 $\Omega(n2^n)$；逐层复制的总和也为 $O(n2^n)$，所以完整实现是 $\Theta(n2^n)$，与[[MIT 6.100L-slides/mit6_100l_lec23.pdf#page=43|讲义第 43 页]]计入连接成本后的结论一致。
<!-- bilingual-en:start -->
For `gen_subsets` on $n$ distinct elements, all subset lists contain $n2^{n-1}$ element references. The [[显式输出下界|explicit-output lower bound]] is therefore already $\Omega(n2^n)$. Summing the copying across levels also gives $O(n2^n)$, so the full implementation is $\Theta(n2^n)$, matching [[MIT 6.100L-slides/mit6_100l_lec23.pdf#page=43|slide 43]] after concatenation is included.
<!-- bilingual-en:end -->

因此即使代码很短，复杂度也可能极高。  
这再次提醒你：代码行数和复杂度没有直接关系。
<!-- bilingual-en:start -->
A short program can therefore require an enormous amount of work. Code length and computational complexity are not directly related.
<!-- bilingual-en:end -->

### 7. Logarithmic：每次都大幅缩小问题规模
<!-- bilingual-en:start -->
*7. Logarithmic Time: Each Step Removes a Large Fraction of the Problem*
<!-- bilingual-en:end -->
在 logarithmic 一类里，老师拿：
<!-- bilingual-en:start -->
To illustrate logarithmic complexity, the instructor uses:
<!-- bilingual-en:end -->

- `digit_sum(n)`（通过位数理解）
- 后面的二分搜索
<!-- bilingual-en:start -->
- `digit_sum(n)` (understood in terms of the number of digits)
- The binary search that follows
<!-- bilingual-en:end -->

来帮助大家建立直觉。
<!-- bilingual-en:start -->
These examples help build the underlying intuition.
<!-- bilingual-en:end -->

这类函数的共同点是：
<!-- bilingual-en:start -->
Functions in this class share a common pattern:
<!-- bilingual-en:end -->

- 每一步都把剩余问题砍掉一大块
- 所以总步数是“能砍多少次才见底”
<!-- bilingual-en:start -->
- Each step eliminates a large fraction of the remaining problem.
- The number of steps is the number of reductions required to reach the base case.
<!-- bilingual-en:end -->

要区别数值与表示长度：正整数 $n$ 的十进制位数是 $d=\lfloor\log_{10}n\rfloor+1$，所以扫描每一位是关于 $d$ 的线性循环、关于数值 $n$ 的对数次迭代；$n=0$ 的表示单独有一位。[[MIT 6.100L-lecture-code/mit6_100l_lec23_code.py|代码 `digit_sum`]]先做 `str(n)`，该转换与新字符串的空间不能凭“有 $d$ 轮循环”当成免费操作。[Python 文档](https://docs.python.org/3/library/stdtypes.html#integer-string-conversion-length-limitation)也指出大整数与十进制字符串间的转换并非线性时间，并设有可配置的位数限制。这里应把课堂的循环计数和实际转换成本分开。
<!-- bilingual-en:start -->
Separate numerical value from representation length: a positive integer $n$ has $d=\lfloor\log_{10}n\rfloor+1$ decimal digits. Scanning the digits is linear in $d$ and uses logarithmically many iterations in the value $n$; zero separately has a one-digit representation. [[MIT 6.100L-lecture-code/mit6_100l_lec23_code.py|The code's `digit_sum`]] first calls `str(n)`, so conversion work and string space are not free. The [Python documentation](https://docs.python.org/3/library/stdtypes.html#integer-string-conversion-length-limitation) explains that large-integer decimal conversion is not linear-time and has a configurable digit limit. Distinguish the classroom loop count from conversion costs.
<!-- bilingual-en:end -->

### 8. 搜索算法再回归：这次重点是复杂度分类
<!-- bilingual-en:start -->
*8. Returning to Search Algorithms with a Complexity Lens*
<!-- bilingual-en:end -->
后半段课堂回到 searching。
<!-- bilingual-en:start -->
The second half of the lecture returns to searching.
<!-- bilingual-en:end -->

老师先放：
<!-- bilingual-en:start -->
The instructor first presents:
<!-- bilingual-en:end -->

- `linear_search(L, e)`：无序列表线性扫
- `search(L, e)`：有序列表上线性扫，但可提前停
<!-- bilingual-en:start -->
- `linear_search(L, e)`: a linear scan of an unordered list.
- `search(L, e)`: a linear scan of a sorted list that may stop early.
<!-- bilingual-en:end -->

然后再引出：
<!-- bilingual-en:start -->
The instructor then introduces:
<!-- bilingual-en:end -->

- `bisect_search1`
- `bisect_search2`

这里的主问题不是“谁更快”这句口号，而是：
<!-- bilingual-en:start -->
The question is not simply which algorithm is faster, but:
<!-- bilingual-en:end -->

- 为什么是这个复杂度
- 有哪些额外成本
<!-- bilingual-en:start -->
- Why each algorithm belongs to its complexity class.
- Which additional operations contribute to its cost.
<!-- bilingual-en:end -->

[[MIT 6.100L-lecture-code/mit6_100l_lec23_code.py|本讲代码]]里的 `linear_search` 即使命中也不 `return`，而是设 `found=True` 后扫完整表，因此对非空规模最好与最坏均为 $\Theta(n)$。第 55 页的提前返回版本才有常数最好情况；有序线性扫描还可在遇到大于目标的元素时停下，但最坏仍需扫完整表。[[最好情况复杂度]]与[[最坏情况复杂度]]必须对应到指定实现。
<!-- bilingual-en:start -->
In [[MIT 6.100L-lecture-code/mit6_100l_lec23_code.py|the lecture code]], `linear_search` sets `found=True` on a match but still scans the whole list, giving $\Theta(n)$ best and worst cases for nonempty sizes. The early-return version on slide 55 has a constant best case. Sorted linear search may additionally stop after passing the target, but still scans the full list in the worst case. [[最好情况复杂度|Best-case]] and [[最坏情况复杂度|worst-case complexity]] must refer to the specified implementation.
<!-- bilingual-en:end -->

### 9. `bisect_search1`：切片版递归会带来复制成本
<!-- bilingual-en:start -->
*9. `bisect_search1`: Recursive Slicing Adds Copying Overhead*
<!-- bilingual-en:end -->
`bisect_search1` 的写法里用到了：
<!-- bilingual-en:start -->
`bisect_search1` uses slicing in its implementation:
<!-- bilingual-en:end -->

- `L[:half]`
- `L[half:]`

这说明每次递归除了逻辑判断，还在做切片复制。  
所以老师把它单独拿出来很有意义，因为它提醒你：
<!-- bilingual-en:start -->
Each recursive call therefore performs both logical checks and a slice copy. The instructor highlights this version because it shows that:
<!-- bilingual-en:end -->

- 递归本身之外，辅助操作也可能影响复杂度和常数项
<!-- bilingual-en:start -->
- Beyond recursion itself, auxiliary operations can affect both asymptotic complexity and constant factors.
<!-- bilingual-en:end -->

这份实现的[[切片二分成本|最坏紧界是 $\Theta(n)$]]：只沿一条分支前进，复制量为 $n/2+n/4+\cdots=\Theta(n)$，再加对数次比较。[[MIT 6.100L-slides/mit6_100l_lec23.pdf#page=61|讲义第 61 页]]先把每层都按原始 $n$ 计，写出课堂粗估，随后也注明线性紧界。粗估可写 $O(n\log n)$，不应写成 $\Theta(n\log n)$。
<!-- bilingual-en:start -->
The [[切片二分成本|tight worst-case cost is $\Theta(n)$]]: one visited branch copies $n/2+n/4+\cdots=\Theta(n)$ references, plus logarithmically many comparisons. [[MIT 6.100L-slides/mit6_100l_lec23.pdf#page=61|Slide 61]] initially charges the original $n$ at every level, then also notes the tighter linear bound. The coarse estimate is $O(n\log n)$, not $\Theta(n\log n)$.
<!-- bilingual-en:end -->

### 10. `bisect_search2`：索引版更贴近真正的二分思路
<!-- bilingual-en:start -->
*10. `bisect_search2`: The Index-Based Version More Closely Matches Binary Search*
<!-- bilingual-en:end -->
相对地，`bisect_search2` 用的是：
<!-- bilingual-en:start -->
In contrast, `bisect_search2` uses index bounds:
<!-- bilingual-en:end -->

- `low`
- `high`
- `mid`

以及一个 helper function。
<!-- bilingual-en:start -->
It also uses a helper function.
<!-- bilingual-en:end -->

它更接近真正的二分搜索实现，因为：
<!-- bilingual-en:start -->
This is closer to a direct implementation of binary search because it:
<!-- bilingual-en:end -->

- 不复制子列表
- 只是缩小索引区间
<!-- bilingual-en:start -->
- It does not copy sublists.
- It only narrows the index range.
<!-- bilingual-en:end -->

这让你看到复杂度分析不只是“这是不是递归”，还要看递归每层具体做了什么。
<!-- bilingual-en:start -->
This shows that complexity analysis is not just about whether a function is recursive; it must also account for the work done at each level of recursion.
<!-- bilingual-en:end -->

索引递归版最坏时间是 $\Theta(\log n)$，峰值栈空间仍有 $\Theta(\log n)$；不切片并不等于不用栈。若改成迭代索引更新，则可用常数[[辅助空间]]。此外，[[MIT 6.100L-slides/mit6_100l_lec23.pdf#page=66|第 66–67 页]]比较先排序再查询：固定长度 $n\ge2$ 的列表、不更新数据，并执行 $K\ge1$ 次查询。若 $S,B,L$ 分别表示所选排序、二分与线性扫描的最坏成本函数，则 $S(n)+K B(n)$ 与 $K L(n)$ 给出两种方案的总成本上界；不要直接对 $\Theta$ 集合做数值减法。在常数成本比较与随机访问模型下，使用标准归并排序及不切片的索引二分，最坏总成本的紧界分别为 $\Theta(n\log n+K\log n)$ 与 $\Theta(Kn)$：反复查找缺失目标可使每次搜索都走到最坏量级。它们不是每组查询的实际等式；例如本讲索引二分版若每次都在首次中点命中，查询部分只有 $\Theta(K)$。有限规模的交叉点仍取决于常数。这里把预处理分配给一串查询，是[[摊还复杂度|序列成本分析]]，不是随机输入上的平均。
<!-- bilingual-en:start -->
Index-based recursion has $\Theta(\log n)$ worst-case time and $\Theta(\log n)$ peak stack space; avoiding slices does not eliminate the stack. Iterative index updates can use constant [[辅助空间|auxiliary space]]. [[MIT 6.100L-slides/mit6_100l_lec23.pdf#page=66|Slides 66–67]] compare sorting once before querying. Fix a list of length $n\ge2$, make no updates, and issue $K\ge1$ queries. If $S,B,L$ denote the chosen sorting, binary-search, and linear-scan worst-case cost functions, then $S(n)+K B(n)$ and $K L(n)$ bound the two total costs; do not numerically subtract $\Theta$ sets. With constant-cost comparisons and random access, standard merge sort followed by index-based binary search without slices has tight worst-case total cost $\Theta(n\log n+K\log n)$, versus $\Theta(Kn)$ for repeated linear scans. Repeated absent-target queries attain the worst-case search orders. These are not actual-cost equalities for every query sequence: if the index-based version finds every target at its first midpoint, the query portion costs only $\Theta(K)$. Constants still determine finite-size crossover points. Allocating preprocessing across queries is [[摊还复杂度|sequence-cost analysis]], not an expectation over random inputs.
<!-- bilingual-en:end -->

### 11. 这节课是在做代码阅读训练
<!-- bilingual-en:start -->
*11. Treating the Lecture as an Exercise in Reading Code*
<!-- bilingual-en:end -->
Lecture 23 的整体感觉会比前一讲更“碎”，因为它几乎没有一个单一大主题例子，而是很多小段代码。
<!-- bilingual-en:start -->
Lecture 23 may feel more fragmented than the previous lecture because it uses many small code examples rather than one large central example.
<!-- bilingual-en:end -->

但这些例子其实都服务同一个目标：
<!-- bilingual-en:start -->
But all these examples serve the same goal:
<!-- bilingual-en:end -->

- 训练你从真实代码结构直接读出 complexity class
<!-- bilingual-en:start -->
- Train you to infer a complexity class directly from real code structure.
<!-- bilingual-en:end -->

所以本讲的正确学习方式不是背完整张表，而是每看到一个函数，都问：
<!-- bilingual-en:start -->
The right way to study this lecture is therefore not to memorize a complete table, but to ask four questions about every function:
<!-- bilingual-en:end -->

1. 输入规模怎么定义
2. 哪些语句依赖输入
3. 是顺序相加还是嵌套相乘
4. 是否有递归分叉或规模折半
<!-- bilingual-en:start -->

&nbsp;
**1.** How is the input size defined?<br>
**2.** Which statements depend on the input?<br>
**3.** Are costs added sequentially or multiplied through nesting?<br>
**4.** Does the recursion branch, or does it halve the problem size?<br>
<!-- bilingual-en:end -->

## Exercise log

> [!example] Finger exercise 23
> 官方练习给三段代码，让你判断 worst-case Theta：
> - `running_product(a)` -> `Theta(n)`
> - `tricky_f(L, L2)` -> `Theta(n^2)`
> - `sum_f(n)` -> `Theta(log n)`
> <!-- bilingual-en:start -->
> The official exercise gives three code segments and asks you to determine their worst-case Theta complexity:
> - `running_product(a)` -> `Theta(n)`
> - `tricky_f(L, L2)` -> `Theta(n^2)`
> - `sum_f(n)` -> `Theta(log n)`
> <!-- bilingual-en:end -->

按[[MIT 6.100L-finger-exercises/mit6_100l_ex23_sol.pdf#page=1|官方题目与解答第 1–2 页]]补上口径：第一题定义 $n=a$，但有提前返回。取 $a=k!/4!$（$k\ge5$），函数在 $i=k$ 时便返回，只运行 $k-4$ 次，所以点态成本不能对所有充分大的 $a$ 都称为 $\Theta(a)$。按单位成本计，它普遍是 $O(a)$；若按 $1\le a\le N$ 内的最坏成本报告，才得到 $\Theta(N)$：取不超过 $N$ 的最大 $a\equiv1\pmod5$，乘积总含因子 $5$，不会提前返回，且 $a\ge N-4$。这说明了官方“最坏路径线性”的答案，不能把它读成每次都执行完整循环；任意精度乘法的实际成本仍要另算。
<!-- bilingual-en:start -->
The [[MIT 6.100L-finger-exercises/mit6_100l_ex23_sol.pdf#page=1|official questions and answers, pages 1–2]] need explicit conventions. Problem 1 sets $n=a$ but can return early: for $a=k!/4!$, $k\ge5$, it returns at $i=k$ after only $k-4$ iterations. Its pointwise cost is therefore not $\Theta(a)$ for all sufficiently large inputs. Under unit costs it is universally $O(a)$; the maximum over $1\le a\le N$ is $\Theta(N)$. Indeed, the largest $a\le N$ with $a\equiv1\pmod5$ is at least $N-4$ and cannot equal the product, which is divisible by five, so all $a$ iterations run. This interprets the official linear worst-path answer without claiming every run follows that path. Arbitrary-precision multiplication needs separate costing.
<!-- bilingual-en:end -->

第二题两表等长 $n$，两次顺序扫描各含一次最坏线性成员测试，合计 $\Theta(n^2)$，而不是三层相乘；第三题对正整数精确逐位除以 $10$，循环轮数为 $\lfloor\log_{10}n\rfloor+1$。
<!-- bilingual-en:start -->
Problem 2 uses two equal-length lists; each sequential scan contains a worst-case linear membership test, giving $\Theta(n^2)$ rather than a product of three loop lengths. Problem 3, when interpreted as exact digit removal on a positive integer, requires $\lfloor\log_{10}n\rfloor+1$ iterations.
<!-- bilingual-en:end -->

第三题原代码写的是 `n = int(n/10)`，会先做浮点除法；对很大的整数可能舍入或溢出，不能无条件当成精确整除。若实现“移去最后一位”的整数算法，更新应写为 `n //= 10`，再按[[算法成本模型]]说明大整数除法成本；原练习文件保留，课堂给出的 $\Theta(\log n)$ 在这里解释为理想逐位循环的次数。
<!-- bilingual-en:start -->
Problem 3's original `n = int(n/10)` performs floating-point division first and can round or overflow for very large integers. It is not unconditionally exact integer division. An integer algorithm that removes the last decimal digit uses `n //= 10`, with large-integer division charged under the stated [[算法成本模型|cost model]]. The original exercise is retained; its $\Theta(\log n)$ is interpreted here as the ideal digit-removal iteration count.
<!-- bilingual-en:end -->

这三题选得很准，因为它们分别覆盖：
<!-- bilingual-en:start -->
These three questions are well chosen because they cover:
<!-- bilingual-en:end -->

- 简单线性循环
- 成员测试嵌套导致的平方级
- 数字按位缩小导致的对数级
<!-- bilingual-en:start -->
- Simple linear loops
- Nested membership tests leading to quadratic complexity
- Shrinking a number one digit at a time, leading to logarithmic complexity
<!-- bilingual-en:end -->

如果这三题你能独立解释为什么，不只是选对答案，那本讲主线就基本吃透了。
<!-- bilingual-en:start -->
If you can explain each answer independently instead of merely choosing the right option, you have understood the lecture's main thread.
<!-- bilingual-en:end -->

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec23.pdf|Lecture 23 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec23_code.py|Lecture 23 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex23_sol.pdf|Lecture 23 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec23_transcript.pdf|Lecture 23 transcript]]
- Recitation 10: [[MIT 6.100L-OCW-offline-site/static_resources/mit6_100l_f22_rec10.zip|Recitation 10 materials]]
- PS 5 halfway hand-in due: [[MIT 6.100L-problem-sets/mit6_100l_ps5.pdf|PS5 statement]], [[MIT 6.100L-OCW-offline-site/static_resources/mit6_100l_f22_ps5_code.zip|PS5 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (本地 Revised and Expanded Edition：Ch 9.3 与 10.1)

## Review checklist
- [ ] 我能从代码里定义输入规模，而不是默认都写成 `n`。
- [ ] 我能给常数、线性、二次、指数、对数几个复杂度类各举一个代码例子。
- [ ] 我能解释为什么短代码也可能是指数级。
- [ ] 我能判断某段代码是顺序成本相加还是嵌套成本相乘。
- [ ] 我能说明递归实现里切片复制为什么会额外花成本。
- [ ] 我能解释 `bisect_search1` 和 `bisect_search2` 的设计差异。
- [ ] 我能分析 search on sorted list 为什么虽然可提前停，本质最坏仍是线性。
- [ ] 我能把 finger exercise 23 的三题说清楚理由，而不是只会选答案。
- [ ] 我能把本讲看成“从代码到 Theta”的翻译训练。
- [ ] 我能按课堂顺序复述：Theta recap -> code classes -> search examples -> bisection variants。
<!-- bilingual-en:start -->
- [ ] I can define input size from code rather than defaulting to `n`.
- [ ] I can give one code example for each of the constant, linear, quadratic, exponential, and logarithmic complexity classes.
- [ ] I can explain why short code can still be exponential in the worst case.
- [ ] I can determine whether a piece of code adds sequential costs or multiplies nested costs.
- [ ] I can explain why slice copies in recursive implementations add extra costs.
- [ ] I can explain the design differences between `bisect_search1` and `bisect_search2`.
- [ ] I can analyze why searching on a sorted list, although it may terminate early, is fundamentally linear in the worst case.
- [ ] I can explain my reasoning for all three questions in Finger Exercise 23 rather than merely select the answers.
- [ ] I can treat this lecture as practice in translating code into Theta notation.
- [ ] I can reconstruct the lecture sequence: Theta recap -> code classes -> search examples -> bisection variants.
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 只看循环层数，不看每层边界和辅助操作。
> - 看到递归就笼统写成指数或线性，不先分析调用结构。
> - 忽略切片、成员测试这类看似小但可能昂贵的操作。
> - 把“平均情况可能提前停”误当成“最坏情况就不是线性”。
> <!-- bilingual-en:start -->
> - Looking only at loop depth while ignoring loop bounds and auxiliary operations.
> - Labeling recursion as exponential or linear without first analyzing the call structure.
> - Ignoring operations such as slicing and membership tests that look small but may be expensive.
> - Assuming that an early exit in some cases means the worst-case complexity is no longer linear.
> <!-- bilingual-en:end -->
