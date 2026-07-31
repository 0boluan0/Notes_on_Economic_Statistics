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

它们的共同点是：
<!-- bilingual-en:start -->
They share the following structure:
<!-- bilingual-en:end -->

- 每层调用会分叉成多个子调用
- 整体展开像一棵快速膨胀的树
<!-- bilingual-en:start -->
- Each call branches into several subcalls.
- The overall expansion resembles a rapidly growing tree.
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

### 11. 这节课是在做代码阅读训练
<!-- bilingual-en:start -->
*11. Treating the Lecture as an Exercise in Reading Code*
<!-- bilingual-en:end -->
Lecture 23 的整体感觉会比前一讲更“碎”，因为它几乎没有一个单一大主题例子，而是很多 ছোট代码。
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
1. How is the input size defined?
2. Which statements depend on the input?
3. Are costs added sequentially or multiplied through nesting?
4. Does the recursion branch, or does it halve the problem size?
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
- Recitation 10: [[MIT 6.100L-recitations/mit6_100l_rec10.zip|Recitation 10 materials]]
- PS 5 halfway hand-in due: [[MIT 6.100L-problem-sets/mit6_100l_ps5.pdf|PS5 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps5_code.zip|PS5 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 12.1)

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
