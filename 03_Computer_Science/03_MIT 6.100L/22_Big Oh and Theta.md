---
aliases:
  - MIT 6.100L Lecture 22
  - 6.100L L22
  - Big Oh and Theta
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 22
---

# Lecture 22: Big Oh and Theta

> [!tip] Hint
> - 这节课前半段继续 timing，但比上节更系统，用 `perf_counter` 提高精度。
> - timing 不再只看数字函数，还开始看 list 相关函数，比如求和、查找、diameter。
> - linear search 和 binary search 会在这讲里被放到一起正式比较。
> - binary search 快，不是因为写法神秘，而是因为每次都把搜索区间砍半。
> - `diameter` 和 `all_binary_numbers` 分别被用来制造 quadratic 和 exponential 的直观对比。
> - 到中后段课程终于正式引入 order of growth，以及 big O / big Theta 语言。
> - 老师更偏爱 Theta，因为它表达的是 asymptotically tight bound，而不是随便一个上界。
> - 判断 Theta 时最重要的是：先说清 `n` 代表什么，再抓 dominant term。
> - 不同变量的式子不能瞎写成 `Theta(n)`，必须说明哪个输入维度在增长。
> - 听完这节课，你应该能把 timing 直觉过渡成“用增长阶比较程序”的理论语言。
> <!-- bilingual-en:start -->
> - This lecture's first half continues with timing measurements but approaches it more systematically using `perf_counter` for higher precision.
> - Timing no longer focuses solely on numerical functions; we also begin examining list-related functions such as sum, search, and diameter.
> - The lecture formally compares linear search with binary search.
> - Binary search is fast not because of a mysterious trick, but because each step halves the search interval.
> - The `diameter` and `all_binary_numbers` functions make the contrast between quadratic and exponential growth concrete.
> - Later in the lecture, order of growth and the language of Big O and Big Theta are formally introduced.
> - The instructor prefers Theta because it expresses asymptotically tight bounds rather than just any upper bound.
> - To determine Theta, first define what `n` represents, then identify the dominant term.
> - An expression involving several variables cannot be labeled `Theta(n)` indiscriminately; you must specify which input dimension is growing.
> - After this lecture, you should be able to turn your intuition from timing experiments into the theoretical language of order of growth.
> <!-- bilingual-en:end -->

## Lecture flow

### 1. 开场继续 timing，但换成更精细的计时器
<!-- bilingual-en:start -->
*1. Continuing with a More Precise Timer*
<!-- bilingual-en:end -->
Lecture 22 开头延续上讲，但老师先把技术细节升级成 `time.perf_counter()`。
<!-- bilingual-en:start -->
Lecture 22 picks up where the previous lecture left off, but upgrades the timer to `time.perf_counter()`.
<!-- bilingual-en:end -->

原因很实际：
<!-- bilingual-en:start -->
The reasons are practical:
<!-- bilingual-en:end -->

- 上讲某些函数太快
- `time.time()` 分辨率不够
- `perf_counter()` 更适合测短代码段
<!-- bilingual-en:start -->
- Some functions in the previous lecture ran too quickly to time reliably.
- `time.time()` lacks sufficient resolution for those measurements.
- `perf_counter()` is better suited to short code segments.
<!-- bilingual-en:end -->

所以这节课虽然还在 timing，但已经更强调测量质量。
<!-- bilingual-en:start -->
Although this lecture continues to use timing, it places greater emphasis on measurement quality.
<!-- bilingual-en:end -->

### 2. 先看简单 numeric 函数：常数 vs 线性
<!-- bilingual-en:start -->
*2. Simple Numeric Functions: Constant versus Linear Growth*
<!-- bilingual-en:end -->
课堂先用两个函数热身：
<!-- bilingual-en:start -->
The class warms up with two functions:
<!-- bilingual-en:end -->

- `convert_to_km(m)`：常数时间
- `compound(invest, interest, n_months)`：随着某个参数变化可能呈线性增长
<!-- bilingual-en:start -->
- `convert_to_km(m)`: constant time.
- `compound(invest, interest, n_months)`: linear growth when `n_months` is the growing input.
<!-- bilingual-en:end -->

这时老师特别提醒一件事：
<!-- bilingual-en:start -->
At this point, the instructor emphasizes an important point:
<!-- bilingual-en:end -->

- 一个函数可能有多个输入
- 但不是每个输入变化都会影响复杂度
<!-- bilingual-en:start -->
- A function can have several inputs.
- Not every input affects the number of operations.
<!-- bilingual-en:end -->

例如 `compound` 中，如果增长的是 `n_months`，复杂度分析就和它最相关；  
如果只是 `invest` 数值变大，循环次数并不会变。
<!-- bilingual-en:start -->
For example, the complexity of `compound` changes when `n_months` grows. Increasing only the value of `invest` does not change the number of loop iterations.
<!-- bilingual-en:end -->

### 3. timing list 函数：输入规模开始从“数值”变成“列表长度”
<!-- bilingual-en:start -->
*3. Timing List Functions: Input Size Becomes List Length*
<!-- bilingual-en:end -->
随后课堂把输入类型切到 list。
<!-- bilingual-en:start -->
The class then switches focus to lists as inputs.
<!-- bilingual-en:end -->

例如：
<!-- bilingual-en:start -->
For example:
<!-- bilingual-en:end -->

```python
def sum_of(L):
    total = 0.0
    for elt in L:
        total += elt
```

这里老师明显开始强调：
<!-- bilingual-en:start -->
Here, the instructor emphasizes that:
<!-- bilingual-en:end -->

- `n` 不是元素值本身
- `n` 是 `len(L)`
<!-- bilingual-en:start -->
- `n` is not the value of an individual element.
- Here, `n` is `len(L)`.
<!-- bilingual-en:end -->

这一步很关键，因为复杂度单元最容易卡住的点之一就是：  
你得先定义“输入规模”。
<!-- bilingual-en:start -->
This point matters because defining “input size” is one of the first sources of confusion in complexity analysis.
<!-- bilingual-en:end -->

### 4. linear search vs binary search：同一任务，不同增长阶
<!-- bilingual-en:start -->
*4. Linear Search versus Binary Search: One Task, Different Growth Rates*
<!-- bilingual-en:end -->
这节课最重要的对比例子之一是查找元素。
<!-- bilingual-en:start -->
One of the most important comparisons in this lecture concerns searching for an element.
<!-- bilingual-en:end -->

老师先给出 brute-force：
<!-- bilingual-en:start -->
The instructor first presents a brute-force linear search:
<!-- bilingual-en:end -->

```python
def is_in(L, x):
    for elt in L:
        if elt == x:
            return True
    return False
```

然后再给出 binary search：
<!-- bilingual-en:start -->
The instructor then presents binary search:
<!-- bilingual-en:end -->

```python
def binary_search(L, x):
    ...
```

这时课堂真正要你看的不是“代码写法差多少”，而是：
<!-- bilingual-en:start -->
The point is not how different the two implementations look, but how much of the search space each step eliminates:
<!-- bilingual-en:end -->

- linear search 每次最坏只排除一个元素
- binary search 每次都排除一半候选空间
<!-- bilingual-en:start -->
- Linear search eliminates at most one element at each step in the worst case.
- Binary search halves the candidate space at each step.
<!-- bilingual-en:end -->

### 5. 为什么 binary search 是 logarithmic
<!-- bilingual-en:start -->
*5. Why Binary Search Is Logarithmic*
<!-- bilingual-en:end -->
老师花了不少时间口头画列表，把二分搜索的动作讲成：
<!-- bilingual-en:start -->
The instructor spends considerable time talking through a list step by step, describing binary search as follows:
<!-- bilingual-en:end -->

- 看中点
- 决定去左半边还是右半边
- 再看那一半的中点
<!-- bilingual-en:start -->
- Inspect the midpoint.
- Decide whether to keep the left or right half.
- Inspect the midpoint of the retained half.
<!-- bilingual-en:end -->

所以搜索区间大小大致经历：
<!-- bilingual-en:start -->
The size of the search space therefore follows roughly this sequence:
<!-- bilingual-en:end -->

- `n`
- `n/2`
- `n/4`
- `n/8`

直到缩到 `1`。
<!-- bilingual-en:start -->
This continues until the search space shrinks to `1`.
<!-- bilingual-en:end -->

这就是 logarithmic growth 的直觉来源。
<!-- bilingual-en:start -->
This is where the intuition behind logarithmic growth comes from.
<!-- bilingual-en:end -->

> [!note]
> 当问题规模每一步按比例缩小，而不是按固定常数减少时，复杂度往往会走向 `log n`。
> <!-- bilingual-en:start -->
> When each step removes a fixed fraction of the remaining problem rather than a fixed number of items, the complexity is often logarithmic: `log n`.
> <!-- bilingual-en:end -->

### 6. `diameter(L)`：嵌套循环制造 quadratic growth
<!-- bilingual-en:start -->
*6. `diameter(L)`: Nested Loops Create Quadratic Growth*
<!-- bilingual-en:end -->
老师随后又拿 `diameter(L)` 这种两两比较点对的函数做对照。
<!-- bilingual-en:start -->
The instructor then uses `diameter(L)`, which compares pairs of points, as a contrasting example.
<!-- bilingual-en:end -->

因为：
<!-- bilingual-en:start -->
Because:
<!-- bilingual-en:end -->

- 外层遍历点
- 内层又遍历剩余点
<!-- bilingual-en:start -->
- The outer loop iterates over the points.
- The inner loop iterates over the remaining points.
<!-- bilingual-en:end -->

所以总比较次数和 `len(L)^2` 同阶。
<!-- bilingual-en:start -->
So the total number of comparisons is on the order of `len(L)^2`.
<!-- bilingual-en:end -->

这时课堂已经在把几种经典增长阶直觉排开：
<!-- bilingual-en:start -->
By this point, the lecture has laid out the intuition behind several standard orders of growth:
<!-- bilingual-en:end -->

- constant
- linear
- logarithmic
- quadratic

### 7. `all_binary_numbers(N)`：指数增长真正变得吓人
<!-- bilingual-en:start -->
*7. `all_binary_numbers(N)`: Exponential Growth Becomes Daunting*
<!-- bilingual-en:end -->
为了让 exponential growth 也变得直观，老师再给出：
<!-- bilingual-en:start -->
To make exponential growth concrete, the instructor next presents:
<!-- bilingual-en:end -->

- 生成所有 N 位二进制串
<!-- bilingual-en:start -->
- Generate all `N`-bit binary strings.
<!-- bilingual-en:end -->

这个任务本身就有：
<!-- bilingual-en:start -->
This task itself has:
<!-- bilingual-en:end -->

- `2^N` 个输出
<!-- bilingual-en:start -->
- `2^N` outputs.
<!-- bilingual-en:end -->

所以无论你实现得多漂亮，规模一大都会迅速爆炸。
<!-- bilingual-en:start -->
No matter how elegant the implementation is, the amount of work explodes as `N` grows.
<!-- bilingual-en:end -->

这一步非常重要，因为它提醒你：
<!-- bilingual-en:start -->
This example makes an important point:
<!-- bilingual-en:end -->

- 有些问题不是“实现写得差”
- 而是任务本身的输出规模就决定了下界非常大
<!-- bilingual-en:start -->
- Some problems are not slow because of a poor implementation.
- The required output size itself imposes a very large lower bound.
<!-- bilingual-en:end -->

### 8. 从 timing 转向理论语言：order of growth
<!-- bilingual-en:start -->
*8. Moving from Timing to Theory: Order of Growth*
<!-- bilingual-en:end -->
做完这么多 timing 和 counting 之后，课堂终于引出：
<!-- bilingual-en:start -->
After all the timing and counting examples, the lecture finally introduces:
<!-- bilingual-en:end -->

- order of growth
- Big O
- Big Theta

老师这里要解决的问题是：
<!-- bilingual-en:start -->
The aim is to answer a more general question:
<!-- bilingual-en:end -->

- 我们不想只记某台机器上的秒数
- 我们想比较输入变大时，增长趋势是什么
<!-- bilingual-en:start -->
- We do not want only the number of seconds recorded on one particular machine.
- We want to compare how runtime grows as the input becomes larger.
<!-- bilingual-en:end -->

这就是 order of growth 的作用。
<!-- bilingual-en:start -->
That is the role of order-of-growth analysis.
<!-- bilingual-en:end -->

### 9. 为什么课程更偏爱 Theta
<!-- bilingual-en:start -->
*9. Why the Course Prefers Theta*
<!-- bilingual-en:end -->
老师明确说更喜欢用 Theta 来描述。
<!-- bilingual-en:start -->
The instructor explicitly prefers Theta for describing order of growth.
<!-- bilingual-en:end -->

原因是：
<!-- bilingual-en:start -->
The reason is:
<!-- bilingual-en:end -->

- Big O 只给上界
- 这个上界可能很松
- Theta 更强调 asymptotically tight bound
<!-- bilingual-en:start -->
- Big O supplies an upper bound.
- That upper bound may be very loose.
- Theta supplies an asymptotically tight bound.
<!-- bilingual-en:end -->

也就是说，Theta 不是随便找个长得更快的函数就完事，而是要抓住真正同阶的增长。
<!-- bilingual-en:start -->
Theta therefore does not merely name a function that grows at least as fast; it captures the same asymptotic growth rate.
<!-- bilingual-en:end -->

### 10. 定义 `n` 代表什么，比写符号更重要
<!-- bilingual-en:start -->
*10. Why Defining `n` Matters More Than Writing Symbols*
<!-- bilingual-en:end -->
这节课里老师反复追问：
<!-- bilingual-en:start -->
Throughout the lecture, the instructor repeatedly asks:
<!-- bilingual-en:end -->

- 这里的 `n` 到底是什么
- 是整数参数本身
- 还是字符串长度
- 还是列表长度
- 还是某两个列表中的某一个长度
<!-- bilingual-en:start -->
- What exactly does `n` denote here?
- The integer argument itself?
- The length of a string?
- The length of a list?
- The length of one of two lists?
<!-- bilingual-en:end -->

这是复杂度分析最基础、也最容易被省略的一步。  
如果 `n` 没定义清楚，`Theta(n)` 这种写法几乎没有意义。
<!-- bilingual-en:start -->
This is the most fundamental yet often overlooked step in complexity analysis. If `n` is not clearly defined, expressions like `Theta(n)` lose almost all meaning.
<!-- bilingual-en:end -->

### 11. dominant term：抓增长最快的那一项
<!-- bilingual-en:start -->
*11. Dominant Term: Identify the Fastest-Growing Term*
<!-- bilingual-en:end -->
讲完符号意义后，老师进入实际简化。
<!-- bilingual-en:start -->
After explaining the notation, the instructor moves on to simplifying expressions in practice.
<!-- bilingual-en:end -->

核心规则是：
<!-- bilingual-en:start -->
The core rule is:
<!-- bilingual-en:end -->

- 抓 dominant term
- 丢掉低阶项
- 丢掉常数系数
<!-- bilingual-en:start -->
- Focus on the dominant term.
- Discard lower-order terms.
- Ignore constant coefficients.
<!-- bilingual-en:end -->

例如：
<!-- bilingual-en:start -->
For example:
<!-- bilingual-en:end -->

- `n^2 + log n + 2` -> `Theta(n^2)`
- `2^n + n log n + n^2` -> `Theta(2^n)`

课堂在这里的目标不是形式化证明，而是建立简化直觉。
<!-- bilingual-en:start -->
The aim here is not a formal proof, but intuition for simplifying growth expressions.
<!-- bilingual-en:end -->

### 12. 组合规则：顺序相加、嵌套相乘
<!-- bilingual-en:start -->
*12. Composition Rules: Sequential Costs Add, Nested Costs Multiply*
<!-- bilingual-en:end -->
老师还开始把代码结构和 Theta 组合联系起来：
<!-- bilingual-en:start -->
The instructor then connects code structure to rules for combining Theta costs:
<!-- bilingual-en:end -->

- 顺序执行的代码块，复杂度大致相加，然后取 dominant one
- 嵌套循环或嵌套成本，复杂度往往相乘
<!-- bilingual-en:start -->
- For sequential code blocks, add the costs and retain the dominant term.
- For nested loops or nested work, the costs usually multiply.
<!-- bilingual-en:end -->

这为下一讲从真实代码直接读复杂度打基础。
<!-- bilingual-en:start -->
This prepares you to infer complexity directly from real code in the next lecture.
<!-- bilingual-en:end -->

## Exercise log

> [!example] Finger exercise 22
> 官方练习是三道“化简成 Theta”：
> - `n*n + log(n) + 2**a` -> `Theta(n^2)`
> - `2**n + n*log(n) + n**2` -> `Theta(2^n)`
> - `f*log(f) + 100000 + 300*a + x*y*z` 在 `n` 这一维下 -> `Theta(1)`
> <!-- bilingual-en:start -->
> The official exercises contain three problems about simplifying expressions to Theta notation:
> - `n*n + log(n) + 2**a` -> `Theta(n^2)`
> - `2**n + n*log(n) + n**2` -> `Theta(2^n)`
> - `f*log(f) + 100000 + 300*a + x*y*z` -> `Theta(1)` with respect to `n`
> <!-- bilingual-en:end -->

这套题的价值非常高，因为它逼你明确区分：
<!-- bilingual-en:start -->
These problems are valuable because they force you to distinguish clearly between:
<!-- bilingual-en:end -->

- 哪个变量才是分析时增长的主变量
- 哪些项其实对这个主变量来说只是常数
<!-- bilingual-en:start -->
- Which variable is allowed to grow in the analysis
- Which terms are constant with respect to that variable
<!-- bilingual-en:end -->

这正是本讲理论部分最容易偷懒、但最不能偷懒的地方。
<!-- bilingual-en:start -->
This is the easiest part of the theory to gloss over, but also the part where precision matters most.
<!-- bilingual-en:end -->

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec22.pdf|Lecture 22 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec22_code.py|Lecture 22 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex22_sol.pdf|Lecture 22 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec22_transcript.pdf|Lecture 22 transcript]]
- Recitation 10: [[MIT 6.100L-recitations/mit6_100l_rec10.zip|Recitation 10 materials]]
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 11)

## Review checklist
- [ ] 我能解释为什么 `perf_counter` 比普通 timing 更适合本讲。
- [ ] 我能为多参数函数说明“哪一个参数变化才决定复杂度”。
- [ ] 我能把 list 函数的输入规模定义成列表长度。
- [ ] 我能解释 linear search 和 binary search 的增长差异来自哪里。
- [ ] 我能说明为什么 `diameter` 是 quadratic、`all_binary_numbers` 是 exponential。
- [ ] 我能解释 order of growth 为什么比单次 timing 更重要。
- [ ] 我能区分 Big O 和 Big Theta 的直觉差别。
- [ ] 我能在写 `Theta(...)` 前先说清 `n` 是什么。
- [ ] 我能做 dominant term simplification。
- [ ] 我能按课堂顺序复述：better timing -> search comparison -> several growth shapes -> Theta notation。
<!-- bilingual-en:start -->
- [ ] I can explain why `perf_counter` is more suitable here than a less precise timer.
- [ ] For a function with several parameters, I can state which parameter's growth determines the complexity.
- [ ] I can define the input size of a list function as the list's length.
- [ ] I can explain where the difference between linear-search and binary-search growth comes from.
- [ ] I can explain why `diameter` is quadratic and `all_binary_numbers` is exponential.
- [ ] I can explain why order of growth matters more than one timing measurement.
- [ ] I can distinguish the intuition behind Big O and Big Theta.
- [ ] Before writing `Theta(...)`, I can define what `n` represents.
- [ ] I can simplify an expression by identifying its dominant term.
- [ ] I can reconstruct the lecture sequence: better timing -> search comparison -> several growth patterns -> Theta notation.
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 没定义清 `n` 就直接写 `Theta(n)`。
> - 把数值大小和输入规模混为一谈。
> - 见到 Big O / Theta 就只顾套公式，不回到代码结构。
> - 化简时把与主变量无关的项也错误保留下来。
> <!-- bilingual-en:start -->
> - Writing `Theta(n)` before defining what `n` represents.
> - Confusing numerical magnitude with input size.
> - Applying Big O or Big Theta formulas mechanically without returning to the code structure.
> - Retaining terms that are constant with respect to the variable being analyzed.
> <!-- bilingual-en:end -->
