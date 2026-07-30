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

## Lecture flow

### 1. 开场继续 timing，但换成更精细的计时器
Lecture 22 开头延续上讲，但老师先把技术细节升级成 `time.perf_counter()`。

原因很实际：

- 上讲某些函数太快
- `time.time()` 分辨率不够
- `perf_counter()` 更适合测短代码段

所以这节课虽然还在 timing，但已经更强调测量质量。

### 2. 先看简单 numeric 函数：常数 vs 线性
课堂先用两个函数热身：

- `convert_to_km(m)`：常数时间
- `compound(invest, interest, n_months)`：随着某个参数变化可能呈线性增长

这时老师特别提醒一件事：

- 一个函数可能有多个输入
- 但不是每个输入变化都会影响复杂度

例如 `compound` 中，如果增长的是 `n_months`，复杂度分析就和它最相关；  
如果只是 `invest` 数值变大，循环次数并不会变。

### 3. timing list 函数：输入规模开始从“数值”变成“列表长度”
随后课堂把输入类型切到 list。

例如：

```python
def sum_of(L):
    total = 0.0
    for elt in L:
        total += elt
```

这里老师明显开始强调：

- `n` 不是元素值本身
- `n` 是 `len(L)`

这一步很关键，因为复杂度单元最容易卡住的点之一就是：  
你得先定义“输入规模”。

### 4. linear search vs binary search：同一任务，不同增长阶
这节课最重要的对比例子之一是查找元素。

老师先给出 brute-force：

```python
def is_in(L, x):
    for elt in L:
        if elt == x:
            return True
    return False
```

然后再给出 binary search：

```python
def binary_search(L, x):
    ...
```

这时课堂真正要你看的不是“代码写法差多少”，而是：

- linear search 每次最坏只排除一个元素
- binary search 每次都排除一半候选空间

### 5. 为什么 binary search 是 logarithmic
老师花了不少时间口头画列表，把二分搜索的动作讲成：

- 看中点
- 决定去左半边还是右半边
- 再看那一半的中点

所以搜索区间大小大致经历：

- `n`
- `n/2`
- `n/4`
- `n/8`

直到缩到 `1`。

这就是 logarithmic growth 的直觉来源。

> [!note]
> 当问题规模每一步按比例缩小，而不是按固定常数减少时，复杂度往往会走向 `log n`。

### 6. `diameter(L)`：嵌套循环制造 quadratic growth
老师随后又拿 `diameter(L)` 这种两两比较点对的函数做对照。

因为：

- 外层遍历点
- 内层又遍历剩余点

所以总比较次数和 `len(L)^2` 同阶。

这时课堂已经在把几种经典增长阶直觉排开：

- constant
- linear
- logarithmic
- quadratic

### 7. `all_binary_numbers(N)`：指数增长真正变得吓人
为了让 exponential growth 也变得直观，老师再给出：

- 生成所有 N 位二进制串

这个任务本身就有：

- `2^N` 个输出

所以无论你实现得多漂亮，规模一大都会迅速爆炸。

这一步非常重要，因为它提醒你：

- 有些问题不是“实现写得差”
- 而是任务本身的输出规模就决定了下界非常大

### 8. 从 timing 转向理论语言：order of growth
做完这么多 timing 和 counting 之后，课堂终于引出：

- order of growth
- Big O
- Big Theta

老师这里要解决的问题是：

- 我们不想只记某台机器上的秒数
- 我们想比较输入变大时，增长趋势是什么

这就是 order of growth 的作用。

### 9. 为什么课程更偏爱 Theta
老师明确说更喜欢用 Theta 来描述。

原因是：

- Big O 只给上界
- 这个上界可能很松
- Theta 更强调 asymptotically tight bound

也就是说，Theta 不是随便找个长得更快的函数就完事，而是要抓住真正同阶的增长。

### 10. 定义 `n` 代表什么，比写符号更重要
这节课里老师反复追问：

- 这里的 `n` 到底是什么
- 是整数参数本身
- 还是字符串长度
- 还是列表长度
- 还是某两个列表中的某一个长度

这是复杂度分析最基础、也最容易被省略的一步。  
如果 `n` 没定义清楚，`Theta(n)` 这种写法几乎没有意义。

### 11. dominant term：抓增长最快的那一项
讲完符号意义后，老师进入实际简化。

核心规则是：

- 抓 dominant term
- 丢掉低阶项
- 丢掉常数系数

例如：

- `n^2 + log n + 2` -> `Theta(n^2)`
- `2^n + n log n + n^2` -> `Theta(2^n)`

课堂在这里的目标不是形式化证明，而是建立简化直觉。

### 12. 组合规则：顺序相加、嵌套相乘
老师还开始把代码结构和 Theta 组合联系起来：

- 顺序执行的代码块，复杂度大致相加，然后取 dominant one
- 嵌套循环或嵌套成本，复杂度往往相乘

这为下一讲从真实代码直接读复杂度打基础。

## Exercise log

> [!example] Finger exercise 22
> 官方练习是三道“化简成 Theta”：
> - `n*n + log(n) + 2**a` -> `Theta(n^2)`
> - `2**n + n*log(n) + n**2` -> `Theta(2^n)`
> - `f*log(f) + 100000 + 300*a + x*y*z` 在 `n` 这一维下 -> `Theta(1)`

这套题的价值非常高，因为它逼你明确区分：

- 哪个变量才是分析时增长的主变量
- 哪些项其实对这个主变量来说只是常数

这正是本讲理论部分最容易偷懒、但最不能偷懒的地方。

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

> [!warning] Common mistakes
> - 没定义清 `n` 就直接写 `Theta(n)`。
> - 把数值大小和输入规模混为一谈。
> - 见到 Big O / Theta 就只顾套公式，不回到代码结构。
> - 化简时把与主变量无关的项也错误保留下来。
