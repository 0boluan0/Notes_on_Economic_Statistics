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

## Lecture flow

### 1. 先把上节规则重新说一遍
Lecture 23 开场先回顾上节最后几条最重要的分析原则：

- 先定义输入规模
- 只关心增长趋势
- 抓 dominant term
- 丢掉加法常数和乘法常数

老师这样做很合理，因为本讲几乎全是在这些规则上做实战。

### 2. Constant class：最快速的是“不随输入规模增长”
老师先从最简单的常数类开始。

典型例子：

```python
def add(x, y):
    return x + y

def convert_to_km(m):
    return m * 1.609
```

关键点在于：

- 无论输入值本身是大是小
- 执行步骤数大致不变

所以它们属于 `Theta(1)`。

### 3. Linear class：输入规模增大一倍，工作量也大致跟着增一倍
接下来老师列出一组线性例子：

- `mul(x, y)` 对 `y` 来说是线性的
- `add_digits(s)` 对字符串长度线性
- `fact_iter(a)` 对 `a` 线性
- `fact_recur(x)` 对 `x` 线性
- `compound(..., n_months)` 对月份数线性
- `fib_iter(n)` 对 `n` 线性

这组例子特别重要，因为它强调：

- “线性”不是指所有参数都线性
- 而是指相对某个输入规模维度线性

> [!note]
> 复杂度符号里的变量不是固定叫 `n` 就完事，必须和具体输入含义对应起来。

### 4. 同样是线性，问题规模定义却可以完全不同
老师在这一部分一直逼你说清：

- `Theta(y)`
- `Theta(len(s))`
- `Theta(n_months)`

为什么这些看起来都像 linear，但不能混写？

因为：

- 输入对象不同
- 增长维度不同
- 所以分析变量必须明说

这一步是本讲最重要的习惯训练之一。

### 5. Polynomial / Quadratic：嵌套扫描开始出现
接着课堂切到二次复杂度。

典型函数有：

- `g(n)`：双重循环
- `is_subset(L1, L2)`
- `intersect(L1, L2)`
- `diameter(L)`

它们共同特征是：

- 某一层工作里又包含一层与输入规模相关的完整扫描

尤其像 `is_subset(L1, L2)` 这类题，老师在强调：

- 不能只看有几个循环
- 还要看每层循环跑多长

### 6. Exponential：最容易失控的一类
老师随后用两类经典函数展示指数复杂度：

- `gen_subsets(L)`
- `fib_recur(x)`

它们的共同点是：

- 每层调用会分叉成多个子调用
- 整体展开像一棵快速膨胀的树

因此即使代码很短，复杂度也可能极高。  
这再次提醒你：代码行数和复杂度没有直接关系。

### 7. Logarithmic：每次都大幅缩小问题规模
在 logarithmic 一类里，老师拿：

- `digit_sum(n)`（通过位数理解）
- 后面的二分搜索

来帮助大家建立直觉。

这类函数的共同点是：

- 每一步都把剩余问题砍掉一大块
- 所以总步数是“能砍多少次才见底”

### 8. 搜索算法再回归：这次重点是复杂度分类
后半段课堂回到 searching。

老师先放：

- `linear_search(L, e)`：无序列表线性扫
- `search(L, e)`：有序列表上线性扫，但可提前停

然后再引出：

- `bisect_search1`
- `bisect_search2`

这里的主问题不是“谁更快”这句口号，而是：

- 为什么是这个复杂度
- 有哪些额外成本

### 9. `bisect_search1`：切片版递归会带来复制成本
`bisect_search1` 的写法里用到了：

- `L[:half]`
- `L[half:]`

这说明每次递归除了逻辑判断，还在做切片复制。  
所以老师把它单独拿出来很有意义，因为它提醒你：

- 递归本身之外，辅助操作也可能影响复杂度和常数项

### 10. `bisect_search2`：索引版更贴近真正的二分思路
相对地，`bisect_search2` 用的是：

- `low`
- `high`
- `mid`

以及一个 helper function。

它更接近真正的二分搜索实现，因为：

- 不复制子列表
- 只是缩小索引区间

这让你看到复杂度分析不只是“这是不是递归”，还要看递归每层具体做了什么。

### 11. 这节课是在做代码阅读训练
Lecture 23 的整体感觉会比前一讲更“碎”，因为它几乎没有一个单一大主题例子，而是很多 ছোট代码。

但这些例子其实都服务同一个目标：

- 训练你从真实代码结构直接读出 complexity class

所以本讲的正确学习方式不是背完整张表，而是每看到一个函数，都问：

1. 输入规模怎么定义
2. 哪些语句依赖输入
3. 是顺序相加还是嵌套相乘
4. 是否有递归分叉或规模折半

## Exercise log

> [!example] Finger exercise 23
> 官方练习给三段代码，让你判断 worst-case Theta：
> - `running_product(a)` -> `Theta(n)`
> - `tricky_f(L, L2)` -> `Theta(n^2)`
> - `sum_f(n)` -> `Theta(log n)`

这三题选得很准，因为它们分别覆盖：

- 简单线性循环
- 成员测试嵌套导致的平方级
- 数字按位缩小导致的对数级

如果这三题你能独立解释为什么，不只是选对答案，那本讲主线就基本吃透了。

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

> [!warning] Common mistakes
> - 只看循环层数，不看每层边界和辅助操作。
> - 看到递归就笼统写成指数或线性，不先分析调用结构。
> - 忽略切片、成员测试这类看似小但可能昂贵的操作。
> - 把“平均情况可能提前停”误当成“最坏情况就不是线性”。
