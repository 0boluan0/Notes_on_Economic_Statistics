---
aliases:
  - MIT 6.100L Lecture 16
  - 6.100L L16
  - Recursion on Non-Numerics
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 16
---

# Lecture 16: Recursion on Non-Numerics

> [!tip] Hint
> - 这节课开头先回顾递归，但很快就把注意力从纯数字转到 list、nested list 这类非数值对象上。
> - Fibonacci 先被拿出来复习，是为了暴露“递归定义很自然，但 naive recursion 可能重复工作得很严重”。
> - memoization 用 dict 存中间结果，正好把前面的 recursion 和 dictionaries 连了起来。
> - 课堂后半段的真正主角是 list recursion：`L[0]` 和 `L[1:]` 这套拆法会反复出现。
> - `total_recur` 是最小模板：空列表或单元素是 base case，其余交给更短的子列表。
> - `in_list` 的错误版本很关键，因为它说明递归不只要把规模缩小，还要保留必要的信息。
> - `flatten`、`deep_rev` 这些例子在告诉你：递归不仅能处理数字，也能处理嵌套结构。
> - 每次写 list recursion 时，都要先问自己 base case 是空列表、单元素列表，还是别的结构。
> - towers of Hanoi 被放在最后，是为了让你感受到递归对“分步骤搬运”类问题的天然契合。
> - 听完这节课，你应该能把 recursion 模板从 numeric 问题迁移到 sequence 和 nested structure 上。
> <!-- bilingual-en:start -->
> - The lecture briefly reviews recursion before moving from purely numerical examples to lists and nested lists.
> - Fibonacci exposes an important contrast: a recursive definition may be natural while naive recursion repeats a great deal of work.
> - Memoization stores intermediate results in a dictionary, linking recursion with the earlier dictionary material.
> - The main topic is list recursion, repeatedly decomposing a list into `L[0]` and `L[1:]`.
> - `total_recur` is the minimal pattern: handle an empty or singleton base case, then delegate the remainder to a shorter list.
> - The incorrect `in_list` is especially useful because it shows that reducing input size is not enough; each level must preserve the information the problem needs.
> - `flatten` and `deep_rev` demonstrate that recursion can traverse nested structures as well as numerical definitions.
> - Before writing list recursion, decide whether the base case is an empty list, a one-element list, or another structural boundary.
> - Towers of Hanoi closes the lecture because staged movement problems naturally decompose into smaller copies of themselves.
> - By the end, you should be able to transfer the recursive pattern from numbers to sequences and nested structures.
> <!-- bilingual-en:end -->

## Lecture flow

### 1. 先用 Fibonacci 回顾递归，但这次重点是效率
<!-- bilingual-en:start -->
*1. Revisiting Fibonacci with an Emphasis on Efficiency*
<!-- bilingual-en:end -->
Lecture 16 一开始没有立刻上 list recursion，而是先回看上一讲的递归思想。

老师选的是 Fibonacci：

- `fib(1) = 1`
- `fib(2) = 1`
- `fib(n) = fib(n-1) + fib(n-2)`

这当然是很典型的递归定义，但这次回顾的重点不再只是：

- base case 是什么
- recursive step 是什么

而是要开始看：

- 递归展开后有没有重复工作
<!-- bilingual-en:start -->
Lecture 16 begins with the familiar Fibonacci recurrence rather than list recursion. This time, the purpose is not merely to identify the base case and recursive step, but to ask whether the expanded computation repeats the same work.
<!-- bilingual-en:end -->

### 2. naive Fibonacci 很自然，但会爆出大量重复调用
<!-- bilingual-en:start -->
*2. Naive Fibonacci Is Natural but Repeats Many Calls*
<!-- bilingual-en:end -->
`fib_recur(n)` 的定义非常顺：
<!-- bilingual-en:start -->
The recursive definition of `fib_recur(n)` is direct:
<!-- bilingual-en:end -->

```python
def fib_recur(n):
    if n == 1 or n == 2:
        return 1
    else:
        return fib_recur(n - 1) + fib_recur(n - 2)
```

问题在于，一旦 trace 稍大的 `n`，你会发现：

- `fib(n-2)` 会被算很多遍
- `fib(n-3)` 会被算得更多

这说明“递归定义自然”不等于“递归实现高效”。  
这也是本讲开场比上讲多出来的一层认识。
<!-- bilingual-en:start -->
Tracing a moderately large input reveals repeated evaluations of `fib(n-2)`, `fib(n-3)`, and smaller subproblems. A natural recursive definition does not guarantee an efficient recursive implementation; that is the additional lesson in this review.
<!-- bilingual-en:end -->

### 3. memoization：把递归结果存起来，别一遍遍重算
<!-- bilingual-en:start -->
*3. Memoization: Store Recursive Results Instead of Recomputing Them*
<!-- bilingual-en:end -->
老师随后给出改进版 `fib_efficient(n, d)`。

关键点是：

- `d` 是一个 dict
- key 是已经算过的 `n`
- value 是对应 Fibonacci 值

于是逻辑变成：
<!-- bilingual-en:start -->
The improved `fib_efficient(n, d)` uses a dictionary whose keys are previously solved inputs and whose values are their Fibonacci numbers. The logic becomes:
<!-- bilingual-en:end -->

```python
if n in d:
    return d[n]
else:
    ans = fib_efficient(n - 1, d) + fib_efficient(n - 2, d)
    d[n] = ans
    return ans
```

这一步非常有代表性，因为它让你看到：

- recursion 不是孤立主题
- 它会和前面学过的 dict 结合，形成更高效的实现
<!-- bilingual-en:start -->
This example shows recursion working together with a dictionary rather than standing as an isolated topic. Cached subproblem results turn the same recurrence into a far more efficient implementation.
<!-- bilingual-en:end -->

### 4. 递归可以不止一个子调用：`score_count`
<!-- bilingual-en:start -->
*4. Recursion Can Make Several Subcalls: `score_count`*
<!-- bilingual-en:end -->
老师还给了 `score_count(x)` 这类例子。

它不是像上一讲乘法那样只有一个递归分支，而是：

- `score_count(x-1)`
- `score_count(x-2)`
- `score_count(x-3)`

一起出现。

这说明递归模式会越来越丰富：

- 有时一层只生成一个更小问题
- 有时一层会分裂成多个更小问题

理解这一点，对后面看复杂度和树状展开非常重要。
<!-- bilingual-en:start -->
`score_count(x)` branches into calls on `x-1`, `x-2`, and `x-3`. Some recursive steps produce one smaller problem; others branch into several. Recognizing that distinction is essential for understanding call trees and time complexity.
<!-- bilingual-en:end -->

### 5. 课堂正式转场：主角从数字变成 list
<!-- bilingual-en:start -->
*5. Moving from Numerical Recursion to Lists*
<!-- bilingual-en:end -->
回顾完数字递归后，老师明确说今天 main event 是 recursion on non-numerics，尤其是 lists。

转场之后，你需要把上讲的递归模板翻译成列表语言：

- 数字版常问 “离 base case 还有多远”
- list 版常问 “列表还剩多少元素没处理”

最常见拆法就是：

- `L[0]`：当前要处理的头部
- `L[1:]`：剩余更小的同类问题
<!-- bilingual-en:start -->
After the numerical review, the lecture moves to recursion on non-numerical objects, especially lists. Numerical recursion often asks how far the input is from a base value; list recursion asks how many elements remain. A common decomposition treats `L[0]` as the current element and `L[1:]` as a smaller instance of the same problem.
<!-- bilingual-en:end -->

### 6. `total_recur(L)`：list recursion 的最小模板
<!-- bilingual-en:start -->
*6. `total_recur(L)`: The Minimal List-Recursion Pattern*
<!-- bilingual-en:end -->
老师先从最简单的例子开始：递归求列表元素和。
<!-- bilingual-en:start -->
The first example recursively sums a list:
<!-- bilingual-en:end -->

```python
def total_recur(L):
    if L == []:
        return 0
    else:
        return L[0] + total_recur(L[1:])
```

这里的模式几乎是后面所有 list recursion 的母版：

- base case：空列表
- recursive step：处理头元素 + 递归处理剩余部分

> [!note]
> list recursion 最常见的套路，就是“先处理一个元素，再把问题交给更短的列表”。
> <!-- bilingual-en:start -->
> The most common list-recursion pattern handles one element and delegates the rest of the problem to a shorter list.
> <!-- bilingual-en:end -->
> <!-- bilingual-en:start -->
> The empty list is the base case; the recursive step adds the head element to the result for the remaining list. This structure becomes the template for many later examples.
> <!-- bilingual-en:end -->

### 7. `total_len_recur`：换掉“处理头元素”的方式，框架不变
<!-- bilingual-en:start -->
*7. `total_len_recur`: Changing the Head Operation without Changing the Structure*
<!-- bilingual-en:end -->
随后老师让你把同样结构用于字符串列表长度和。

核心思想完全不变，只是头元素的处理方式从：

- `L[0]`

变成：

- `len(L[0])`

这一步是在训练你别把递归模板和某个具体运算绑死。  
真正稳定的是结构，而不是里面那一个动作。
<!-- bilingual-en:start -->
The same structure can sum string lengths by replacing the contribution `L[0]` with `len(L[0])`. The exercise separates the stable recursive structure from the particular operation performed on the head element.
<!-- bilingual-en:end -->

### 8. `in_list(L, e)`：递归不仅要缩小，还要保持逻辑正确
<!-- bilingual-en:start -->
*8. `in_list(L, e)`: A Smaller Input Must Still Preserve the Logic*
<!-- bilingual-en:end -->
老师接下来故意展示了一个错误版本：
<!-- bilingual-en:start -->
The instructor deliberately presents an incorrect version:
<!-- bilingual-en:end -->

```python
def in_list(L, e):
    if len(L) == 1:
        return L[0] == e
    else:
        return in_list(L[1:], e)
```

它的问题在于：

- 每次都只看尾部递归结果
- 却没有检查当前头元素是不是 `e`

所以如果 `e` 正好在前面，它会被跳过去。

这是本讲非常值得记住的一幕，因为它说明：

- 递归不是“只会把输入切小”
- 你还必须确保每一层保留了问题真正需要的信息
<!-- bilingual-en:start -->
This function recurses on the tail but never checks whether the current head equals `e`, so an early match is discarded. Recursive input reduction is not enough; each level must perform or preserve the work required for correctness.
<!-- bilingual-en:end -->

### 9. 正确版 `in_list`：头元素检查是不可省的一层工作
<!-- bilingual-en:start -->
*9. Correcting `in_list`: The Head Check Cannot Be Omitted*
<!-- bilingual-en:end -->
修好之后的写法通常是：
<!-- bilingual-en:start -->
A corrected version is:
<!-- bilingual-en:end -->

```python
def in_list(L, e):
    if len(L) == 0:
        return False
    elif L[0] == e:
        return True
    else:
        return in_list(L[1:], e)
```

这正好展示了递归设计时两个常见判断：

- base case 是什么
- 当前层自己必须做什么

只有把这两个问题都回答清楚，递归才会既终止又正确。
<!-- bilingual-en:start -->
This version answers both design questions: what terminates the recursion, and what the current level must do. Both are necessary for termination and correctness.
<!-- bilingual-en:end -->

### 10. `flatten(L)`：开始处理嵌套结构
<!-- bilingual-en:start -->
*10. `flatten(L)`: Beginning to Process Nested Structures*
<!-- bilingual-en:end -->
课堂后半段递归真正开始变有趣的是 `flatten(L)`。

如果 `L` 的元素本身就是子列表，那么你希望输出：

- 把所有子列表里的元素按顺序拼成一个平坦列表

在简单版本里：
<!-- bilingual-en:start -->
If each element of `L` is itself a sublist, `flatten` should concatenate those elements into one flat list. A simplified version is:
<!-- bilingual-en:end -->

```python
def flatten(L):
    if len(L) == 1:
        return L[0]
    else:
        return L[0] + flatten(L[1:])
```

这里你已经不只是拆“列表长度”，还在依赖：

- 列表拼接
- 子列表本身的结构
<!-- bilingual-en:start -->
The recursion now relies not only on a shorter outer list but also on list concatenation and the structure of each sublist.
<!-- bilingual-en:end -->

### 11. `in_lists_of_list`：先看当前子列表，再看剩余子列表
<!-- bilingual-en:start -->
*11. `in_lists_of_list`: Check the Current Sublist, Then the Rest*
<!-- bilingual-en:end -->
老师又给出一个很自然的迁移题：

- `L` 是 list of lists
- 判断 `e` 是否出现在任何一个子列表里

这时递归思路是：

- 如果当前子列表里已经有 `e`，立即返回 `True`
- 否则把问题缩成“剩余子列表里有没有”

这比单层 `in_list` 多了一层嵌套，但骨架依然熟悉。
<!-- bilingual-en:start -->
For a list of lists, first test whether `e` appears in the current sublist; if not, recurse on the remaining sublists. The data add one level of nesting, but the recursive skeleton remains familiar.
<!-- bilingual-en:end -->

### 12. `my_rev`：递归也能做顺序重排
<!-- bilingual-en:start -->
*12. `my_rev`: Recursion Can Reorder a Sequence*
<!-- bilingual-en:end -->
接下来老师把递归用在反转列表顺序上：
<!-- bilingual-en:start -->
The next example reverses a list recursively:
<!-- bilingual-en:end -->

```python
def my_rev(L):
    if len(L) == 1:
        return L
    else:
        return my_rev(L[1:]) + [L[0]]
```

这个例子很重要，因为它说明递归不仅能累计数值，还能重组顺序。

写法的直觉是：

- 先把后面的部分反过来
- 再把当前头元素接到最后
<!-- bilingual-en:start -->
This shows that recursion can reorganize a sequence rather than merely accumulate a number: reverse the tail first, then append the current head to the end.
<!-- bilingual-en:end -->

### 13. `deep_rev`：递归处理“嵌套里的嵌套”
<!-- bilingual-en:start -->
*13. `deep_rev`: Recursing through Nested Levels*
<!-- bilingual-en:end -->
真正把 recursion on non-numerics 推到深水区的是 `deep_rev`。

此时列表元素可能本身还是 list，  
于是你不仅要反转顶层顺序，还要递归反转内部子列表。

这里课堂重点变成：

- 先判断当前元素是不是 list
- 如果不是，按普通元素处理
- 如果是，再对这个子列表递归调用 `deep_rev`

这类题已经很接近“树状结构递归”了。

> [!example]
> `deep_rev` 的价值在于让你真正看到：  
> 递归不是只会对一个方向不断切片，它可以随着数据结构本身的嵌套层级一起深入。
> <!-- bilingual-en:start -->
> `deep_rev` shows that recursion need not repeatedly slice in only one direction; it can descend according to the nesting of the data structure itself.
> <!-- bilingual-en:end -->
> <!-- bilingual-en:start -->
> Here, an element may itself be a list. The function must distinguish atomic elements from sublists and recursively reverse every nested list as well as the top-level order. This is already close to recursion over a tree.
> <!-- bilingual-en:end -->

### 14. Towers of Hanoi：递归适合“分阶段搬运”问题
<!-- bilingual-en:start -->
*14. Towers of Hanoi: Recursion for Staged Movement Problems*
<!-- bilingual-en:end -->
老师最后还给了 Towers of Hanoi。

它的思想特别适合递归：

1. 先把上面 `n-1` 个盘子移到 spare
2. 再把最大的盘子移到 target
3. 再把 `n-1` 个盘子从 spare 移到 target

这里递归的美感在于：

- 大问题可以拆成几个更小的同类问题
- 每一步拆法结构完全一致

所以这节课最后是在用一个更“算法味”的问题巩固递归直觉。
<!-- bilingual-en:start -->
Towers of Hanoi decomposes moving `n` disks into moving the top `n-1` disks to the spare peg, moving the largest disk to the target, and moving those `n-1` disks again. The large problem consists of smaller problems with exactly the same structure, making it a natural final reinforcement of recursive thinking.
<!-- bilingual-en:end -->

## Exercise log

> [!example] Finger exercise 16
> 官方练习就是 `flatten(L)`：
> - 输入是可能含嵌套子列表的 list
> - 返回一个完全 flatten 过的新列表
> <!-- bilingual-en:start -->
> The official exercise is `flatten(L)`:
> - The input is a list that may contain nested sublists.
> - The result is a new, completely flattened list.
> <!-- bilingual-en:end -->

官方解法很能代表本讲后半段的标准写法：
<!-- bilingual-en:start -->
The official solution captures the standard pattern from the second half of the lecture:
<!-- bilingual-en:end -->

```python
result = []
for i in L:
    if type(i) == list:
        result.extend(flatten(i))
    else:
        result.append(i)
return result
```

它和课堂上的简化版 `flatten` 相比，多了一层真正的“类型判断后递归进入子列表”。  
所以这题是本讲最核心的一道迁移练习。
<!-- bilingual-en:start -->
Compared with the simplified classroom version, it genuinely tests each element's type and recursively enters a sublist. It is the lecture's central transfer exercise for recursion over nested structures.
<!-- bilingual-en:end -->

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec16.pdf|Lecture 16 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec16_code.py|Lecture 16 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex16_sol.pdf|Lecture 16 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec16_transcript.pdf|Lecture 16 transcript]]
- Recitation 8: [[MIT 6.100L-recitations/mit6_100l_rec08.zip|Recitation 08 materials]]
- PS 4 out: [[MIT 6.100L-problem-sets/mit6_100l_ps4.pdf|PS4 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps4_code.zip|PS4 starter code]]
- PS 3 due: [[MIT 6.100L-problem-sets/mit6_100l_ps3.pdf|PS3 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps3_code.zip|PS3 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 6.2-6.4)

## Review checklist
- [ ] 我能解释为什么 naive Fibonacci 会重复计算，以及 memoization 如何缓解。
- [ ] 我能把 numeric recursion 的模板迁移到 list recursion。
- [ ] 我能说明 `L[0]` / `L[1:]` 这套拆法为什么常见。
- [ ] 我能解释 `in_list` 错误版本到底漏掉了哪一步信息。
- [ ] 我能设计 list recursion 的 base case，是空列表还是单元素列表。
- [ ] 我能读懂并手写 `flatten`、`in_lists_of_list`、`my_rev` 这类递归函数。
- [ ] 我能解释 `deep_rev` 为什么比普通 reverse 多一层递归。
- [ ] 我能说明 Towers of Hanoi 为什么天然适合递归。
- [ ] 我能把 finger exercise 16 视为“递归处理嵌套结构”的最小练习。
- [ ] 我能按课堂顺序复述：Fibonacci efficiency -> list recursion -> nested list recursion -> Hanoi。
<!-- bilingual-en:start -->
- [ ] I can explain why naive Fibonacci repeats work and how memoization reduces it.
- [ ] I can transfer the numerical-recursion pattern to list recursion.
- [ ] I can explain why the `L[0]` / `L[1:]` decomposition is common.
- [ ] I can identify the information omitted by the incorrect `in_list`.
- [ ] I can choose whether a list-recursion base case should be an empty list or a singleton.
- [ ] I can read and write recursive functions such as `flatten`, `in_lists_of_list`, and `my_rev`.
- [ ] I can explain why `deep_rev` requires another level of recursion beyond an ordinary reverse.
- [ ] I can explain why Towers of Hanoi naturally fits recursion.
- [ ] I can use finger exercise 16 as a minimal exercise in recursive processing of nested structures.
- [ ] I can reconstruct the lecture sequence: Fibonacci efficiency -> list recursion -> nested-list recursion -> Hanoi.
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 只会把 list 切小，却忘了当前层还需要检查或处理头元素。
> - 看到嵌套列表时没有先判断元素类型。
> - base case 设计得太弱，导致空列表或单元素列表处理出错。
> - 把 recursion on lists 机械理解成“永远只写 `L[0] + f(L[1:])`”，不会根据任务调整。
> <!-- bilingual-en:start -->
> - Shortening a list without checking or processing the current head element.
> - Encountering a nested list without first distinguishing a sublist from an atomic element.
> - Choosing a base case that fails on an empty or one-element list.
> - Treating list recursion as the fixed formula `L[0] + f(L[1:])` instead of adapting the decomposition to the task.
> <!-- bilingual-en:end -->
