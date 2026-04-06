---
aliases:
  - MIT 6.100L Lecture 10
  - 6.100L L10
  - Lists and Mutability
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 10
---

# Lecture 10: Lists and Mutability

> [!tip] Hint
> - 这节课开头先把 tuple 和 list 放在一起回顾，但立刻指出今天真正的新词是 mutability。
> - `L[3] = 10` 这种写法的重点不是语法，而是“同一个 list object 在内存里被改掉了”。
> - 列表的可变性会让 assignment、append、sort、clear 这些操作都带上副作用。
> - `L = L.append(5)` 是课堂里专门拿来打你的错误示范，因为 `append` 返回 `None`。
> - 很多 list 操作会原地修改对象，而不是返回一个新列表；这和字符串、tuple 很不一样。
> - `split`/`join` 是老师用来把 string 和 list 联系起来的桥。
> - `sorted(L)` 和 `L.sort()` 的差别，是这一讲理解 mutability 的最好窗口。
> - `square_list(L)` 这类例子第一次让你正面面对“函数调用后实参本身变了”。
> - 边迭代边 append 的几个 tricky examples，是在告诉你 mutability 会和循环相互作用，产生很难查的 bug。
> - 这节课的核心不是记 API，而是建立“一个名字指向的同一个 list 可能在原地变形”的直觉。

## Lecture flow

### 1. 先复习 list，但马上把焦点转到 mutability
Lecture 10 一开场，老师先承认一件事：

- tuple 上节已经介绍过
- list 上节也看过一些基础操作

但今天不会再把它们并排讲太久，因为真正新、也真正麻烦的地方是：

> [!note]
> list 是 mutable object。

这意味着：

- 创建以后还能改
- 改的是对象本身
- 不是“生成一个改过的新值再替换掉旧值”

课程后面所有关于 aliasing、cloning、side effects 的讨论，都是从这里长出来的。

### 2. `L[i] = value`：第一次正式看到“原地修改”
老师最先抓住的例子就是：

```python
L = [2, 4, 3]
L[1] = 5
```

这行代码之所以重要，是因为它和我们之前熟悉的 assignment 不完全一样。

在前几讲里，左边通常是变量名；  
现在左边变成了“列表中某个位置”。

这代表：

- 不是给 `L` 重新绑定一个新对象
- 而是进入 `L` 指向的那个 list object
- 把其中一个元素换掉

老师在这里专门说会在内存层面解释，因为如果不把“对象本身被改掉”这件事想清楚，后面会一直混乱。

### 3. list 和 tuple 看起来像，但行为逻辑已经分叉
课堂接着回顾 list 和 tuple 的相似点：

- 都能装多种类型
- 都能 indexing
- 都能 slicing
- 都能遍历

但现在需要强行把差异钉牢：

- tuple：immutable
- list：mutable

这不是一个小注释，而是会改变你对很多操作的预期。

例如字符串和 tuple 的很多操作，本质上是在生成新对象；  
list 上的很多操作，却是在原对象上直接修改。

### 4. `append` 的教学重点是副作用，不是“多了一个元素”
老师很快拿 `append` 做示范：

```python
L = [2, 4, 3]
L.append(5)
```

这时最该记住的不是 list 尾部多了一个元素，而是：

- `append` 是 method
- 它改变 `L` 这个对象本身
- 它的返回值是 `None`

所以这一讲最经典的坑才会成立：

```python
L = L.append(5)
```

因为 `append` 并不会把“新列表”返回给你，结果 `L` 被重新绑定成 `None`。

> [!warning]
> 这行错代码之所以常见，不是因为你不会 `append`，而是因为你还在下意识把 list 操作想成“返回新值”的风格。

### 5. 第一批 you-try-it：先用 list 造新结构
在真正进入更棘手的副作用之前，课堂先给了两个更平稳的函数题：

- `make_ordered_list(n)`
- `remove_elem(L, e)`

这两题的用处是让你熟悉：

- 新建空列表
- 循环扫描已有元素
- 有条件地 `append`
- 最后返回新列表

这里其实有一个很重要的过渡：

- 你可以写“返回新列表”的 list 函数
- 也可以写“原地修改原列表”的 list 函数

这两种风格在后面课程里会不断被拿来对比。

### 6. string 和 list 之间来回切换：`split` / `join`
老师中段花了一块内容讲 string-list operations。

例如：

- `list(s)`：把字符串拆成字符列表
- `s.split(' ')`：按空格切成单词列表
- `''.join(L)`：把字符列表重新拼成字符串

这一段课堂很像技术细节，但其实在训练一种数据表示转换思维：

- 有时字符串方便存储和显示
- 有时 list 方便逐元素处理
- 你需要知道如何在两种表示之间切换

比如 `count_words(sen)` 的最自然解法就是：

```python
words = sen.split(' ')
return len(words)
```

### 7. `sorted(L)` vs `L.sort()`：同样叫“排序”，语义完全不同
老师随后把 mutability 讲得更具体，方法是对比两个排序接口：

```python
a = sorted(L)
```

和

```python
a = L.sort()
```

它们的表面效果都和“排序”有关，但语义不同：

- `sorted(L)`：返回一个排好序的新列表，不改原列表
- `L.sort()`：原地排序，改原列表，返回 `None`

这组对比几乎就是理解 mutability 的最佳练习。

如果你能稳定区分这两个写法，后面就更容易理解：

- 为什么有些函数能安全写进表达式
- 为什么有些 method 只该单独写成一行

### 8. `square_list(L)`：函数也能带来 list 的副作用
课堂后面进入一个更关键的例子：

```python
def square_list(L):
    for i in range(len(L)):
        L[i] = L[i]**2
```

如果你调用：

```python
Lin = [2, 3, 4]
square_list(Lin)
```

调用结束后，`Lin` 本身变成 `[4, 9, 16]`。

这说明：

- 函数参数传进来的是对同一个 list object 的引用
- 在函数内部做原地修改
- 外部那个对象也会跟着变

这一步实际上把“函数”和“mutability”两条线合并了。  
从这里开始，写函数时你必须思考：

- 我是要返回新对象
- 还是要修改传入对象

### 9. 课堂故意安排了几个 tricky examples
老师后面连续放了几个边迭代边修改列表的例子，因为这是 mutability 最容易制造 bug 的地方。

典型情况包括：

- 对 `range(len(L))` 迭代时 append
- 对 `for e in L` 迭代时 append
- 在循环里不断 `L = L + L`

这些例子有的会：

- 无限增长
- 无限循环
- 或者行为和直觉不一样

它们想传达的核心不是“记住哪一段坏代码”，而是：

> [!warning]
> 只要你一边遍历 list，一边又修改这个 list，本轮循环到底会看到哪些元素，就必须重新分析，不能再凭直觉猜。

### 10. `clear()` 和 `L = []`：看起来都清空，实际上不同
老师最后又安排了一个非常重要的对比：

```python
L.clear()
```

和

```python
L = []
```

它们看起来都像“把列表变空”，但从对象角度完全不同：

- `clear()`：清空同一个 list object
- `L = []`：让名字 `L` 重新绑定到一个新空列表

所以如果别的名字也指向原来的那个 list，  
这两种写法的后果会完全不同。

这正是下一讲 aliasing/cloning 的正式入口。

### 11. 这节课建立的是对象直觉，不只是 list API
Lecture 10 如果只记成“学了 append、sort、split、join”，会丢掉最重要的部分。

这节课真正完成的是：

- 让你接受 list 是 mutable object
- 让你看到 method 常常带副作用
- 让你意识到函数也可能通过参数修改原对象
- 让你开始对“同一个对象”和“新建对象”保持敏感

## Exercise log

> [!example] Finger exercise 10
> 官方题目是 `all_true(n, Lf)`：
> - `n` 是整数
> - `Lf` 是“接收 int、返回 bool”的函数列表
> - 如果列表里的每个函数在输入 `n` 时都返回 `True`，才返回 `True`

这题放在 Lecture 10 里很有意思，因为它同时用到了：

- list 作为“装函数对象的容器”
- 遍历 list
- 布尔累积逻辑

官方解法大意是：

```python
flag = True
for f in Lf:
    if not f(n):
        flag = False
        break
return flag
```

它提醒你，list 不只是装数字和字符串，也可以装“行为”。  
所以这题正好把 Lecture 8 的 functions-as-objects 和 Lecture 10 的 lists 接上了。

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec10.pdf|Lecture 10 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec10_code.py|Lecture 10 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex10_sol.pdf|Lecture 10 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec10_transcript.pdf|Lecture 10 transcript]]
- Recitation 5: [[MIT 6.100L-recitations/mit6_100l_rec05.zip|Recitation 05 materials]]
- PS 2 halfway hand-in due: [[MIT 6.100L-problem-sets/mit6_100l_ps2.pdf|PS2 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps2_code.zip|PS2 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 5.3-5.5)

## Review checklist
- [ ] 我能解释 list 的 mutability 到底意味着什么。
- [ ] 我能说明 `L[i] = value` 为什么是原地修改而不是重建整个列表。
- [ ] 我能区分 `append` / `sort` / `clear` 这类原地 method 和返回新对象的函数。
- [ ] 我能解释为什么 `L = L.append(5)` 会出错。
- [ ] 我能区分 `sorted(L)` 和 `L.sort()` 的语义差异。
- [ ] 我能说明 `square_list(L)` 为什么会让调用者手里的列表一起改变。
- [ ] 我能解释为什么边遍历边修改 list 容易出错。
- [ ] 我能说清 `clear()` 和 `L = []` 的差别。
- [ ] 我能把 finger exercise 10 和“list 也可以装函数对象”联系起来。
- [ ] 我能按课堂顺序复述这节课：mutability -> methods with side effects -> list/string conversions -> tricky iteration cases。

> [!warning] Common mistakes
> - 把 list method 一律当成“返回新列表”的操作。
> - 在不知道返回值的情况下写 `x = some_list_method(...)`。
> - 一边遍历列表一边修改它，却没有重新分析循环行为。
> - 写函数时没说明自己到底是在返回新列表还是修改原列表。
