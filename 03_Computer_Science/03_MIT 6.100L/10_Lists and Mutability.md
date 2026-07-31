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
> <!-- bilingual-en:start -->
> - This lesson begins by reviewing tuples and lists together, but quickly emphasizes mutability as today's key concept.
> - The point of writing `L[3] = 10` is not about syntax, but about modifying the same list object in memory.
> - Because lists are mutable, element assignment and methods such as `append`, `sort`, and `clear` have side effects.
> - `L = L.append(5)` is a classic error example because `append()` returns `None`, not the modified list.
> - Many list operations mutate an existing object rather than returning a new list; strings and tuples cannot be modified in place.
> - `split` and `join` provide a bridge between string and list representations.
> - The contrast between `sorted(L)`, which returns a new list, and `L.sort()`, which sorts in place, is a particularly clear window into mutability.
> - Examples such as `square_list(L)` confront you with the fact that a function can change the argument object held by its caller.
> - Deliberately tricky examples that append during iteration show how mutation and loops can interact to create subtle bugs.
> - The central goal is not memorizing APIs, but learning to see a name as a reference to a list object whose contents may change in place.
> <!-- bilingual-en:end -->

## Lecture flow

### 1. 先复习 list，但马上把焦点转到 mutability
<!-- bilingual-en:start -->
*1. Reviewing Lists Before Focusing on Mutability*
<!-- bilingual-en:end -->
Lecture 10 一开场，老师先承认一件事：
<!-- bilingual-en:start -->
The instructor begins by locating the lecture relative to the previous one:
<!-- bilingual-en:end -->

- tuple 上节已经介绍过
- list 上节也看过一些基础操作
<!-- bilingual-en:start -->
- Tuples were introduced in the previous lecture.
- Some basic list operations have already appeared as well.
<!-- bilingual-en:end -->

但今天不会再把它们并排讲太久，因为真正新、也真正麻烦的地方是：
<!-- bilingual-en:start -->
The comparison is brief because the genuinely new—and difficult—idea is:
<!-- bilingual-en:end -->

> [!note]
> list 是 mutable object。
> <!-- bilingual-en:start -->
> Lists are mutable objects.
> <!-- bilingual-en:end -->

这意味着：
<!-- bilingual-en:start -->
This means:
<!-- bilingual-en:end -->

- 创建以后还能改
- 改的是对象本身
- 不是“生成一个改过的新值再替换掉旧值”
<!-- bilingual-en:start -->
- A list can be changed after it is created.
- The operation changes the existing object itself.
- It does not merely construct a modified value and replace the old one.
<!-- bilingual-en:end -->

课程后面所有关于 aliasing、cloning、side effects 的讨论，都是从这里长出来的。
<!-- bilingual-en:start -->
The later discussions of aliasing, cloning, and side effects all follow from this fact.
<!-- bilingual-en:end -->

### 2. `L[i] = value`：第一次正式看到“原地修改”
<!-- bilingual-en:start -->
*2. `L[i] = value`: The First Explicit In-Place Modification*
<!-- bilingual-en:end -->
老师最先抓住的例子就是：
<!-- bilingual-en:start -->
The instructor begins with:
<!-- bilingual-en:end -->

```python
L = [2, 4, 3]
L[1] = 5
```

这行代码之所以重要，是因为它和我们之前熟悉的 assignment 不完全一样。
<!-- bilingual-en:start -->
This assignment differs from the forms used earlier in the course.
<!-- bilingual-en:end -->

在前几讲里，左边通常是变量名；  
现在左边变成了“列表中某个位置”。
<!-- bilingual-en:start -->
Previously, the left-hand side was usually a variable name; here it designates a position within a list.
<!-- bilingual-en:end -->

这代表：
<!-- bilingual-en:start -->
The operation therefore means:
<!-- bilingual-en:end -->

- 不是给 `L` 重新绑定一个新对象
- 而是进入 `L` 指向的那个 list object
- 把其中一个元素换掉
<!-- bilingual-en:start -->
- It does not rebind `L` to a new object.
- It follows `L` to the existing list object.
- It replaces one element within that object.
<!-- bilingual-en:end -->

老师在这里专门说会在内存层面解释，因为如果不把“对象本身被改掉”这件事想清楚，后面会一直混乱。
<!-- bilingual-en:start -->
The memory-level account matters because the rest of the topic remains confusing unless the mutation of the object itself is clear.
<!-- bilingual-en:end -->

### 3. list 和 tuple 看起来像，但行为逻辑已经分叉
<!-- bilingual-en:start -->
*3. Lists and Tuples Look Similar but Behave Differently*
<!-- bilingual-en:end -->
课堂接着回顾 list 和 tuple 的相似点：
<!-- bilingual-en:start -->
The lecture reviews the similarities between lists and tuples:
<!-- bilingual-en:end -->

- 都能装多种类型
- 都能 indexing
- 都能 slicing
- 都能遍历
<!-- bilingual-en:start -->
- Both can contain objects of different types.
- Both support indexing.
- Both support slicing.
- Both can be traversed.
<!-- bilingual-en:end -->

但现在需要强行把差异钉牢：
<!-- bilingual-en:start -->
The crucial difference must now remain explicit:
<!-- bilingual-en:end -->

- tuple：immutable
- list：mutable

这不是一个小注释，而是会改变你对很多操作的预期。
<!-- bilingual-en:start -->
This is not a minor qualification; it changes what many operations can be expected to do.
<!-- bilingual-en:end -->

例如字符串和 tuple 的很多操作，本质上是在生成新对象；  
list 上的很多操作，却是在原对象上直接修改。
<!-- bilingual-en:start -->
For example, many operations on strings and tuples create new objects; however, many list operations directly modify the original object.
<!-- bilingual-en:end -->

### 4. `append` 的教学重点是副作用，不是“多了一个元素”
<!-- bilingual-en:start -->
*4. `append` Is Primarily a Lesson About Side Effects*
<!-- bilingual-en:end -->
老师很快拿 `append` 做示范：
<!-- bilingual-en:start -->
The instructor demonstrates `append`:
<!-- bilingual-en:end -->

```python
L = [2, 4, 3]
L.append(5)
```

这时最该记住的不是 list 尾部多了一个元素，而是：
<!-- bilingual-en:start -->
The important point is not merely that one element has been added, but that:
<!-- bilingual-en:end -->

- `append` 是 method
- 它改变 `L` 这个对象本身
- 它的返回值是 `None`
<!-- bilingual-en:start -->
- `append` is a method.
- It mutates the object referenced by `L`.
- It returns `None`.
<!-- bilingual-en:end -->

所以这一讲最经典的坑才会成立：
<!-- bilingual-en:start -->
These facts produce the lecture's classic mistake:
<!-- bilingual-en:end -->

```python
L = L.append(5)
```

因为 `append` 并不会把“新列表”返回给你，结果 `L` 被重新绑定成 `None`。
<!-- bilingual-en:start -->
Because `append` does not return a new list, the assignment rebinds `L` to `None`.
<!-- bilingual-en:end -->

> [!warning]
> 这行错代码之所以常见，不是因为你不会 `append`，而是因为你还在下意识把 list 操作想成“返回新值”的风格。
> <!-- bilingual-en:start -->
> This mistake is common not because `append` is obscure, but because it is easy to assume unconsciously that every list operation returns a new value.
> <!-- bilingual-en:end -->

### 5. 第一批 you-try-it：先用 list 造新结构
<!-- bilingual-en:start -->
*5. First “You Try It” Exercises: Constructing New Lists*
<!-- bilingual-en:end -->
在真正进入更棘手的副作用之前，课堂先给了两个更平稳的函数题：
<!-- bilingual-en:start -->
Before confronting subtler side effects, the lecture uses two simpler function exercises:
<!-- bilingual-en:end -->

- `make_ordered_list(n)`
- `remove_elem(L, e)`

这两题的用处是让你熟悉：
<!-- bilingual-en:start -->
They provide practice in:
<!-- bilingual-en:end -->

- 新建空列表
- 循环扫描已有元素
- 有条件地 `append`
- 最后返回新列表
<!-- bilingual-en:start -->
- Creating an empty list.
- Traversing existing elements.
- Calling `append` conditionally.
- Returning the completed new list.
<!-- bilingual-en:end -->

这里其实有一个很重要的过渡：
<!-- bilingual-en:start -->
The exercises also introduce an important distinction:
<!-- bilingual-en:end -->

- 你可以写“返回新列表”的 list 函数
- 也可以写“原地修改原列表”的 list 函数
<!-- bilingual-en:start -->
- A function can return a newly constructed list.
- A function can instead mutate the list supplied by its caller.
<!-- bilingual-en:end -->

这两种风格在后面课程里会不断被拿来对比。
<!-- bilingual-en:start -->
Later lectures repeatedly compare these two interface styles.
<!-- bilingual-en:end -->

### 6. string 和 list 之间来回切换：`split` / `join`
<!-- bilingual-en:start -->
*6. Moving Between String and List Representations with `split` and `join`*
<!-- bilingual-en:end -->
老师中段花了一块内容讲 string-list operations。
<!-- bilingual-en:start -->
The middle of the lecture covers conversions between strings and lists.
<!-- bilingual-en:end -->

例如：
<!-- bilingual-en:start -->
For example:
<!-- bilingual-en:end -->

- `list(s)`：把字符串拆成字符列表
- `s.split(' ')`：按空格切成单词列表
- `''.join(L)`：把字符列表重新拼成字符串
<!-- bilingual-en:start -->
- `list(s)` converts a string into a list of characters.
- `s.split(' ')` separates it into words at spaces.
- `''.join(L)` combines a character list into a string again.
<!-- bilingual-en:end -->

这一段课堂很像技术细节，但其实在训练一种数据表示转换思维：
<!-- bilingual-en:start -->
These details develop a broader habit of changing data representations deliberately:
<!-- bilingual-en:end -->

- 有时字符串方便存储和显示
- 有时 list 方便逐元素处理
- 你需要知道如何在两种表示之间切换
<!-- bilingual-en:start -->
- Strings may be convenient for storage and display.
- Lists may be convenient for element-by-element processing.
- A program often needs to move between the two representations.
<!-- bilingual-en:end -->

比如 `count_words(sen)` 的最自然解法就是：
<!-- bilingual-en:start -->
For example, the most natural solution for `count_words(sen)` is:
<!-- bilingual-en:end -->

```python
words = sen.split(' ')
return len(words)
```

### 7. `sorted(L)` vs `L.sort()`：同样叫“排序”，语义完全不同
<!-- bilingual-en:start -->
*7. `sorted(L)` versus `L.sort()`: Similar Purpose, Different Semantics*
<!-- bilingual-en:end -->
老师随后把 mutability 讲得更具体，方法是对比两个排序接口：
<!-- bilingual-en:start -->
The instructor makes mutability more concrete by comparing two sorting interfaces:
<!-- bilingual-en:end -->

```python
a = sorted(L)
```

和
<!-- bilingual-en:start -->
with
<!-- bilingual-en:end -->

```python
a = L.sort()
```

它们的表面效果都和“排序”有关，但语义不同：
<!-- bilingual-en:start -->
Both concern sorting, but their semantics differ:
<!-- bilingual-en:end -->

- `sorted(L)`：返回一个排好序的新列表，不改原列表
- `L.sort()`：原地排序，改原列表，返回 `None`
<!-- bilingual-en:start -->
- `sorted(L)` returns a new sorted list without changing the original.
- `L.sort()` sorts the original list in place and returns `None`.
<!-- bilingual-en:end -->

这组对比几乎就是理解 mutability 的最佳练习。
<!-- bilingual-en:start -->
This is one of the clearest exercises for understanding mutability.
<!-- bilingual-en:end -->

如果你能稳定区分这两个写法，后面就更容易理解：
<!-- bilingual-en:start -->
Distinguishing them reliably also clarifies:
<!-- bilingual-en:end -->

- 为什么有些函数能安全写进表达式
- 为什么有些 method 只该单独写成一行
<!-- bilingual-en:start -->
- Why some function calls can safely participate in larger expressions.
- Why certain mutating methods should normally appear as standalone statements.
<!-- bilingual-en:end -->

### 8. `square_list(L)`：函数也能带来 list 的副作用
<!-- bilingual-en:start -->
*8. `square_list(L)`: A Function Can Mutate Its List Argument*
<!-- bilingual-en:end -->
课堂后面进入一个更关键的例子：
<!-- bilingual-en:start -->
The lecture then turns to a more consequential example:
<!-- bilingual-en:end -->

```python
def square_list(L):
    for i in range(len(L)):
        L[i] = L[i]**2
```

如果你调用：
<!-- bilingual-en:start -->
Given the call:
<!-- bilingual-en:end -->

```python
Lin = [2, 3, 4]
square_list(Lin)
```

调用结束后，`Lin` 本身变成 `[4, 9, 16]`。
<!-- bilingual-en:start -->
After the call, `Lin` itself contains `[4, 9, 16]`.
<!-- bilingual-en:end -->

这说明：
<!-- bilingual-en:start -->
This demonstrates that:
<!-- bilingual-en:end -->

- 函数参数传进来的是对同一个 list object 的引用
- 在函数内部做原地修改
- 外部那个对象也会跟着变
<!-- bilingual-en:start -->
- The parameter refers to the same list object held by the caller.
- The function mutates that object in place.
- The caller consequently observes the changed contents.
<!-- bilingual-en:end -->

这一步实际上把“函数”和“mutability”两条线合并了。  
从这里开始，写函数时你必须思考：
<!-- bilingual-en:start -->
This example joins the topics of functions and mutability. From now on, every list-processing interface should answer:
<!-- bilingual-en:end -->

- 我是要返回新对象
- 还是要修改传入对象
<!-- bilingual-en:start -->
- Does the function return a new object?
- Or does it mutate the object supplied by the caller?
<!-- bilingual-en:end -->

### 9. 课堂故意安排了几个 tricky examples
<!-- bilingual-en:start -->
*9. Deliberately Tricky Examples of Mutation During Iteration*
<!-- bilingual-en:end -->
老师后面连续放了几个边迭代边修改列表的例子，因为这是 mutability 最容易制造 bug 的地方。
<!-- bilingual-en:start -->
The instructor presents several examples that mutate a list during traversal, one of the easiest ways for mutability to produce bugs.
<!-- bilingual-en:end -->

典型情况包括：
<!-- bilingual-en:start -->
The cases include:
<!-- bilingual-en:end -->

- 对 `range(len(L))` 迭代时 append
- 对 `for e in L` 迭代时 append
- 在循环里不断 `L = L + L`
<!-- bilingual-en:start -->
- Appending while iterating over `range(len(L))`.
- Appending during `for e in L`.
- Repeatedly evaluating `L = L + L` inside a loop.
<!-- bilingual-en:end -->

这些例子有的会：
<!-- bilingual-en:start -->
Depending on the iteration mechanism, the result may be:
<!-- bilingual-en:end -->

- 无限增长
- 无限循环
- 或者行为和直觉不一样
<!-- bilingual-en:start -->
- Unbounded growth.
- A nonterminating loop.
- Behavior that differs from a casual intuition.
<!-- bilingual-en:end -->

它们想传达的核心不是“记住哪一段坏代码”，而是：
<!-- bilingual-en:start -->
The point is not to memorize particular bad snippets, but to adopt this rule:
<!-- bilingual-en:end -->

> [!warning]
> 只要你一边遍历 list，一边又修改这个 list，本轮循环到底会看到哪些元素，就必须重新分析，不能再凭直觉猜。
> <!-- bilingual-en:start -->
> Whenever you modify a list while iterating over it, you must reanalyze which elements the loop will visit on that pass; intuition alone is no longer reliable.
> <!-- bilingual-en:end -->

### 10. `clear()` 和 `L = []`：看起来都清空，实际上不同
<!-- bilingual-en:start -->
*10. `clear()` versus `L = []`: Similar Appearance, Different Object Semantics*
<!-- bilingual-en:end -->
老师最后又安排了一个非常重要的对比：
<!-- bilingual-en:start -->
The instructor closes with another important contrast:
<!-- bilingual-en:end -->

```python
L.clear()
```

和
<!-- bilingual-en:start -->
and
<!-- bilingual-en:end -->

```python
L = []
```

它们看起来都像“把列表变空”，但从对象角度完全不同：
<!-- bilingual-en:start -->
Both appear to empty a list, but they do fundamentally different things to objects and names:
<!-- bilingual-en:end -->

- `clear()`：清空同一个 list object
- `L = []`：让名字 `L` 重新绑定到一个新空列表
<!-- bilingual-en:start -->
- `clear()` empties the existing list object.
- `L = []` rebinds the name `L` to a newly created empty list.
<!-- bilingual-en:end -->

所以如果别的名字也指向原来的那个 list，  
这两种写法的后果会完全不同。
<!-- bilingual-en:start -->
If other names still refer to the original list, the two operations therefore have different observable consequences.
<!-- bilingual-en:end -->

这正是下一讲 aliasing/cloning 的正式入口。
<!-- bilingual-en:start -->
This distinction leads directly into the next lecture on aliasing and cloning.
<!-- bilingual-en:end -->

### 11. 这节课建立的是对象直觉，不只是 list API
<!-- bilingual-en:start -->
*11. Building an Object Model Rather Than Memorizing List APIs*
<!-- bilingual-en:end -->
Lecture 10 如果只记成“学了 append、sort、split、join”，会丢掉最重要的部分。
<!-- bilingual-en:start -->
Reducing Lecture 10 to `append`, `sort`, `split`, and `join` misses its central contribution.
<!-- bilingual-en:end -->

这节课真正完成的是：
<!-- bilingual-en:start -->
The lecture instead:
<!-- bilingual-en:end -->

- 让你接受 list 是 mutable object
- 让你看到 method 常常带副作用
- 让你意识到函数也可能通过参数修改原对象
- 让你开始对“同一个对象”和“新建对象”保持敏感
<!-- bilingual-en:start -->
- Establishes that a list is a mutable object.
- Shows that methods often have side effects.
- Demonstrates that a function may mutate an object through its parameter.
- Builds sensitivity to the difference between changing an existing object and creating a new one.
<!-- bilingual-en:end -->

## Exercise log

> [!example] Finger exercise 10
> 官方题目是 `all_true(n, Lf)`：
> - `n` 是整数
> - `Lf` 是“接收 int、返回 bool”的函数列表
> - 如果列表里的每个函数在输入 `n` 时都返回 `True`，才返回 `True`
> <!-- bilingual-en:start -->
> The official problem is `all_true(n, Lf)`:
> - `n` is an integer
> - `Lf` is a list of functions from integers to Boolean values.
> - Return `True` only if every function in the list returns `True` when applied to `n`.
> <!-- bilingual-en:end -->

这题放在 Lecture 10 里很有意思，因为它同时用到了：
<!-- bilingual-en:start -->
Its placement in Lecture 10 is revealing because it combines:
<!-- bilingual-en:end -->

- list 作为“装函数对象的容器”
- 遍历 list
- 布尔累积逻辑
<!-- bilingual-en:start -->
- A list used as a container for function objects.
- Traversal of that list.
- Boolean accumulation with early termination.
<!-- bilingual-en:end -->

官方解法大意是：
<!-- bilingual-en:start -->
The official solution is essentially:
<!-- bilingual-en:end -->

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
<!-- bilingual-en:start -->
A list can contain behavior in the form of function objects, not only numbers and strings. The exercise therefore connects Lecture 8's functions-as-objects idea with Lecture 10's lists.
<!-- bilingual-en:end -->

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
<!-- bilingual-en:start -->
- [ ] I can explain what the mutability of lists really means.
- [ ] I can explain why `L[i] = value` modifies the list in place rather than creating a new one.
- [ ] I can distinguish between methods like `append`, `sort`, and `clear` (which modify the list in place) and functions that return new objects.
- [ ] I can explain why `L = L.append(5)` leads to an error.
- [ ] I can clarify the semantic difference between `sorted(L)` and `L.sort()`.
- [ ] I can explain why calling `square_list(L)` would change the list the caller has.
- [ ] I can explain why modifying a list while iterating over it is problematic.
- [ ] I can distinguish between `clear()` and `L = []`.
- [ ] I can connect finger exercise 10 with the idea that lists can hold function objects.
- [ ] I can reconstruct the lecture sequence: mutability -> methods with side effects -> list/string conversions -> tricky iteration cases.
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 把 list method 一律当成“返回新列表”的操作。
> - 在不知道返回值的情况下写 `x = some_list_method(...)`。
> - 一边遍历列表一边修改它，却没有重新分析循环行为。
> - 写函数时没说明自己到底是在返回新列表还是修改原列表。
> <!-- bilingual-en:start -->
> - Treating all list methods as operations that return new lists.
> - Writing `x = some_list_method(...)` without knowing what it returns.
> - Modifying a list while iterating over it without re-evaluating the loop behavior.
> - Failing to specify in functions whether you are returning a new list or modifying the original one.
> <!-- bilingual-en:end -->
