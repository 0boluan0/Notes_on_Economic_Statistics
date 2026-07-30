---
aliases:
  - MIT 6.100L Lecture 09
  - 6.100L L09
  - Lambda Functions, Tuples, and Lists
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 09
---

# Lecture 09: Lambda Functions, Tuples and Lists

> [!tip] Hint
> - 这节课前半段先把上节“函数作为对象”推到一个极简形式，也就是 lambda。
> - lambda 的动机不是炫技，而是有些函数简单到只用一次，没必要专门写一个完整 `def`。
> - `lambda x: x % 2 == 0` 这种写法本质上还是函数对象，只是匿名。
> - `do_twice(n, fn)` 这种例子在课堂上是为了让你真正接受“函数可以像值一样被传来传去”。
> - 讲完 lambda 后，课程突然切到 tuples，不是跳话题，而是开始引入新的复合数据类型。
> - tuple 的关键词是 ordered、indexed、immutable，经常适合用来把多个返回值打包。
> - `quotient_and_remainder` 这类函数展示了 tuple 最自然的用法：一次返回多个结果。
> - `*args` 是另一个课堂转折，它说明函数参数个数本身也可以更灵活。
> - 课程最后才引入 list，故意把它先讲成“像 tuple 一样的序列”，暂时还不强调 mutability。
> - 听完这节课，你应该能解释为什么 tuples 和 lists 都是 sequence，但后面课程会把它们分开对待。

## Lecture flow

### 1. 先把上节高阶函数收尾：为什么需要 lambda
Lecture 9 的开头不是凭空冒出 lambda，而是从上节的 `apply(criteria, n)` 继续走。

上节已经学到：

- 函数是对象
- 函数可以作为参数
- 所以像 `apply(is_even, 10)` 这样的调用是合理的

但老师立刻指出一个实际问题：  
如果某个函数特别简单，而且只会用一次，那为了它单独写一个完整 `def`，看起来就有点笨重。

这时才轮到 lambda 出场。

### 2. lambda：匿名函数，只保留最小必要结构
lambda 的课堂定位非常明确：

- 它是一个函数
- 但没有名字
- 适合写非常短、非常局部的逻辑

例如：

```python
lambda x: x % 2 == 0
```

它和：

```python
def is_even(x):
    return x % 2 == 0
```

在行为上等价，只是前者没有名字。

老师在这里拆解 lambda 的时候，重点是让你读懂它的组成：

- `lambda` 关键字
- 参数列表
- 冒号
- 一个表达式

> [!note]
> lambda 不是“迷你版程序块”；它只能写一个表达式，因此天然适合极简单、临时性的函数。

### 3. `apply(lambda ..., n)`：把临时标准直接塞进去
最自然的使用方式，就是把 lambda 直接传给上节的高阶函数。

例如：

```python
apply(lambda x: x == 5, 100)
```

或者：

```python
apply(lambda x: x % 10 == 0, 100)
```

这样写时，你要真正意识到：

- `criteria` 位置需要的是一个函数对象
- lambda 表达式的结果正好就是一个函数对象

所以这里不是“语法魔法”，只是把一个临时函数直接放到了参数位置。

### 4. `do_twice(n, fn)`：函数对象真的可以被重复调用
老师接着给出：

```python
def do_twice(n, fn):
    return fn(fn(n))
```

这类代码第一次看常常会让人卡住，因为它把函数对象和普通值混在一起使用了。

正确的读法是：

1. 先对 `n` 调用一次 `fn`
2. 把结果再交给同一个 `fn`
3. 返回最终结果

例如：

```python
do_twice(3, lambda x: x**2)
```

执行过程就是：

- 先算 `fn(3)`，得到 `9`
- 再算 `fn(9)`，得到 `81`

这一步很重要，因为它让“函数能当参数”不再停留在一句口号上。

### 5. 课程此时切到 tuples：新的数据组织方式
lambda 部分讲完后，老师把课堂推进到新的 object types：**tuples** 和 **lists**。

先讲 tuple，是因为它相对简单。老师把 tuple 介绍成一种 sequence：

- 有顺序
- 可以索引
- 可以切片
- 可以遍历

但它和 list 的重要区别先被提前说出来：  
tuple 是 **immutable**。

所以课堂这里真正希望你先形成的印象是：

- 它像字符串那样可以 indexing/slicing
- 它又能装不同类型的对象
- 但创建后不能原地修改

### 6. tuple 最自然的用途：打包多个返回值
tuple 在这节课里最重要的登场方式不是“存一堆数据”，而是配合函数返回多个结果。

老师用的典型例子是：

```python
def quotient_and_remainder(x, y):
    q = x // y
    r = x % y
    return (q, r)
```

这让函数调用不再被迫只能给一个数字，而是可以给一个结构化结果。

后面再写：

```python
quot, rem = quotient_and_remainder(5, 2)
```

你就能同时拿到两个值。

> [!example]
> 这时 tuple 的意义不是“多了一个数据类型”，而是函数接口一下子灵活了很多。

### 7. `char_counts`：tuple 让返回结果更像一个小记录
课堂随后的 you-try-it `char_counts(s)` 就是在练这一点。

它要求返回：

- 元音个数
- 辅音个数

如果没有 tuple，你很难把这两个紧密相关的结果优雅地一起返回。  
有了 tuple，函数接口就可以很自然地写成：

```python
return (vowels, consonants)
```

这里也顺手强化了一件事：

- sequence 可以按位置取值
- 所以 tuple 返回值经常依赖“第一个位置是什么、第二个位置是什么”的约定

### 8. `*args`：参数个数也可以灵活
老师接下来又引入一个与函数接口相关的新点：**variable number of arguments**。

代码里最核心的对比是：

```python
def mean(*args):
    ...
```

和

```python
def mean(args):
    ...
```

这两个写法看起来很像，但意义完全不同：

- `*args`：把任意多个位置参数打包成一个 tuple
- `args`：只是普通的单个参数名

所以：

- `mean(1, 2, 3)` 对应 `*args`
- `mean((1, 2, 3))` 对应普通单参数版本

这部分的课堂重点不是背语法，而是接受：

- tuple 不只会出现在返回值里
- 也会出现在参数收集里

### 9. 先讲 list，但先把它当作“更灵活的 sequence”
讲完 tuple 后，老师才开始引入 list。

在这节课里，list 还没有被拿来重点讲 mutability；  
它先被放在“sequence 家族”里和 tuple 并列介绍。

也就是说，课堂此时先强调的是这些相似点：

- 都能 indexing
- 都能遍历
- 都能装多种类型
- 都能作为一组对象传来传去

于是代码顺着就出现了：

```python
def list_sum(L):
    total = 0
    for e in L:
        total += e
    return total
```

以及：

```python
def len_sum(L):
    total = 0
    for s in L:
        total += len(s)
    return total
```

老师在这里其实是在训练你：  
一旦进入 sequence 语境，很多处理模式都会重复出现。

### 10. `sum_and_prod`、`sublist_sum`：list 可以承载更复杂的嵌套结构
这节课最后的例子开始把 list 的灵活性往前推。

例如：

- `sum_and_prod(L)`：对同一个 list 同时累计两个量
- `sublist_sum(L)`：list 的元素本身还是 list

这在课堂上有两个效果：

- 你开始习惯 sequence 可以嵌套 sequence
- 你开始区分“返回一个值”和“返回一个结构”

到这一步，课程已经为下节的 list mutability 做好了地基。

### 11. 这节课的主线其实是“函数接口更自由了”
如果只看标题，会觉得 Lecture 9 是几块碎内容拼在一起。  
但按课堂推进来看，主线其实很清楚：

1. 函数作为对象，可以匿名化成 lambda
2. 函数接口可以接受行为本身作为参数
3. 函数接口也可以返回 tuple 这样的复合结果
4. 接下来我们需要更通用的 sequence 来装数据，于是引出 list

所以它不是跳跃，而是在慢慢把“函数接口”和“数据组织”同时扩展。

## Exercise log

> [!example] Finger exercise 09
> 官方练习是 `dot_product(tA, tB)`：
> - 输入两个等长数字 tuple
> - 返回一个 tuple
> - 第一个元素是长度
> - 第二个元素是 pairwise products 的总和

这题跟本讲高度对齐，因为它同时检查三件事：

- 你会不会按 index 并行访问两个 tuple
- 你能不能把循环累计结果写对
- 你有没有接受“函数返回值本身也可以是 tuple”

官方解法是：

```python
tot = 0
for i in range(len(tA)):
    tot += tA[i] * tB[i]
return (len(tA), tot)
```

如果你写的时候总想拆成两个函数或把长度和结果分别打印，说明这节课关于 tuple 返回值的直觉还不够稳。

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec09.pdf|Lecture 09 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec09_code.py|Lecture 09 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex09_sol.pdf|Lecture 09 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec09_transcript.pdf|Lecture 09 transcript]]
- Recitation 4: [[MIT 6.100L-recitations/mit6_100l_rec04.zip|Recitation 04 materials]]
- PS 2 out: [[MIT 6.100L-problem-sets/mit6_100l_ps2.pdf|PS2 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps2_code.zip|PS2 starter code]]
- PS 1 due: [[MIT 6.100L-problem-sets/mit6_100l_ps1.pdf|PS1 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps1_code.zip|PS1 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 5.1-5.3)

## Review checklist
- [ ] 我能说明 lambda 是怎么从上节“函数作为对象”自然生长出来的。
- [ ] 我能解释 lambda 和普通 `def` 在适用场景上的差异。
- [ ] 我能手动 trace `do_twice(3, lambda x: x**2)` 这种调用。
- [ ] 我能说清 tuple 的三个关键词：ordered、indexable、immutable。
- [ ] 我能解释为什么 tuple 很适合做多个返回值的打包。
- [ ] 我能区分 `mean(*args)` 和 `mean(args)` 的含义。
- [ ] 我能说明 list 和 tuple 在这节课里先被当成什么共同类别来看待。
- [ ] 我能读懂 `sublist_sum` 这类“list of lists”结构。
- [ ] 我能把 finger exercise 09 和 tuple 返回值的用途联系起来。
- [ ] 我能按课堂顺序复述：lambda -> tuple -> `*args` -> lists。

> [!warning] Common mistakes
> - 把 lambda 当成一种和函数完全不同的东西，而不是匿名函数。
> - 忘了 tuple 的不可变性，写出想原地修改 tuple 的代码。
> - 没区分 `*args` 是打包多个参数，而不是普通单参数。
> - 看到 list/tuple 都能索引，就忽略了后面课程会重点区分它们的 mutability。
