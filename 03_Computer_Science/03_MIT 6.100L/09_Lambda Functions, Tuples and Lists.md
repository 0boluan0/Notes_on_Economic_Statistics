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
> <!-- bilingual-en:start -->
> - The first half pushes the previous lecture's idea of functions as objects into the minimal form of a lambda.
> - Lambdas are not introduced for show. Some functions are so small and so local that a complete `def` would be unnecessarily heavy.
> - `lambda x: x % 2 == 0` still produces a function object; the function simply has no name.
> - Examples such as `do_twice(n, fn)` make concrete the claim that functions can be passed around like other values.
> - The move from lambdas to tuples begins the introduction of new compound data types rather than abandoning the earlier topic.
> - A tuple is ordered, indexed, and immutable, and it is especially useful for packaging several return values.
> - `quotient_and_remainder` demonstrates that natural use by returning two related results together.
> - `*args` marks another transition: even the number of arguments accepted by a function can be flexible.
> - Lists appear only at the end and are initially presented as sequences like tuples; their mutability is deliberately left for the next lecture.
> - By the end, you should understand why tuples and lists are both sequences while anticipating why their different mutability will matter.
> <!-- bilingual-en:end -->

## Lecture flow

### 1. 先把上节高阶函数收尾：为什么需要 lambda
<!-- bilingual-en:start -->
*1. Completing the Higher-Order Function Story: Why Lambdas Are Useful*
<!-- bilingual-en:end -->
Lecture 9 的开头不是凭空冒出 lambda，而是从上节的 `apply(criteria, n)` 继续走。
<!-- bilingual-en:start -->
Lecture 9 introduces lambdas by continuing directly from the earlier `apply(criteria, n)` example.
<!-- bilingual-en:end -->

上节已经学到：

- 函数是对象
- 函数可以作为参数
- 所以像 `apply(is_even, 10)` 这样的调用是合理的
<!-- bilingual-en:start -->
The previous lecture established that functions are objects and may be supplied as arguments, making a call such as `apply(is_even, 10)` meaningful.
<!-- bilingual-en:end -->

但老师立刻指出一个实际问题：  
如果某个函数特别简单，而且只会用一次，那为了它单独写一个完整 `def`，看起来就有点笨重。
<!-- bilingual-en:start -->
The practical problem is that writing a complete `def` is cumbersome for a function that is extremely simple and used only once.
<!-- bilingual-en:end -->

这时才轮到 lambda 出场。
<!-- bilingual-en:start -->
That is the role filled by a lambda.
<!-- bilingual-en:end -->

### 2. lambda：匿名函数，只保留最小必要结构
<!-- bilingual-en:start -->
*2. Lambdas: Anonymous Functions with Only the Essential Structure*
<!-- bilingual-en:end -->
lambda 的课堂定位非常明确：

- 它是一个函数
- 但没有名字
- 适合写非常短、非常局部的逻辑
<!-- bilingual-en:start -->
A lambda is a function without a name, suited to very short logic needed only in a local context.
<!-- bilingual-en:end -->

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
<!-- bilingual-en:start -->
The lambda and the named `is_even` function behave equivalently; only the former is anonymous.
<!-- bilingual-en:end -->

老师在这里拆解 lambda 的时候，重点是让你读懂它的组成：

- `lambda` 关键字
- 参数列表
- 冒号
- 一个表达式
<!-- bilingual-en:start -->
Its parts are the `lambda` keyword, a parameter list, a colon, and one expression.
<!-- bilingual-en:end -->

> [!note]
> lambda 不是“迷你版程序块”；它只能写一个表达式，因此天然适合极简单、临时性的函数。
> <!-- bilingual-en:start -->
> A lambda is not a miniature statement block. It contains one expression, which makes it naturally suited to simple, temporary functions.
> <!-- bilingual-en:end -->

### 3. `apply(lambda ..., n)`：把临时标准直接塞进去
<!-- bilingual-en:start -->
*3. `apply(lambda ..., n)`: Supplying a Temporary Criterion Directly*
<!-- bilingual-en:end -->
最自然的使用方式，就是把 lambda 直接传给上节的高阶函数。
<!-- bilingual-en:start -->
The most natural first use is to pass a lambda directly to the preceding lecture's higher-order function.
<!-- bilingual-en:end -->

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
<!-- bilingual-en:start -->
The `criteria` argument requires a function object, and evaluating a lambda expression produces exactly such an object.
<!-- bilingual-en:end -->

所以这里不是“语法魔法”，只是把一个临时函数直接放到了参数位置。
<!-- bilingual-en:start -->
There is no syntactic magic: a temporary function has simply been placed in an argument position.
<!-- bilingual-en:end -->

### 4. `do_twice(n, fn)`：函数对象真的可以被重复调用
<!-- bilingual-en:start -->
*4. `do_twice(n, fn)`: Calling a Function Object Repeatedly*
<!-- bilingual-en:end -->
老师接着给出：
<!-- bilingual-en:start -->
The instructor then gives:
<!-- bilingual-en:end -->

```python
def do_twice(n, fn):
    return fn(fn(n))
```

这类代码第一次看常常会让人卡住，因为它把函数对象和普通值混在一起使用了。
<!-- bilingual-en:start -->
This code can initially be disorienting because function objects and ordinary values appear in the same expression.
<!-- bilingual-en:end -->

正确的读法是：

1. 先对 `n` 调用一次 `fn`
2. 把结果再交给同一个 `fn`
3. 返回最终结果
<!-- bilingual-en:start -->
Read it in three steps: apply `fn` to `n`, apply the same `fn` to that result, and return the second result.
<!-- bilingual-en:end -->

例如：

```python
do_twice(3, lambda x: x**2)
```

执行过程就是：

- 先算 `fn(3)`，得到 `9`
- 再算 `fn(9)`，得到 `81`
<!-- bilingual-en:start -->
For `do_twice(3, lambda x: x**2)`, the first call gives `9` and the second gives `81`.
<!-- bilingual-en:end -->

这一步很重要，因为它让“函数能当参数”不再停留在一句口号上。
<!-- bilingual-en:start -->
The example turns “functions can be arguments” from a slogan into an execution trace.
<!-- bilingual-en:end -->

### 5. 课程此时切到 tuples：新的数据组织方式
<!-- bilingual-en:start -->
*5. Turning to Tuples: A New Way to Organize Data*
<!-- bilingual-en:end -->
lambda 部分讲完后，老师把课堂推进到新的 object types：**tuples** 和 **lists**。
<!-- bilingual-en:start -->
After lambdas, the lecture moves to two new object types: **tuples** and **lists**.
<!-- bilingual-en:end -->

先讲 tuple，是因为它相对简单。老师把 tuple 介绍成一种 sequence：

- 有顺序
- 可以索引
- 可以切片
- 可以遍历
<!-- bilingual-en:start -->
Tuples come first because they are relatively simple sequences: they are ordered, support indexing and slicing, and can be traversed.
<!-- bilingual-en:end -->

但它和 list 的重要区别先被提前说出来：  
tuple 是 **immutable**。
<!-- bilingual-en:start -->
The essential contrast with a list is stated immediately: a tuple is **immutable**.
<!-- bilingual-en:end -->

所以课堂这里真正希望你先形成的印象是：

- 它像字符串那样可以 indexing/slicing
- 它又能装不同类型的对象
- 但创建后不能原地修改
<!-- bilingual-en:start -->
Like a string, it supports indexing and slicing; unlike a homogeneous numeric array, it can contain objects of different types; once created, however, it cannot be changed in place.
<!-- bilingual-en:end -->

### 6. tuple 最自然的用途：打包多个返回值
<!-- bilingual-en:start -->
*6. A Natural Use of Tuples: Packaging Multiple Return Values*
<!-- bilingual-en:end -->
tuple 在这节课里最重要的登场方式不是“存一堆数据”，而是配合函数返回多个结果。
<!-- bilingual-en:start -->
The tuple's most important role here is not merely storing several items, but allowing a function to return several related results together.
<!-- bilingual-en:end -->

老师用的典型例子是：

```python
def quotient_and_remainder(x, y):
    q = x // y
    r = x % y
    return (q, r)
```

这让函数调用不再被迫只能给一个数字，而是可以给一个结构化结果。
<!-- bilingual-en:start -->
The function can now return a structured result rather than a single number.
<!-- bilingual-en:end -->

后面再写：

```python
quot, rem = quotient_and_remainder(5, 2)
```

你就能同时拿到两个值。
<!-- bilingual-en:start -->
Tuple unpacking then assigns the two results at once.
<!-- bilingual-en:end -->

> [!example]
> 这时 tuple 的意义不是“多了一个数据类型”，而是函数接口一下子灵活了很多。
> <!-- bilingual-en:start -->
> The tuple matters here because it makes the function interface substantially more expressive, not merely because it adds another data type.
> <!-- bilingual-en:end -->

### 7. `char_counts`：tuple 让返回结果更像一个小记录
<!-- bilingual-en:start -->
*7. `char_counts`: Using a Tuple as a Small Record*
<!-- bilingual-en:end -->
课堂随后的 you-try-it `char_counts(s)` 就是在练这一点。
<!-- bilingual-en:start -->
The subsequent `char_counts(s)` exercise practices exactly this use.
<!-- bilingual-en:end -->

它要求返回：

- 元音个数
- 辅音个数
<!-- bilingual-en:start -->
It must return both the number of vowels and the number of consonants.
<!-- bilingual-en:end -->

如果没有 tuple，你很难把这两个紧密相关的结果优雅地一起返回。  
有了 tuple，函数接口就可以很自然地写成：
<!-- bilingual-en:start -->
Without a tuple, returning those closely related results together would be awkward. With one, the interface is simply:
<!-- bilingual-en:end -->

```python
return (vowels, consonants)
```

这里也顺手强化了一件事：

- sequence 可以按位置取值
- 所以 tuple 返回值经常依赖“第一个位置是什么、第二个位置是什么”的约定
<!-- bilingual-en:start -->
The example also reinforces that sequences are accessed by position, so a tuple return value relies on an agreed meaning for its first, second, and later fields.
<!-- bilingual-en:end -->

### 8. `*args`：参数个数也可以灵活
<!-- bilingual-en:start -->
*8. `*args`: Allowing a Variable Number of Arguments*
<!-- bilingual-en:end -->
老师接下来又引入一个与函数接口相关的新点：**variable number of arguments**。
<!-- bilingual-en:start -->
The instructor next extends function interfaces to accept a **variable number of arguments**.
<!-- bilingual-en:end -->

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
<!-- bilingual-en:start -->
The two signatures look similar but mean very different things.
<!-- bilingual-en:end -->

- `*args`：把任意多个位置参数打包成一个 tuple
- `args`：只是普通的单个参数名
<!-- bilingual-en:start -->
- `*args` packages any number of positional arguments into a tuple.
- `args` is merely the name of one ordinary argument.
<!-- bilingual-en:end -->

所以：

- `mean(1, 2, 3)` 对应 `*args`
- `mean((1, 2, 3))` 对应普通单参数版本
<!-- bilingual-en:start -->
Thus, `mean(1, 2, 3)` matches the `*args` version, whereas `mean((1, 2, 3))` supplies one tuple to the ordinary single-argument version.
<!-- bilingual-en:end -->

这部分的课堂重点不是背语法，而是接受：

- tuple 不只会出现在返回值里
- 也会出现在参数收集里
<!-- bilingual-en:start -->
The important point is that tuples package not only return values, but also collections of supplied arguments.
<!-- bilingual-en:end -->

### 9. 先讲 list，但先把它当作“更灵活的 sequence”
<!-- bilingual-en:start -->
*9. Introducing Lists First as More Flexible Sequences*
<!-- bilingual-en:end -->
讲完 tuple 后，老师才开始引入 list。
<!-- bilingual-en:start -->
Lists are introduced only after tuples.
<!-- bilingual-en:end -->

在这节课里，list 还没有被拿来重点讲 mutability；  
它先被放在“sequence 家族”里和 tuple 并列介绍。
<!-- bilingual-en:start -->
This lecture does not yet center their mutability; instead, it places lists beside tuples in the broader sequence family.
<!-- bilingual-en:end -->

也就是说，课堂此时先强调的是这些相似点：

- 都能 indexing
- 都能遍历
- 都能装多种类型
- 都能作为一组对象传来传去
<!-- bilingual-en:start -->
The immediate similarities are that both support indexing and traversal, can contain several types of object, and can be passed around as a collection.
<!-- bilingual-en:end -->

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
<!-- bilingual-en:start -->
The examples train you to recognize that the same processing patterns recur across sequence types.
<!-- bilingual-en:end -->

### 10. `sum_and_prod`、`sublist_sum`：list 可以承载更复杂的嵌套结构
<!-- bilingual-en:start -->
*10. `sum_and_prod` and `sublist_sum`: Lists Can Contain Nested Structure*
<!-- bilingual-en:end -->
这节课最后的例子开始把 list 的灵活性往前推。
<!-- bilingual-en:start -->
The final examples develop the flexibility of lists further.
<!-- bilingual-en:end -->

例如：

- `sum_and_prod(L)`：对同一个 list 同时累计两个量
- `sublist_sum(L)`：list 的元素本身还是 list
<!-- bilingual-en:start -->
- `sum_and_prod(L)` accumulates two quantities from one list.
- In `sublist_sum(L)`, the elements of the list are themselves lists.
<!-- bilingual-en:end -->

这在课堂上有两个效果：

- 你开始习惯 sequence 可以嵌套 sequence
- 你开始区分“返回一个值”和“返回一个结构”
<!-- bilingual-en:start -->
You begin to treat sequences as nestable and to distinguish returning one scalar value from returning a structure.
<!-- bilingual-en:end -->

到这一步，课程已经为下节的 list mutability 做好了地基。
<!-- bilingual-en:start -->
This establishes the groundwork for the next lecture on list mutability.
<!-- bilingual-en:end -->

### 11. 这节课的主线其实是“函数接口更自由了”
<!-- bilingual-en:start -->
*11. The Unifying Thread: More Expressive Function Interfaces*
<!-- bilingual-en:end -->
如果只看标题，会觉得 Lecture 9 是几块碎内容拼在一起。  
但按课堂推进来看，主线其实很清楚：
<!-- bilingual-en:start -->
The title can make Lecture 9 look fragmented, but its progression has a clear theme:
<!-- bilingual-en:end -->

1. 函数作为对象，可以匿名化成 lambda
2. 函数接口可以接受行为本身作为参数
3. 函数接口也可以返回 tuple 这样的复合结果
4. 接下来我们需要更通用的 sequence 来装数据，于是引出 list
<!-- bilingual-en:start -->

&nbsp;
**1.** Because functions are objects, an anonymous lambda can represent a small one.<br>
**2.** A function interface can accept behavior itself as an argument.<br>
**3.** It can return a compound result packaged in a tuple.<br>
**4.** The need for more general-purpose sequences then motivates lists.<br>
<!-- bilingual-en:end -->

所以它不是跳跃，而是在慢慢把“函数接口”和“数据组织”同时扩展。
<!-- bilingual-en:start -->
The lecture is not jumping between unrelated topics; it is expanding function interfaces and data organization in parallel.
<!-- bilingual-en:end -->

## Exercise log

> [!example] Finger exercise 09
> 官方练习是 `dot_product(tA, tB)`：
> - 输入两个等长数字 tuple
> - 返回一个 tuple
> - 第一个元素是长度
> - 第二个元素是 pairwise products 的总和
> <!-- bilingual-en:start -->
> The official exercise is `dot_product(tA, tB)`:
> - It receives two equal-length numeric tuples.
> - It returns a tuple.
> - The first returned element is the length.
> - The second is the sum of the pairwise products.
> <!-- bilingual-en:end -->

这题跟本讲高度对齐，因为它同时检查三件事：

- 你会不会按 index 并行访问两个 tuple
- 你能不能把循环累计结果写对
- 你有没有接受“函数返回值本身也可以是 tuple”
<!-- bilingual-en:start -->
The exercise tests parallel indexed access to two tuples, correct loop accumulation, and acceptance of a tuple as the function's return value.
<!-- bilingual-en:end -->

官方解法是：
<!-- bilingual-en:start -->
The official solution is:
<!-- bilingual-en:end -->

```python
tot = 0
for i in range(len(tA)):
    tot += tA[i] * tB[i]
return (len(tA), tot)
```

如果你写的时候总想拆成两个函数或把长度和结果分别打印，说明这节课关于 tuple 返回值的直觉还不够稳。
<!-- bilingual-en:start -->
If you keep trying to split the task into two functions or print the length and result separately, the idea of a tuple-valued return has not yet settled.
<!-- bilingual-en:end -->

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
<!-- bilingual-en:start -->
- [ ] I can explain how lambdas grow naturally from the idea that functions are objects.
- [ ] I can distinguish the appropriate use cases for a lambda and a regular `def`.
- [ ] I can manually trace a call such as `do_twice(3, lambda x: x**2)`.
- [ ] I can name the three key properties of a tuple: ordered, indexable, and immutable.
- [ ] I can explain why tuples are useful for packaging multiple return values.
- [ ] I can distinguish `mean(*args)` from `mean(args)`.
- [ ] I can identify the common sequence category through which lists and tuples are introduced here.
- [ ] I can read a nested list structure such as `sublist_sum`.
- [ ] I can connect finger exercise 09 to the purpose of tuple return values.
- [ ] I can reconstruct the lecture sequence: lambdas -> tuples -> `*args` -> lists.
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 把 lambda 当成一种和函数完全不同的东西，而不是匿名函数。
> - 忘了 tuple 的不可变性，写出想原地修改 tuple 的代码。
> - 没区分 `*args` 是打包多个参数，而不是普通单参数。
> - 看到 list/tuple 都能索引，就忽略了后面课程会重点区分它们的 mutability。
> <!-- bilingual-en:start -->
> - Treating a lambda as something fundamentally different from a function rather than as an anonymous function.
> - Forgetting that tuples are immutable and attempting to modify one in place.
> - Failing to distinguish `*args`, which packages multiple arguments, from an ordinary single parameter.
> - Seeing that lists and tuples both support indexing and overlooking the mutability distinction developed in the next lecture.
> <!-- bilingual-en:end -->
