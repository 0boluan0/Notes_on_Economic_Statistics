---
aliases:
  - MIT 6.100L Lecture 08
  - 6.100L L08
  - Functions as Objects
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 08
---

# Lecture 08: Functions as Objects

> [!tip] Hint
> - 这节课先回顾上节函数语法，但立刻把重点移到“函数调用之后到底返回了什么”。
> - `return` 被老师讲得很具体：它会立刻终止函数执行，并把一个值传回调用点。
> - 即使你没写 `return`，函数也不是“不返回”，而是隐式返回 `None`。
> - `print` 是给人看的，`return` 是给程序后续步骤用的，这里开始被正式拆开。
> - `is_triangular` 的 buggy 代码在训练的不是三角数，而是“什么时候该 return，什么时候只是 print”。
> - 课堂把上节的 bisection 包成 `bisection_root`，是在示范如何把一整段算法变成一个函数对象。
> - scope 例子不是枝节，而是在说明局部变量、外层变量、读与改为什么不同。
> - 真正的新东西是：函数和 int、str 一样，也是 object，所以可以作为参数传给别的函数。
> - `calc(op, x, y)`、`apply(criteria, n)` 这些例子在为后面的 lambda、高阶函数和 testing 做准备。
> - 听完这节课，你应该能解释“函数为什么能当参数”而不只是背一句“Python 万物皆对象”。

## Lecture flow

### 1. 开场先回顾函数，但重点放在“函数调用的结果”
这节课一开始确实回顾了上节的函数语法，不过老师的重心已经变了。

上节课重点是：

- 为什么要 decomposition
- 函数怎么定义
- docstring 是什么

这节课开始把注意力放到函数运行时：

- 调用一个函数时到底发生了什么
- 它返回什么
- 这个返回值会怎样进入后续计算

所以 Lecture 8 不是单纯继续讲函数语法，而是在把“函数是程序部件”推进成“函数是可以被传递和操控的对象”。

### 2. `is_even_with_return` vs `is_even_without_return`
老师第一个正式对比的例子非常直接：  
两个几乎一样的函数，唯一差别是有没有 `return`。

```python
def is_even_with_return(i):
    print('with return')
    remainder = i % 2
    return remainder == 0

def is_even_without_return(i):
    print('without return')
    remainder = i % 2
    has_rem = (remainder == 0)
    print(has_rem)
```

课堂在这里强调了两个判断：

- 写了 `return`，函数调用会被替换成那个返回值
- 没写 `return`，函数会隐式返回 `None`

这就是为什么：

```python
print(is_even_with_return(3))
print(is_even_without_return(3))
```

打印出来的行为完全不同。

> [!note]
> Python 中“没有显式 return”不等于“没有返回值”，而是“返回 `None`”。

### 3. `print` 和 `return` 在程序里服务不同对象
老师接着把这个差异讲得更彻底。

- `print(...)` 是副作用，目的是把东西显示到终端
- `return ...` 是把值交给调用者

所以如果你写：

```python
def mult(x, y):
    print(x * y)
```

那它确实会把乘积显示出来，但它**没有把这个值交给别的代码**。  
一旦你把它嵌进更大的表达式，就会出问题，因为函数调用的结果其实是 `None`。

这时课堂里的核心转折是：

> [!warning]
> 初学者最容易犯的错之一，是把“屏幕上看到了结果”误当成“程序里真的得到了这个值”。

### 4. `is_triangular` 的 bug：错误不在数学，而在控制流
接下来的 you-try-it 是 `is_triangular(n)`。

表面上它是在判断一个数是不是 triangular number，  
但这道题课堂真正想训练的是函数控制流。

原始 buggy 版本的问题是：

- 在循环里找到答案时只是 `print(True)`
- 循环结束后无论如何又 `print(False)`
- 它没有正确地利用 `return` 来结束函数

所以修复这类 bug 的关键不是“再多写几个 if”，而是想清楚：

- 什么时候已经足够确定答案
- 该不该立刻终止函数
- 结果应不应该返回而不是打印

### 5. 把旧算法封装成函数：`bisection_root`
接着老师把前几讲已经学过的二分平方根包装成函数：

```python
def bisection_root(x):
    epsilon = 0.01
    low = 0
    high = x
    ans = (high + low) / 2.0
    while abs(ans**2 - x) >= epsilon:
        if ans**2 < x:
            low = ans
        else:
            high = ans
        ans = (high + low) / 2.0
    return ans
```

这个例子很重要，因为它第一次把一整段算法变成可复用部件。  
课堂这里其实在说：

- 之前你会写算法
- 现在你要学会把算法封成函数
- 这样别的函数才能直接调用它

后面 `count_nums_with_sqrt_close_to` 这种题就是建立在这一步上的。

### 6. 一个函数可以调用另一个函数，于是程序开始真正组合起来
一旦 `bisection_root` 被封好，后面的函数就可以站在更高层去写。

例如：

- 我不再关心二分法内部怎么收缩区间
- 我只把它当成“给我一个数，返回它平方根近似值”的工具

这其实就是上节 abstraction 的兑现版本。  
Lecture 7 讲的是理念，Lecture 8 开始把它变成一种实际写法。

### 7. scope：局部名字、外层名字、能读不能随便改
讲完 `return` 之后，课堂转去讲 **scope**。

老师用几个很短的函数例子说明：

- 函数内部创建的变量默认是局部的
- 可以读取外层已经存在的名字
- 但如果你在函数内部给某个名字赋值，Python 会把它当局部变量处理
- 所以“读外层变量”和“改外层变量”不是一回事

例如：

```python
def g(y):
    print(x)
    print(x + 1)

def h(y):
    x += 1
```

`g` 里只是读 `x`，如果外面有 `x`，它能工作；  
`h` 里既想读又想写 `x`，Python 会把它判定成局部变量，于是报错。

> [!note]
> scope 这部分的意义不是记语法细节，而是理解每个函数调用都有自己独立的小环境。

### 8. 真正的新内容：函数和别的值一样，也是 object
到这里课程才进入标题里的核心：**functions as objects**。

老师明确说出一个观念：

- int 是对象
- str 是对象
- list 是对象
- function 也是对象

既然函数有名字、可以被引用，那它就可以像别的对象一样被：

- 赋给变量
- 作为参数传递
- 从一个函数传进另一个函数

这并不是神秘规则，而是 Python 对函数的一种统一处理方式。

### 9. `calc(op, x, y)`：把“做什么操作”当成输入
老师用 `calc` 例子把这个观念真正落地：

```python
def calc(op, x, y):
    return op(x, y)
```

然后再定义：

```python
def add(a, b):
    return a + b

def sub(a, b):
    return a - b
```

这里最关键的理解是：

- `op` 不是字符串
- `op` 不是运算符符号
- `op` 是一个函数对象

所以 `calc(add, 2, 3)` 的意思就是：

1. 把函数 `add` 传进去
2. 在 `calc` 内部调用 `op(x, y)`
3. 实际上也就是调用 `add(2, 3)`

### 10. `apply(criteria, n)`：把“判断标准”本身变成参数
更典型的高阶函数例子是：

```python
def apply(criteria, n):
    count = 0
    for i in range(0, n + 1):
        if criteria(i):
            count += 1
    return count
```

这段代码非常值得停下来理解，因为它第一次把“行为”当成输入。

`criteria` 在这里不是一个布尔值，而是：

- 一个接受数字
- 返回布尔值
- 表示筛选标准的函数

于是：

- `apply(is_even, 10)` 统计的是 0 到 10 里偶数有多少个
- 如果以后传入别的条件函数，就能统计别的东西

> [!example]
> 这就是抽象层级提升的瞬间：  
> `apply` 不再关心“偶数”是什么，它只关心“给我一个判定标准，我就按这个标准数数”。

### 11. 课堂最后已经在为 lambda 铺路
Lecture 8 虽然还没有系统讲 lambda，但已经在制造那个需求。

比如如果某个标准非常简单，只会用一次，那么为了配合 `apply` 去专门写一个完整 `def`，看起来就有点重。

这正是 Lecture 9 要继续解决的问题。  
所以这节课最后的重要收束其实是：

- 函数可以返回值
- 函数可以作为参数
- 函数调用发生在独立 scope 中
- 因此函数本身已经变成可以操作的对象

## Exercise log

> [!example] Finger exercise 08
> 官方练习是 `same_chars(s1, s2)`：如果 `s1` 中的每个字符都在 `s2` 中出现，且 `s2` 中的每个字符也都在 `s1` 中出现，则返回 `True`。

这题表面上是字符串扫描，实际却正好卡在本讲的两个重点上：

- 你要写的是一个真正 **返回布尔值** 的函数，而不是打印中间判断
- 你要清楚函数接口只承诺“比较字符集合关系”，不承诺比较出现次数

官方解法是两段对称循环：

- 先检查 `s1` 的字符是否都在 `s2`
- 再检查 `s2` 的字符是否都在 `s1`

它适合放在本讲后面，因为这就是最典型的“按 specification 写函数并正确 return”。

如果你写着写着想 `print(True)` 或 `print(False)`，说明本讲最核心的区分还没站稳。

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec08.pdf|Lecture 08 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec08_code.py|Lecture 08 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex08_sol.pdf|Lecture 08 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec08_transcript.pdf|Lecture 08 transcript]]
- Recitation 4: [[MIT 6.100L-recitations/mit6_100l_rec04.zip|Recitation 04 materials]]
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 4.3-4.6)

## Review checklist
- [ ] 我能解释显式 `return`、隐式 `None`、`print` 三者的区别。
- [ ] 我能说明为什么 `print(mult(...))` 这种写法经常暴露出函数没有返回值的问题。
- [ ] 我能解释 `is_triangular` 这类 bug 为什么本质上是控制流错误而不只是公式错误。
- [ ] 我能说明为什么把 bisection 封成函数后，别的函数就能站在更高层组织代码。
- [ ] 我能说清 scope 的基本规则：局部变量、读取外层变量、在函数里赋值会触发什么结果。
- [ ] 我能解释为什么函数可以作为参数传给别的函数。
- [ ] 我能读懂 `calc(op, x, y)` 和 `apply(criteria, n)` 这类高阶函数。
- [ ] 我能解释“函数作为对象”到底带来了什么编程上的好处，而不是只会背定义。
- [ ] 我能把 finger exercise 08 和本讲的 `return` / 函数接口联系起来。
- [ ] 我能按课堂顺序复述：回顾函数 -> return vs print -> scope -> functions as objects。

> [!warning] Common mistakes
> - 函数内部把结果打印出来，就误以为调用者已经拿到了那个值。
> - 没写 `return` 却以为函数“什么都没返回”。
> - 搞不清 scope，看到函数里出现同名变量就以为和外面的一定是同一个东西。
> - 传给高阶函数的不是函数对象本身，而是不小心先把函数调用掉了。
