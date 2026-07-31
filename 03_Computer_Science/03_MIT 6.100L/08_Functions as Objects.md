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
> <!-- bilingual-en:start -->
> - The lecture briefly reviews function syntax, then turns immediately to what a function call actually returns.
> - The instructor gives `return` a precise operational meaning: it ends the current call immediately and sends a value back to the call site.
> - A function without an explicit `return` still returns a value—`None`.
> - `print` displays information for a person; `return` supplies a value for the program's subsequent computation.
> - The faulty `is_triangular` code is not mainly about triangular numbers. It teaches when to return and when merely printing is insufficient.
> - Wrapping the earlier bisection algorithm in `bisection_root` shows how an entire algorithm becomes a reusable function object.
> - The scope examples explain why local and enclosing names differ and why reading an outer name is not the same as assigning to it.
> - The genuinely new idea is that a function, like an integer or string, is an object and can therefore be passed as an argument.
> - `calc(op, x, y)` and `apply(criteria, n)` prepare the way for lambdas, higher-order functions, and testing.
> - By the end, you should be able to explain why a function can be an argument rather than merely repeat that “everything in Python is an object.”
> <!-- bilingual-en:end -->

## Lecture flow

### 1. 开场先回顾函数，但重点放在“函数调用的结果”
<!-- bilingual-en:start -->
*1. Reviewing Functions with a New Focus on the Result of a Call*
<!-- bilingual-en:end -->
这节课一开始确实回顾了上节的函数语法，不过老师的重心已经变了。
<!-- bilingual-en:start -->
The lecture does review the preceding function syntax, but its emphasis has changed.
<!-- bilingual-en:end -->

上节课重点是：

- 为什么要 decomposition
- 函数怎么定义
- docstring 是什么
<!-- bilingual-en:start -->
Lecture 7 centered on decomposition, function definitions, and docstrings.
<!-- bilingual-en:end -->

这节课开始把注意力放到函数运行时：

- 调用一个函数时到底发生了什么
- 它返回什么
- 这个返回值会怎样进入后续计算
<!-- bilingual-en:start -->
Lecture 8 asks what happens during a call, what value the call returns, and how that value participates in later computation.
<!-- bilingual-en:end -->

所以 Lecture 8 不是单纯继续讲函数语法，而是在把“函数是程序部件”推进成“函数是可以被传递和操控的对象”。
<!-- bilingual-en:start -->
The subject is no longer merely function syntax: a function moves from being a program component to being an object that can itself be referenced, passed, and manipulated.
<!-- bilingual-en:end -->

### 2. `is_even_with_return` vs `is_even_without_return`
<!-- bilingual-en:start -->
*2. `is_even_with_return` versus `is_even_without_return`*
<!-- bilingual-en:end -->
老师第一个正式对比的例子非常直接：  
两个几乎一样的函数，唯一差别是有没有 `return`。
<!-- bilingual-en:start -->
The first comparison uses two nearly identical functions whose only difference is the presence of `return`.
<!-- bilingual-en:end -->

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
<!-- bilingual-en:start -->
With `return`, the function call evaluates to the returned value. Without one, the function implicitly returns `None`.
<!-- bilingual-en:end -->

这就是为什么：

```python
print(is_even_with_return(3))
print(is_even_without_return(3))
```

打印出来的行为完全不同。
<!-- bilingual-en:start -->
That difference explains why printing the results of the two calls produces different output.
<!-- bilingual-en:end -->

> [!note]
> Python 中“没有显式 return”不等于“没有返回值”，而是“返回 `None`”。
> <!-- bilingual-en:start -->
> In Python, “no explicit return” does not mean “no return value”; it means the call returns `None`.
> <!-- bilingual-en:end -->

### 3. `print` 和 `return` 在程序里服务不同对象
<!-- bilingual-en:start -->
*3. `print` and `return` Serve Different Consumers*
<!-- bilingual-en:end -->
老师接着把这个差异讲得更彻底。
<!-- bilingual-en:start -->
The instructor then sharpens the distinction.
<!-- bilingual-en:end -->

- `print(...)` 是副作用，目的是把东西显示到终端
- `return ...` 是把值交给调用者
<!-- bilingual-en:start -->
- `print(...)` is a side effect that displays something in the terminal.
- `return ...` hands a value to the caller.
<!-- bilingual-en:end -->

所以如果你写：

```python
def mult(x, y):
    print(x * y)
```

那它确实会把乘积显示出来，但它**没有把这个值交给别的代码**。  
一旦你把它嵌进更大的表达式，就会出问题，因为函数调用的结果其实是 `None`。
<!-- bilingual-en:start -->
The function displays the product, but it does **not make that product available to other code**. In a larger expression, the call contributes `None`, not the displayed number.
<!-- bilingual-en:end -->

这时课堂里的核心转折是：

> [!warning]
> 初学者最容易犯的错之一，是把“屏幕上看到了结果”误当成“程序里真的得到了这个值”。
> <!-- bilingual-en:start -->
> A common beginner error is to mistake “I saw the result on screen” for “the program received that value.”
> <!-- bilingual-en:end -->

### 4. `is_triangular` 的 bug：错误不在数学，而在控制流
<!-- bilingual-en:start -->
*4. The `is_triangular` Bug: A Control-Flow Error, Not a Mathematical One*
<!-- bilingual-en:end -->
接下来的 you-try-it 是 `is_triangular(n)`。

表面上它是在判断一个数是不是 triangular number，  
但这道题课堂真正想训练的是函数控制流。
<!-- bilingual-en:start -->
Although the exercise appears to test whether a number is triangular, its real subject is function control flow.
<!-- bilingual-en:end -->

原始 buggy 版本的问题是：

- 在循环里找到答案时只是 `print(True)`
- 循环结束后无论如何又 `print(False)`
- 它没有正确地利用 `return` 来结束函数
<!-- bilingual-en:start -->
The faulty version prints `True` when it finds an answer, prints `False` after the loop regardless, and never uses `return` to end the function at the decisive moment.
<!-- bilingual-en:end -->

所以修复这类 bug 的关键不是“再多写几个 if”，而是想清楚：

- 什么时候已经足够确定答案
- 该不该立刻终止函数
- 结果应不应该返回而不是打印
<!-- bilingual-en:start -->
Repairing it requires deciding when enough evidence is available, whether the function should stop immediately, and whether the result belongs in a return value rather than terminal output.
<!-- bilingual-en:end -->

### 5. 把旧算法封装成函数：`bisection_root`
<!-- bilingual-en:start -->
*5. Encapsulating an Existing Algorithm in `bisection_root`*
<!-- bilingual-en:end -->
接着老师把前几讲已经学过的二分平方根包装成函数：
<!-- bilingual-en:start -->
The instructor next wraps the earlier bisection square-root algorithm in a function:
<!-- bilingual-en:end -->

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
<!-- bilingual-en:start -->
For the first time, a complete algorithm becomes a reusable component.
<!-- bilingual-en:end -->

- 之前你会写算法
- 现在你要学会把算法封成函数
- 这样别的函数才能直接调用它
<!-- bilingual-en:start -->
Knowing how to write an algorithm is followed by knowing how to encapsulate it so that other functions can call it directly.
<!-- bilingual-en:end -->

后面 `count_nums_with_sqrt_close_to` 这种题就是建立在这一步上的。
<!-- bilingual-en:start -->
Later functions such as `count_nums_with_sqrt_close_to` depend on this step.
<!-- bilingual-en:end -->

### 6. 一个函数可以调用另一个函数，于是程序开始真正组合起来
<!-- bilingual-en:start -->
*6. Function Composition Begins When One Function Calls Another*
<!-- bilingual-en:end -->
一旦 `bisection_root` 被封好，后面的函数就可以站在更高层去写。
<!-- bilingual-en:start -->
Once `bisection_root` has a stable interface, later functions can operate at a higher level.
<!-- bilingual-en:end -->

例如：

- 我不再关心二分法内部怎么收缩区间
- 我只把它当成“给我一个数，返回它平方根近似值”的工具
<!-- bilingual-en:start -->
They no longer need to manage how the bisection interval shrinks; they can treat the function as a tool that maps a number to an approximate square root.
<!-- bilingual-en:end -->

这其实就是上节 abstraction 的兑现版本。  
Lecture 7 讲的是理念，Lecture 8 开始把它变成一种实际写法。
<!-- bilingual-en:start -->
This realizes the previous lecture's abstraction principle: Lecture 7 supplied the idea, and Lecture 8 turns it into a practical composition technique.
<!-- bilingual-en:end -->

### 7. scope：局部名字、外层名字、能读不能随便改
<!-- bilingual-en:start -->
*7. Scope: Local Names, Enclosing Names, and the Difference Between Reading and Assigning*
<!-- bilingual-en:end -->
讲完 `return` 之后，课堂转去讲 **scope**。
<!-- bilingual-en:start -->
After `return`, the lecture turns to **scope**.
<!-- bilingual-en:end -->

老师用几个很短的函数例子说明：

- 函数内部创建的变量默认是局部的
- 可以读取外层已经存在的名字
- 但如果你在函数内部给某个名字赋值，Python 会把它当局部变量处理
- 所以“读外层变量”和“改外层变量”不是一回事
<!-- bilingual-en:start -->
Variables created inside a function are local by default. A function may read an existing name from an enclosing scope, but assigning to that name inside the function makes Python treat it as local; reading an outer variable and changing one are therefore different operations.
<!-- bilingual-en:end -->

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
<!-- bilingual-en:start -->
`g` merely reads `x` and can use an enclosing definition. `h` both reads and assigns to `x`; Python classifies it as a local name, so the read occurs before that local value exists and raises an error.
<!-- bilingual-en:end -->

> [!note]
> scope 这部分的意义不是记语法细节，而是理解每个函数调用都有自己独立的小环境。
> <!-- bilingual-en:start -->
> The point of scope is not a collection of syntax rules, but the fact that every function call has its own local environment.
> <!-- bilingual-en:end -->

### 8. 真正的新内容：函数和别的值一样，也是 object
<!-- bilingual-en:start -->
*8. The New Idea: Functions Are Objects Like Other Values*
<!-- bilingual-en:end -->
到这里课程才进入标题里的核心：**functions as objects**。
<!-- bilingual-en:start -->
The lecture now reaches its title concept: **functions as objects**.
<!-- bilingual-en:end -->

老师明确说出一个观念：

- int 是对象
- str 是对象
- list 是对象
- function 也是对象
<!-- bilingual-en:start -->
Integers, strings, lists, and functions are all objects.
<!-- bilingual-en:end -->

既然函数有名字、可以被引用，那它就可以像别的对象一样被：

- 赋给变量
- 作为参数传递
- 从一个函数传进另一个函数
<!-- bilingual-en:start -->
Because a function can be named and referenced, it can be assigned to a variable, supplied as an argument, and passed from one function to another.
<!-- bilingual-en:end -->

这并不是神秘规则，而是 Python 对函数的一种统一处理方式。
<!-- bilingual-en:start -->
This is not a special exception but part of Python's uniform treatment of functions as values.
<!-- bilingual-en:end -->

### 9. `calc(op, x, y)`：把“做什么操作”当成输入
<!-- bilingual-en:start -->
*9. `calc(op, x, y)`: Supplying the Operation Itself as Input*
<!-- bilingual-en:end -->
老师用 `calc` 例子把这个观念真正落地：
<!-- bilingual-en:start -->
The `calc` example makes the idea concrete:
<!-- bilingual-en:end -->

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
<!-- bilingual-en:start -->
The crucial point is that `op` is neither a string nor an operator symbol; it is a function object.
<!-- bilingual-en:end -->

所以 `calc(add, 2, 3)` 的意思就是：

1. 把函数 `add` 传进去
2. 在 `calc` 内部调用 `op(x, y)`
3. 实际上也就是调用 `add(2, 3)`
<!-- bilingual-en:start -->
Thus, `calc(add, 2, 3)` passes the function `add`, invokes `op(x, y)` inside `calc`, and thereby evaluates `add(2, 3)`.
<!-- bilingual-en:end -->

### 10. `apply(criteria, n)`：把“判断标准”本身变成参数
<!-- bilingual-en:start -->
*10. `apply(criteria, n)`: Making the Criterion Itself an Argument*
<!-- bilingual-en:end -->
更典型的高阶函数例子是：
<!-- bilingual-en:start -->
A more representative higher-order function is:
<!-- bilingual-en:end -->

```python
def apply(criteria, n):
    count = 0
    for i in range(0, n + 1):
        if criteria(i):
            count += 1
    return count
```

这段代码非常值得停下来理解，因为它第一次把“行为”当成输入。
<!-- bilingual-en:start -->
This is the first example that treats behavior itself as input.
<!-- bilingual-en:end -->

`criteria` 在这里不是一个布尔值，而是：

- 一个接受数字
- 返回布尔值
- 表示筛选标准的函数
<!-- bilingual-en:start -->
`criteria` is not a Boolean value. It is a function that accepts a number, returns a Boolean, and represents a selection rule.
<!-- bilingual-en:end -->

于是：

- `apply(is_even, 10)` 统计的是 0 到 10 里偶数有多少个
- 如果以后传入别的条件函数，就能统计别的东西
<!-- bilingual-en:start -->
`apply(is_even, 10)` counts the even numbers from 0 through 10; supplying another predicate makes the same loop count something else.
<!-- bilingual-en:end -->

> [!example]
> 这就是抽象层级提升的瞬间：  
> `apply` 不再关心“偶数”是什么，它只关心“给我一个判定标准，我就按这个标准数数”。
> <!-- bilingual-en:start -->
> This raises the level of abstraction: `apply` no longer needs to know what “even” means. Given any criterion, it counts the values satisfying that criterion.
> <!-- bilingual-en:end -->

### 11. 课堂最后已经在为 lambda 铺路
<!-- bilingual-en:start -->
*11. Preparing the Ground for Lambdas*
<!-- bilingual-en:end -->
Lecture 8 虽然还没有系统讲 lambda，但已经在制造那个需求。
<!-- bilingual-en:start -->
Lecture 8 does not yet teach lambdas systematically, but it creates the need for them.
<!-- bilingual-en:end -->

比如如果某个标准非常简单，只会用一次，那么为了配合 `apply` 去专门写一个完整 `def`，看起来就有点重。
<!-- bilingual-en:start -->
If a criterion is very simple and used only once, defining a full named function solely for `apply` feels excessive.
<!-- bilingual-en:end -->

这正是 Lecture 9 要继续解决的问题。  
所以这节课最后的重要收束其实是：

- 函数可以返回值
- 函数可以作为参数
- 函数调用发生在独立 scope 中
- 因此函数本身已经变成可以操作的对象
<!-- bilingual-en:start -->
That is the problem Lecture 9 addresses. The closing picture is that functions return values, can themselves be arguments, execute in independent scopes, and are therefore manipulable objects.
<!-- bilingual-en:end -->

## Exercise log

> [!example] Finger exercise 08
> 官方练习是 `same_chars(s1, s2)`：如果 `s1` 中的每个字符都在 `s2` 中出现，且 `s2` 中的每个字符也都在 `s1` 中出现，则返回 `True`。
> <!-- bilingual-en:start -->
> The official exercise is `same_chars(s1, s2)`: return `True` if every character in `s1` appears in `s2` and every character in `s2` appears in `s1`.
> <!-- bilingual-en:end -->

这题表面上是字符串扫描，实际却正好卡在本讲的两个重点上：

- 你要写的是一个真正 **返回布尔值** 的函数，而不是打印中间判断
- 你要清楚函数接口只承诺“比较字符集合关系”，不承诺比较出现次数
<!-- bilingual-en:start -->
Although it looks like a string-scanning task, it tests two lecture themes: writing a function that **returns a Boolean** instead of printing intermediate judgments, and reading the interface precisely—the function compares character membership, not occurrence counts.
<!-- bilingual-en:end -->

官方解法是两段对称循环：

- 先检查 `s1` 的字符是否都在 `s2`
- 再检查 `s2` 的字符是否都在 `s1`
<!-- bilingual-en:start -->
The official solution uses two symmetric loops, first checking the characters of `s1` against `s2` and then the characters of `s2` against `s1`.
<!-- bilingual-en:end -->

它适合放在本讲后面，因为这就是最典型的“按 specification 写函数并正确 return”。
<!-- bilingual-en:start -->
It is a direct exercise in implementing a specification and returning the result correctly.
<!-- bilingual-en:end -->

如果你写着写着想 `print(True)` 或 `print(False)`，说明本讲最核心的区分还没站稳。
<!-- bilingual-en:start -->
An impulse to write `print(True)` or `print(False)` signals that the central print-versus-return distinction is still unstable.
<!-- bilingual-en:end -->

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
<!-- bilingual-en:start -->
- [ ] I can distinguish an explicit `return`, an implicit `None`, and `print`.
- [ ] I can explain why `print(mult(...))` often reveals that a function has no useful return value.
- [ ] I can explain why a bug such as the one in `is_triangular` is fundamentally a control-flow error, not merely a formula error.
- [ ] I can explain how encapsulating bisection in a function lets other functions organize code at a higher level.
- [ ] I can state the basic scope rules for local variables, reading enclosing names, and assigning inside a function.
- [ ] I can explain why a function can be supplied as an argument to another function.
- [ ] I can read higher-order functions such as `calc(op, x, y)` and `apply(criteria, n)`.
- [ ] I can explain the practical benefit of functions as objects rather than merely recite the definition.
- [ ] I can connect finger exercise 08 to return values and function interfaces.
- [ ] I can reconstruct the lecture sequence: function review -> return versus print -> scope -> functions as objects.
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 函数内部把结果打印出来，就误以为调用者已经拿到了那个值。
> - 没写 `return` 却以为函数“什么都没返回”。
> - 搞不清 scope，看到函数里出现同名变量就以为和外面的一定是同一个东西。
> - 传给高阶函数的不是函数对象本身，而是不小心先把函数调用掉了。
> <!-- bilingual-en:start -->
> - Printing a result inside a function and assuming that the caller has received that value.
> - Assuming that a function with no explicit `return` returns nothing at all.
> - Confusing scope and assuming that equal names inside and outside a function necessarily identify the same variable.
> - Calling a function prematurely instead of passing the function object itself to a higher-order function.
> <!-- bilingual-en:end -->
