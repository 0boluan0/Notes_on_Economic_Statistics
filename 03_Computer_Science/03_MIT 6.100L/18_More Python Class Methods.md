---
aliases:
  - MIT 6.100L Lecture 18
  - 6.100L L18
  - More Python Class Methods
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 18
---

# Lecture 18: More Python Class Methods

> [!tip] Hint
> - 这节课一开始先区分两种视角：implementing a class 和 using a class。
> - Coordinate 会先被拿来复习，然后逐步加 `to_origin`、`__str__` 这种更像“对象自己知道怎么表现自己”的方法。
> - `__init__`、`__str__` 这些 dunder methods 开始正式登场，它们让 Python 的内置语法和你的类接起来。
> - Circle 是用 Coordinate 作为组成部件 built out 的，这节课第一次明显体现 composition。
> - `type(...) == Coordinate` 或 `isinstance(...)` 的检查，是在讲如何在类内部保护自己的输入前提。
> - `is_inside(point)` 展示了“对象方法可以调用另一个对象的方法”，比如 point.distance(center)。
> - Fraction 例子一开始故意用普通方法 `plus`、`minus`、`times`，然后才过渡到运算符重载。
> - `__add__`、`__mul__`、`__float__`、`__str__` 让对象能参与 `+`、`*`、`float(...)`、`print(...)` 这些语法。
> - `reduce` 的例子提醒你：方法设计时还要考虑返回值类型是否一致。
> - 听完这节课，你应该能看懂“一个对象类型如何借助 dunder methods 接入 Python 语法生态”。
> <!-- bilingual-en:start -->
> - This lecture begins by distinguishing two perspectives: implementing a class and using a class.
> - The lecture first reviews `Coordinate`, then adds methods such as `to_origin` and `__str__`, allowing an object to control its own state and presentation.
> - Dunder methods such as `__init__` and `__str__` connect a user-defined class to Python's built-in syntax.
> - `Circle` contains a `Coordinate`, providing the lecture's first clear example of composition.
> - Checks such as `type(...) == Coordinate` or `isinstance(...)` are about how to protect input assumptions within a class.
> - The `is_inside(point)` method shows that one object's method can call another object's method, as in `point.distance(center)`.
> - The Fraction example starts with regular methods `plus`, `minus`, and `times`, then transitions to operator overloading.
> - Methods like `__add__`, `__mul__`, `__float__`, and `__str__` enable objects to participate in operations like `+`, `*`, `float(...)`, and `print(...)`.
> - The `reduce` example reminds you to consider return value types when designing methods.
> - By the end, you should understand how dunder methods make a user-defined type participate naturally in Python syntax.
> <!-- bilingual-en:end -->

## Lecture flow

### 1. 先强调两种视角：写类的人 vs 用类的人
<!-- bilingual-en:start -->
*1. Two Perspectives: Implementing a Class and Using It*
<!-- bilingual-en:end -->
Lecture 18 开场先不是新代码，而是把上讲隐含的两种视角正式说破：
<!-- bilingual-en:start -->
Lecture 18 begins not with new code but by making explicit the two perspectives implied by the previous lecture:
<!-- bilingual-en:end -->

- implementing the class
- using the class

写类的人关心：
<!-- bilingual-en:start -->
The class writer cares about:
<!-- bilingual-en:end -->

- 这个类型有哪些属性
- 方法怎么定义
- 输入要不要检查
<!-- bilingual-en:start -->
- Which attributes the type has
- How its methods are defined
- Whether inputs need validation
<!-- bilingual-en:end -->

用类的人关心：
<!-- bilingual-en:start -->
The class user cares about:
<!-- bilingual-en:end -->

- 怎么创建实例
- 怎么调用方法
- 这个类型看起来像什么、能做什么
<!-- bilingual-en:start -->
- How to create instances
- How to call methods
- How the type is represented and what it can do
<!-- bilingual-en:end -->

这一点很重要，因为后面 dunder methods 的设计，本质上就是在照顾“用类的人”的体验。
<!-- bilingual-en:start -->
This distinction matters because dunder methods are largely about making the class natural for its users.
<!-- bilingual-en:end -->

### 2. 先回到 Coordinate，巩固类的基本结构
<!-- bilingual-en:start -->
*2. Returning to `Coordinate` to Reinforce Class Structure*
<!-- bilingual-en:end -->
老师先把上节的 Coordinate 拿回来。
<!-- bilingual-en:start -->
The instructor begins by returning to the `Coordinate` class from the previous lecture.
<!-- bilingual-en:end -->

核心仍然是：
<!-- bilingual-en:start -->
The core remains:
<!-- bilingual-en:end -->

- `__init__` 写入 `x`、`y`
- `distance(other)` 计算两点距离
<!-- bilingual-en:start -->
- `__init__` assigns `x` and `y`.
- `distance(other)` computes the distance between two points.
<!-- bilingual-en:end -->

然后在此基础上逐步增强，而不是一口气换到新类。  
这种课堂推进说明：类的学习不是一个大跳跃，而是在原有 blueprint 上不断加功能。
<!-- bilingual-en:start -->
The class is then extended incrementally instead of being replaced immediately. This progression shows that learning classes involves adding capabilities step by step to an existing blueprint rather than making one large conceptual leap.
<!-- bilingual-en:end -->

### 3. 新方法 `to_origin`：方法也可以改对象内部状态
<!-- bilingual-en:start -->
*3. `to_origin`: A Method Can Modify Object State*
<!-- bilingual-en:end -->
Coordinate 新增的第一个方法是：
<!-- bilingual-en:start -->
The first added method to Coordinate is:
<!-- bilingual-en:end -->

```python
def to_origin(self):
    self.x = 0
    self.y = 0
```

它的重要性在于：
<!-- bilingual-en:start -->
Its importance lies in:
<!-- bilingual-en:end -->

- 你现在不只是“读取对象属性”
- 还开始通过方法去 **改变对象状态**
<!-- bilingual-en:start -->
- You are no longer merely reading attributes.
- A method now changes the object's state.
<!-- bilingual-en:end -->

这让类方法和前面学过的 list mutation 直觉连上了：  
对象方法同样可以带 side effect。
<!-- bilingual-en:start -->
This reconnects class methods with the earlier intuition for list mutation: an object method can also have a side effect.
<!-- bilingual-en:end -->

### 4. `__str__`：让对象知道自己怎么被打印
<!-- bilingual-en:start -->
*4. `__str__`: Giving an Object Its Printed Representation*
<!-- bilingual-en:end -->
老师紧接着加了一个非常关键的 dunder method：
<!-- bilingual-en:start -->
The instructor then adds an important dunder method:
<!-- bilingual-en:end -->

```python
def __str__(self):
    return "<" + str(self.x) + "," + str(self.y) + ">"
```

它的教学意义远大于“打印好看一点”。
<!-- bilingual-en:start -->
Its significance goes well beyond making the output look nicer.
<!-- bilingual-en:end -->

因为一旦实现了 `__str__`：
<!-- bilingual-en:start -->
Once you implement `__str__`:
<!-- bilingual-en:end -->

- `print(c)` 就不再输出一串内存地址
- 而会调用你定义的字符串表示
<!-- bilingual-en:start -->
- `print(c)` no longer displays Python's default object representation.
- It uses the string representation defined by the class.
<!-- bilingual-en:end -->

> [!note]
> `__str__` 让对象类型参与到 Python 原生的 `print` 语法里。
> <!-- bilingual-en:start -->
> `__str__` enables object types to participate in Python's native `print` syntax.
> <!-- bilingual-en:end -->

这也是 dunder methods 的第一次真正落地体验。
<!-- bilingual-en:start -->
This is the lecture's first concrete use of a dunder method.
<!-- bilingual-en:end -->

### 5. Circle：用一个类去构造更复杂的类
<!-- bilingual-en:start -->
*5. `Circle`: Building a More Complex Class through Composition*
<!-- bilingual-en:end -->
Coordinate 复习完后，老师开始用它搭 Circle。
<!-- bilingual-en:start -->
After reviewing `Coordinate`, the instructor uses it to build `Circle`.
<!-- bilingual-en:end -->

Circle 的数据不是两个普通数字，而是：
<!-- bilingual-en:start -->
The data stored by a `Circle` are not simply two unrelated numbers:
<!-- bilingual-en:end -->

- `center`：一个 Coordinate object
- `radius`：一个半径值
<!-- bilingual-en:start -->
- `center`: a `Coordinate` object
- `radius`: a numerical radius
<!-- bilingual-en:end -->

这一步特别重要，因为它让你看到 **composition**：
<!-- bilingual-en:start -->
This example makes **composition** concrete:
<!-- bilingual-en:end -->

- 类可以把别的对象类型当作自己的组成部分
<!-- bilingual-en:start -->
- A class can contain an instance of another type as one of its components.
<!-- bilingual-en:end -->

所以面向对象不只是“新语法”，也是把已有抽象组合成更复杂抽象的方法。
<!-- bilingual-en:start -->
Object-oriented programming is therefore not merely new syntax; composition combines existing abstractions into more complex ones.
<!-- bilingual-en:end -->

### 6. 在 `__init__` 里做类型检查
<!-- bilingual-en:start -->
*6. Type Checking in `__init__`*
<!-- bilingual-en:end -->
老师接着修改 Circle 的 `__init__`，要求：
<!-- bilingual-en:start -->
The instructor then modifies `Circle.__init__` to require that:
<!-- bilingual-en:end -->

- `center` 必须是 Coordinate 对象
- `radius` 必须是 int
<!-- bilingual-en:start -->
- `center` is a `Coordinate` object.
- `radius` is an `int`.
<!-- bilingual-en:end -->

否则 raise `ValueError`。
<!-- bilingual-en:start -->
Otherwise, the constructor raises `ValueError`.
<!-- bilingual-en:end -->

课堂里老师在这里还提到了：
<!-- bilingual-en:start -->
The instructor compares two checks:
<!-- bilingual-en:end -->

- `type(center) == Coordinate`
- `isinstance(center, Coordinate)`

这些检查的意义，是在类定义内部保护自己的输入前提。  
也就是把“这个类允许什么样的初始化方式”写得更显式。
<!-- bilingual-en:start -->
These checks enforce the constructor's preconditions inside the class definition, making the permitted forms of initialization explicit.
<!-- bilingual-en:end -->

### 7. `is_inside(point)`：对象之间也会相互协作
<!-- bilingual-en:start -->
*7. `is_inside(point)`: Objects Collaborate with Each Other*
<!-- bilingual-en:end -->
老师随后给 Circle 增加：
<!-- bilingual-en:start -->
The instructor then adds this method to `Circle`:
<!-- bilingual-en:end -->

```python
def is_inside(self, point):
    return point.distance(self.center) < self.radius
```

这段代码很值得停一下，因为它体现了几个 OOP 观念同时在工作：
<!-- bilingual-en:start -->
This code is worth pausing over because it demonstrates several OOP concepts at work:
<!-- bilingual-en:end -->

- `point` 是 Coordinate object
- `self.center` 也是 Coordinate object
- 你可以调用一个对象的方法来帮助另一个对象完成判断
<!-- bilingual-en:start -->
- `point` is a `Coordinate` object.
- `self.center` is also a `Coordinate` object.
- One object's method can be used to complete another object's computation.
<!-- bilingual-en:end -->

所以类方法不是孤岛，它们会建立对象之间的协作关系。
<!-- bilingual-en:start -->
Methods are therefore not isolated; they allow objects to collaborate.
<!-- bilingual-en:end -->

### 8. Fraction 第一版：先把行为写出来，不急着上运算符
<!-- bilingual-en:start -->
*8. First Version of `Fraction`: Define the Behavior before Overloading Operators*
<!-- bilingual-en:end -->
讲完 Circle 之后，课堂切到 Fraction。
<!-- bilingual-en:start -->
After `Circle`, the lecture turns to `Fraction`.
<!-- bilingual-en:end -->

老师先故意用普通方法名：
<!-- bilingual-en:start -->
The instructor deliberately begins with ordinary method names:
<!-- bilingual-en:end -->

- `times`
- `divide`
- `plus`
- `minus`

来写一个 `SimpleFraction` 类。
<!-- bilingual-en:start -->
These methods form a `SimpleFraction` class.
<!-- bilingual-en:end -->

这一步很聪明，因为它先让你想清楚：
<!-- bilingual-en:start -->
This sequence first asks you to decide:
<!-- bilingual-en:end -->

- fraction 对象的 data 是什么
- fraction 应该有哪些数学行为
<!-- bilingual-en:start -->
- Which data represent a fraction?
- Which mathematical operations should it support?
<!-- bilingual-en:end -->

然后才进入“怎么让它和 Python 的 `+` `*` 对接”的话题。
<!-- bilingual-en:start -->
Only then does the lecture ask how those operations should connect to Python's `+` and `*` syntax.
<!-- bilingual-en:end -->

### 9. `get_inverse` vs `invert`：返回新值和修改自身要分清
<!-- bilingual-en:start -->
*9. `get_inverse` versus `invert`: Returning a New Value or Mutating the Object*
<!-- bilingual-en:end -->
在 `SimpleFraction` 上，老师安排了一个很典型的对比：
<!-- bilingual-en:start -->
For `SimpleFraction`, the instructor sets up a revealing comparison:
<!-- bilingual-en:end -->

- `get_inverse`：返回 `1/self`
- `invert`：直接交换 numerator 和 denominator
<!-- bilingual-en:start -->
- `get_inverse`: returns `1/self`
- `invert`: directly swaps the numerator and denominator
<!-- bilingual-en:end -->

这组方法是在重演一条整个课程都在强调的区分：
<!-- bilingual-en:start -->
This pair of methods reiterates a key distinction emphasized throughout the course:
<!-- bilingual-en:end -->

- 返回一个值
- 修改对象自身
<!-- bilingual-en:start -->
- Return a new value.
- Modify the object itself.
<!-- bilingual-en:end -->

如果你把这两种风格混在一起，类方法就会越来越难用。
<!-- bilingual-en:start -->
Mixing these two styles makes class methods increasingly difficult to use.
<!-- bilingual-en:end -->

### 10. 运算符重载：让对象接入 `+`、`*`、`print`
<!-- bilingual-en:start -->
*10. Operator Overloading: Connecting Objects to `+`, `*`, and `/`*
<!-- bilingual-en:end -->
前面的普通方法讲清楚之后，老师才切到真正的 Pythonic 写法。
<!-- bilingual-en:start -->
After establishing the operations with ordinary methods, the instructor introduces Python's operator-overloading protocol.
<!-- bilingual-en:end -->

比如：
<!-- bilingual-en:start -->
For example:
<!-- bilingual-en:end -->

```python
def __mul__(self, other):
    ...

def __add__(self, other):
    ...

def __truediv__(self, other):
    ...
```

实现之后：
<!-- bilingual-en:start -->
After implementation:
<!-- bilingual-en:end -->

- `a * b`
- `a + b`
- `a / b`

这些语法就会自动映射到你定义的 dunder methods。
<!-- bilingual-en:start -->
Each expression is dispatched to the corresponding dunder method.
<!-- bilingual-en:end -->

课堂在这里反复强调的是：
<!-- bilingual-en:start -->
The lecture repeatedly emphasizes that:
<!-- bilingual-en:end -->

- 运算符背后其实也是方法调用
- 只是 Python 帮你写成了更自然的形式
<!-- bilingual-en:start -->
- An operator expression is still a method call underneath.
- Python presents that call in a more natural form.
<!-- bilingual-en:end -->

### 11. 三种等价调用方式
<!-- bilingual-en:start -->
*11. Three Equivalent Call Styles*
<!-- bilingual-en:end -->
老师还专门展示了下面三种调用是等价的：
<!-- bilingual-en:start -->
The instructor also shows that the following three calls are equivalent:
<!-- bilingual-en:end -->

```python
a * b
a.__mul__(b)
Fraction.__mul__(a, b)
```

这一步和上一讲 `c.distance(origin)` 的等价调用一起，构成了理解 OOP 语法糖的关键。
<!-- bilingual-en:start -->
Together with the previous lecture's `c.distance(origin)` example, this equivalence explains the syntactic sugar used for object-oriented calls.
<!-- bilingual-en:end -->

如果你能看懂这里，就会知道：
<!-- bilingual-en:start -->
Once this mechanism is clear:
<!-- bilingual-en:end -->

- 类方法没有神秘力量
- 只是 Python 在不同场合替你做了不同的绑定和调度
<!-- bilingual-en:start -->
- Class methods need no mysterious special mechanism.
- Python performs the appropriate binding and dispatch for each syntax.
<!-- bilingual-en:end -->

### 12. `__float__`、`__str__`：对象可以参与更多内置转换
<!-- bilingual-en:start -->
*12. `__float__`, `__str__`: Objects Can Participate in More Built-in Conversions*
<!-- bilingual-en:end -->
老师继续往 Fraction 里加：
<!-- bilingual-en:start -->
The instructor continues by adding these methods to `Fraction`:
<!-- bilingual-en:end -->

- `__float__`
- `__str__`

于是：
<!-- bilingual-en:start -->
Thus:
<!-- bilingual-en:end -->

- `float(c)` 会调用 `__float__`
- `print(c)` 会调用 `__str__`
<!-- bilingual-en:start -->
- `float(c)` calls `__float__`
- `print(c)` calls `__str__`
<!-- bilingual-en:end -->

这让对象类型不只是能做数学运算，还能更自然地和 Python 自带函数配合。
<!-- bilingual-en:start -->
The type can now participate not only in mathematical expressions but also in Python's built-in conversions and printing functions.
<!-- bilingual-en:end -->

### 13. `reduce`：方法返回值类型也需要设计
<!-- bilingual-en:start -->
*13. `reduce`: Return Type Is Part of Method Design*
<!-- bilingual-en:end -->
Fraction 的 `reduce` 方法引出了一个更细的问题：  
约分之后到底返回什么类型？
<!-- bilingual-en:start -->
The `Fraction.reduce` method raises a subtler design question: which type should simplification return?
<!-- bilingual-en:end -->

如果分母变成 `1`，是：
<!-- bilingual-en:start -->
If the denominator becomes `1`, then:
<!-- bilingual-en:end -->

- 返回一个 `int`
- 还是仍然返回一个 `Fraction`
<!-- bilingual-en:start -->
- Return an `int`.
- Continue to return a `Fraction`.
<!-- bilingual-en:end -->

老师专门把这个点拿出来，是因为这涉及接口一致性。
<!-- bilingual-en:start -->
The instructor highlights this choice because it affects interface consistency.
<!-- bilingual-en:end -->

如果一个方法有时返回 `Fraction`，有时返回 `int`，调用者的使用体验会变得很不稳定。  
所以课堂后来通过 you-try-it 去修这个设计。
<!-- bilingual-en:start -->
A method that sometimes returns a `Fraction` and sometimes an `int` forces callers to handle an unstable interface. A later “you try it” exercise revises this design.
<!-- bilingual-en:end -->

### 14. 这节课真正推进的是“对象与语言语法的接缝”
<!-- bilingual-en:start -->
*14. The Interface between Objects and Language Syntax*
<!-- bilingual-en:end -->
Lecture 18 表面上是继续写类方法，实际上它完成了更深的一步：
<!-- bilingual-en:start -->
Lecture 18 goes beyond adding more class methods:
<!-- bilingual-en:end -->

- 你的对象开始接入 Python 的运算符、打印、类型转换
- 你的类也开始由简单数据容器，变成更自然的语言级对象
<!-- bilingual-en:start -->
- User-defined objects begin to participate in Python's operators, printing, and type conversions.
- A class evolves from a simple data container into a type that behaves naturally within the language.
<!-- bilingual-en:end -->

这是从“我能定义类”走向“我能定义用起来像原生对象的类”的关键一步。
<!-- bilingual-en:start -->
This is the key transition from “I can define a class” to “I can define a class that behaves like a built-in type.”
<!-- bilingual-en:end -->

## Exercise log

> [!example] Finger exercise 18
> 官方题目还是 `Circle`，但这次要求实现：
> - `__init__`
> - `get_radius`
> - `__add__`
> - `__str__`
> <!-- bilingual-en:start -->
> The official problem still revolves around `Circle`, but this time you're required to implement:
> - `__init__`
> - `get_radius`
> - `__add__`
> - `__str__`
> <!-- bilingual-en:end -->

它非常贴合本讲，因为它直接检查你是否理解：
<!-- bilingual-en:start -->
The exercise closely matches the lecture because it checks whether you understand that:
<!-- bilingual-en:end -->

- dunder methods 不是装饰
- 它们会决定对象如何参与 `+` 和 `print`
<!-- bilingual-en:start -->
- Dunder methods aren't mere decorations.
- They determine how objects participate in operations like `+` and `print`.
<!-- bilingual-en:end -->

官方 `__add__` 的意思是：
<!-- bilingual-en:start -->
The specified `__add__` implementation is:
<!-- bilingual-en:end -->

```python
return Circle(self.r + c.r)
```

这说明运算符重载本质上还是“返回一个新对象”，而不是必须原地修改自己。
<!-- bilingual-en:start -->
This shows that an overloaded operator can return a new object rather than mutating the current instance in place.
<!-- bilingual-en:end -->

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec18.pdf|Lecture 18 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec18_code.py|Lecture 18 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex18_sol.pdf|Lecture 18 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec18_transcript.pdf|Lecture 18 transcript]]
- Recitation: none attached to this lecture week
- PS 4 halfway hand-in due: [[MIT 6.100L-problem-sets/mit6_100l_ps4.pdf|PS4 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps4_code.zip|PS4 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 10.1)

## Review checklist
- [ ] 我能区分 implementing a class 和 using a class 两种视角。
- [ ] 我能解释 `to_origin` 这类方法如何修改对象自身状态。
- [ ] 我能说明 `__str__` 为什么能改变 `print(obj)` 的行为。
- [ ] 我能解释 Circle 如何把 Coordinate 作为组成部分。
- [ ] 我能判断什么时候需要在 `__init__` 中做类型检查。
- [ ] 我能说明 `is_inside(point)` 里为什么是对象和对象协作。
- [ ] 我能区分 `get_inverse` 和 `invert` 这类“返回值 vs 修改自身”的方法设计。
- [ ] 我能解释 `__add__`、`__mul__`、`__float__` 这类 dunder methods 的作用。
- [ ] 我能理解 `reduce` 方法中“返回类型一致性”为什么重要。
- [ ] 我能按课堂顺序复述：Coordinate recap -> Circle -> SimpleFraction -> operator overloading -> reduce。
<!-- bilingual-en:start -->
- [ ] I can distinguish between the perspectives of `implementing a class` and `using a class`.
- [ ] I can explain how methods like `to_origin` modify the object's state.
- [ ] I can describe why `__str__` affects the behavior of `print(obj)`.
- [ ] I can explain how `Circle` incorporates `Coordinate` as part of its structure.
- [ ] I can determine when type checking should be performed in `__init__`.
- [ ] I can explain how `is_inside(point)` makes two objects collaborate.
- [ ] I can distinguish methods such as `get_inverse` and `invert`: returning a new value versus mutating the current instance.
- [ ] I can explain the purpose of dunder methods such as `__add__`, `__mul__`, and `__float__`.
- [ ] I can explain why a consistent return type makes the `reduce` interface more stable.
- [ ] I can reconstruct the lecture sequence: `Coordinate` recap -> `Circle` -> `SimpleFraction` -> operator overloading -> `reduce`.
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 把 dunder methods 当成要死记的特殊名字，却不理解它们对应哪种语言行为。
> - 设计类方法时混淆“返回新对象”和“修改当前对象”。
> - 用类组合类时，没有先想清楚内部属性本身是不是别的对象。
> - 让同一个方法在不同情况下返回完全不同类型，导致接口不稳定。
> <!-- bilingual-en:start -->
> - Memorizing dunder method names without understanding the language behavior attached to each one.
> - Confusing returning a new object with mutating the current object when designing a method.
> - Composing classes without first clarifying whether internal attributes are objects themselves.
> - Letting one method return unrelated types in different cases and thereby creating an unstable interface.
> <!-- bilingual-en:end -->
