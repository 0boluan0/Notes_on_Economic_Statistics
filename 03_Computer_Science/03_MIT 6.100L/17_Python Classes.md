---
aliases:
  - MIT 6.100L Lecture 17
  - 6.100L L17
  - Python Classes
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 17
---

# Lecture 17: Python Classes

> [!tip] Hint
> - 这节课一开始先从“我们已经在用 int、str、list、dict 这些 object types”退一步，看 object type 到底是什么。
> - 老师反复强调一个对象类型由两部分决定：data representation 和 behaviors。
> - `1234` 是一个 instance of int，`"hello"` 是一个 instance of string，这种说法是为了给“自己定义类型”铺语言基础。
> - classes 的目标不是让你背新语法，而是让你能造自己的 object type。
> - Coordinate 被选作第一个类，因为它的数据和行为都很直观：x、y 和 distance。
> - `class Coordinate(object):` 里的 `object` 是 parent class，今天先把它当最通用基类。
> - `__init__` 是 constructor，负责在创建实例时把数据属性装进去。
> - `self` 不是魔法字，它表示“当前这个实例”，方法通过它访问自己的属性。
> - `c.distance(origin)` 和 `Coordinate.distance(c, origin)` 的等价，是课堂里解释 method call 机制的关键。
> - 听完这节课，你应该能区分“定义类的蓝图”和“创建类的实例”。
> <!-- bilingual-en:start -->
> - This lecture steps back from the familiar object types we have already used, such as `int`, `str`, `list`, and `dict`, to ask what an object type actually is.
> - The instructor emphasizes that an object type is defined by two things: its data representation and the behavior it supports.
> - Saying that `1234` is an instance of `int` and `"hello"` is an instance of `str` establishes the vocabulary needed to define our own types.
> - The purpose of classes is not to introduce more syntax to memorize, but to let you define your own object types.
> - `Coordinate` is chosen as the first class because its data (`x` and `y`) and its behavior (distance calculation) are intuitive.
> - In `class Coordinate(object):`, `object` is the parent class; for now, treat it as the most general base class.
> - `__init__` initializes a newly created instance by assigning its data attributes.
> - `self` is not magic; it refers to the current instance, allowing a method to access that instance's attributes.
> - `c.distance(origin)` and `Coordinate.distance(c, origin)` are equivalent, illustrating how method calls work.
> - By the end of the lecture, you should be able to distinguish the blueprint defined by a class from a concrete instance created from that blueprint.
> <!-- bilingual-en:end -->

## Lecture flow

### 1. 先回到已经熟悉的对象：int、float、string、list、dict
<!-- bilingual-en:start -->
*1. Returning to Familiar Objects: `int`, `float`, `str`, `list`, and `dict`*
<!-- bilingual-en:end -->
Lecture 17 开场并没有立刻写类，而是先提醒你：
<!-- bilingual-en:start -->
Lecture 17 does not begin by defining a class. It first reminds you that:
<!-- bilingual-en:end -->

- 这门课从一开始就在用 objects
- 只是以前这些 object types 都是 Python 预先帮你定义好的
<!-- bilingual-en:start -->
- This course has used objects from the very beginning.
- Until now, every object type used in the course has been predefined by Python.
<!-- bilingual-en:end -->

老师列了很多例子：
<!-- bilingual-en:start -->
The instructor lists several examples:
<!-- bilingual-en:end -->

- `1234`
- `3.14159`
- `"hello"`
- `[1, 2, 3]`
- `{'a': 1}`

这些都不是“裸值”，而是某种 type 的 instance。
<!-- bilingual-en:start -->
None of these is a “bare value”; each is an instance of a type.
<!-- bilingual-en:end -->

这一步很关键，因为课程不是要凭空加一个新主题，而是要把“对象类型”这件事从使用层推进到定义层。
<!-- bilingual-en:start -->
This transition matters because the course is not introducing an unrelated topic; it is moving from using object types to defining them.
<!-- bilingual-en:end -->

### 2. object type 由什么决定：data + behaviors
<!-- bilingual-en:start -->
*2. What Defines an Object Type: Data + Behaviors*
<!-- bilingual-en:end -->
老师接着给出全讲最重要的一句抽象：
<!-- bilingual-en:start -->
The instructor then states the lecture's central abstraction:
<!-- bilingual-en:end -->

> [!note]
> 一个 object type 由两件事共同定义：  
> 它的数据如何表示，它拥有哪些行为。
> <!-- bilingual-en:start -->
> An object type is defined by two things:
> its data representation and the behavior it supports.
> <!-- bilingual-en:end -->

例如：
<!-- bilingual-en:start -->
For example:
<!-- bilingual-en:end -->

- list 的数据表示是一个有序序列
- list 的行为包括 append、sort、indexing 等
<!-- bilingual-en:start -->
- A list's data representation is an ordered sequence.
- Its supported operations include appending, sorting, and indexing.
<!-- bilingual-en:end -->

同理，如果我们要定义自己的 object type，也必须回答：
<!-- bilingual-en:start -->
Similarly, if we want to define our own object type, we must answer:
<!-- bilingual-en:end -->

- 我用哪些数据属性表示它
- 我希望它能做什么
<!-- bilingual-en:start -->
- What data properties will it use?
- What actions will it perform?
<!-- bilingual-en:end -->

### 3. instance 的语言先立起来
<!-- bilingual-en:start -->
*3. Establishing the Vocabulary of Instances*
<!-- bilingual-en:end -->
老师随后专门强调 “instance of a type” 这种说法。
<!-- bilingual-en:start -->
The instructor then emphasizes the phrase “instance of a type.”
<!-- bilingual-en:end -->

这不是咬文嚼字，而是为了后面区分：
<!-- bilingual-en:start -->
This is not mere pedantry; it prepares the distinction between:
<!-- bilingual-en:end -->

- class：蓝图、类型定义
- instance：按这个蓝图造出来的具体对象
<!-- bilingual-en:start -->
- A class: a blueprint or type definition.
- An instance: a concrete object created from that blueprint.
<!-- bilingual-en:end -->

所以：
<!-- bilingual-en:start -->
So:
<!-- bilingual-en:end -->

- `1234` 是 int 的一个 instance
- `"hello"` 是 string 的一个 instance
<!-- bilingual-en:start -->
- `1234` is an instance of `int`.
- `"hello"` is an instance of `str`.
<!-- bilingual-en:end -->

而今天开始，我们也会自己造某种 type 的 instance。
<!-- bilingual-en:start -->
From this point on, we also create instances of types that we define ourselves.
<!-- bilingual-en:end -->

### 4. 先设计一个简单世界：二维坐标
<!-- bilingual-en:start -->
*4. Designing a Simple World: Two-Dimensional Coordinates*
<!-- bilingual-en:end -->
为了让类的设计足够透明，老师选了二维坐标 `Coordinate` 作为第一种自定义类型。
<!-- bilingual-en:start -->
To keep the class design transparent, the instructor chooses a two-dimensional coordinate, `Coordinate`, as the first custom type.
<!-- bilingual-en:end -->

对一个 coordinate object，最自然的数据是：
<!-- bilingual-en:start -->
The natural data attributes of a coordinate object are:
<!-- bilingual-en:end -->

- `x`
- `y`

最自然的行为则可能包括：
<!-- bilingual-en:start -->
The most natural behaviors might include:
<!-- bilingual-en:end -->

- 返回 `x`
- 返回 `y`
- 计算到另一个点的距离
<!-- bilingual-en:start -->
- Return its `x` coordinate.
- Return its `y` coordinate.
- Compute the distance to another point.
<!-- bilingual-en:end -->

这个例子非常适合入门，因为：
<!-- bilingual-en:start -->
This example is particularly suitable for beginners because:
<!-- bilingual-en:end -->

- 数据属性直观
- 行为也很容易用已有数学知识表达
<!-- bilingual-en:start -->
- Its data attributes are intuitive.
- Its behavior is easy to express with familiar mathematics.
<!-- bilingual-en:end -->

### 5. class definition：先写蓝图，再谈实例
<!-- bilingual-en:start -->
*5. Class Definition: First the Blueprint, Then the Instance*
<!-- bilingual-en:end -->
老师这时才正式写下：
<!-- bilingual-en:start -->
Only then does the instructor write:
<!-- bilingual-en:end -->

```python
class Coordinate(object):
    ...
```

这里课堂的重点不是记住括号，而是理解你现在在做的事情：
<!-- bilingual-en:start -->
The point is not to memorize the parentheses, but to understand what the definition does:
<!-- bilingual-en:end -->

- 你还没有创建任何具体点
- 你只是在定义“什么叫 Coordinate”
<!-- bilingual-en:start -->
- No particular point has been created yet.
- The code defines what it means to be a `Coordinate`.
<!-- bilingual-en:end -->

所以类定义和函数定义很像，都是先写一个 blueprint，供后面重复使用。
<!-- bilingual-en:start -->
Like a function definition, a class definition creates a blueprint for later reuse.
<!-- bilingual-en:end -->

### 6. `__init__`：构造实例时把数据装进去
<!-- bilingual-en:start -->
*6. `__init__`: Initializing Instances with Data*
<!-- bilingual-en:end -->
接着老师介绍类中第一个必须会的方法：`__init__`。
<!-- bilingual-en:start -->
Next, the instructor introduces the first essential class method: `__init__`.
<!-- bilingual-en:end -->

例如：
<!-- bilingual-en:start -->
For example:
<!-- bilingual-en:end -->

```python
class Coordinate(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
```

这里的课堂理解应该是：
<!-- bilingual-en:start -->
The model to keep in mind is:
<!-- bilingual-en:end -->

- 当你写 `Coordinate(3, 4)` 时，Python 创建一个新对象
- 然后自动调用 `__init__`
- 把 `3` 和 `4` 放进这个对象的属性里
<!-- bilingual-en:start -->
- When you write `Coordinate(3, 4)`, Python creates a new object.
- Python then calls `__init__` automatically.
- `__init__` assigns `3` and `4` to that object's attributes.
<!-- bilingual-en:end -->

所以 `__init__` 更像是“实例初始化规则”。
<!-- bilingual-en:start -->
Thus, `__init__` specifies how a new instance is initialized.
<!-- bilingual-en:end -->

### 7. `self`：当前这个对象自己
<!-- bilingual-en:start -->
*7. `self`: The Current Instance*
<!-- bilingual-en:end -->
这节课里老师多次停下来解释 `self`。
<!-- bilingual-en:start -->
The instructor pauses several times to explain `self`.
<!-- bilingual-en:end -->

`self` 不是保留字魔法，而是方法中的第一个参数约定，用来表示：
<!-- bilingual-en:start -->
`self` is not a special keyword. By convention, it is the first method parameter and refers to:
<!-- bilingual-en:end -->

- 当前调用这个方法的实例
<!-- bilingual-en:start -->
- The instance on which the method was called.
<!-- bilingual-en:end -->

所以在 `__init__` 里写：
<!-- bilingual-en:start -->
So when you write in `__init__`:
<!-- bilingual-en:end -->

```python
self.x = x
self.y = y
```

意思就是把传进来的参数写进当前这个坐标对象。
<!-- bilingual-en:start -->
This assigns the incoming arguments to attributes on the current `Coordinate` instance.
<!-- bilingual-en:end -->

后面你写别的方法时，也都是通过 `self` 去读写当前实例的数据。
<!-- bilingual-en:start -->
Other methods likewise use `self` to read and write the current instance's data.
<!-- bilingual-en:end -->

### 8. 先加简单方法：getter 和 distance
<!-- bilingual-en:start -->
*8. Adding Simple Methods: Getters and Distance*
<!-- bilingual-en:end -->
定义完数据属性后，老师继续给 Coordinate 加行为。
<!-- bilingual-en:start -->
After defining the data attributes, the instructor adds behavior to `Coordinate`.
<!-- bilingual-en:end -->

典型方法有：
<!-- bilingual-en:start -->
Typical methods include:
<!-- bilingual-en:end -->

```python
def getX(self):
    return self.x

def getY(self):
    return self.y

def distance(self, other):
    x_diff_sq = (self.x - other.x) ** 2
    y_diff_sq = (self.y - other.y) ** 2
    return (x_diff_sq + y_diff_sq) ** 0.5
```

这里第一次让你真正把“对象的行为”写进类型定义里。
<!-- bilingual-en:start -->
This is the first time the object's behavior is encoded directly in the type definition.
<!-- bilingual-en:end -->

### 9. method call 的真正机制：点号只是语法糖
<!-- bilingual-en:start -->
*9. How Method Calls Work: Dot Notation Is Syntactic Sugar*
<!-- bilingual-en:end -->
老师在这部分花了不少时间解释下面两种写法为什么等价：
<!-- bilingual-en:start -->
The instructor spends substantial time explaining why these two forms are equivalent:
<!-- bilingual-en:end -->

```python
c.distance(origin)
Coordinate.distance(c, origin)
```

这件事非常关键，因为它揭示了 method call 的本质：
<!-- bilingual-en:start -->
This matters because it exposes the mechanism behind a method call:
<!-- bilingual-en:end -->

- 点号调用只是把实例自动塞进方法的第一个参数位置
- 也就是 `self`
<!-- bilingual-en:start -->
- Dot notation automatically supplies the instance as the method's first argument.
- That first parameter is conventionally named `self`.
<!-- bilingual-en:end -->

所以当你写 `c.distance(origin)` 时，Python 实际上是在背后做：
<!-- bilingual-en:start -->
Thus, when you write `c.distance(origin)`, Python effectively does the following:
<!-- bilingual-en:end -->

- 找到 `Coordinate` 类里的 `distance`
- 把 `c` 当成 `self`
- 把 `origin` 当成 `other`
<!-- bilingual-en:start -->
- Find the `distance` method on the `Coordinate` class.
- Pass `c` as `self`.
- Pass `origin` as `other`.
<!-- bilingual-en:end -->

> [!note]
> method 和函数不是两个完全不同的机制；method 本质上是“绑定到某个对象上的函数调用约定”。
> <!-- bilingual-en:start -->
> Methods and functions do not use entirely separate mechanisms; a method call is a function call in which an object is bound as the first argument.
> <!-- bilingual-en:end -->

### 10. 先有类定义，再有实例使用
<!-- bilingual-en:start -->
*10. Defining a Class versus Using Its Instances*
<!-- bilingual-en:end -->
老师随后来回切换两种视角：
<!-- bilingual-en:start -->
The instructor then alternates between two perspectives:
<!-- bilingual-en:end -->

- implementing the class
- using the class

这点在 OOP 里非常重要。
<!-- bilingual-en:start -->
This is crucial in OOP.
<!-- bilingual-en:end -->

写类定义时你关心：
<!-- bilingual-en:start -->
When writing a class definition, you consider:
<!-- bilingual-en:end -->

- 这个类型有哪些属性
- 哪些方法应该存在
<!-- bilingual-en:start -->
- What properties does this type have?
- Which methods should exist?
<!-- bilingual-en:end -->

创建实例时你关心：
<!-- bilingual-en:start -->
When creating an instance, you consider:
<!-- bilingual-en:end -->

- 我要用哪些具体值初始化它
- 我要调用哪些方法
<!-- bilingual-en:start -->
- What specific values do I use to initialize it?
- Which methods do I call?
<!-- bilingual-en:end -->

这两种视角如果混在一起，会导致一开始学类时非常容易糊涂。
<!-- bilingual-en:start -->
Conflating these perspectives is a common source of confusion when first learning classes.
<!-- bilingual-en:end -->

### 11. data attributes 和 procedural attributes
<!-- bilingual-en:start -->
*11. Data Attributes and Procedural Attributes*
<!-- bilingual-en:end -->
老师还专门把属性分成两类来说：
<!-- bilingual-en:start -->
The instructor also divides attributes into two categories:
<!-- bilingual-en:end -->

- data attributes：例如 `x`、`y`
- procedural attributes：例如 `distance`
<!-- bilingual-en:start -->
- Data attributes: for example `x`, `y`
- Procedural attributes: for example `distance`
<!-- bilingual-en:end -->

这种说法的价值在于，它让你从“变量和函数”升级到“对象内部既带数据也带方法”的统一视角。
<!-- bilingual-en:start -->
This vocabulary replaces a separate “variables and functions” view with one unified model in which an object carries both data and methods.
<!-- bilingual-en:end -->

### 12. Vehicle 练习：开始让你自己设计类
<!-- bilingual-en:start -->
*12. Vehicle Exercise: Designing a Class Yourself*
<!-- bilingual-en:end -->
课堂后半段给了 Vehicle 的 at-home 设计题：
<!-- bilingual-en:start -->
The second half of the lecture assigns an at-home design problem for a `Vehicle` class:
<!-- bilingual-en:end -->

- 轮子数
- 乘客数
- 颜色
- 再加方法比如 `add_n_occupants`
- 再加默认颜色、最大载客量等约束
<!-- bilingual-en:start -->
- Number of wheels
- Number of occupants
- Color
- Add methods like `add_n_occupants`
- Constraints such as a default color and maximum occupancy
<!-- bilingual-en:end -->

这部分虽然不是课程最核心的演示代码，但它的教学意义很强：  
你已经不只是在看 Coordinate，而是开始尝试自己做设计决策。
<!-- bilingual-en:start -->
Although this is not the lecture's central demonstration, it has a clear teaching purpose: you move from observing `Coordinate` to making class-design decisions yourself.
<!-- bilingual-en:end -->

### 13. 这节课真正建立的是 OOP 的第一层词汇
<!-- bilingual-en:start -->
*13. Establishing the First Layer of OOP Vocabulary*
<!-- bilingual-en:end -->
Lecture 17 结束时，课程并没有急着把类讲复杂。  
它真正完成的是让下面这些词有了稳定含义：
<!-- bilingual-en:start -->
Lecture 17 does not rush into advanced class features. Instead, it gives stable meanings to the following terms:
<!-- bilingual-en:end -->

- class
- instance
- attribute
- method
- constructor / `__init__`
- `self`

有了这套词汇，后面 Circle、Fraction、Inheritance 才能顺利展开。
<!-- bilingual-en:start -->
With this vocabulary established, subsequent topics like Circle, Fraction, and Inheritance can proceed smoothly.
<!-- bilingual-en:end -->

## Exercise log

> [!example] Finger exercise 17
> 官方题目要求实现一个 `Circle` 类，包含：
> - `__init__(radius)`
> - `get_radius`
> - `set_radius`
> - `get_area`
> - `equal`
> - `bigger`
> <!-- bilingual-en:start -->
> The official problem requires implementing a `Circle` class with:
> - `__init__(radius)`
> - `get_radius`
> - `set_radius`
> - `get_area`
> - `equal`
> - `bigger`
> <!-- bilingual-en:end -->

这题很适合作为 Lecture 17 的收尾，因为它会检查你有没有真正掌握：
<!-- bilingual-en:start -->
This problem is a fitting conclusion to Lecture 17 because it checks whether you can:
<!-- bilingual-en:end -->

- 数据属性该放在哪
- 方法里如何通过 `self` 访问属性
- 一个对象方法如何接收另一个同类对象作为参数
<!-- bilingual-en:start -->
- Place data attributes correctly.
- Access attributes through `self` inside methods.
- Pass another instance of the same class to an instance method.
<!-- bilingual-en:end -->

比如 `equal(self, c)` 和 `bigger(self, c)` 都要求你已经接受“方法参数也可以是另一个对象实例”。
<!-- bilingual-en:start -->
For example, `equal(self, c)` and `bigger(self, c)` both require understanding that a method argument can itself be another object instance.
<!-- bilingual-en:end -->

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec17.pdf|Lecture 17 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec17_code.py|Lecture 17 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex17_sol.pdf|Lecture 17 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec17_transcript.pdf|Lecture 17 transcript]]
- Recitation: none attached to this lecture week
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 10.1)

## Review checklist
- [ ] 我能解释 object type 为什么由 data representation 和 behaviors 共同定义。
- [ ] 我能区分 class 和 instance。
- [ ] 我能说明 `__init__` 的作用。
- [ ] 我能解释 `self` 在方法里到底表示什么。
- [ ] 我能手写一个最小 Coordinate 类，并加上一个简单方法。
- [ ] 我能解释为什么 `c.distance(origin)` 和 `Coordinate.distance(c, origin)` 等价。
- [ ] 我能区分 data attribute 和 procedural attribute。
- [ ] 我能自己为一个简单现实对象设计 class 的属性和方法。
- [ ] 我能把 finger exercise 17 看成类设计的最小综合练习。
- [ ] 我能按课堂顺序复述：objects we already know -> define our own types -> Coordinate -> methods -> using instances。
<!-- bilingual-en:start -->
- [ ] I can explain why object types are defined by both data representation and behaviors.
- [ ] I can distinguish between class and instance.
- [ ] I can describe the purpose of `__init__`.
- [ ] I can explain what `self` represents in a method.
- [ ] I can write a minimal Coordinate class with a simple method.
- [ ] I can explain why `c.distance(origin)` is equivalent to `Coordinate.distance(c, origin)`.
- [ ] I can distinguish between data attributes and procedural attributes.
- [ ] I can design the properties and methods for a class that models a simple real-world object.
- [ ] I can use finger exercise 17 as a compact, integrated class-design exercise.
- [ ] I can reconstruct the lecture sequence: familiar objects -> defining our own types -> `Coordinate` -> methods -> using instances.
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 把 class 和 instance 混成一回事。
> - 在方法里忘记通过 `self` 访问当前对象的属性。
> - 把类定义阶段和实例使用阶段混在一起理解。
> - 看到 `self` 就把它当神秘语法，而不理解它只是当前实例引用。
> <!-- bilingual-en:start -->
> - Confusing a class with one of its instances.
> - Forgetting to access the current object's attributes through `self` inside a method.
> - Mixing up class definition with instance use.
> - Treating `self` as mysterious syntax rather than a reference to the current instance.
> <!-- bilingual-en:end -->
