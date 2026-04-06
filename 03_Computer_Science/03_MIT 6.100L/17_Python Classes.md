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
> - 我能解释 class 为什么是数据抽象工具，而不是‘更复杂的函数集合’。
> - 我能区分 class、instance、attribute、method。
> - 我能说明 `__init__` 和 `self` 各自负责什么。
> - 我能把对象状态与对象行为放进同一个抽象里思考。
> - 我能围绕本讲的主轴 “Class 是 data abstraction 的另一种容器” / “Attributes、methods、`self` 与 `__init__`” / “对象设计的重点是接口与状态一致性”，不翻 slides 也把整节课重新讲一遍。
> - 我能解释为什么 class 适合管理相关状态与行为。
> - 我能说明 `__init__` 与 `self` 的角色。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 10.1
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: Class 是 data abstraction 的另一种容器 / Attributes、methods、`self` 与 `__init__` / 对象设计的重点是接口与状态一致性
> - 这讲开始从‘把函数组织起来’转向‘把状态与行为一起组织起来’。
> - class 的目标是为一类对象定义统一接口和内部状态，而不是单纯为了写面向对象而写面向对象。
> - 后面的 dunder methods、inheritance 都建立在这里的 class / instance 关系上。

## Core ideas
### Class 是 data abstraction 的另一种容器
如果函数把动作抽象出来，那么 class 则把‘一类东西的状态 + 作用在它身上的动作’一起抽象出来。
- 当多个函数总在围绕同一份相关数据工作时，往往就是引入 class 的信号。
- class 描述的是一类对象的共同结构，instance 是按这个模板创建出的具体对象。
- 对象让你不必到处传一大串相关变量，而是把它们放回一个有语义的实体里。
- 好的 class 不只是存数据，还通过方法维护对象的不变量和操作方式。

> [!note] What to internalize
> - One-sentence takeaway: 如果函数把动作抽象出来，那么 class 则把‘一类东西的状态 + 作用在它身上的动作’一起抽象出来。
> - Review anchor: 当多个函数总在围绕同一份相关数据工作时，往往就是引入 class 的信号。
> - Review anchor: class 描述的是一类对象的共同结构，instance 是按这个模板创建出的具体对象。

从做题角度看，只要题目在考“Class 是 data abstraction 的另一种容器”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：如果函数把动作抽象出来，那么 class 则把‘一类东西的状态 + 作用在它身上的动作’一起抽象出来。

### Attributes、methods、`self` 与 `__init__`
学习 class 时最需要搞清的是：状态放在哪里，方法操作的是谁，以及对象是在什么时候被初始化的。
- instance attribute 存的是对象自己的状态，例如余额、名称、位置等。
- method 本质上还是函数，只是它约定第一个参数是对象本身，通常写作 `self`。
- `__init__` 负责在对象创建时建立初始状态，它不是对象本身，而是初始化逻辑。
- 一旦理解 `self` 代表‘当前这个对象’，大部分方法调用语法都会变得自然。

> [!note] What to internalize
> - One-sentence takeaway: 学习 class 时最需要搞清的是：状态放在哪里，方法操作的是谁，以及对象是在什么时候被初始化的。
> - Review anchor: instance attribute 存的是对象自己的状态，例如余额、名称、位置等。
> - Review anchor: method 本质上还是函数，只是它约定第一个参数是对象本身，通常写作 `self`。

从做题角度看，只要题目在考“Attributes、methods、`self` 与 `__init__`”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：学习 class 时最需要搞清的是：状态放在哪里，方法操作的是谁，以及对象是在什么时候被初始化的。

### 对象设计的重点是接口与状态一致性
一个 class 最重要的质量标准不是花哨程度，而是：外部接口是否清楚，内部状态是否始终保持合法。
- 如果对象状态可以被任意外部代码改坏，class 的抽象价值就会大幅下降。
- 方法应该围绕对象职责设计，而不是把所有相关函数都机械塞进去。
- 在面向对象设计里，问题通常不是‘能不能做’，而是‘这件事该由哪个对象负责做’。
- 把状态与行为绑在一起，正是 class 相比纯函数式组织的最大优势。

> [!note] What to internalize
> - One-sentence takeaway: 一个 class 最重要的质量标准不是花哨程度，而是：外部接口是否清楚，内部状态是否始终保持合法。
> - Review anchor: 如果对象状态可以被任意外部代码改坏，class 的抽象价值就会大幅下降。
> - Review anchor: 方法应该围绕对象职责设计，而不是把所有相关函数都机械塞进去。

从做题角度看，只要题目在考“对象设计的重点是接口与状态一致性”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：一个 class 最重要的质量标准不是花哨程度，而是：外部接口是否清楚，内部状态是否始终保持合法。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - Example: simple Coordinate class
> - c = Coordinate(3,4)
> - a = 0
> - origin = Coordinate(a,a)
> - print(c.x)
> - print(origin.x)
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 定义一个最小可用的类
> ```python
> class Counter:
>     def __init__(self, start=0):
>         self.value = start
>
>     def increment(self):
>         self.value += 1
>
> c = Counter(10)
> c.increment()
> print(c.value)
> ```
> 这个例子同时展示了 class、instance、attribute、method、`self` 和 `__init__` 的基本协作关系。

> [!example] 对象把相关状态收回到一个抽象里
> ```python
> class Point:
>     def __init__(self, x, y):
>         self.x = x
>         self.y = y
>
> p = Point(3, 4)
> print(p.x, p.y)
> ```
> 与其在程序里单独维护一堆 `x`、`y` 变量，不如把它们放回 `Point` 这个更有语义的对象里。

## Exercise log
> [!note] Finger exercise snapshot
> - Official prompt: Write the class according to the specifications below: class Circle(): def __init__(self, radius): """ Initializes self with radius """ # your code here def get_radius(self): """ Returns the radius of self """ # your...
> - Official solution sketch:
> ```python
> class Circle():
> def __init__(self, radius):
> self.r = radius
> def get_radius(self):
> return self.r
> def set_radius(self, radius):
> ```
> - What this is really testing: whether you can compress the lecture into one small, high-frequency coding move without needing the slides beside you.
> - Where to revisit if this feels shaky: go back to the first two Core ideas sections in this note, then rerun the official lecture code once with your own input.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: this lecture does not have a direct calendar milestone attached, so use it as a consolidation lecture rather than a sprint lecture.
> - Recitation connection: there is no recitation attached to this lecture week in the official calendar.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec17.pdf|Lecture 17 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec17_code.py|Lecture 17 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex17_sol.pdf|Lecture 17 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec17_transcript.pdf|Lecture 17 transcript]]
- Recitation: none attached to this lecture week
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 10.1)

## Review checklist
- [ ] 我能解释为什么 class 适合管理相关状态与行为。
- [ ] 我能区分 class、instance、attribute、method。
- [ ] 我能说明 `__init__` 与 `self` 的角色。
- [ ] 我能判断一个问题是否值得抽象成 class。
- [ ] 我能为一个简单实体设计最小接口。
- [ ] 我能围绕“Class 是 data abstraction 的另一种容器”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“Attributes、methods、`self` 与 `__init__`”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：把 class 当成‘写法更绕的函数集合’，没有看到状态抽象的价值。
- [ ] 我能说出并避免这个高频误区：分不清 class 与 instance，或者不知道 `self` 指向谁。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 把 class 当成‘写法更绕的函数集合’，没有看到状态抽象的价值。
> - 分不清 class 与 instance，或者不知道 `self` 指向谁。
> - 对象初始化阶段没有建立好合法状态。
