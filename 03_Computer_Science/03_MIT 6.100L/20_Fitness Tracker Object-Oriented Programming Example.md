---
aliases:
  - MIT 6.100L Lecture 20
  - 6.100L L20
  - Fitness Tracker Object-Oriented Programming Example
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 20
---

# Lecture 20: Fitness Tracker Object-Oriented Programming Example

> [!tip] Hint
> - 我能把一个真实场景拆成多个对象与职责，而不是只写一个大类。
> - 我能解释对象设计里 state、behavior、interaction 三个维度。
> - 我能说明为什么案例课的重点是建模与接口，而不是某个具体业务细节。
> - 我能从完整例子里看出前面 class / inheritance / mutability 的知识如何组合。
> - 我能围绕本讲的主轴 “从需求出发划分对象职责” / “对象之间的交互决定了系统结构” / “综合案例能暴露设计弱点”，不翻 slides 也把整节课重新讲一遍。
> - 我能从需求描述中识别候选对象与职责。
> - 我能说明对象交互如何影响系统结构。
> - 我能判断一个类是否已经承担过多责任。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 10.4
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: 从需求出发划分对象职责 / 对象之间的交互决定了系统结构 / 综合案例能暴露设计弱点
> - 这一讲不是引入全新概念，而是把前面 OOP 的知识放进一个更完整、更接近真实应用的案例里。
> - Fitness tracker 例子提醒你：工程问题通常不是一招鲜，而是多个抽象一起协作。
> - 这也是课程从‘学单个概念’走向‘组合概念做设计’的重要一步。

## Core ideas
### 从需求出发划分对象职责
真实案例里最难的不是写方法，而是决定系统里应该有哪些对象、每个对象负责什么、对象之间如何协作。
- 如果一类数据总是和同一组操作一起出现，它们往往属于同一个对象。
- 设计对象时先问‘谁拥有这份状态’，再问‘谁最适合负责这件行为’。
- 职责划分清楚后，代码会自然拆开；职责混乱时，类就会越写越臃肿。
- 案例课的价值在于训练这种分工眼光。

> [!note] What to internalize
> - One-sentence takeaway: 真实案例里最难的不是写方法，而是决定系统里应该有哪些对象、每个对象负责什么、对象之间如何协作。
> - Review anchor: 如果一类数据总是和同一组操作一起出现，它们往往属于同一个对象。
> - Review anchor: 设计对象时先问‘谁拥有这份状态’，再问‘谁最适合负责这件行为’。

从做题角度看，只要题目在考“从需求出发划分对象职责”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：真实案例里最难的不是写方法，而是决定系统里应该有哪些对象、每个对象负责什么、对象之间如何协作。

### 对象之间的交互决定了系统结构
一个对象设计得再好，如果对象之间耦合混乱，整体系统仍然会难以维护。
- 好的交互方式通常让每个对象只暴露必要接口，而不是彼此随意操作对方内部状态。
- 对象协作要围绕业务流程组织，例如记录一次运动、更新统计、汇总结果等。
- 当某个对象知道了太多别人的内部细节，往往说明抽象边界不够稳。
- 案例中真正值得学的是接口分工，而不是某个变量名。

> [!note] What to internalize
> - One-sentence takeaway: 一个对象设计得再好，如果对象之间耦合混乱，整体系统仍然会难以维护。
> - Review anchor: 好的交互方式通常让每个对象只暴露必要接口，而不是彼此随意操作对方内部状态。
> - Review anchor: 对象协作要围绕业务流程组织，例如记录一次运动、更新统计、汇总结果等。

从做题角度看，只要题目在考“对象之间的交互决定了系统结构”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：一个对象设计得再好，如果对象之间耦合混乱，整体系统仍然会难以维护。

### 综合案例能暴露设计弱点
一旦把概念放进稍复杂的真实例子里，哪些接口太脆弱、哪些状态约束没想清楚，会立刻暴露出来。
- 综合案例会迫使你思考更新路径、数据一致性、扩展性，而不是只让代码在一个例子上跑通。
- 如果一个类在案例里承担过多责任，通常应该继续 decomposition 或重新设计抽象。
- 从案例中回看前面概念时，你会更清楚为什么 class 设计、mutability 管理、inheritance 边界都那么重要。
- 这也是为什么案例课通常比单概念讲解更接近真正的程序设计。

> [!note] What to internalize
> - One-sentence takeaway: 一旦把概念放进稍复杂的真实例子里，哪些接口太脆弱、哪些状态约束没想清楚，会立刻暴露出来。
> - Review anchor: 综合案例会迫使你思考更新路径、数据一致性、扩展性，而不是只让代码在一个例子上跑通。
> - Review anchor: 如果一个类在案例里承担过多责任，通常应该继续 decomposition 或重新设计抽象。

从做题角度看，只要题目在考“综合案例能暴露设计弱点”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：一旦把概念放进稍复杂的真实例子里，哪些接口太脆弱、哪些状态约束没想清楚，会立刻暴露出来。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - EXAMPLE: Simple workout class
> - TEST: Inspect internal state of classes
> - print(SimpleWorkout.__dict__.keys()) # dict_keys(['__module__', '__init__', 'get_calories', 'get_start', 'set_calories', 'set_start', '__dict__', '__weakref__', '__doc__'])
> - print(SimpleWorkout.__dict__.values())
> - print(my_workout.__dict__.keys()) # dict_keys(['start', 'end', 'calories', 'icon', 'kind'])
> - print(my_workout.__dict__.values())
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 把记录与统计拆到对象里
> ```python
> class Workout:
>     def __init__(self, minutes):
>         self.minutes = minutes
>
> class Tracker:
>     def __init__(self):
>         self.workouts = []
>
>     def add_workout(self, workout):
>         self.workouts.append(workout)
> ```
> 即使是最小例子，也能看出职责划分：`Workout` 表达单次记录，`Tracker` 管理集合与汇总入口。

> [!example] 对象方法维护自身状态
> ```python
> class Tracker:
>     def __init__(self):
>         self.total_minutes = 0
>
>     def log_minutes(self, minutes):
>         self.total_minutes += minutes
> ```
> 比起在系统别处直接改 `total_minutes`，由对象方法集中管理状态更容易维护一致性。

## Exercise log
> [!note] Finger exercise snapshot
> - Official prompt: In this problem, you will implement two classes according to the specification below: one Container class and one Queue class (a subclass of Container ). Our Container class will initialize an empty list. The two...
> - Official solution sketch:
> ```python
> class Container(object):
> def __init__(self):
> self.myList = []
> def size(self):
> return len(self.myList)
> def add(self, elem):
> ```
> - What this is really testing: whether you can compress the lecture into one small, high-frequency coding move without needing the slides beside you.
> - Where to revisit if this feels shaky: go back to the first two Core ideas sections in this note, then rerun the official lecture code once with your own input.
> - Follow-on practice path: after this finger exercise, the most natural next stop is Recitation 09.
> - Homework bridge: this lecture is directly connected to the following calendar milestones: PS 5 out, PS 4 due.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: calendar shows this lecture touching the following milestones: PS 5 out；PS 4 due。读完本讲后，不应只会解释概念，还应能把它们搬到更长的程序里。
> - Recitation connection: Recitation 09 is the best place to turn the lecture ideas into shorter solved exercises.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec20.pdf|Lecture 20 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec20_code.zip|Lecture 20 code (zip)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex20_sol.pdf|Lecture 20 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec20_transcript.pdf|Lecture 20 transcript]]
- Recitation 9: [[MIT 6.100L-recitations/mit6_100l_rec09.zip|Recitation 09 materials]]
- PS 5 out: [[MIT 6.100L-problem-sets/mit6_100l_ps5.pdf|PS5 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps5_code.zip|PS5 starter code]]
- PS 4 due: [[MIT 6.100L-problem-sets/mit6_100l_ps4.pdf|PS4 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps4_code.zip|PS4 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 10.4)

## Review checklist
- [ ] 我能从需求描述中识别候选对象与职责。
- [ ] 我能说明对象交互如何影响系统结构。
- [ ] 我能判断一个类是否已经承担过多责任。
- [ ] 我能把前面学过的 OOP 概念放回综合案例中理解。
- [ ] 我能设计一个更小但职责清晰的追踪系统。
- [ ] 我能围绕“从需求出发划分对象职责”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“对象之间的交互决定了系统结构”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：把真实案例误解为‘业务代码演示’，忽略它真正训练的是对象建模。
- [ ] 我能说出并避免这个高频误区：一个类包揽所有责任，导致系统没有清晰边界。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 把真实案例误解为‘业务代码演示’，忽略它真正训练的是对象建模。
> - 一个类包揽所有责任，导致系统没有清晰边界。
> - 对象之间直接操作彼此内部状态，破坏抽象。
