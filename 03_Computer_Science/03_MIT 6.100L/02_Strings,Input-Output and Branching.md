---
date: 2026-02-04
科目: MIT 6.100L
---

# Strings, Input/Output and Branching

## Strings（字符串）

- 字符串是**按顺序排列的字符序列**（区分大小写）
- 用单引号或双引号包裹，风格保持一致
- 基本操作：拼接 `+`、重复 `*`

```python
s = "me"
b = "myself"
print(s + b)        # memyself
print(s + " " + b)  # me myself
print(s * 3)        # mememe
```

```python
f= "a"
g=" b"
h="3"
S2=(f+g)*int(h)
print(S2)
```

### 长度与索引

- `len(s)` 返回字符串长度
- 索引从 0 开始；最后一个索引是 `len(s)-1` 或 `-1`

```python
s = "abc"
print(len(s))  # 3
print(s[0])    # a
print(s[-1])   # c
```

### 切片（Slicing）

- 形式：`s[start:stop:step]`
- ==`stop` 不包含在结果里（到 `stop-1`）==
- `step` 可为负，表示从右往左

```python
s = "abcdefgh"
print(s[3:6])     # def
print(s[3:6:2])   # df
print(s[:])       # abcdefgh
print(s[::-1])    # hgfedcba
print(s[4:1:-2])  # ec
```
 
> [!important] 字符串不可变（Immutable）
> 不能直接修改某个位置的字符，只能创建新字符串再绑定。
>
> ```python
> s = "car"
> # s[0] = 'b'  # 报错
> s = 'b' + s[1:]
> print (s)
> ```

> [!tip] Big Idea
> “如果我改成这样会怎样？”——**直接在控制台试**。

## Input / Output

### 输出（print）

- 交互式 shell 会显示表达式结果，但脚本里需要 `print()` 才可见
- `print(a, b, c)` 用空格分隔；也可先拼接成字符串

```python
a = "the"
b = 3
c = "musketeers"
print(a, b, c)
print(a + str(b) + c)
```

### 输入（input）

- `input(s)` 会显示提示字符串 `s`，并**始终返回字符串**
- 处理数字需要显式类型转换

```python
num1 = input("Type a number: ")
print(5 * num1)       # 字符串重复
num2 = int(input("Type a number: "))
print(5 * num2)       # 数值乘法
```

> [!important] 输入永远是 str
> 参与数值计算前先 `int()` / `float()`。

### f-strings

- `f"...{expr}..."`：花括号里放表达式，运行时求值并转成字符串

```python
num = 3000
fraction = 1/3
print(f"{num*fraction} is {fraction*100}% of {num}")
```

> [!tip] Big Idea
> **表达式可以放在任何需要值的位置**，Python 会先求值。

## Branching（条件分支）

### 赋值 vs 相等判断

- 赋值：`var = value`（改变绑定）
- 判断：`expr1 == expr2`（得到 True/False）

### 比较与逻辑运算

- 比较运算：`> >= < <= == !=` → 返回布尔值
- 逻辑运算：`not`、`and`、`or`

```python
pset_time = 15
sleep_time = 8
print(sleep_time > pset_time)  # False

derive = True
drink = False
print(derive and drink)        # False
```

### if / elif / else

- 条件必须是布尔表达式
- **缩进决定代码块**（Python 语法核心）
- `if-elif-else`：只执行第一个为 True 的分支

```python
if (pset_time + sleep_time) > 24:
    print("impossible!")
elif (pset_time + sleep_time) >= 24:
    print("full schedule!")
else:
    leftover = abs(24 - pset_time - sleep_time)
    print(leftover, "h of free time!")
print("end of day")
```

> [!important] 缩进就是控制流
> 语义结构必须匹配视觉结构；缩进错误会导致逻辑错误。

> [!tip] Big Idea
> 先写一点、测一点；尽早调试。遇到意外，用 Python Tutor 逐步走代码。

## 课堂练习（You Try It）

- [ ] 字符串拼接与重复：`b = ":"; c = ")"; s1 = b + 2*c` 的结果是什么？
- [ ] `s = "ABC d3f ghi"`：分别求 `s[3:len(s)-1]`、`s[4:0:-1]`、`s[6:3]`
- [ ] 输入动词并输出：`I can <verb> better than you!`，再打印该动词 5 次
- [ ] 保存 secret number，输入猜测，打印 True/False 是否相等
- [ ] 猜数字：判断“太低/太高/相等”
- [ ] 修正缩进错误的 if 代码（语义结构与视觉结构一致）

## 小结

- 字符串是字符序列：可索引、可切片、不可变
- `input()` 总是返回字符串，数值运算需显式类型转换
- `print()` 控制输出；`f"...{expr}..."` 简化格式化
- 比较与逻辑运算产生布尔值，驱动 if/elif/else 分支
- 缩进决定程序结构，写一点就测试一点

## 资料

- ![[MIT 6.100L-slides/mit6_100l_lec02.pdf]]
