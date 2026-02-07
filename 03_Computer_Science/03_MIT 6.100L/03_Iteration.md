---
date: 2026-02-07
科目: MIT 6.100L
---

# Iteration（循环）

## 上节回顾（连接到本讲）

- 字符串可以索引/切片，且是 immutable
- `input()` 总是返回 `str`
- `if / elif / else` 用布尔条件控制分支，缩进决定代码块
- 本讲在分支基础上加入“重复执行”：循环（loops）

## while loops（条件驱动循环）

### 基本结构

```python
while <condition>:
    <code>
    <code>
```

- 每轮先判断 `<condition>`
- 若为 `True`，执行缩进代码块，再回到条件判断
- 若为 `False`，退出循环

> [!important] Big Idea
> `while` 可以无限重复执行。  
> 条件若永远不变成 `False`，程序就会进入 infinite loop。

### 例子：Lost Forest

```python
where = input("You're in the Lost Forest. Go left or right? ")
while where == "right":
    where = input("You're in the Lost Forest. Go left or right? ")
print("You got out of the Lost Forest!")
```

- 只有输入精确的 `"right"`（小写）才会继续循环
- 如果输入 `"RIGHT"`，条件为 `False`，会直接退出

### 终止与调试 infinite loop

```python
n = int(input("Enter a non-negative integer: "))
while n > 0:
    print("x")
    n = n - 1
```

- 循环变量必须更新，否则可能无限循环
- 在 IDE/shell 中可用 `Ctrl-C`（或 `Cmd-C`）强制中断

### 常见模式 1：计数循环（counter pattern）

```python
n = 0
while n < 5:
    print(n)
    n = n + 1
```

### 常见模式 2：累乘（factorial）

```python
x = 4
i = 1
factorial = 1
while i <= x:
    factorial *= i
    i += 1
print(f"{x} factorial is {factorial}")
```

## for loops（序列驱动循环）

### 基本结构

```python
for <variable> in <sequence>:
    <code>
```

- 每次循环，`<variable>` 依次取序列中的下一个值
- 序列耗尽后，循环自动结束（天然有限）

> [!tip] Big Idea
> `for` 的循环次数由“序列长度”决定，通常更不容易写出无限循环。

### while 与 for 对照

```python
# while: 更冗长
n = 0
while n < 5:
    print(n)
    n = n + 1

# for: 更紧凑
for n in range(5):
    print(n)
```

## `range()`：最常用的整数序列

- `range(stop)`：从 `0` 到 `stop-1`
- `range(start, stop)`：从 `start` 到 `stop-1`
- `range(start, stop, step)`：按步长 `step` 递增/递减
- `stop` 永远是“右开区间”（不包含）

```python
for i in range(1, 4, 1):
    print(i)        # 1, 2, 3

for j in range(1, 4, 2):
    print(j * 2)    # 2, 6

for me in range(4, 0, -1):
    print("$" * me) # $$$$, $$$, $$, $
```

## 常见模式：running sum（滚动求和）

```python
mysum = 0
for i in range(10):
    mysum += i
print(mysum)  # 45
```

- `mysum` 作为“累计状态”在每一轮更新
- `range(10)` 实际遍历 `0..9`

### 区间求和的边界修正

如果要“包含 end”，应写成 `range(start, end + 1)`：

```python
mysum = 0
start = 3
end = 5
for i in range(start, end + 1):
    mysum += i
print(mysum)  # 12
```

## for 重写 factorial

```python
x = 4
factorial = 1
for i in range(1, x + 1, 1):
    factorial *= i
print(f"{x} factorial is {factorial}")
```

## while vs for：何时用哪个

- `while`：当“结束条件”是逻辑条件（未知要循环几次）时更自然
- `for`：当“要遍历一个已知序列（如 range、字符串）”时更自然
- 写 `while` 时重点检查：循环变量是否更新、终止条件是否可达

## 课堂练习（You Try It）

- [ ] 运行 Lost Forest 代码，输入 `"RIGHT"` 时会输出什么？
- [ ] 给 Lost Forest 加计数器：若进入循环超过 2 次，显示 sad face
- [ ] 运行并手动停止 `while True:` 的无限循环
- [ ] 口算/实测 `range(1,4,1)`、`range(1,4,2)`、`range(4,0,-1)` 的输出
- [ ] 修改区间求和代码，使其包含 `start` 与 `end`

## 小结

- 循环让程序具备“重复执行”能力：`while` 看条件，`for` 看序列
- `while` 功能灵活，但要警惕 infinite loop
- `for + range` 是整数序列遍历的高频组合
- 计数、求和、阶乘是本讲的三个核心模板

## 资料

- ![[MIT 6.100L-slides/mit6_100l_lec03.pdf]]
