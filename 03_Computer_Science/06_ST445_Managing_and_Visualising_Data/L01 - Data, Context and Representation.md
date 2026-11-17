---
course: ST445
lecture: 1
week: 1
unit_kind: lecture
lecture_date: 2026-09-29
aliases:
  - ST445 Lecture 1 - Data, Context and Representation
  - ST445 L01 - Data Preparation
---

<!-- bilingual-en:start -->
# L01 — Data, Context and Representation
<!-- bilingual-en:end -->

### 第一讲：让数据表示保留原来的含义

<!-- review-entry:start -->
<!-- bilingual-en:start -->
## Recall
<!-- bilingual-en:end -->

> [!info] Topic · Records and meaning / 记录与含义
> <!-- bilingual-en:start -->
> Reconstruct a hospital record, its schema and the unit represented by one row.
> <!-- bilingual-en:end -->
> 从一条医院记录出发，回想字段、结构约定以及“一行是什么”。
>
> > [!question]- Prompt · Repeated measurements / 重复测量
> > <!-- bilingual-en:start -->
> > A patient has one temperature reading at 08:15 and another at 12:15. Why can `patient_id` alone fail as a key for the measurement table?
> > <!-- bilingual-en:end -->
> > 同一患者在两个时刻各测一次体温，为什么患者编号不能单独标识测量表中的一行？
> >
> > > [!success]- Answer · 答案
> > > <!-- bilingual-en:start -->
> > > It identifies the patient, while the table records measurement events. Both rows have that identifier. A key must distinguish the events, for example patient plus timestamp if that combination is unique. See [[#8. Define an observation before reshaping|observations and keys]].
> > > <!-- bilingual-en:end -->
> > > 编号认出患者，却没有区分两次测量；需要能唯一识别测量事件的字段组合。
>
> > [!question]- Prompt · A missing field value / 字段没有值
> > <!-- bilingual-en:start -->
> > A temperature table has the same defined columns and types in every row, but one temperature is missing. Does that alone make the data semi-structured?
> > <!-- bilingual-en:end -->
> > 表的字段和类型约定没有改变，只缺了一次温度值，这是否就变成半结构化数据？
> >
> > > [!success]- Answer · 答案
> > > <!-- bilingual-en:start -->
> > > No. The field still exists under the fixed schema; this record lacks its value. See [[#1. Give the value a context and a schema|the concrete schema]].
> > > <!-- bilingual-en:end -->
> > > 不会。缺少字段的值与改变字段结构，是两件事。

> [!info] Topic · Representation and operations / 表示与运算
> <!-- bilingual-en:start -->
> Recall what integer width, floating-point precision and container operations preserve.
> <!-- bilingual-en:end -->
> 回想位宽、浮点精度和容器操作分别会怎样影响保存的信息。
>
> > [!question]- Prompt · The order of conversion / 转换的先后
> > <!-- bilingual-en:start -->
> > A `uint32` array contains `2**32 - 1`. Why can widening to `uint64` before adding 1 differ from widening afterwards?
> > <!-- bilingual-en:end -->
> > 为什么先扩大类型再加一，与先加一再扩大类型，会得到不同结果？
> >
> > > [!success]- Answer · 答案
> > > <!-- bilingual-en:start -->
> > > The `uint32` addition wraps to zero. Widening afterwards preserves that already altered result; widening first provides room for `2**32`. See [[#2. Match numerical representation to the job|integer width]].
> > > <!-- bilingual-en:end -->
> > > 后转换无法恢复已经丢掉的进位；先转换才给正确结果留下空间。

> [!info] Topic · Reading and checking data / 读取与检查
> <!-- bilingual-en:start -->
> Recall the source agreements needed for missing values, text and time.
> <!-- bilingual-en:end -->
> 回想缺失值、文本和时间各自需要什么来源约定。
>
> > [!question]- Prompt · Reading an identifier / 读取标识符
> > <!-- bilingual-en:start -->
> > A source defines `00731` as a patient identifier. Why does converting it to an integer and then back to a string fail to preserve the identifier?
> > <!-- bilingual-en:end -->
> > 将患者编号先转成整数再转回字符串，为什么不能保留原来的编号？
> >
> > > [!success]- Answer · 答案
> > > <!-- bilingual-en:start -->
> > > Integer conversion retains the numerical value 731 but discards the two leading zeros. The later string conversion produces `"731"`. Read identity-bearing codes as strings. See [[#1. Give the value a context and a schema|field meaning and types]].
> > > <!-- bilingual-en:end -->
> > > 数值没有记录原来写了几个前导零，转回字符串也不能自动找回它们。

<!-- bilingual-en:start -->
## Quick self-test

Allow 5–10 minutes for the three questions. Try each one first. Open the hint if you get stuck, then check your answer.
<!-- bilingual-en:end -->

三题约需 5–10 分钟。先自己做，卡住时看提示，做完再对答案。

> [!example] Test 1 · Trace two variables / 跟踪两个变量
> <!-- bilingual-en:start -->
> After the code below, what are `readings` and `result`? Starting again from the original list, replace the second line so that `readings` keeps its order and `result` holds a sorted list.
>
> ```python
> readings = [38.1, 37.9, 38.1]
> result = readings.sort()
> ```
> <!-- bilingual-en:end -->
> 执行后两个变量各是什么？重新从原列表开始，怎样只改第二行，保留原顺序并得到排序结果？
>
> > [!question]- Hint · 提示
> > <!-- bilingual-en:start -->
> > Trace the effect of the method call on the existing object, then separately trace the value assigned to `result`.
> > <!-- bilingual-en:end -->
> > 先看原对象受到什么操作，再看赋值语句接收什么返回值。
>
> > [!success]- Solution and check · 答案与核对
> > <!-- bilingual-en:start -->
> > `readings` becomes `[37.9, 38.1, 38.1]`; `result` is `None`. From the original list, use `result = sorted(readings)`. Check both the object's change and the returned value. See [[#5. Choose containers by the information needed|mutation and sorting]].
> > <!-- bilingual-en:end -->
> > 原列表已经排好，返回值却是 `None`；`sorted(readings)` 才会返回新列表。两种写法都会保留重复值。

> [!example] Test 2 · Read a small extract / 读取一小段数据
> <!-- bilingual-en:start -->
> The source defines `patient_id` as an exact string, `country` as an ISO country code, and an empty `temperature_c` as unrecorded. State the values or missing states that should be retained from these two rows. Does `keep_default_na=False` prevent explicitly supplied `na_values` from identifying missing temperatures?
>
> ```csv
> patient_id,temperature_c,country
> 4411,37.9,NA
> 00731,,GB
> ```
> <!-- bilingual-en:end -->
> 按给定字段规则，哪些内容应保留为原值，哪里应识别为缺失？关闭默认缺失词表后，显式指定的缺失规则还会不会生效？
>
> > [!question]- Hint · 提示
> > <!-- bilingual-en:start -->
> > Apply the source's meaning to each column separately. Then distinguish the default missing-value list from the list supplied by the caller.
> > <!-- bilingual-en:end -->
> > 分列判断，不要把一种标记的解释套到整份文件；再区分默认规则和自己传入的规则。
>
> > [!success]- Solution and check · 答案与核对
> > <!-- bilingual-en:start -->
> > Keep IDs `"4411"` and `"00731"`, country codes `"NA"` and `"GB"`, and temperature 37.9 in the first row. The second temperature is missing. `NA` denotes Namibia here. Explicit `na_values` still apply when `keep_default_na=False` under normal missing-value parsing. Check that no row or leading zero is lost. See [[#6. Make missing-value rules field-specific|the checked CSV example]].
> > <!-- bilingual-en:end -->
> > 保留两个编号、两个国家代码和 37.9；只有第二条温度缺失。关闭默认词表后，仍可用 `na_values` 指定空白温度为缺失。

> [!example] Test 3 · Reorganise observations / 重新组织观察
> <!-- bilingual-en:start -->
> On the same date, each patient has exactly one reading at each listed time. Write the fields and the four records of a tidy table. Why does changing two storage rows into four rows not mean that new measurements were collected?
>
> | patient_id | 08:15 | 12:15 |
> |---|---:|---:|
> | 4411 | 37.9 | 38.1 |
> | 00731 | 36.8 | 37.0 |
> <!-- bilingual-en:end -->
> 同一天内，每位患者在表列的两个时刻各测一次。写出 tidy 表的字段和四条记录，并解释行数增加的含义。
>
> > [!question]- Hint · 提示
> > <!-- bilingual-en:start -->
> > Choose one cell containing a temperature. Identify all the labels needed to say what that measurement is.
> > <!-- bilingual-en:end -->
> > 先选一个温度单元格，找出说明这次测量所需的行标和列标。
>
> > [!success]- Solution and check · 答案与核对
> > <!-- bilingual-en:start -->
> > Use `patient_id`, `time`, `temperature_c`: `(4411, 08:15, 37.9)`, `(4411, 12:15, 38.1)`, `(00731, 08:15, 36.8)`, `(00731, 12:15, 37.0)`, keeping identifiers as strings. The four measurements already occupied four cells; reshaping exposes their time labels as values. Check that each original cell appears once with its original patient and time. See [[#8. Define an observation before reshaping|the observation rule]].
> > <!-- bilingual-en:end -->
> > 把列名里的时间变成字段，四次测量各占一行；没有新增测量。编号的字符形式、时间和温度的对应关系都要保留。
<!-- review-entry:end -->

<!-- bilingual-en:start -->
## 1. Give the value a context and a schema
<!-- bilingual-en:end -->

### 1. 先给数值语境，再写清结构约定

<!-- bilingual-en:start -->
The lecture asks how recorded data can become a trustworthy, communicable analysis. A question guides data exploration, including exploratory data analysis (EDA); what the data reveal may refine the question. Before that exploration, we must know what the records mean.

The slide's number **37.9** becomes interpretable when accompanied by a measurement, unit, subject and time:

| Field | Example value | Meaning or representation rule |
|---|---|---|
| `patient_id` | `"4411"` | Exact identity code; preserve as a string |
| `temperature_c` | `37.9` | Measured body temperature in degrees Celsius |
| `measured_at` | `2026-10-05T08:15Z` | Timestamp explicitly expressed in UTC |
| `triage_category` | `2` | Integer category code interpreted through its codebook |

These agreements form a **schema**: which fields exist, how their values are represented, and how records are organised. For this table, one record describes one measurement event. A missing temperature leaves the field in place; it does not by itself change the schema.

**Data** are recorded facts or values. **Information** is their organisation and interpretation in relation to a question. An identifier that looks numerical is still used for matching, not arithmetic: `str(int("00731"))` produces `"731"`. An ordinal category may have an order without equal numerical gaps. The source's meaning determines whether arithmetic is appropriate.
<!-- bilingual-en:end -->

这张小表具体展示了 schema 的三类约定：有哪些字段、各字段怎样表示、每条记录如何组织。单位、对象和时间把 37.9 连到现实中的一次测量；患者编号和分诊编码即使长得像数字，也不因此具有普通数量的算术含义。

<!-- bilingual-en:start -->
For a yes/no field, map the source's stated codes to `True` and `False`, while preserving a distinct missing state if needed. `bool("False")` is `True` because a nonempty string is truthy; it does not interpret the English word. Type conversion therefore needs the field's meaning, not only a convenient constructor.
<!-- bilingual-en:end -->

Boolean 转换也应按来源约定。没有回答不等于回答“否”；`bool()` 判断非空字符串的规则，不是问卷答案的解释规则。

<!-- bilingual-en:start -->
A structured table follows fixed fields and types. Semi-structured records use keys or tags but may vary or nest. For example, `{"os":"iOS","app":"3.2"}` and `{"os":"Android"}` share a key but do not have identical fields. Unstructured text, such as a free-form clinical note, needs interpretation before symptoms or times can become analysis variables. A free-text column does not become semantically structured merely because it sits in a spreadsheet; the slide's survey combines ratings, comments and JSON in one table.

The three V's motivate careful preparation: **volume** is growing quantity, **velocity** is the rate of arrival or change, and **variety** is diversity of sources and forms. Their importance here is practical: a large, changing mixture needs explicit interpretation rules.
<!-- bilingual-en:end -->

结构化、半结构化与非结构化描述的是内容怎样组织和解释，不是文件扩展名的机械分类。三种 V 也只用于说明为何不能逐条猜测数据含义。

[[ST445 - Week 01 - Lecture.pdf#page=17|PPT pp. 16–18]] · [[ST445 - Week 01 - Lecture.pdf#page=29|PPT pp. 29–32]]

<!-- bilingual-en:start -->
## 2. Match numerical representation to the job
<!-- bilingual-en:end -->

### 2. 区分数值范围、精度与字段用途

<!-- bilingual-en:start -->
A **bit** has two possible states; eight bits form one **byte**. With $n$ bits there are $2^n$ patterns. Their interpretation determines the range:

| Representation | Range | Eight-bit example |
|---|---|---|
| Unsigned integer | $0$ to $2^n-1$ | 0 to 255 |
| Signed two's-complement integer | $-2^{n-1}$ to $2^{n-1}-1$ | −128 to 127 |

Python's built-in `int` grows as needed, subject to memory and computation resources. NumPy can instead use fixed-width types. A `uint32` addition that exceeds its range retains the result modulo $2^{32}$; `2**32 - 1` plus 1 becomes 0. Widen **before** the operation:

```python
import numpy as np

a = np.array([2**32 - 1], dtype=np.uint32)
print((a.astype(np.uint64) + 1).tolist())  # [4294967296]
print((a + 1).astype(np.uint64).tolist())  # [0]
```

The second conversion cannot restore an already lost carry. Signed fixed-width overflow can instead produce a negative result; it is not a rule that all integer overflow becomes zero.
<!-- bilingual-en:end -->

位宽决定可用模式的数量。Python `int` 没有统一的固定宽度；NumPy 的指定整数类型有。类型扩大必须在可能溢出的运算之前完成，事后转换只是把已经错误的结果放进更宽的容器。

<!-- bilingual-en:start -->
Floating-point **precision** is a different issue from range. A finite binary fraction has a denominator that is a power of two; decimal fractions such as 3.14 usually need rounding. Thus `6.14 - 3.14` gives `2.9999999999999996` on the usual binary64 representation. For finite values, `math.isclose` compares the difference using

$$|a-b|\le\max\{\text{rel\_tol}\max(|a|,|b|),\ \text{abs\_tol}\}.$$

`rel_tol` scales with the numbers; `abs_tol` is an allowance in their units and is especially useful near zero. Supply both by keyword:

```python
import math

x = 6.14 - 3.14
print(x == 3.0)                                      # False
print(math.isclose(x, 3.0, rel_tol=1e-9, abs_tol=0.0)) # True
print(math.isclose(1e-12, 0.0, abs_tol=1e-10))        # True
```

The tolerance must reflect the required accuracy. Exact comparison remains meaningful when the question is whether two stored values are identical. [Python's `math.isclose` reference](https://docs.python.org/3/library/math.html#math.isclose) specifies the comparison and keyword-only arguments.
<!-- bilingual-en:end -->

比较计算结果与目标值时，先确定能接受多大的误差。目标值为零时，用 `abs_tol` 按测量单位给出容许差距。若要判断两个存储的数值是否恰好相等，就用 `==`。

<!-- bilingual-en:start -->
Binary64 has 53 bits of significand precision. All integers with absolute value at most $2^{53}$ are exactly representable, but immediately above this boundary the spacing becomes 2. Therefore `int(float(9007199254740993))` becomes `9007199254740992`. Larger exact integers still exist, such as $2^{53}+2$; the boundary concerns **consecutive** integers, not the maximum magnitude of a float.

An identifier should remain exact, using a suitable integer representation or a string. Also distinguish precision loss from floating-point overflow: an operation such as `1e308 * 10` produces `inf`, whereas converting an extremely large Python integer with `float(10**400)` raises `OverflowError` in CPython. The operation and type determine the behaviour.
<!-- bilingual-en:end -->

大整数转浮点可能先丢掉末位，而不是等数值“特别大”才出问题。`inf` 是浮点值，不是 Python `int` 变大之后的下一种状态；转换与浮点运算也可能有不同的越界表现。

[[ST445 - Week 01 - Lecture.pdf#page=34|PPT pp. 34–39]] · [[ST445 - Week 01 - Lecture.pdf#page=48|PPT p. 48]]

<!-- bilingual-en:start -->
## 3. See what vectorisation changes
<!-- bilingual-en:end -->

### 3. 用同一个计算比较向量化与 Python 循环

<!-- bilingual-en:start -->
A numeric NumPy array stores homogeneous values, and its elementwise operations run in compiled code. **Vectorisation** expresses a whole-array operation without writing the element loop in Python. The loop still happens, with less repeated Python-object handling. **Broadcasting** is the compatibility rule that lets, for example, a scalar such as 32 participate in an operation on an entire array. [NumPy's explanation](https://numpy.org/doc/stable/user/whatisnumpy.html) connects these mechanisms.

This example converts the same 200,000 temperatures from Celsius to Fahrenheit. It first checks that the two implementations agree, then prints measured execution times:

```python
import timeit
import numpy as np

values = [37.9] * 200_000
array = np.array(values, dtype=np.float64)

def python_loop():
    return [x * 9 / 5 + 32 for x in values]

def numpy_vector():
    return array * 9 / 5 + 32

assert np.allclose(python_loop(), numpy_vector())
for name, function in [("Python loop", python_loop),
                       ("NumPy", numpy_vector)]:
    seconds = min(timeit.repeat(function, number=1, repeat=3))
    print(name, f"{seconds:.6f} seconds")
```

Both methods give 100.22 for the first temperature. In one run, the list comprehension took 0.00594 seconds and NumPy took 0.000121 seconds. Timing starts with both inputs already in memory. The speedup depends on the hardware, array size and operation.
<!-- bilingual-en:end -->

NumPy 在内部执行循环，按 broadcasting 规则把标量运算应用到每个元素。先准备好两份输入，再核对结果并计时。这里的加速倍数会随硬件、数组大小和运算方式变化。

[[ST445 - Week 01 - Lecture.pdf#page=38|PPT p. 38]]

<!-- bilingual-en:start -->
## 4. Separate records, text and bytes
<!-- bilingual-en:end -->

### 4. 区分记录结构、文本格式和字节编码

<!-- bilingual-en:start -->
CSV represents rows and fields in text, with a delimiter and quoting rules. It does not by itself tell a reader that a field is an identifier, a measurement or a missing marker. JSON represents values through objects, arrays and primitive values; an object can hold a nested device record. For example:

```json
{
  "patient_id": "00731",
  "temperature_c": 37.9,
  "device": {"id": "T1", "unit": "degrees Celsius"}
}
```

The quotes make the identifier a string; the nested object keeps device fields together. The data still need agreements about units, required fields and valid values. A JSON object is a text representation, while a Python dictionary is an in-memory object; they are related by parsing or serialisation, not identical entities. See the [Python JSON reference](https://docs.python.org/3/library/json.html) and [CSV reference](https://docs.python.org/3/library/csv.html).
<!-- bilingual-en:end -->

CSV/JSON 解决怎样写下记录结构的问题，字段含义仍要另外约定。JSON 的大括号可以表达嵌套对象，但文件里的 JSON 文本与程序中的 `dict` 是不同层次。

<!-- bilingual-en:start -->
**Unicode** assigns code points; **UTF-8** maps them to bytes. One code point uses one to four UTF-8 bytes:

| Character | Code point, decimal | UTF-8 bytes, hexadecimal |
|---|---:|---|
| `A` | 65 | `41` |
| `é` | 233 | `C3 A9` |
| `€` | 8364 | `E2 82 AC` |

```python
text = "café"
raw = text.encode("utf-8")
print(len(text), len(raw))       # 4 5
print(raw.decode("utf-8"))      # café
print(raw.decode("latin-1"))    # cafÃ©
```

The bytes do not change when the wrong decoder is chosen; their interpretation does. Python `len(str)` counts code points, not bytes or, in every writing system, visible characters. Confirm the source's encoding instead of deleting accents to hide a decoding problem. Mismatched decoding can break joins and grouping even when the program raises no exception.
<!-- bilingual-en:end -->

`é` 的编号是 233，UTF-8 编码却是两个 bytes：`C3 A9`。`café` 因此是 4 个 code points、5 个 UTF-8 bytes。编码和解码必须匹配；乱码不是“字符本来就坏了”，也不能靠随意删掉重音符号解决。

[[ST445 - Week 01 - Lecture.pdf#page=31|PPT pp. 29–31]] · [[ST445 - Week 01 - Lecture.pdf#page=40|PPT p. 40]] · [[ST445 - Week 01 - Lecture.pdf#page=50|PPT p. 50]]

<!-- bilingual-en:start -->
## 5. Choose containers by the information needed
<!-- bilingual-en:end -->

### 5. 容器保留什么，方法改变什么

<!-- bilingual-en:start -->
| Container | Example | What it is useful for |
|---|---|---|
| `dict` | `{"patient_id":"4411", "temperature_c":37.9}` | Look up a field by its unique key |
| `list` | `["4411", "00731", "4411"]` | Keep an adjustable sequence, including repeated visits |
| `tuple` | `(37.9, "degrees Celsius")` | Keep a fixed combination of positions |
| `set` | `{"4411", "00731"}` | Keep distinct values and test membership |

Setting an existing dictionary key replaces its value. Using only patient ID as a key for multiple measurements can therefore overwrite an earlier event. A list preserves order and repetitions; converting the visit list above to a set loses both occurrence counts and any dependable positional order.

An empty dictionary is `{}`; an empty set is `set()`. A one-item tuple is `(37.9,)`: the comma creates it. A tuple prevents rebinding its positions, but an object inside it can still be mutable.

For lists, `sort()`, `append()` and `reverse()` mutate the object and return `None`. `sorted()` returns a new list:

```python
readings = [38.1, 37.9, 38.1]
ordered = sorted(readings)
print(readings)  # [38.1, 37.9, 38.1]
print(ordered)   # [37.9, 38.1, 38.1]
```

The distinction is between a method's effect and its return value. Assigning `readings = readings.sort()` first sorts the object, then rebinds the name to `None`.
<!-- bilingual-en:end -->

选择容器是在选择要保留的信息。重复患者编号可能代表真实的多次就诊；去重会主动丢掉次数。原地修改与返回值也必须分开看：列表改好了，不代表方法返回了这个列表。

[[ST445 - Week 01 - Lecture.pdf#page=41|PPT pp. 41–44]]

<!-- bilingual-en:start -->
## 6. Make missing-value rules field-specific
<!-- bilingual-en:end -->

### 6. 先解释缺失，再决定如何处理

<!-- bilingual-en:start -->
Missingness means that a required value is unavailable. A blank, `NA`, `-999` or `0` is only a spelling until the source defines it. A valid zero count differs from an unrecorded count; an unmeasured temperature differs from a measured value. Preserve reasons such as “not measured” or “not applicable” when the analysis needs that distinction. Recognising missingness is separate from deleting or imputing records.

In the hospital example, assume the source has confirmed three rules: IDs are exact strings; `country` uses ISO country codes; only blank or `-999` temperatures are missing. A blank country is also missing, but `NA` is the valid code for Namibia.

```python
from io import StringIO
import pandas as pd

csv_text = """patient_id,temperature_c,country
4411,37.9,NA
00731,,GB
"""

records = pd.read_csv(
    StringIO(csv_text),
    dtype={"patient_id": "string", "temperature_c": "float64",
           "country": "string"},
    keep_default_na=False,
    na_values={"temperature_c": ["", "-999"], "country": [""]}
)
print(records["patient_id"].tolist())             # ['4411', '00731']
print(records["country"].tolist())                # ['NA', 'GB']
print(records["temperature_c"].isna().tolist())    # [False, True]
```

`keep_default_na=False` turns off pandas' default list of missing markers, so `NA` stays the country code for Namibia. The per-column `na_values` rules mark the second temperature as missing. The string `dtype` keeps both leading zeros in patient ID `00731`. Both rows appear in the result. [pandas read_csv reference](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)
<!-- bilingual-en:end -->

关闭默认词表，是为了避免国家代码 `NA` 被误删；随后仍要写出真正适用的缺失规则。字符串 dtype 保护编号，`na_values` 负责识别指定标记，这两件事互不替代。识别缺失后，也还没有决定应删行还是填补。

[[ST445 - Week 01 - Lecture.pdf#page=45|PPT pp. 45–46]] · [[ST445 - Week 01 - Lecture.pdf#page=51|PPT p. 51]]

<!-- bilingual-en:start -->
## 7. Apply standards without inventing context
<!-- bilingual-en:end -->

### 7. 标准传递已知含义，不补造缺少的信息

<!-- bilingual-en:start -->
| Standard | What it specifies | What still needs checking |
|---|---|---|
| IEEE 754 | Floating-point representation and arithmetic | Whether the chosen precision suits the field |
| Unicode / UTF-8 | Code points and their byte encoding | Which encoding the source actually used |
| ISO 8601 | An unambiguous date/time representation | Whether a timezone or offset was supplied |
| ISO 3166 | Country codes such as `GB`, `DE`, `NA` | Whether missing-value defaults destroy a real code |
| ISO 4217 | Currency codes such as `GBP`, `EUR` | The unit and currency intended by an amount |

`05/10/2026` can mean 5 October or 10 May under different conventions. `2026-10-05` fixes the date order, but supplies neither a clock time nor a timezone. `2026-10-05T08:15Z` explicitly denotes 08:15 UTC. Appending `Z` to a local time asserts that it was UTC; it is not a conversion between timezones.

Lexicographic sorting agrees with chronological sorting only under suitable shared format, precision and timezone conventions. Parse timestamps with their timezone information when computing differences or combining sources.
<!-- bilingual-en:end -->

标准化首先要有来源依据。明确的年月日顺序不等于明确的时区；仅有日期时，也不能自己补出某个时刻。国家代码和货币代码虽然都是文字，却描述不同的语义对象。

[[ST445 - Week 01 - Lecture.pdf#page=47|PPT pp. 47–52]]

<!-- bilingual-en:start -->
## 8. Define an observation before reshaping
<!-- bilingual-en:end -->

### 8. 先定义一次观察，再决定表格长什么样

<!-- bilingual-en:start -->
An **observational unit** is what counts as one observation in the analysis. In a temperature table it may be one patient's measurement at one time. A patient can have several valid rows. If a patient can be measured more than once at the same timestamp, even patient plus time is insufficient; a measurement identifier or another distinguishing field is needed.

The slide's borough example contains two boroughs and three years:

| borough | 2023 | 2024 | 2025 |
|---|---:|---:|---:|
| Camden | 412 | 398 | 405 |
| Hackney | 377 | 391 | 402 |

For annual borough counts, one observation is a **borough–year** combination. There are six observations stored across two rows. In tidy form, `borough`, `year` and `count` become columns; the cell 398 becomes the row `(Camden, 2024, 398)`.

```python
import pandas as pd

wide = pd.DataFrame({
    "borough": ["Camden", "Hackney"],
    "2023": [412, 377], "2024": [398, 391], "2025": [405, 402]
})
tidy = wide.melt(id_vars="borough", var_name="year", value_name="count")
tidy["year"] = tidy["year"].astype("int64")

assert len(tidy) == 6
assert not tidy.duplicated(["borough", "year"]).any()
assert tidy.loc[(tidy.borough == "Camden") & (tidy.year == 2024),
                "count"].item() == 398
```

Tidy data puts each variable in a column, each observation in a row, and each type of unit in a table. A hospital can use one table for patients and another for measurements, linked by patient ID. Temperature and heart rate measured at the same event can remain separate variables in separate columns. “Tidy” does not mean making every table as long as possible.
<!-- bilingual-en:end -->

六次 borough–year 观察原来就存在，只是年份藏在列名里。reshape 把这种对应关系写明，没有增加信息。检查应追踪原始单元格、观察数和键的唯一性，而不只是确认代码没有报错。医院的患者表与测量事件表也应按各自的观察单位组织。

[[ST445 - Week 01 - Lecture.pdf#page=53|PPT pp. 53–54]]

<!-- bilingual-en:start -->
## 9. Leave a checkable preparation record
<!-- bilingual-en:end -->

### 9. 让每个整理决定都有依据和检查

<!-- bilingual-en:start -->
A useful record connects the source's meaning to the transformation:

| Item | Hospital CSV example |
|---|---|
| Evidence | The source defines country codes and temperature missing markers |
| Rule | Preserve IDs as strings; use explicit missing markers by field |
| Reason | Keep Namibia's `NA` while recognising genuinely unavailable temperatures |
| Check | Both IDs and rows remain; `NA` survives; the blank temperature is missing |

Before use, establish **provenance**—who produced the data, when, how and with what changes—and the permitted uses and sharing arrangements. Check whether people are identifiable and which safeguards apply. Public visibility does not itself establish permission to reuse data.

For reproduction, retain raw inputs, rerunnable transformation code, source agreements and relevant environment versions. Another authorised reader should be able to rebuild the processed result and check the same invariants. Private records and secrets need not be made public to provide a documented reproduction path. Git tracks changes and GitHub supports hosting and collaboration; their practical workflow belongs to the Seminar.
<!-- bilingual-en:end -->

“清洗过”不是充分说明。要能指出来源依据、处理规则、理由和检查结果，并保留重跑所需的输入、代码和环境。是否可用、可共享，也应在使用数据前弄清楚。

[[ST445 - Week 01 - Lecture.pdf#page=55|PPT p. 55]] · [[ST445 - Week 01 - Lecture.pdf#page=59|PPT p. 59]]

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

### 来源与核验

<!-- bilingual-en:start -->
| Coverage | Source location |
|---|---|
| Analysis questions, context, data and information | PPT pp. 10–11, 16–18; recording about 00:48:18–00:50:29 and 01:03:36–01:10:26 |
| Schema and field types | PPT pp. 29–32; about 01:23:35–01:30:24 |
| Bits, integer width, floats and text | PPT pp. 34–40, 48, 50; about 01:31:42–01:47:14 |
| Collections and mutation | PPT pp. 41–44; about 01:47:16–01:52:50 |
| Missingness and standards | PPT pp. 45–52; about 01:53:01–01:56:30 |
| Observation units and tidy data | PPT pp. 53–54 |
| Data origin and use | PPT p. 55; about 01:56:47–01:57:34 |
| Reproducibility and Seminar boundary | PPT pp. 59–60, 70–71; the recording assigns GitHub practice to Seminar at about 01:56:32–01:56:46 |

Recording times are approximate and follow the lecture transcript. The code examples were run with NumPy 2.1.3 and pandas 2.2.3.
<!-- bilingual-en:end -->

页码对应 Week 1 的 75 页课件。录播时间取自课堂字幕，可用来定位相应讲解。代码运行环境使用 NumPy 2.1.3 和 pandas 2.2.3。

- [[03_Computer_Science/06_ST445_Managing_and_Visualising_Data/2026_Course_Materials/02_Lecture_Notes/Week_01/ST445 - Week 01 - Lecture.pdf|ST445 Week 1 lecture slides]]
- [ST445 Lecture recording, 29 September 2026](https://echo360.org.uk/media/4ce70ea5-3820-4059-8257-0c1dd336c212/public)
- [Python data structures](https://docs.python.org/3/tutorial/datastructures.html)
- [Python Unicode HOWTO](https://docs.python.org/3/howto/unicode.html)

<!-- studyroom-ink:bb0cfebe02b673093b4484116aa9f3f20361984a9f7513a0d2e0725a92a3143f -->
<!-- bilingual-en:start -->
## Additional explanation

Consistent fields and types usually make data easier to prepare: the same parsing rules can be applied across records.

A tuple permits repeated values: `(2, 2)` has two elements.

In `pd.read_csv`, `keep_default_na=False` does not disable type inference: a numeric-looking identifier can still lose its leading zeros. With missing-value parsing enabled, `keep_default_na=True` recognises both pandas’ default markers and any explicitly supplied `na_values`. The resulting missing value depends on the column type: a nullable `string` column uses `pd.NA`, not floating-point `NaN`.
<!-- bilingual-en:end -->

### 补充讲解

字段及其类型越一致，数据通常越容易准备，因为不同记录可以共用同一套解析规则。

元组允许重复值：`(2, 2)` 有两个元素。

在 `pd.read_csv` 中，`keep_default_na=False` 不会关闭类型推断：形似数字的编号仍可能丢失前导零。启用缺失值解析时，`keep_default_na=True` 会同时识别 pandas 的默认标记和通过 `na_values` 显式指定的标记。缺失值的表示取决于列类型：可空的 `string` 列使用 `pd.NA`，而非浮点值 `NaN`。
