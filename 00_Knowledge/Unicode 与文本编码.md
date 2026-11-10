---
student_os: knowledge-atom
atom_id: CS-TEXT-CODEPOINT-ENCODING-001
aliases:
  - Unicode code points and byte encodings
  - Code points versus bytes
  - Unicode、UTF-8 与 Latin-1
status: source-checked
atom_type: distinction
---

<!-- bilingual-en:start -->
# Code points identify text; encodings represent it as bytes
<!-- bilingual-en:end -->

码点标识文字，编码规定怎样存成字节。

<!-- bilingual-en:start -->
Unicode assigns code points to characters. An encoding determines how text is represented by bytes for storage or transfer. These are different layers: the character `é` has code point U+00E9, which is decimal 233, while UTF-8 and Latin-1 encode it differently.
<!-- bilingual-en:end -->

Unicode 给字符分配码点；编码决定文字在保存或传输时变成哪些字节。这是两个层次：`é` 的码点为 U+00E9，十进制是 233，但 UTF-8 与 Latin-1 会把它存成不同的字节。

```python
assert ord("é") == 233
assert "é".encode("utf-8").hex(" ") == "c3 a9"
assert "é".encode("latin-1").hex(" ") == "e9"
assert len("Café") == 4
assert len("Café".encode("utf-8")) == 5
```

<!-- bilingual-en:start -->
In Python, `str` stores Unicode text and `bytes` stores byte values. `.encode(...)` goes from text to bytes; `.decode(...)` interprets bytes using an encoding. Latin-1 covers only a limited character range; it cannot encode every Unicode character. `len(str)` counts code points, which need not match the number of visible symbols: a base letter and a combining accent may be separate code points.
<!-- bilingual-en:end -->

Python 的 `str` 表示 Unicode 文字，`bytes` 表示字节。`.encode(...)` 把文字变成字节，`.decode(...)` 按指定编码解释字节。Latin-1 只能表示有限范围的字符，不能编码全部 Unicode 字符。`len(str)` 数的是码点，也未必等于眼睛看到的字数：一个字母加一个组合重音，可能占两个码点。

<!-- bilingual-en:start -->
Using the wrong decoder can produce plausible-looking but incorrect text without an exception. UTF-8 bytes for `Café`, decoded as Latin-1, become `CafÃ©`. A join on the station name then fails even if the source files have the same row counts. Confirm the source encoding instead of guessing from successful decoding.
<!-- bilingual-en:end -->

解码规则用错时，程序未必报错。把 `Café` 的 UTF-8 字节按 Latin-1 读，会得到 `CafÃ©`。即使两个文件的行数相同，按站名连接也会失败。程序能解码，不代表规则正确；要确认来源实际采用的编码。

## 来源与核验

- [Python Unicode HOWTO](https://docs.python.org/3/howto/unicode.html): code points, encodings, `str`/`bytes`, and combining characters / 码点、编码、字符串与字节、组合字符。
- [[ST445 - Week 01 - Lecture.pdf#page=40|ST445 Lecture 1, pp. 40, 50]]: character/byte examples and encoding standards / 课程中的字符、字节例子及编码标准。
