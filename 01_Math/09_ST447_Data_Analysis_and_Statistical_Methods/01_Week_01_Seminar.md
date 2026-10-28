<!-- bilingual-en:start -->
# Week 1 Seminar — Descriptive Data Analysis in R
<!-- bilingual-en:end -->

第一周小班课：R 与描述性数据分析

<!-- bilingual-en:start -->
This reading follows Chapter 1 (all 29 pages) and the Week 1 R script. Its technical workflow is **create objects → select and transform values → import data → describe groups → plot distributions → preserve the analysis**. The original files remain the reference for classroom scope. The explanations below clarify behaviour where the slides simplify R or the script comments disagree with its executable lines.
<!-- bilingual-en:end -->

本页沿 Chapter 1 全 29 页及第一周 R 脚本整理技术内容：**建立对象 → 选择与变换 → 导入数据 → 描述组别 → 查看分布 → 保存分析**。课堂范围以原文件为准；课件对 R 的简写，以及脚本注释与可执行行不一致之处，在相应段落澄清。

## 1. R, the editor and the Console

R、编辑器与控制台

<!-- bilingual-en:start -->
R performs the calculations; an editor provides a place to write and run code. In the course’s RStudio examples, Source edits scripts, Console executes commands, Environment shows session objects, and Files, Plots, Help and History provide their corresponding views. The base R commands also run from an R session in another editor. Opening a script and executing its contents are different actions.
<!-- bilingual-en:end -->

R 负责计算，编辑器提供编写和运行代码的界面。课程以 RStudio 为例：Source 编辑脚本，Console 执行命令，Environment 显示会话对象，其他面板分别处理文件、图形、帮助和历史。同样的 base R 命令也能在其他编辑器连接的 R 会话里运行。打开脚本不等于执行脚本。

```r
sqrt(9) / 3 - 1       # 0
# A comment starts with # outside a string.
# The Console prompt > is not part of the code to copy.
```
<!-- bilingual-en:start -->
A function receives arguments and returns a result. In `sqrt(9)`, the argument is 9 and the result is 3. A named argument such as `by = 2` selects a setting. A semicolon separates commands on one line. R and contributed packages are installed separately from the editor; package installation and loading into a session are also separate steps.
<!-- bilingual-en:end -->

函数接收参数并返回结果；`sqrt(9)` 的参数是 9，结果是 3。`by = 2` 这样的命名参数指定设置。分号用于在一行内分开命令。R、扩展包与编辑器分别安装；安装包和在会话中载入包也是两步。

```r
# install.packages("ggplot2")  # install when needed
# library(ggplot2)              # attach it in the current session
# ?mean
# help(mean)
# help(package = "ggplot2")
```
<!-- bilingual-en:start -->
Use the help page’s Arguments and Details to check an unfamiliar call. The examples in this note require only base R and its standard packages.
<!-- bilingual-en:end -->

查看帮助页的 Arguments 和 Details 可以确认陌生命令的含义。本页示例仅使用 base R 及随附的标准包。

## 2. Objects, values and vectors

对象、值与向量

<!-- bilingual-en:start -->
First distinguish storing a result from testing it. This prevents an apparent equality sign from being read as a permanent mathematical equation.
<!-- bilingual-en:end -->

先区分“保存结果”和“比较结果”，避免把程序赋值读成永久成立的数学等式。

![[R 赋值与相等比较]]

![[R 数值与字符字面量]]
<!-- bilingual-en:start -->
The vector carries this reasoning from one value to a collection. An element’s type comes from the object that contains it, not just from how the printed value looks.
<!-- bilingual-en:end -->

向量把单个值的规则扩展到一组值。元素类型取决于包含它的对象，不能只凭打印出来的外观判断。

![[R 原子向量]]
<!-- bilingual-en:start -->
The script constructs sequences and repeated patterns before selecting elements. `:` uses a step of 1 or −1; `seq()` can specify another step. With a positive step, the sequence stops before exceeding the endpoint. In `rep()`, a scalar `times` repeats the whole vector; `each` repeats each element. A vector of counts supplies one repetition count per element, and `length.out` fixes the final length.
<!-- bilingual-en:end -->

脚本先构造序列和重复模式，再选择元素。`:` 的步长为 1 或 −1；`seq()` 可以指定其他步长。正步长序列不会超过终点。`rep()` 中，单个 `times` 值重复整条向量，`each` 重复各元素；计数向量分别指定各元素的重复次数，`length.out` 决定最终长度。

```r
numbers <- c(1, 2, 3)
people <- c("Ali", "Bet", "Cat")
1:5                              # 1 2 3 4 5
seq(1, 9, by = 2)                # 1 3 5 7 9
seq(1, 10, by = 2)               # 1 3 5 7 9
rep(2, 5)                        # 2 2 2 2 2
rep(numbers, times = 2)           # 1 2 3 1 2 3
rep(numbers, each = 2)            # 1 1 2 2 3 3
rep(numbers, c(2, 1, 2))          # 1 1 2 3 3
rep(numbers, times = 3, length.out = 7) # 1 2 3 1 2 3 1
```

## 3. Select, transform and summarize

选择、变换与汇总

<!-- bilingual-en:start -->
Selection asks which entries to keep. An index can specify positions directly or express a condition on each entry. Keep that distinction separate from the values returned.
<!-- bilingual-en:end -->

选择操作回答“保留哪些元素”。索引可以直接给位置，也可以为每个元素给出条件；索引和返回的值需要分清。

![[R 向量位置索引]]

![[R 逻辑筛选]]

![[R 向量回收]]
<!-- bilingual-en:start -->
Element-wise operations produce values for each position; aggregation combines them. Operation order matters: squaring before summing is a different calculation from squaring the sum. `log()` without a base argument is the natural logarithm.
<!-- bilingual-en:end -->

逐元素运算为各位置产生值，汇总则将这些值合并。顺序很重要：先平方后求和，与先求和后平方不同。不指定底数的 `log()` 是自然对数。

```r
numbers <- c(1, 2, 3)
numbers^2              # 1 4 9
sum(numbers^2)         # 14
sum(numbers)^2         # 36
log(numbers)
x <- 1:100
sum(x)                 # 5050
y <- x^2
y[14]                  # 196
```

## 4. Bring files into the analysis

把文件读入分析

<!-- bilingual-en:start -->
Relative paths are resolved from R’s working directory, which `getwd()` reports and `setwd()` changes. Opening a script does not necessarily set that directory. The examples below use the verified course data folder explicitly so imports do not depend on the current editor folder.
<!-- bilingual-en:end -->

相对路径以 R 的工作目录为基准；`getwd()` 查看，`setwd()` 修改。打开脚本不一定改变工作目录。下面直接使用已核对的课程数据目录，避免导入依赖编辑器当前目录。

```r
data_dir <- "/Users/fengyihang/Library/Mobile Documents/iCloud~md~obsidian/Documents/Academic/01_Math/09_ST447_Data_Analysis_and_Statistical_Methods/2026_Course_Materials/06_Datasets"
getwd()
x <- scan(file.path(data_dir, "simpleData.txt"), skip = 2)
length(x)              # 12
mean(x); range(x)      # 6.5; 1 12
summary(x)             # 1 3.75 6.5 6.5 9.25 12
d <- read.table(file.path(data_dir, "simpleData.txt"), skip = 2)
dim(d)                 # 3 4
d[, 2]                 # 2 6 10
d[2, 4]                # 8
```
<!-- bilingual-en:start -->
The file has two explanatory lines followed by three rows of four numbers. `skip = 2` skips the explanations. Here `scan()` reads a flat numeric vector; `read.table()` returns a data frame. Chapter 1 calls the latter a matrix and prints matrix-style output, but the actual return class is `data.frame`.
<!-- bilingual-en:end -->

文件先有两行说明，再有三行、每行四个数字。`skip = 2` 跳过说明；此处 `scan()` 读成扁平数值向量，`read.table()` 返回数据框。Chapter 1 把后者称为矩阵，并使用矩阵式输出，但实际返回类是 `data.frame`。

![[R 数据框列选择]]

## 5. Give the Jobs columns their intended meaning

让 Jobs 各列具有正确含义

<!-- bilingual-en:start -->
Jobs contains 253 questionnaire returns from business-school graduates. The file’s six columns distinguish an identifier, categories and numerical measurements. Salary is measured in **thousands of pounds**; Search counts jobs after graduation.
<!-- bilingual-en:end -->

Jobs 包含商学院毕业生的 253 份问卷回复。原文件六列中，标识符、类别和数值测量各有用途。Salary 单位为**千英镑**，Search 是毕业后的工作份数。

| File column | Meaning / 含义 |
|---|---|
| IDNo. | Graduate identifier / 毕业生编号 |
| JobType | 1 Accounting; 2 Finance; 3 Management; 4 Marketing and sales; 5 Other / 会计、金融、管理、市场与销售、其他 |
| Sex | 1 Male; 2 Female, as coded in this dataset / 数据原有编码：男、女 |
| Satisfaction | Documented: 1 Very satisfied; 2 Satisfied; 3 Not satisfied. Code 4 also occurs, with meaning unspecified / 课件解释 1–3；数据另有含义未说明的编码 4 |
| Salary | Salary in £000 / 工资，千英镑 |
| Search | Number of jobs after graduation / 毕业后的工作份数 |

![[R 导入表头与行名]]

```r
jobs_path <- file.path(data_dir, "Jobs.txt")
dim(read.table(jobs_path))                  # 254 6
dim(read.table(jobs_path, header = TRUE))   # 253 6
jobs <- read.table(jobs_path, header = TRUE, row.names = 1,
  colClasses = c("integer", "factor", "factor", "factor", "numeric", "integer"))
dim(jobs)                                  # 253 5
names(jobs)
sapply(jobs, class)
jobs$Salary[1:4]                            # 51 38 51 52
table(jobs$Satisfaction)                    # codes 1, 2, 3, 4: counts 81, 102, 60, 10
```
<!-- bilingual-en:start -->
With no header setting, the column-name line is read as an extra observation. With the header recognized and the ID moved to row names, Salary is column 4 in the data frame, although it was column 5 in the file. The six explicit classes align with the original six input columns. Named access such as `jobs$Salary` expresses the intended variable more clearly than a bare column number.
<!-- bilingual-en:end -->

未声明表头时，列名行被当成多出的一条观测。识别表头、将 ID 转为行名后，Salary 从文件第 5 列变成数据框第 4 列。这里六个明确指定的类型逐一对应原文件六列。`jobs$Salary` 这样的名称访问比裸列号更清楚。

![[R 因子与水平]]
<!-- bilingual-en:start -->
The first three data columns are imported as factors. Check the data against the codebook: Chapter 1 p.13 documents Satisfaction codes 1–3, but the supplied file also contains **10 observations with code 4**. Its meaning is not specified in that source. The documented three categories have an order; the position or meaning of code 4 cannot be inferred from its number. Keep that discrepancy explicit before labelling or ordering the full variable. This import creates an unordered factor. `summary(jobs$JobType)` reports level counts, whereas a numeric `summary()` reports minimum, quartiles, median, mean and maximum. A numeric summary includes the mean and is not merely a five-number summary.
<!-- bilingual-en:end -->

前三个数据列按 factor 导入。需要对照编码说明与实际数据：Chapter 1 第 13 页只解释满意度编码 1–3，提供的数据却另有 **10 条编码为 4 的观测**；该来源未说明其含义。已解释的三档有顺序，但不能凭数字给编码 4 推定含义或位置。为整个变量命名或排序前，应保留这处未解决的差异。这次导入得到无序 factor。`summary(jobs$JobType)` 给出水平计数；数值型 `summary()` 则给出最小值、四分位数、中位数、均值和最大值。它包含均值，不能简称为只含五数的概括。

## 6. Count categories and state the reference group

计数类别，并说明比例的参照群体


```r
table(jobs$JobType)
tt <- table(jobs$Sex, jobs$JobType)
tt
```

| Sex / 性别 | Accounting | Finance | Management | Marketing/sales | Other | Total / 合计 |
|---|---:|---:|---:|---:|---:|---:|
| Male / 男 | 35 | 30 | 18 | 39 | 13 | 135 |
| Female / 女 | 38 | 22 | 18 | 25 | 15 | 118 |
| Total / 合计 | 73 | 52 | 36 | 64 | 28 | 253 |
<!-- bilingual-en:start -->
`table()` counts categories. In `tt`, sex is the row variable and job type is the column variable. The two group sizes differ, so raw counts and within-group distributions answer different questions.
<!-- bilingual-en:end -->

`table()` 统计类别频数。`tt` 以性别为行、工作类别为列。两组人数不同，因此原始计数与组内分布回答不同的问题。

![[R 列联表条件比例]]

```r
100 * tt[1, ] / sum(tt[1, ])
# 25.92593 22.22222 13.33333 28.88889 9.62963
100 * tt[2, ] / sum(tt[2, ])
# 32.20339 18.64407 15.25424 21.18644 12.71186
100 * prop.table(tt, 1)[1, ]     # same as the first percentage row
```
<!-- bilingual-en:start -->
For example, 35/135 is the accounting share among male respondents, whereas 35/73 is the male share among accounting respondents. The numerator is the same, but the reference group changes. These describe the questionnaire sample; a table alone does not establish a causal effect.
<!-- bilingual-en:end -->

例如，35/135 是男性受访者中会计工作的比例，35/73 是会计工作受访者中男性的比例。分子相同，参照群体不同。这些结果描述问卷样本，不能单凭一张表得出因果结论。

## 7. Plot category counts and shares

画出类别计数与份额


![[R 矩阵条形图]]

```r
job_labels <- c("Accounting", "Finance", "Management", "Marketing/sales", "Other")
barplot(tt, names.arg = job_labels, legend.text = c("Male", "Female"),
        col = c("blue", "red"), main = "Graduates by job category")
pie(table(jobs$JobType), labels = job_labels)
text(0, 1, "Total", cex = 2)
pie(tt[1, ], labels = job_labels)
pie(tt[2, ], labels = job_labels)
```
<!-- bilingual-en:start -->
The bar chart has five stacked bars, one per job category. Male counts form the lower segments and female counts the upper segments. The explicit colours above reproduce the slide’s colour mapping; the original script omits `col`. Pie charts display shares of a total. `text(0, 1, ...)` places a label at those plot coordinates; `cex = 2` increases its size. Prefer a common-baseline bar chart when you need to judge small differences between categories.
<!-- bilingual-en:end -->

条形图有五根堆叠柱，每根对应一类工作；下段为男性计数，上段为女性计数。上面明确指定颜色，以对应课件配色；原脚本未指定 `col`。饼图显示总体份额。`text(0, 1, ...)` 在图上该坐标放置文字，`cex = 2` 放大文字。需要判断类别间细小差异时，使用有共同基线的条形图更直接。

![[01_Math/09_ST447_Data_Analysis_and_Statistical_Methods/2026_Course_Materials/02_Lecture_Notes/ST447 - Chapter 01 - Descriptive Data Analysis in R.pdf#page=18]]

## 8. Describe numerical distributions

描述数值分布


```r
mSalary <- jobs$Salary[jobs$Sex == "1"]
fSalary <- jobs$Salary[jobs$Sex == "2"]
summary(jobs$Salary)
summary(mSalary)
summary(fSalary)
```

| Group / 组别 | Min | Q1 | Median | Mean | Q3 | Max |
|---|---:|---:|---:|---:|---:|---:|
| All / 全部 | 26 | 43 | 47 | 47.13 | 52 | 65 |
| Male / 男 | 34 | 44 | 48 | 48.11 | 53 | 65 |
| Female / 女 | 26 | 42.25 | 46 | 46.00 | 51 | 61 |
<!-- bilingual-en:start -->
The median describes the middle of the ordered sample; Q1 and Q3 mark its lower and upper quartiles. The mean adds values and divides by their count. Here 48.11 is approximately £48,110. To compare distributions, inspect **centre, spread, shape and unusual values**. A mean difference alone neither describes the whole shape nor explains its cause.
<!-- bilingual-en:end -->

中位数描述排序样本的中间位置，Q1、Q3 是下、上四分位数；均值是总和除以个数。这里 48.11 约为 48,110 英镑。比较分布需要同时看**中心、离散程度、形状和异常值**；均值差既不能概括整个形状，也不能解释原因。

```r
op <- par(mfrow = c(3, 1))
hist(jobs$Salary, col = "gray", nclass = 15, xlim = c(25, 66), main = "All graduates")
hist(mSalary, col = "blue", nclass = 15, xlim = c(25, 66), main = "Male graduates")
hist(fSalary, col = "red", nclass = 15, xlim = c(25, 66), main = "Female graduates")
par(op)
```
<!-- bilingual-en:start -->
These are the course calls, arranged vertically with `par(mfrow = c(3, 1))`. On the checked local R run, the suggested 15 classes produced 20, 16 and 18 bins respectively. Their bin widths happen to agree, but their plotted vertical scales differ. Equal-looking bar heights across panels need not represent equal counts.
<!-- bilingual-en:end -->

这是课程中的调用，用 `par(mfrow = c(3, 1))` 纵向排列。本次本地 R 核对中，建议的 15 箱实际产生 20、16、18 箱。三者箱宽碰巧相同，但图上的纵轴尺度不同；不同面板中看起来同高的柱子，未必对应同样人数。

![[01_Math/09_ST447_Data_Analysis_and_Statistical_Methods/2026_Course_Materials/02_Lecture_Notes/ST447 - Chapter 01 - Descriptive Data Analysis in R.pdf#page=22]]

![[R 直方图分组比较]]
<!-- bilingual-en:start -->
For a deliberate comparison of the two group shapes, the following uses common breaks, density scaling and a common vertical range. This is a clarified comparison of the course data, not an additional classroom result.
<!-- bilingual-en:end -->

要有意比较两组形状，下例使用共同断点、密度尺度和共同纵轴范围。这是对课程数据的补充比较方法，不冒充额外课堂结果。

```r
common_breaks <- seq(25, 67, by = 2)
hm <- hist(mSalary, breaks = common_breaks, plot = FALSE)
hf <- hist(fSalary, breaks = common_breaks, plot = FALSE)
ymax <- max(hm$density, hf$density)
op <- par(mfrow = c(2, 1))
plot(hm, freq = FALSE, ylim = c(0, ymax), col = "blue", main = "Male", xlab = "Salary (£000)")
plot(hf, freq = FALSE, ylim = c(0, ymax), col = "red", main = "Female", xlab = "Salary (£000)")
par(op)
stem(jobs$Salary)
```
<!-- bilingual-en:start -->
A stem-and-leaf display gives more detail about individual values than histogram bins. Read the decimal-point key printed **above** the stems. The scale and rounding can limit precision, so do not assume every original decimal digit is retained. Filtering can combine conditions: `jobs$Salary[jobs$Sex == "1" & jobs$Salary > 50]` selects male respondents’ salaries above £50,000.
<!-- bilingual-en:end -->

茎叶图比直方图分箱保留更多个体取值细节。先读茎列**上方**打印的小数点说明；尺度与舍入会限制精度，不能认为原始每位小数都会保留。筛选可以组合条件：`jobs$Salary[jobs$Sex == "1" & jobs$Salary > 50]` 选择男性受访者中超过 50,000 英镑的工资。

## 9. Export values without changing their meaning

导出时保留数值含义

<!-- bilingual-en:start -->
The export demonstration below writes to a temporary folder, displayed by `out_dir`. Use a chosen persistent output directory when you intend to keep the files. `write()` arranges a vector into columns. `write.table()` controls separators, names and quoting independently.
<!-- bilingual-en:end -->

下面的导出演示写入临时目录，可用 `out_dir` 查看位置；要长期保留时，改用选定的持久输出目录。`write()` 把向量排成列；`write.table()` 分别控制分隔符、名称和引号。

```r
out_dir <- file.path(tempdir(), "st447-week1-export")
dir.create(out_dir, showWarnings = FALSE)
write(1:10, file.path(out_dir, "x.txt"), ncolumns = 2)
write.table(jobs, file.path(out_dir, "Jobs1.txt"))
write.table(jobs, file.path(out_dir, "Jobs2.txt"), row.names = FALSE, col.names = FALSE)
write.table(jobs, file.path(out_dir, "Jobs3.txt"), sep = ",")
```
<!-- bilingual-en:start -->
The first table keeps row and column names and uses spaces; the second omits both; the third uses commas. Factor labels are normally written as quoted text. Removing quotes and converting categories to numbers are different operations.
<!-- bilingual-en:end -->

第一份保留行列名称并使用空格，第二份去掉两类名称，第三份使用逗号。factor 标签通常作为带引号的文本写出。去掉引号与把类别转为数值是不同操作。

![[R 因子标签转数值]]
<!-- bilingual-en:start -->
For these numeric category labels, convert through their character labels if numerical columns are intended. The original script’s direct `as.numeric()` happens to give the same values here because labels and codes align. Alternatively, preserve category labels and include the identifier explicitly.
<!-- bilingual-en:end -->

这些类别标签本身是数字；若确实要数值列，可经字符标签转换。原脚本直接 `as.numeric()` 在这里碰巧得到相同值，因为标签和编码一致。另一做法是保留类别标签，并明确加入编号。

```r
numeric_jobs <- data.frame(
  JobType = as.numeric(as.character(jobs$JobType)),
  Sex = as.numeric(as.character(jobs$Sex)),
  Satisfaction = as.numeric(as.character(jobs$Satisfaction)),
  Salary = jobs$Salary, Search = jobs$Search)
write.table(numeric_jobs, file.path(out_dir, "Jobs4.txt"))
jobsExport <- data.frame(IDNo. = row.names(jobs), jobs)
write.table(jobsExport, file.path(out_dir, "Jobs5.txt"), row.names = FALSE, quote = FALSE)
```

## 10. Preserve a reproducible analysis

保存可复现的分析过程

![[R 工作区历史与脚本]]
<!-- bilingual-en:start -->
Collect the intended imports, transformations, summaries and plots in a saved analysis script. Supply required package loading and data paths explicitly. Save the script before executing it. In RStudio, History → To Source transfers selected history entries for editing; Run executes selected lines. In another editor, use its existing R execution route. `q()` exits the R session.
<!-- bilingual-en:end -->

将预期的导入、变换、摘要和绘图整理为分析脚本，明确写出所需包和数据路径，保存后再执行。RStudio 可用 History → To Source 把历史命令送入编辑区，用 Run 执行所选行；其他编辑器沿用其已有 R 执行方式。`q()` 退出 R 会话。

```r
# After saving the intended analysis as jobsAnalysis.R:
# source("jobsAnalysis.R", echo = TRUE)
# Save the current global objects only when wanted:
# save.image(file.path(out_dir, "week1.RData"))
# In an interactive front end that supports it:
# savehistory(file.path(out_dir, "week1.Rhistory"))
```
<!-- bilingual-en:start -->
In the supplied `week1 (3).R`, `write()` (line 394), five `write.table()` calls (398, 401, 404, 419, 425), `save.image()` (437) and `savehistory()` (441) are active. The nearby claim that the saving demonstrations are commented out does not match those lines. Commenting only the two save calls still leaves six text-file exports active. The two `source()` examples and `q()` are commented out. Execute the sections you intend, or adapt a copy with explicit output paths; do not infer side effects from that comment.
<!-- bilingual-en:end -->

原 `week1 (3).R` 中，`write()`（394 行）、五个 `write.table()`（398、401、404、419、425 行）、`save.image()`（437 行）和 `savehistory()`（441 行）均为活动代码。旁边“保存示例已注释”的说明与代码不符。仅注释两个 save 调用，仍会留下六个文本文件导出。两处 `source()` 和 `q()` 已注释。应执行需要的段落，或在副本里明确安排输出路径，不能根据那条注释判断副作用。

## 11. Consolidated check

集中检查

<!-- bilingual-en:start -->
Use the course data and provide code together with brief interpretations. The first two tasks follow the Chapter 1 quizzes; the count, mean and denominator checks make the result interpretable.
<!-- bilingual-en:end -->

使用课程数据，提交代码及简短解释。前两项承接 Chapter 1 的 quiz；数量、均值和分母检查使结果具有明确含义。
<!-- bilingual-en:start -->
**1. Original Quiz 1:** Extract Accounting and Finance salaries. Compare them using `summary()` and `stem()`. Describe their centres, spreads and shapes.

**2. Original Quiz 2:** Extract salaries for **male graduates in Finance**. Also report how many observations you selected and their mean salary.

**3.** Explain what each of these describes: `tt[1, 2] / sum(tt[1, ])` and `tt[1, 2] / sum(tt[, 2])`.
<!-- bilingual-en:end -->

**1. 原 Quiz 1：** 提取会计与金融类别工资，用 `summary()` 和 `stem()` 比较中心、离散程度及形状。

**2. 原 Quiz 2：** 提取金融类别中男性毕业生的工资，并报告所选观测数量与工资均值。

**3.** 分别说明 `tt[1, 2] / sum(tt[1, ])` 与 `tt[1, 2] / sum(tt[, 2])` 描述什么。

## Sources and verification

来源与核验

- [[01_Math/09_ST447_Data_Analysis_and_Statistical_Methods/2026_Course_Materials/02_Lecture_Notes/ST447 - Chapter 01 - Descriptive Data Analysis in R.pdf|Chapter 1, pp. 1–29]]: course sequence, examples, charts and quizzes / 课堂顺序、示例、图形与习题。
- [[01_Math/09_ST447_Data_Analysis_and_Statistical_Methods/2026_Course_Materials/03_Seminars/Week_01/week1 (3).R|Week 1 R script]]: the detailed teaching examples and executable export/save lines / 详细例子及实际导出、保存代码。
- [[01_Math/09_ST447_Data_Analysis_and_Statistical_Methods/2026_Course_Materials/06_Datasets/Jobs.txt|Jobs.txt]] and [[01_Math/09_ST447_Data_Analysis_and_Statistical_Methods/2026_Course_Materials/06_Datasets/simpleData.txt|simpleData.txt]]: numerical examples / 数值核对数据。
- [R: histograms](https://stat.ethz.ch/R-manual/R-patched/library/graphics/html/hist.html), [stem-and-leaf plots](https://stat.ethz.ch/R-manual/R-patched/library/graphics/html/stem.html), [data output](https://stat.ethz.ch/R-manual/R-patched/library/utils/html/write.table.html): plotting and export boundaries / 绘图与导出边界。
<!-- bilingual-en:start -->
The numerical results and example behaviour were checked against the archived files with R 4.6.1. The original PDF charts on pages 18 and 22 were inspected. Each embedded atom cites the source for its general rule. Source verification concerns the content; independent performance is assessed through the consolidated check.
<!-- bilingual-en:end -->

数值结果与示例行为用 R 4.6.1 对照归档文件核对；原 PDF 第 18、22 页图形已查看。各嵌入原子列出通用规则来源。内容核验与独立操作表现分别判断，后者由集中检查体现。
