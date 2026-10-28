---
student_os: knowledge-atom
atom_id: r-base-histogram-group-comparison
aliases:
  - "Group histograms need shared breaks and a clearly chosen vertical scale"
  - "比较分组直方图需要统一分箱，并明确纵轴尺度"
status: source-checked
---
<!-- bilingual-en:start -->
# Group histograms need shared breaks and a clearly chosen vertical scale
<!-- bilingual-en:end -->

比较分组直方图需要统一分箱，并明确纵轴尺度

<!-- bilingual-en:start -->
A histogram groups numeric values into intervals. To compare distributions, choose common breakpoints and state whether heights show counts or densities. With density scaling, rectangle areas represent proportions and total area is one; this removes the direct effect of unequal sample counts on total area.
<!-- bilingual-en:end -->

直方图把数值分入区间。比较分布时应使用共同断点，并说明柱高表示计数还是密度。密度尺度下，矩形面积表示比例、总面积为 1，从而消除样本量不同对总面积的直接影响。

```r
a <- c(1, 2, 2, 3)
b <- c(1, 1, 2, 3, 3, 4)
breaks <- 0:4
ha <- hist(a, breaks = breaks, plot = FALSE)
hb <- hist(b, breaks = breaks, plot = FALSE)
sum(ha$density * diff(ha$breaks)) # 1
sum(hb$density * diff(hb$breaks)) # 1
```
<!-- bilingual-en:start -->
A numeric bin count, including the course’s `nclass = 15`, is a suggestion; R can choose different breakpoints. `xlim` changes the display range, not the bins. Use an explicit `breaks` vector and inspect axes when comparability matters.
<!-- bilingual-en:end -->

包括课程 `nclass = 15` 在内的数值箱数只是建议，R 可能另选断点。`xlim` 改变显示范围，不决定分箱。需要可比性时，应明确给出 `breaks` 并检查坐标轴。

## 来源与核验

- [[01_Math/09_ST447_Data_Analysis_and_Statistical_Methods/2026_Course_Materials/02_Lecture_Notes/ST447 - Chapter 01 - Descriptive Data Analysis in R.pdf#page=22|Chapter 1, pp. 21–22]], salary histograms / 工资直方图。
- [R: histograms](https://stat.ethz.ch/R-manual/R-patched/library/graphics/html/hist.html): breaks and density / 断点与密度。
