---
aliases:
  - Fisher价格指数是同一比较下拉氏与派氏价格指数的几何平均
  - Fisher ideal price index
student_os: knowledge-atom
atom_id: FI-MKT-074
atom_type: definition
status: source-checked
---

# Fisher价格指数是同一比较下拉氏与派氏价格指数的几何平均
<!-- bilingual-en:start -->
*The Fisher price index is the geometric mean of the Laspeyres and Paasche price indices for the same comparison.*
<!-- bilingual-en:end -->

Fisher 指数兼顾“旧数量篮子”和“新数量篮子”两个视角。先对同一组对象、同两期价格计算[[拉氏价格指数|拉氏]]和[[派氏价格指数|派氏]]，再开平方；两指数必须都为正且采用同一基值：
<!-- bilingual-en:start -->
Fisher combines the old-basket and current-basket perspectives. Calculate [[拉氏价格指数|Laspeyres]] and [[派氏价格指数|Paasche]] for the same objects and price periods, then take their geometric mean. Both must be positive and use the same base value.
<!-- bilingual-en:end -->

$$I_F=\sqrt{I_LI_P}.$$

两股价格从 10、20 变为 12、18；旧数量为 2、1，新数量为 1、3。$I_L=105$、$I_P=660/7\approx94.286$，于是 $I_F=\sqrt{9900}\approx99.499$。若先算不乘 100 的价格比，最后再乘 100；若两指数已以 100 为基值，开方后不可再乘 100。
<!-- bilingual-en:start -->
For prices moving from 10 and 20 to 12 and 18, old quantities 2 and 1 and current quantities 1 and 3 give $I_L=105$ and $I_P=660/7$. Thus $I_F=\sqrt{9900}\approx99.499$. Apply the base scaling once: do not multiply by 100 again when both inputs already have base 100.
<!-- bilingual-en:end -->

课堂 PDF p.235 标作“加权几何平均法”的式子实际就是这一结构。它平均的是两个汇总指数，不是[[价格相对数几何平均|各股票的价格比]]，后者在这个例子中为约 103.923。判断方法应看公式中的对象，不能只看“几何平均”四个字。
<!-- bilingual-en:start -->
The formula labelled “weighted geometric averaging” on course PDF p.235 has this structure. Its inputs are two aggregate indices, not [[价格相对数几何平均|individual stock-price relatives]], whose geometric index here is about 103.923. Identify the operands, not just the word “geometric”.
<!-- bilingual-en:end -->

## 来源与核验

- [[02_Economy/06_证券投资学/证券投资学.pdf#page=235|证券投资课程 PDF p.235]]：已逐式目视辨认拉氏、派氏乘积根式。
- [IMF, An Introduction to PPI Methodology](https://www.imf.org/external/np/sta/tegppi/ch1.pdf)，§1.51，PDF p.15 / printed p.11：Fisher 名称和公式；缩放及算例重算。
<!-- bilingual-en:start -->
- The course formula and IMF definition are checked visually. Scaling and example arithmetic are independently recomputed.
<!-- bilingual-en:end -->
