"""Rebuild the four tutorial notebooks from reviewed, versioned source cells."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parent

def md(text): return nbf.v4.new_markdown_cell(text.strip())
def code(text): return nbf.v4.new_code_cell(text.strip())
SETUP='''
from pathlib import Path
import sys
candidates = [Path.cwd(), Path.cwd().parent, Path.cwd() / "python"]
source_dir = next((p for p in candidates if (p / "lab_reference.py").is_file()), None)
if source_dir is None:
    raise FileNotFoundError("从 python/notebooks 或 python 目录运行此 Notebook")
sys.path.insert(0, str(source_dir))
import numpy as np
from scipy.stats import norm, beta as beta_distribution
from numpy.polynomial.legendre import legvander
import matplotlib.pyplot as plt
import lab_reference as lab
%matplotlib inline
plt.rcParams.update({"figure.figsize": (9, 4.4), "figure.dpi": 125,
                     "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .18, "axes.axisbelow": True})
BLUE, ORANGE, OLIVE, INK = "#24658c", "#bc6327", "#69743a", "#24323c"
SEED = 42
print("NumPy", np.__version__, "· seed", SEED, "· independent PCG64 draws")
'''
intro='''
## 准备 · Setup

先在网页预测结果、拖动参数，再回到这里复算。代码只使用合成数据，不读取学习记录。
所有随机数使用 NumPy PCG64；固定种子可重现本 Notebook，**不意味着与网页的随机样本逐项一致**。
数学定义、参数和边界与网页一致。独立核验脚本另外直接读取网页原始样本，再用 NumPy/SciPy 复算。

运行环境：Python 3.12+，依赖见 `../requirements.txt`。从顶部顺序运行全部单元格；图表标签用英语，讲解用中文。
'''

books=[]
books.append(('01_geometry_optimization',[
md(r'''# 01 · 从局部变化到向量几何与优化

## 目标 · Goal

用四个实验连接导数、线性变换、投影和梯度下降。每一步先写预测，再运行，再用等式核对图像。
需要的准备：坐标、向量点积、矩阵乘法；不要求预先掌握优化。
'''),md(intro),code(SETUP),
md(r'''## 实验 1 · 切线和有限差分

对 $f(x)=x^3-x$，在 $x=1$ 的精确导数为 $2$。前向差分误差为 $3xh+h^2$，中心差分误差为 $h^2$。
先预测：将 $h$ 减半，哪一种误差缩小更快？这里讨论截断误差；极小步长还会受到浮点消减影响。
'''),code('''
x, h = 1., .5
result = lab.compute("tangent", x=x, h=h)
f = lambda t: t**3 - t
xx = np.linspace(-.2, 2.1, 240)
steps = np.logspace(-5, 0, 80)
forward = np.array([abs(lab.compute("tangent", x=x, h=t)["forward"]-2) for t in steps])
central = np.array([abs(lab.compute("tangent", x=x, h=t)["central"]-2) for t in steps])
fig, ax = plt.subplots(1, 2, figsize=(11,4.3), layout="constrained")
ax[0].plot(xx, f(xx), color=INK, label="f(x)")
ax[0].plot(xx, f(x)+2*(xx-x), "--", color=BLUE, label="Tangent")
ax[0].plot(xx, f(x)+result["forward"]*(xx-x), ":", color=ORANGE, label="Forward secant")
ax[0].scatter([x,x+h], [f(x),f(x+h)], color=ORANGE)
ax[0].set(xlabel="x", ylabel="f(x)", title="Local slope at x = 1"); ax[0].legend()
ax[1].loglog(steps,forward,color=ORANGE,label="Forward absolute error")
ax[1].loglog(steps,central,"--",color=BLUE,label="Central absolute error")
ax[1].set(xlabel="Step h",ylabel="Absolute derivative error",title="Truncation error"); ax[1].legend()
plt.show()
print({k: round(result[k],6) for k in ["exact","forward","central"]})
assert np.isclose(result["central"] - result["exact"], h*h)
'''),md(r'''中心差分在此消去了关于 $h$ 的一次误差项。图中的 log-log 斜率对应误差阶数；这不保证对任意函数、任意步长都更准确。

**自己试：** 把 `x` 改成 0，再比较两种差分。为什么它们对这个奇函数恰好相同？
<details><summary>核对思路</summary>在 x=0，前向误差中的 3xh 消失，两个误差均为 h²。</details>
'''),
md(r'''## 实验 2 · 矩阵改变方向与面积

采用列向量约定 $Av$。矩阵两列分别是两个标准基向量的像。行列式给出**带符号面积倍率**：负值表示方向翻转，零表示平面压缩到低维。
'''),code('''
matrices = [("Shear",dict(a=1,b=.7,c=0,d=1)),
            ("Reflection",dict(a=-1,b=0,c=0,d=1)),
            ("Singular collapse",dict(a=1,b=1,c=1,d=1))]
square = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]])
fig, axes = plt.subplots(1,3,figsize=(12,4),layout="constrained")
for ax,(name,params) in zip(axes,matrices):
    r=lab.compute("linear",**params); pts=r["vertices"]
    ax.plot(*square.T,"--",color=INK,label="Unit square")
    ax.plot(*pts.T,"o-",color=BLUE,label="Transformed")
    ax.set(xlim=(-1.5,2.5),ylim=(-.5,2.5),xlabel="x",ylabel="y",
           title=f"{name}: det = {r['determinant']:.1f}")
    ax.set_aspect("equal"); ax.legend(fontsize=9)
plt.show()
assert lab.compute("linear",a=1,b=1,c=1,d=1)["singular"]
'''),md(r'''面积相同不代表变换相同：剪切仍可改变角度。奇异变换没有全平面的逆。

**自己试：** 找两个不同矩阵，行列式都为 1，但把同一单位向量送到不同位置。'''),
md(r'''## 实验 3 · 投影把误差分成正交部分

对单位向量 $u$，投影为 $(u^Tv)u$。残差 $r=v-(u^Tv)u$ 满足 $u^Tr=0$，因此平方长度满足勾股分解。
投影长度可以为负；这表示沿 $u$ 的反方向，不是长度范数为负。
'''),code('''
r=lab.compute("projection",angle=30,vx=2,vy=1)
fig,ax=plt.subplots(layout="constrained")
for key,color,style in [("v",INK,"-"),("projection",BLUE,"--")]:
    value=r[key]; ax.plot([0,value[0]],[0,value[1]],style,color=color,label=key,linewidth=2.5)
ax.plot([r["projection"][0],r["v"][0]],[r["projection"][1],r["v"][1]],":",color=ORANGE,label="residual",linewidth=2.5)
ax.axhline(0,color="#aaa",linewidth=.7); ax.axvline(0,color="#aaa",linewidth=.7)
ax.set(xlabel="x",ylabel="y",title="Orthogonal projection",xlim=(-.2,2.5),ylim=(-.3,1.7))
ax.set_aspect("equal"); ax.legend(); plt.show()
print(f"u dot residual = {r['orthogonality']:.2e}")
print(f"||v||² = {r['norm2']:.8f}; ||projection||² + ||residual||² = {r['decomposedNorm2']:.8f}")
assert abs(r["orthogonality"])<1e-12
assert np.isclose(r["norm2"],r["decomposedNorm2"])
'''),md(r'''**自己试：** 令向量与 $u$ 垂直，再令向量与 $u$ 平行；先预测投影和残差各为多少。
'''),
md(r'''## 实验 4 · 梯度下降何时收敛

对 $f(x,y)=(x^2+Ly^2)/2$，每一步分别乘以 $1-\eta$ 和 $1-\eta L$。
对所有初始点收敛需要 $0<\eta<2/L$。等号处通常振荡；初始点恰好没有不稳定方向分量时可能例外。
'''),code('''
L=4
fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout="constrained")
for rate,color,style in [(.2,BLUE,"-"),(.5,OLIVE,"--"),(.6,ORANGE,":")]:
    r=lab.compute("gradient",rate=rate,condition=L,steps=25)
    if rate <= .5:
        axes[0].plot(r["x"],r["y"],style,color=color,marker=".",label=f"eta={rate}")
    axes[1].semilogy(r["step"],np.maximum(r["loss"],1e-15),style,color=color,label=f"eta={rate}")
axes[0].set(xlabel="x",ylabel="y",title="Stable and boundary trajectories")
axes[0].set_aspect("equal",adjustable="datalim")
axes[1].set(xlabel="Iteration",ylabel="Loss (log scale)",title="Stable, boundary, unstable")
for ax in axes: ax.legend()
plt.show()
stable=lab.compute("gradient",rate=.2)
assert np.all(np.diff(stable["loss"])<=1e-12)
boundary=lab.compute("gradient",rate=.5)
assert np.allclose(np.abs(boundary["y"]),2)
print("Stability bound:",2/L,"; boundary retains a nonzero y component")
'''),md(r'''## 检查 · Checks

已运行的断言检验了中心差分、奇异变换、正交分解和步长边界。图像提供直觉，等式决定是否成立。

## 接下来 · Next steps

把梯度下降的 `condition` 从 4 改成 20；固定步长和按 $1/L$ 调整步长分别比较。回到网页拖动参数后，在这里写下稳定条件的推导。

来源：[MIT OCW 18.02SC 微积分](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/)、[MIT OCW 18.06 线性代数](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)。精确二次递推在本 Notebook 直接代数推导；数值实现独立于网页。
''')]))

books.append(('02_sampling_inference',[
md(r'''# 02 · 抽样波动、置信区间与检验功效

## 目标 · Goal

区分总体分布、样本均值的抽样分布、区间程序的覆盖率和检验的功效。相同图形中的横轴含义必须说清楚。
需要的准备：均值、方差、正态分布、概率事件。
'''),md(intro),code(SETUP),
md(r'''## 实验 5 · 大数定律与中心极限定理的条件

正态与 Exp(1) 的总体方差都为 1，均值的理论标准误差为 $1/\sqrt n$。
标准 Cauchy 的总体均值和方差均未定义；不能把它代入这里的有限方差中心极限定理。
Cauchy 样本均值仍是标准 Cauchy，因此增大样本量不会让它按同样方式集中。
'''),code('''
fig,axes=plt.subplots(1,3,figsize=(12,4),layout="constrained")
for ax,distribution in zip(axes,["normal","exponential","cauchy"]):
    for n,color,style in [(5,ORANGE,"--"),(80,BLUE,"-")]:
        r=lab.compute("sampling",n=n,reps=2000,distribution=distribution,seed=SEED)
        values=np.sort(r["means"]); probabilities=np.arange(1,len(values)+1)/len(values)
        ax.plot(values,probabilities,style,color=color,label=f"n={n}")
    ax.set(xlabel="Sample mean",ylabel="Empirical CDF",title=distribution)
    ax.set_xlim((-8,8) if distribution=="cauchy" else (-1.5,2.5)); ax.legend()
plt.show()
print("Cauchy CDF panel shows x in [-8, 8]; tails extend beyond the displayed window.")
for distribution in ["normal","exponential"]:
    r=lab.compute("sampling",n=80,reps=2000,distribution=distribution,seed=SEED)
    print(distribution,"empirical SE",round(r["means"].std(ddof=1),4),"theory",round(r["theoreticalSE"],4))
assert lab.compute("sampling",distribution="cauchy")["theoreticalSE"] is None
'''),md(r'''这里用经验累积分布避免 Cauchy 的极端值把直方图主体压成一条线；尾部仍然存在，并未被删除后重新归一化。

**自己试：** 固定种子改变 `n`，观察正态与 Cauchy 的中间 50% 宽度。解释为什么有限样本里总能算出一个数，不等于总体期望存在。
'''),
md(r'''## 实验 6 · 95% 是区间程序的长期覆盖率

已知 $\sigma$，且 $X_i\overset{iid}{\sim}N(\mu,\sigma^2)$ 时，
$\bar X\pm z_{(1+c)/2}\sigma/\sqrt n$ 是精确覆盖率为 $c$ 的区间程序。
本实验固定真实 $\mu=0$；重复生成数据，每次区间不同。单个已观察区间不对固定参数赋予频率派概率。
'''),code('''
r=lab.compute("confidence",n=25,reps=100,level=.95,sigma=1,seed=SEED)
fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
for i,(low,high,covers) in enumerate(zip(r["low"],r["high"],r["covered"])):
    ax.plot([low,high],[i+1,i+1],"-" if covers else "--",color=BLUE if covers else ORANGE,linewidth=1.1)
ax.plot([],[],"-",color=BLUE,label="Covers true mean")
ax.plot([],[],"--",color=ORANGE,label="Misses true mean")
ax.axvline(0,color=INK,linewidth=1.8,label="True mu = 0")
ax.set(xlabel="Mean parameter",ylabel="Replicate (all 100 shown)",title="Known-sigma z intervals"); ax.legend()
plt.show()
print(f"Coverage {r['covered'].sum()}/100 = {r['coverage']:.1%}; interval width {r['width']:.4f}")
print(f"Monte Carlo SE under nominal 95% coverage = {r['monteCarloSE']:.4f}")
assert np.allclose(r["high"]-r["low"],r["width"])
assert np.isclose(lab.compute("confidence",n=100)["width"],r["width"]/2)
'''),md(r'''一次重复实验的覆盖率可以高于或低于 95%。重复次数控制模拟估计的精度，样本量控制每个区间的宽度；这两个数量不能互换。

**自己试：** 固定样本量，把重复次数从 100 改成 300；再固定重复次数，把样本量变为四倍。分别预测 Monte Carlo 标准误差和区间宽度如何变化。
'''),
md(r'''## 实验 7 · p 值与功效回答不同的问题

$H_0:\mu=0$，已知 $\sigma=1$，双侧拒绝域为 $|Z|>z_{1-\alpha/2}$。
当真实均值为 $\delta$ 时，$Z\sim N(\delta\sqrt n,1)$；两侧尾面积之和给出**精确功效**。
p 值在零假设分布下计算，不是零假设为真的概率。
'''),code('''
n, effect, alpha, observed = 30, .5, .05, .3
r=lab.compute("testing",n=n,effect=effect,alpha=alpha,observed=observed)
x=np.linspace(-1.1,1.5,600); se=1/np.sqrt(n)
fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout="constrained")
axes[0].plot(x,norm.pdf(x,0,se),color=INK,label="H0: mu=0")
axes[0].plot(x,norm.pdf(x,effect,se),"--",color=BLUE,label=f"Alternative: mu={effect}")
rejected=np.abs(x)>r["criticalMean"]
axes[0].fill_between(x,0,norm.pdf(x,effect,se),where=rejected,color=BLUE,alpha=.18,label="Power (both tails)")
for cutoff in [-r["criticalMean"],r["criticalMean"]]: axes[0].axvline(cutoff,color=ORANGE,linestyle=":")
axes[0].set(xlabel="Sample mean",ylabel="Density",title="Exact normal sampling densities"); axes[0].legend(fontsize=9)
effects=np.linspace(-1,1,100)
for size,color,style in [(10,ORANGE,"--"),(30,BLUE,"-"),(100,OLIVE,":")]:
    power=[lab.compute("testing",n=size,effect=e)["power"] for e in effects]
    axes[1].plot(effects,power,style,color=color,label=f"n={size}")
axes[1].set(xlabel="True mean (sigma = 1)",ylabel="Rejection probability",ylim=(0,1.03),title="Two-sided exact power"); axes[1].legend()
plt.show()
print(f"Observed p-value = {r['pValue']:.6f}; power at effect {effect} = {r['power']:.6f}")
assert np.isclose(lab.compute("testing",effect=0)["power"],.05)
assert np.isclose(lab.compute("testing",effect=-.5)["power"],r["power"])
'''),md(r'''## 检查 · Checks

核对了 Cauchy 未定义的理论标准误差、区间宽度的 $1/\sqrt n$ 比例、零效应下功效等于显著性水平，以及双侧功效的对称性。

## 接下来 · Next steps

固定 `observed` 改变假定真实效应 `effect`：功效变化，p 值不应变化。解释这为什么不是矛盾。

来源：[SciPy 正态分布定义与 CDF/SF](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)、[NIST 置信区间解释](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm)（用于覆盖率含义；该页面公式本身是未知方差 t 区间，本实验已知方差 z 公式直接由正态标准化得到）、[NIST Cauchy 分布](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3663.htm)。
''')]))

books.append(('03_bayes_bootstrap',[
md(r'''# 03 · 后验不确定性与 Bootstrap

## 目标 · Goal

比较两种不同的不确定性计算：给定模型与先验的后验分布，和给定已观察样本的经验重抽样分布。
它们的条件、随机对象和区间解释不同。需要的准备：条件概率、概率密度、分位数。
'''),md(intro),code(SETUP),
md(r'''## 实验 8 · Beta–Binomial 共轭更新

独立 Bernoulli 观测出现 $s$ 次成功、$f$ 次失败。先验 $\theta\sim\mathrm{Beta}(a,b)$ 时，后验为 $\mathrm{Beta}(a+s,b+f)$。
95% 等尾可信区间由后验 CDF 的 0.025 与 0.975 分位数定义；它不是“均值加减两倍标准差”的默认替代品。
似然作为参数的函数不自动是归一化概率密度，本图为便于比较将其峰值归一为 1，并使用独立纵轴。
'''),code('''
a,b,success,failure=2,2,7,3
r=lab.compute("bayes",a=a,b=b,success=success,failure=failure)
theta=np.linspace(.0001,.9999,600)
prior=beta_distribution.pdf(theta,a,b); posterior=beta_distribution.pdf(theta,*r["posterior"])
likelihood=theta**success*(1-theta)**failure; likelihood/=likelihood.max()
fig,ax=plt.subplots(figsize=(10,4.8),layout="constrained")
ax.plot(theta,prior,"--",color=ORANGE,label="Prior density")
ax.plot(theta,posterior,color=BLUE,label="Posterior density",linewidth=2)
inside=(theta>=r["interval"][0])&(theta<=r["interval"][1])
ax.fill_between(theta,0,posterior,where=inside,color=BLUE,alpha=.16,label="Central 95% credible region")
second=ax.twinx(); second.plot(theta,likelihood,":",color=OLIVE,label="Likelihood / maximum")
second.set(ylabel="Relative likelihood (peak = 1)",ylim=(0,1.25)); second.grid(False)
ax.set(xlabel="Success probability theta",ylabel="Probability density",title="Beta(2,2) updated by 7 successes and 3 failures")
ax.legend(loc="upper left",fontsize=9); second.legend(loc="upper right",fontsize=9)
plt.show()
print("Posterior parameters:",r["posterior"],"; posterior mean:",round(r["mean"],6))
print("Central 95% credible interval:",np.round(r["interval"],6))
assert np.allclose(beta_distribution.cdf(r["interval"],*r["posterior"]),[.025,.975])
assert lab.compute("bayes",a=1,b=1,success=0,failure=0)["mean"]==.5
'''),md(r'''后验密度的纵轴可以大于 1，积分才是概率。右侧相对似然的峰值是人为缩放后的 1。

**自己试：** 用 0 成功、10 失败与均匀先验计算可信区间。为什么区间不应以样本比例 0 为中心做对称扩展？
<details><summary>核对思路</summary>后验为 Beta(1,11)，支撑集为 [0,1] 且右偏，等尾分位数自然不对称。</details>
'''),
md(r'''## 实验 12 · Bootstrap 估计的是条件重抽样分布

先抽取一份大小为 $n$ 的样本，再从这份样本有放回抽取 $n$ 个数。重复 $B$ 次计算均值。
Bootstrap 标准误差为这些均值的样本标准差；百分位区间取 2.5% 与 97.5% 分位数，不保证有限样本精确覆盖 95%。
固定样本后，增加 $B$ 只降低模拟噪声，不会增加原始数据的信息。
'''),code('''
r=lab.compute("bootstrap",n=30,reps=2000,distribution="exponential",seed=SEED)
fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout="constrained")
axes[0].hist(r["sample"],bins=12,color=ORANGE,edgecolor="white")
axes[0].set(xlabel="Observation",ylabel="Count (n=30)",title="One observed sample from Exp(1)")
axes[1].hist(r["bootstrapMeans"],bins=35,density=True,color=BLUE,alpha=.8,edgecolor="white")
for i,q in enumerate(r["interval"]): axes[1].axvline(q,color=INK,linestyle="--",label="Percentile 95% endpoints" if i==0 else None)
axes[1].axvline(np.mean(r["sample"]),color=ORANGE,linestyle=":",label="Observed sample mean")
axes[1].set(xlabel="Bootstrap sample mean",ylabel="Conditional density estimate",title="2,000 resamples of size 30"); axes[1].legend(fontsize=9)
plt.show()
print("Bootstrap SE:",round(r["bootstrapSE"],6),"; s / sqrt(n):",round(r["analyticSE"],6))
print("Percentile interval:",np.round(r["interval"],6))
# Exact conditional SE of a bootstrap mean uses the empirical variance with divisor n.
exact_conditional=r["sample"].std(ddof=0)/np.sqrt(len(r["sample"]))
print("Exact conditional SE for the empirical distribution:",round(exact_conditional,6))
assert np.array_equal(lab.compute("bootstrap",reps=100)["sample"],lab.compute("bootstrap",reps=2000)["sample"])
'''),md(r'''Bootstrap SE 与 $s/\sqrt n$ 不必逐位相等。即使无限重抽样次数，前者对经验分布使用的是分母 $n$ 的方差，后者使用分母 $n-1$；再加上有限重复次数的模拟误差。

**自己试：** 保持种子与 `n` 不变，把 `reps` 改为 100、500、2000。确认原始样本完全相同，再比较区间端点。

## 检查 · Checks

核对了后验区间的精确 CDF 值、无观测时均匀先验的均值、以及改变 Bootstrap 重复次数不会改变原始样本。

## 接下来 · Next steps

用一句话分别回答：可信区间的概率对什么条件化？Bootstrap 分布对什么条件化？为什么它们都不能自动代替前一本中的频率派覆盖率实验？

来源：[SciPy Beta 密度、CDF 和逆 CDF](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html)、[SciPy Bootstrap 方法说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)。本实验采用 percentile 方法，而非该函数默认的 BCa 方法。
''')]))

books.append(('04_models_representation',[
md(r'''# 04 · 回归稳定性、过拟合与 PCA

## 目标 · Goal

理解三个不同问题：系数能否稳定估计、预测能否泛化、低维表示保留了多少样本方差。
需要的准备：最小二乘、向量投影、样本协方差；可先完成第 01 本。
'''),md(intro),code(SETUP),
md(r'''## 实验 9 · 相关预测变量与 Ridge

预测变量先用训练样本均值与样本标准差标准化。目标函数为
$\|y-\beta_0\mathbf1-X\beta\|^2+\lambda\|\beta\|^2$，**截距不惩罚**，损失没有除以样本量。
两个标准化预测变量时，经验 VIF 为 $1/(1-r^2)$；这里 $r$ 是实现样本的相关系数，未必恰好等于生成参数 $\rho$。
'''),code('''
rho=.98
penalties=[0,.1,1,10]
fits=[lab.compute("regression",rho=rho,**{"lambda":penalty},seed=SEED) for penalty in penalties]
fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout="constrained")
for i,color,style in [(0,BLUE,"-o"),(1,ORANGE,"--s")]:
    axes[0].plot(penalties,[r["beta"][i] for r in fits],style,color=color,label=f"beta {i+1}")
axes[0].axhline(1,color=INK,linestyle=":",label="True coefficients")
axes[0].set(xlabel="Ridge lambda (sum-loss)",ylabel="Estimated coefficient",title="Coefficients under collinearity"); axes[0].legend()
r=fits[-1]
axes[1].scatter(r["predictions"],r["residuals"],facecolors="none",edgecolors=BLUE,s=32)
axes[1].axhline(0,color=INK,linestyle=":")
axes[1].set(xlabel="Fitted response",ylabel="Residual",title="Residuals at lambda = 10")
plt.show()
print(f"Empirical r={r['sampleCorrelation']:.6f}; VIF={r['vif']:.4f}; design condition={r['conditionNumber']:.4f}")
assert abs(np.mean(r["residuals"]))<1e-12
assert np.allclose(fits[0]["beta"],fits[0]["olsBeta"])
print("OLS coefficients:",np.round(fits[0]["beta"],4),"; Ridge coefficients:",np.round(r["beta"],4))
'''),md(r'''Ridge 用偏差换稳定性，但不保证每个系数都单调向零，也不保证某一次数据上的预测更好。

**自己试：** 将 `rho` 从 0.98 改为 0，比较经验 VIF；再换种子比较系数波动。'''),
md(r'''## 实验 10 · 多项式拟合与验证误差

真实函数为 $\sin(\pi x)$，训练和验证样本独立生成。使用 Legendre 基，截距不惩罚；独立实现用增广设计矩阵的 SVD 最小二乘求解。
训练误差衡量已见数据上的拟合，验证误差衡量这一固定验证集上的预测。反复看验证集选择模型后，需要另留测试集评价最终性能。
'''),code('''
degrees=[1,3,5,9,15]
fits=[lab.compute("overfit",degree=d,n=20,noise=.25,**{"lambda":.001},seed=SEED) for d in degrees]
xx=np.linspace(-1,1,400)
fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
axes[0].plot(xx,np.sin(np.pi*xx),color=INK,label="True sin(pi x)")
axes[0].scatter(fits[0]["train"]["x"],fits[0]["train"]["y"],s=28,facecolors="none",edgecolors=INK,label="Training n=20")
for degree,color,style in [(1,ORANGE,":"),(5,BLUE,"--"),(15,OLIVE,"-")]:
    fit=fits[degrees.index(degree)]
    axes[0].plot(xx,legvander(xx,degree)@fit["coefficients"],style,color=color,label=f"Degree {degree}")
axes[0].set(xlabel="x",ylabel="Response",title="One fixed train / validation split"); axes[0].legend(fontsize=9)
axes[1].plot(degrees,[r["trainMSE"] for r in fits],"o-",color=BLUE,label="Training MSE")
axes[1].plot(degrees,[r["validationMSE"] for r in fits],"s--",color=ORANGE,label="Validation MSE (n=300)")
axes[1].set(xlabel="Polynomial degree",ylabel="Mean squared error",title="Observed errors; no enforced U shape"); axes[1].legend(fontsize=9)
plt.show()
for d,r in zip(degrees,fits): print(f"degree {d:2d}: train={r['trainMSE']:.5f}, validation={r['validationMSE']:.5f}")
assert all(np.array_equal(fits[0]["train"]["y"],r["train"]["y"]) for r in fits)
assert np.allclose(fits[2]["design"],legvander(fits[2]["train"]["x"],5))
'''),md(r'''改变次数时保持数据不变，才能把可见变化归因于模型。误差曲线是这份数据的观测结果，不要求每一次都出现教科书式 U 形。

**自己试：** 用 `n=10, degree=15, lambda=0` 创建欠定问题。NumPy 会返回最小范数解；拟合存在不等于系数唯一，也不等于网页一定采用同样的非唯一解约定。'''),
md(r'''## 实验 11 · PCA 与重建误差

对中心化后的数据做样本协方差分解；可选标准化后得到相关矩阵 PCA。
特征向量的正负号任意；重复特征值对应的方向不唯一。比较实现时，应比较特征子空间、重建和解释方差，不能只比较向量逐项符号。
这里重建误差定义为每个观测的**两个坐标平方误差之和**再取平均。
'''),code('''
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
for ax,standardize in zip(axes,[0,1]):
    r=lab.compute("pca",angle=35,ratio=4,scale=2,n=120,standardize=standardize,seed=SEED)
    observed=r["processed"]; rebuilt=r["reconstruction"]
    ax.scatter(*observed.T,s=14,color=BLUE,alpha=.6,label="Processed observations")
    ax.scatter(*rebuilt.T,s=15,facecolors="none",edgecolors=ORANGE,label="Rank-1 reconstruction")
    for p,q in zip(observed[:20],rebuilt[:20]): ax.plot([p[0],q[0]],[p[1],q[1]],color="#999",linewidth=.65)
    ax.set_aspect("equal",adjustable="datalim")
    ax.set(xlabel="Processed coordinate 1",ylabel="Processed coordinate 2",
           title=("Correlation" if standardize else "Covariance")+f" PCA: {r['explainedRatio']:.1%}")
    ax.legend(fontsize=9)
    expected=r["eigenvalues"][1]*(len(observed)-1)/len(observed)
    assert np.isclose(r["reconstructionMSE"],expected)
    print(("Standardized" if standardize else "Centered"),"eigenvalues",np.round(r["eigenvalues"],5),"; reconstruction error",round(r["reconstructionMSE"],5))
plt.show()
'''),md(r'''由于协方差使用分母 $n-1$，每观测重建平方误差等于 $(n-1)\lambda_2/n$。标准化改变了距离的含义，两个面板误差的单位也不同，不能直接作为优劣排名。

**自己试：** 固定生成样本，改变第二坐标的 `scale`。协方差 PCA 会受单位缩放影响；标准化后的相关矩阵 PCA 对正的逐坐标缩放不变。

## 检查 · Checks

核对截距导致残差均值为零、零惩罚 Ridge 等于 OLS、改变多项式次数保留同一训练样本、Legendre 设计矩阵和 PCA 特征值—重建误差恒等式。

## 接下来 · Next steps

选择一个你无法口头解释的参数，先在网页试验，再修改 Notebook、保留一个结果和一句解释。实验记录可单独保存；本文件不记录个人掌握程度。

来源：[NumPy 最小二乘及最小范数约定](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)、[NumPy 对称矩阵特征分解](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html)、[NumPy Legendre 设计矩阵](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legvander.html)、[scikit-learn Ridge 目标](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)。
''')]))

for name,cells in books:
    notebook=nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    notebook.metadata.language_info={'name':'python','version':'3'}
    nbf.validate(notebook)
    nbf.write(notebook,ROOT/'notebooks'/f'{name}.ipynb')
    print(name,len(cells),'cells')
