# Python 复算与教学 Notebook

四本 Notebook 覆盖基础组 12 个交互实验，已保存从头执行的输出。只想阅读时直接打开 HTML；想修改参数和复算时打开 Notebook。

| 内容 | 可运行 Notebook | 离线阅读 |
|---|---|---|
| 导数、线性变换、投影、梯度下降 | [01_geometry_optimization.ipynb](notebooks/01_geometry_optimization.ipynb) | [含公式与图表](previews/01_geometry_optimization.html) |
| 抽样、置信区间、检验功效 | [02_sampling_inference.ipynb](notebooks/02_sampling_inference.ipynb) | [含公式与图表](previews/02_sampling_inference.html) |
| 贝叶斯更新、Bootstrap | [03_bayes_bootstrap.ipynb](notebooks/03_bayes_bootstrap.ipynb) | [含公式与图表](previews/03_bayes_bootstrap.html) |
| Ridge、过拟合、PCA | [04_models_representation.ipynb](notebooks/04_models_representation.ipynb) | [含公式与图表](previews/04_models_representation.html) |

## 安装与运行

在本 `python` 目录打开终端。首次安装使用独立虚拟环境，Python 3.12+：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

安装完成后，一个命令重新执行四本 Notebook 并更新 HTML：

```sh
.venv/bin/python execute_notebooks.py
```

运行不需要网络、凭据或个人数据。首次安装依赖需要网络。已保存的 HTML 不需要 Python；数学公式使用项目随附的本地 KaTeX，须保留项目相对目录结构。

也可用 IDE 的 Jupyter 支持打开 `.ipynb`，选择上面的 `.venv` Python 解释器，从顶部运行全部单元。固定种子为 42，参数在每个实验单元开头。`requirements-lock.txt` 锁定本次验证的全部依赖；`requirements.txt` 给出兼容范围。Notebook 重新生成源为 `build_notebooks.py`，它会覆盖 Notebook 内容，日常学习只运行 `execute_notebooks.py`。

## 单个实验与数据导出

```sh
.venv/bin/python lab_reference.py --lab testing --params '{"n":100,"effect":0.3}'
.venv/bin/python lab_reference.py --lab all --output results.json
.venv/bin/python ../tests/independent_numerics.py
```

最后一条命令需要 Node.js（默认调用 `node`，可通过 `NODE` 指定路径）。它直接调用网页的数学内核，然后使用 SciPy 与 NumPy 对原始样本、设计矩阵、分布函数和预测重新计算，结果写入 `../tests/independent-report.json`。

## 与网页对应的约定

- 所有计算使用合成数据。Python 使用 NumPy PCG64，网页有自己的伪随机数算法；相同种子在各自实现内可复现，**两边样本不承诺逐项相同**。需要逐项核对网页时使用独立测试脚本，它直接读取网页的原始数据。
- Ridge 与多项式拟合目标均为残差平方和加惩罚；截距不惩罚。Python 用 SVD 增广最小二乘，网页用自己的稳定求解器。
- Python `regression.conditionNumber` 指设计矩阵 `X` 的二范数条件数；网页显示 `XᵀX` 的条件数，两者是平方关系，Notebook 明确标记 design condition。
- Python 与网页的多项式验证集均为 300 点，都是与训练集独立的固定样本。比较次数时保持各自数据不变。验证误差不是最终测试误差。
- Python PCA 的 `eigenvectors` 每列为一个方向；网页每行为一个方向。符号任意，比较时使用子空间和重建。
- PCA 重建误差在当前分析坐标中计算，等于每观测的两个坐标平方误差之和的均值；不是再除以 2 的逐坐标 MSE。相关矩阵 PCA 和协方差 PCA 的误差单位不同。
- Cauchy 总体均值、方差与均值标准误差保持未定义。已知方差 z 区间、两侧 z 检验、等尾 Beta 可信区间和 percentile Bootstrap 保留各自假设与解释。

执行证据在 [execution-report.json](previews/execution-report.json)，预览核验在 [preview-report.json](previews/preview-report.json)。详细范围与边界见 [数学核验](../docs/数学核验.md)。
