# Optimization Algorithm

## [Levenberg–Marquardt algorithm](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm)

* **非线性最小二乘**：问题拆分为**最小二乘**和**非线性**。
  目标函数可以写成**多项平方和**的形式，则该问题就是最小二乘问题：

$$
\begin{aligned}
F(\mathbf{x}) &= \frac{1}{2} \sum_{i} f^{2}_{i}(\mathbf{x}) = \frac{1}{2} \| \mathbf{f}(\mathbf{x})\|^{2} \\
\mathbf{x} &= \underset{\mathbf{x}}{\min}\ F(\mathbf{x})
\end{aligned}
$$

其中$F(\mathbf{x})$是**目标函数**，而$f_{i}(\mathbf{x})$是**残差**函数（residual）。

如果残差函数$f_{i}(\mathbf{x})$是关于待优化参数$\mathbf{x}$的线性函数，则是线性最小二乘问题，可以直接变成正规方程求解，不需要迭代函求解。如果残差函数$f_{i}(\mathbf{x})$包含了指数、对数、三角函数或者参数之间相乘等非线性关系，则是非线性最小二乘问题，需要通过迭代进行求解。

---

### 高斯-牛顿法

* 只适用于**最小二乘问题**，因为要对残差函数做一阶泰勒展开：

$$
\mathbf{f}(\mathbf{x}_{k} + \Delta \mathbf{x}) \approx \mathbf{f}(\mathbf{x}_{k}) + \mathbf{J}_{k} \Delta \mathbf{x}
$$

将一阶展开代入数目标函数$F(\mathbf{x})$中：

$$
F(\mathbf{x}_{k} + \Delta \mathbf{x}) = \frac{1}{2} \| \mathbf{f}(\mathbf{x}_{k}) + \mathbf{J}_{k} \Delta \mathbf{x} \|^{2}
$$

展开上式，得到关于$\Delta \mathbf{x}$的二次函数，二次函数的极值点在其导数为0的地方取到，最后化简得到：

$$
\mathbf{J}_{k}^{\top} \mathbf{J}_{k} \Delta \mathbf{x} = -\mathbf{J}_{k}^{\top} \mathbf{f}(\mathbf{x}_{k})
$$

虽然是对**残差函数**做一阶泰勒展开，但是目标函数最终变成关于$\Delta \mathbf{x}$的二次函数，这就是为什么高斯牛顿法能表现出类似二阶优化（牛顿法）的快速收敛特性。

* 特点：拥有接近牛顿法的收敛速度，但是计算任何二阶导数，但是求逆可能存在数值稳定性问题。

---

### 梯度下降法（一阶梯度法）

* 对**目标函数**做**一阶泰勒**展开：

$$
F(\mathbf{x}_{k} + \Delta \mathbf{x}) \approx F(\mathbf{x}_{k}) + \nabla F(\mathbf{x}_{k})^{\top} \Delta \mathbf{x}
$$

这是一个线性函数，没有极小值点，一般取其步长为：

$$
\Delta \mathbf{x} = - \alpha \cdot \nabla F(\mathbf{x}_{k})
$$

* 特点：一阶收敛，慢，但是稳定。

---

### 牛顿法

* 对**目标函数**做**二阶泰勒**展开：

$$
F(\mathbf{x}_{k} + \Delta \mathbf{x}) \approx F(\mathbf{x}_{k}) + \nabla F(\mathbf{x}_{k})^{\top} \Delta \mathbf{x} + \frac{1}{2} \Delta \mathbf{x}_{k}^{\top} \mathbf{H}_{k} \Delta \mathbf{x}_{k}
$$

其是关于$\Delta \mathbf{x}$的二阶导数，求导得0，最终得到：

$$
\mathbf{H}_{k} \Delta \mathbf{x} = - \nabla F(\mathbf{x}_{k})
$$

其中$\nabla F(\mathbf{x}_{k})$是目标h函数的一阶导数（梯度），$\mathbf{H}_{k}$是目标函数的二阶导数（海森矩阵）。

* 特点：二阶收敛，极快，但是计算二阶导数非常困难且昂贵。

---

* 在求解形如$\mathbf{Ax}=\mathbf{b}$的问题时不会显式计算矩阵逆$\mathbf{A}^{-1}$，而是使用**矩阵分解**的技术：
  当$\mathbf{A}$是**正定**时，使用Cholesky分解，将其分解为下三角矩阵和及转置的乘积，进而进行“前向/后向”替换求解。
  当$\mathbf{A}$的可逆性存疑时，直接对雅克比矩阵进行QR分解或者奇异值分解，进而剔除不可逆的零空间。

### LM详解

* LM算法的核心更新方程为：

$$
(\mathbf{J}^{\top} \mathbf{J} + \lambda \mathbf{I}) = -\mathbf{J}^{\top} \mathbf{f}
$$

其中$\mathbf{f}$是残差函数，$\mathbf{J}$是残差函数的雅克比矩阵，因此决定了其只能用在**最小二乘**问题上。

* 当$\lambda \to 0$时，LM算法退化为高斯-牛顿法。其中的$\lambda \mathbf{I}$保证了矩阵一定可逆。因此当优化过程比较顺利（目标函数值在迭代过程中不断下降），则可以调小$\lambda$，加快收敛。
* 当$\lambda$比较大时，则$\lambda \mathbf{I}$主导矩阵，公式近似变为$\Delta \mathbf{x} = - \frac{1}{\lambda} \mathbf{J}^{\top} \mathbf{f}$，本质就是步长为$\frac{1}{\lambda}$的梯度下降法。因此当优化过程不顺利（目标函数并不稳定下降），说明离极小值还远，需要调大$\lambda$。
* LM算法又称**阻尼最小二乘法**（Damping Least Square）。

## [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS)

### BFGS

* 牛顿法得到：

$$
\mathbf{H}_{k} \Delta \mathbf{x} = - \nabla F(\mathbf{x}_{k}) \to
\Delta \mathbf{x} = - \mathbf{H}_{k}^{-1} \nabla F(\mathbf{x}_{k})
$$

* 海森矩阵（曲率）的逆的计算代价太昂贵，因此通过**拟牛顿条件**去**估算**海森矩阵。
  位置的变化量记为$\mathbf{s}=\mathbf{x}_{k+1} - \mathbf{x}_{k}$，一阶导数（梯度）$\nabla F(\mathbf{x}_{k})$记为$\mathbf{g}_{k}$，则梯度的变化量记为$\mathbf{y}=\mathbf{g}_{k+1} - \mathbf{g}_{k}$。二阶导数（曲率）为$\mathbf{H}_{k}$，其逆$\mathbf{B}_{k} = \mathbf{H}_{k}^{-1}$：

$$
\begin{aligned}
&\text{曲率=梯度变化量/位置变化量} \to \\
&\text{曲率*位置变化量=梯度变化量} \to \\
&\mathbf{H}\mathbf{s} = \mathbf{y}
\end{aligned}
$$

拟牛顿条件就是使用两点之间的平均割线斜率，来拟合瞬时切线曲率。

* $\mathbf{Hs}=\mathbf{y}$中，$\mathbf{H}$含有$n \times n$个未知数，但是只有$n$个等式，因此是一个欠定问题。既然解不唯一，那么就“继承历史，最小修正”，也即新矩阵要尽可能靠近上一步的阵矩。新矩阵$\mathbf{H}_{k+1}$和旧矩阵$\mathbf{H}_{k}$满足以下关系：

$$
\underset{\mathbf{H}_{k+1}}{\min} \| \mathbf{H_{k+1}} - \mathbf{H}_{k} \|^{2}
$$

同时为了使用“历史足迹”（$\mathbf{s}_{k}$和$\mathbf{y}_{k}$）去更新“未来地图”（$\mathbf{H}_{k+1}$），拟牛顿条件变成$\mathbf{H}_{k+1} \mathbf{s}_{k} = \mathbf{y}_{k}$。

假设新矩阵是在旧矩阵的基础上，添加一个修正矩阵：

$$
\mathbf{H}_{k+1} = \mathbf{H}_{k} + \Delta \mathbf{H}
$$

引入**秩-2修正**，也即修正矩阵$\Delta \mathbf{H}$由两对向量外积组合而成：

$$
\mathbf{H}_{k+1} = \mathbf{H}_{k} + a \mathbf{u} \mathbf{u}^{\top} + b \mathbf{v} \mathbf{v}^{\top}
$$

代回拟牛顿条件：

$$
\mathbf{H}_{k} \mathbf{s}_{k} + a \mathbf{u}( \mathbf{u}^{\top} \mathbf{s}_{k}) + b \mathbf{v} (\mathbf{v}^{\top} \mathbf{s}_{k}) = \mathbf{y}_{k}
$$

为了等式成立，最简单的就是让第二项$a \mathbf{u} (\mathbf{u}^{\top} \mathbf{s}_{k})$消除掉旧信息$\mathbf{H}_{k} \mathbf{s}_{k}$，第三项$b \mathbf{v}(\mathbf{v}^{\top} \mathbf{s}_{k})$变成梯度变化量$\mathbf{y}_{k}$。因此选择：

$$
\mathbf{u} = \mathbf{H}_{k} \mathbf{s}_{k},\ \mathbf{v} = \mathbf{y}_{k}
$$

得到：

$$
a = - \frac{1}{\mathbf{s}_{k}^{\top} \mathbf{H}_{k} \mathbf{s}_{k}},\ b = \frac{1}{\mathbf{y}_{k}^{\top} \mathbf{s}_{k}}
$$

最后可得：

$$
\mathbf{H}_{k+1} = \mathbf{H}_{k} - \frac{\mathbf{H}_{k} \mathbf{s}_{k} \mathbf{s}_{k}^{\top} \mathbf{H}_{k}}{\mathbf{s}_{k}^{\top} \mathbf{H}_{k} \mathbf{s}_{k}} + \frac{\mathbf{y}_{k} \mathbf{y}_{k}^{\top}}{\mathbf{y}_{k}^{\top} \mathbf{s}_{k}}
$$

但是求步长最终需要的是海森矩阵的逆$\mathbf{B}_{k}$，利用矩阵求逆公式（Sherman-Morrison-Woodbury公式），对上式直接求逆得到：

$$
\mathbf{B}_{k+1} = (\mathbf{I} - \rho_{k} \mathbf{s}_{k} \mathbf{y}_{k}^{\top}) \mathbf{B}_{k} (\mathbf{I} - \rho_{k} \mathbf{y}_{k}\mathbf{s}_{k}^{\top}) + \rho_{k} \mathbf{s}_{k} \mathbf{s}_{k}^{\top},\ \rho_{k} = \frac{1}{\mathbf{y}^{\top} \mathbf{s}_{k}}
$$

* 公式的物理意象：

第一项$(\mathbf{I} - \rho_{k} \mathbf{s}_{k} \mathbf{y}_{k}^{\top}) \mathbf{B}_{k} (\mathbf{I} - \rho_{k} \mathbf{y}_{k}\mathbf{s}_{k}^{\top})$的作用是“选择性擦除”和“空间扭曲”。其中$\mathbf{B}_{k}$是对全山各处弧度的“旧地图”，当迈出$\mathbf{s}_{k}=\mathbf{x}_{k+1}-\mathbf{x}_{k}$这一步后，**旧地图在这个方向上的预测就不够准了**。

当旧地图的梯度变化信息$\mathbf{y}_{k}$试图通过第一项时，$\mathbf{y}_{k}$的信息被抹成0。只有和$\mathbf{y}_{k}$平行的运动方向会被抹除。

$$
(\mathbf{I} - \rho_{k} \mathbf{y}_{k} \mathbf{s}_{k}^{\top}) \mathbf{y}_{k} = \mathbf{y}_{k} - \rho_{k} \mathbf{y}_{k}(\mathbf{s}_{k}^{\top} \mathbf{y}_{k}) = \mathbf{0}
$$

当输入与当前完全垂直的方向向量$\mathbf{v}$（满足$\mathbf{s}_{k}^{\top} \mathbf{v}=0$），旧地图的记忆会被完好无损的保存下来：

$$
(\mathbf{I} - \rho_{k} \mathbf{y}_{k} \mathbf{s}_{k}^{\top}) \mathbf{v} = \mathbf{v}
$$

记$\mathbf{v}_{\text{old}} = \mathbf{B}_{k} \mathbf{v}$，则得到：

$$
(\mathbf{I} - \rho_{k} \mathbf{s}_{k} \mathbf{y}_{k}^{\top}) \mathbf{v}_{\text{old}} = \mathbf{v}_{\text{old}} - (\rho_{k} \mathbf{y}_{k}^{\top} \mathbf{v}_{\text{old}}) \mathbf{s}_{k}
$$

即在原本输出$\mathbf{v}_{\text{old}}$的基础上，减去一个沿着当前位移方向的刚性分量$\mathbf{s}_{k}$，其含义是“不管旧地图在其他交叉方向上算出了什么结果，此次在$\mathbf{s}_{k}$方向上有了新的空间发现，要沿着$\mathbf{s}_{k}$的反方向弯曲，修正一下”。

最后一项是**注入实践得出来的“绝对真理”**：

$$
\rho_{k} \mathbf{s}_{k} \mathbf{s}_{k}^{\top} (\mathbf{y}_{k} + \mathbf{v}) = \mathbf{s}_{k}
$$

强制把最新一次实践得出来的弧度信息，钉死在当前方向上，确保新地图$\mathbf{B}_{k+1}$在应对刚刚走过的这段路时绝对精准。

### L-BFGS推导

* BFGS需要存储海森矩阵的逆$\mathbf{B}$，因此其空间复杂度是$O(N^{2})$。为了减少内存的使用，注意到$\mathbf{B}_{k+1}$是由$\mathbf{B}_{k}$和向量对$(\mathbf{s}_{k}, \mathbf{y}_{k})$计算得到，如果只**保存最近的m步向量对**，其空间复杂度为$O(mN)$。由于$m \ll N$，因此内存用量大大降低。
* L-BFGS使用空间换时间的代价是：**牺牲了远期记忆**，因为只保存最近m步，对整个大山的认知是不完整的，其收敛速度比BFGS慢一些。
* BFGS中海森矩阵的逆的公式为：

$$
\mathbf{B}_{k+1} = (\mathbf{I} - \rho_{k} \mathbf{s}_{k} \mathbf{y}_{k}^{\top}) \mathbf{B}_{k} (\mathbf{I} - \rho_{k} \mathbf{y}_{k}\mathbf{s}_{k}^{\top}) + \rho_{k} \mathbf{s}_{k} \mathbf{s}_{k}^{\top},\ \rho_{k} = \frac{1}{\mathbf{y}_{k}^{\top} \mathbf{s}_{k}} = \frac{1}{\mathbf{s}_{k}^{\top} \mathbf{y}_{k}}
$$

记$\mathbf{V}_{k}= \mathbf{I} - \rho_{k} \mathbf{y}_{k} \mathbf{s}_{k}^{\top}$，则BFGS公式可以改写为：

$$
\mathbf{B}_{k+1} = \mathbf{V}_{k}^{\top} \mathbf{B}_{k} \mathbf{V}_{k} + \rho_{k} \mathbf{s}_{k} \mathbf{s}_{k}^{\top}
$$

展开迭代：

$$
\begin{aligned}
\mathbf{B}_{1} &= \mathbf{V}_{0}^{\top} \mathbf{B}_{0} \mathbf{V}_{0} \\
&+ \rho_{0} \mathbf{s}_{0} \mathbf{s}_{0}^{\top} \\
\mathbf{B}_{2} &= \mathbf{V}_{1}^{\top} \mathbf{V}_{0}^{\top} \mathbf{B}_{0} \mathbf{V}_{0} \mathbf{V}_{1} \\
&+ \rho_{0} \mathbf{V}_{1}^{\top} \mathbf{s}_{0} \mathbf{s}_{0}^{\top} \mathbf{V}_{1} + \rho_{1} \mathbf{s}_{1} \mathbf{s}_{1}^{\top} \\
\mathbf{B}_{3} &= \mathbf{V}_{2}^{\top} \mathbf{V}_{1}^{\top} \mathbf{V}_{0}^{\top} \mathbf{B}_{0} \mathbf{V}_{0} \mathbf{V}_{1} \mathbf{V}_{2} \\
&+ \rho_{0} \mathbf{V}_{2}^{\top} \mathbf{V}_{1}^{\top} \mathbf{s}_{0} \mathbf{s}_{0}^{\top} \mathbf{V}_{1} \mathbf{V}_{2} + \rho_{1} \mathbf{V}_{2}^{\top} \mathbf{s}_{1} \mathbf{s}_{1}^{\top} \mathbf{V}_{2} + \rho_{2} \mathbf{s}_{2} \mathbf{s}_{2}^{\top} \\
\mathbf{B}_{k+1} &= (\mathbf{V}_{k}^{\top} \cdots \mathbf{V}_{0}^{\top}) \mathbf{B}_{0} (\mathbf{V}_{0} \cdots \mathbf{V}_{k}) \\
&+ \sum_{i=0}^{k} \rho_{i} (\mathbf{V}_{k}^{\top} \cdots \mathbf{V}_{i+1}^{\top}) \mathbf{s}_{i} \mathbf{s}_{i}^{\top} (\mathbf{V}_{i+1} \cdots \mathbf{V}_{k})
\end{aligned}
$$

L-BFGS引入新的假设：**遥远的信息对当前的影响很小，只保留最近的m步历史记录**。因此不再从$\mathbf{B}_{0}$开始递推，而是从当前步回溯m步的某个初始矩阵$\mathbf{B}_{k}^{0}$（一般选取为矩阵$\gamma_{k} \mathbf{I}$）开始递推：

$$
\begin{aligned}
\mathbf{B}_{k+1} &= (\mathbf{V}_{k}^{\top} \cdots \mathbf{V}_{k-m+1}^{\top}) \mathbf{B}_{k}^{0} (\mathbf{V}_{k-m+1} \cdots \mathbf{V}_{k}) \\
&+ \sum_{i=k-m+1}^{k} \rho_{i} (\mathbf{V}_{k}^{\top} \cdots \mathbf{V}_{i+1}^{\top}) \mathbf{s}_{i} \mathbf{s}_{i}^{\top} (\mathbf{V}_{i+1} \cdots \mathbf{V}_{k})
\end{aligned}
$$

使用双循环递归算法递推上式，搜索方向为$r=\mathbf{B}_{k+1} \nabla f(\mathbf{x}_{k+1})=\mathbf{B}_{k+1} \mathbf{g}_{k+1}$

* 第一层循环的计算过程为：

$$
\begin{aligned}
\mathbf{B}_{k+1} \mathbf{g}_{k+1} &= (\mathbf{V}_{k}^{\top} \cdots \mathbf{V}_{k-m+1}^{\top}) \mathbf{B}_{k}^{0} \color{red}{\mathbf{V}_{k-m+1} \cdots \mathbf{V}_{k} \mathbf{g}_{k+1}} \\
&+ \sum_{i=k-m+1}^{k} \rho_{i} (\mathbf{V}_{k}^{\top} \cdots \mathbf{V}_{i+1}^{\top}) \mathbf{s}_{i} \mathbf{s}_{i}^{\top} \color{red}{\mathbf{V}_{i+1} \cdots \mathbf{V}_{k} \mathbf{g}_{k+1}}
\end{aligned}
$$

记$\mathbf{q}_{k}=\mathbf{g}_{k+1}$，观察得到递推公式为$\mathbf{q}_{i}=\mathbf{V}_{i} \mathbf{q}_{i+1}$:

$$
\begin{aligned}
\mathbf{q}_{i} &= (\mathbf{I} -\rho_{i} \mathbf{y}_{i} \mathbf{s}_{i}^{\top}) \mathbf{q}_{i+1} \\
&= \mathbf{q}_{i+1} - \rho_{i} (\mathbf{s}_{i}^{\top} \mathbf{q}_{i+1}) \mathbf{y}_{i} \\
&= \mathbf{q}_{i+1} - {\color{red}\alpha_{i}} \mathbf{y}_{i} \\
\alpha_{i} &= \rho_{i} \mathbf{s}_{i}^{\top} \mathbf{q}_{i+1} \\
\end{aligned}
$$

**注意$\alpha_{i}$可用在第二阶段的计算中**，根据上式能从$\mathbf{q}_{k}$递推回$\mathbf{q}_{k-m+1}$，代回上式得到：

$$
\begin{aligned}
\mathbf{B}_{k+1} \mathbf{g}_{k+1} &= (\mathbf{V}_{k}^{\top} \cdots \mathbf{V}_{k-m+1}^{\top}) \mathbf{B}_{k}^{0} \color{red}{\mathbf{q}_{k-m+1}} \\
&+ \sum_{i=k-m+1}^{k} \rho_{i} (\mathbf{V}_{k}^{\top} \cdots \mathbf{V}_{i+1}^{\top}) \mathbf{s}_{i} \mathbf{s}_{i}^{\top} \color{red}{\mathbf{q}_{i+1}}
\end{aligned}
$$

* 第二层循环的计算过程为：

$$
\begin{aligned}
\mathbf{B}_{k+1} \mathbf{g}_{k+1} &= (\mathbf{V}_{k}^{\top} \cdots \mathbf{V}_{k-m+2}^{\top} \mathbf{V}_{k-m+1}^{\top}) \mathbf{B}_{k}^{0} \color{red}{\mathbf{q}_{k-m+1}} \\
&+ \rho_{k-m+1} (\mathbf{V}_{k}^{\top} \cdots \mathbf{V}_{k-m+2}^{\top}) \mathbf{s}_{k-m+1} \mathbf{s}_{k-m+1}^{\top} \color{red}{\mathbf{q}_{k-m+2}} \\
&+ \sum_{i=k-m+2}^{k} \rho_{i} (\mathbf{V}_{k}^{\top} \cdots \mathbf{V}_{i+1}^{\top}) \mathbf{s}_{i} \mathbf{s}_{i}^{\top} \color{red}{\mathbf{q}_{i+1}} \\
&= (\mathbf{V}_{k}^{\top} \cdots \mathbf{V}_{k-m+2}^{\top}) \{\mathbf{V}_{k-m+1}^{\top} \mathbf{B}_{k}^{0} \mathbf{q}_{k-m+1} + \rho_{k-m+1} \mathbf{s}_{k-m+1} \mathbf{s}_{k-m+1}^{\top} \mathbf{q}_{k-m+2}\} \\
&+ \sum_{i=k-m+2}^{k} \rho_{i} (\mathbf{V}_{k}^{\top} \cdots \mathbf{V}_{i+1}^{\top}) \mathbf{s}_{i} \mathbf{s}_{i}^{\top} \color{red}{\mathbf{q}_{i+1}} \\
\end{aligned}
$$

提取出大括号中那一项：

$$
\begin{aligned}
&\mathbf{V}_{k-m+1}^{\top} \mathbf{B}_{k}^{0} \mathbf{q}_{k-m+1} + \rho_{k-m+1} \mathbf{s}_{k-m+1} \mathbf{s}_{k-m+1}^{\top} \mathbf{q}_{k-m+2} \\
&=\mathbf{V}_{k-m+1}^{\top} (\mathbf{B}_{k}^{0} \mathbf{q}_{k-m+1}) + \mathbf{s}_{k-m+1} (\rho_{k-m+1} \mathbf{s}_{k-m+1}^{\top} \mathbf{q}_{k-m+2}) \\
&=\mathbf{V}_{k-m+1}^{\top} {\color{red}{\mathbf{r}_{k-m+1}}} + \mathbf{s}_{k-m+1} {\color{red}{\alpha_{k-m+1}}} \\
&=\mathbf{r}_{k-m+1} - \rho_{k-m+1} \mathbf{s}_{k-m+1} \mathbf{y}_{k-m+1}^{\top} \mathbf{r}_{k-m+1} + \mathbf{s}_{k-m+1} {\color{red}{\alpha_{k-m+1}}} \\
&=\mathbf{r}_{k-m+1} - \mathbf{s}_{k-m+1} (\rho_{k-m+1} \mathbf{y}_{k-m+1}^{\top} \mathbf{r}_{k-m+1}) + \mathbf{s}_{k-m+1} {\color{red}{\alpha_{k-m+1}}} \\
&=\mathbf{r}_{k-m+1} - \mathbf{s}_{k-m+1} {\color{red}\beta_{k-m+1}} + \mathbf{s}_{k-m+1} {\color{red}{\alpha_{k-m+1}}} \\
&=\mathbf{r}_{k-m+1} + \mathbf{s}_{k-m+1}(\alpha_{k-m+1} - \beta_{k-m+1}) \\
&= \color{red} \mathbf{r}_{k-m+2}
\end{aligned}
$$

代回，得到：

$$
\begin{aligned}
\mathbf{B}_{k+1} \mathbf{g}_{k+1} &= (\mathbf{V}_{k}^{\top} \cdots \mathbf{V}_{k-m+2}^{\top}) \mathbf{r}_{k-m+2} \\
&+ \sum_{i=k-m+2}^{k} \rho_{i} (\mathbf{V}_{k}^{\top} \cdots \mathbf{V}_{i+1}^{\top}) \mathbf{s}_{i} \mathbf{s}_{i}^{\top} \mathbf{q}_{i+1} \\
\end{aligned}
$$

按照上面的就算过程不断递推，最终得到：

$$
\mathbf{B}_{k+1} \mathbf{g}_{k+1} = \mathbf{r}_{k+1}
$$

* 总结L-BFGS的两步循环：

第一步循环：

> 从右向左（从新到旧）：从当前最新的第k步梯度开始，逆序回溯到最旧的k-m+1步，计算**梯度在历史位移方向上的投影系数$\alpha$**，在此过程中不断剥离$\mathbf{V}_{i}$矩阵。
> 输入：当前梯度$\mathbf{g}_{k+1}$，最近保存的m步向量对$(\mathbf{s}_{i}, \mathbf{y}_{i})$和$\rho_{i}$
> 初始化：$\mathbf{q} = \mathbf{g}_{k+1}$
> 循环：计算标量投影$\alpha_{i}=\rho_{i} \mathbf{s}_{i}^{\top} \mathbf{q}_{i+1}$，更新状态向量$\mathbf{q}_{i} = \mathbf{q}_{i+1} - \alpha_{i} \mathbf{y}_{i}$
> 输出：剥离完的残余梯度$\mathbf{q}_{k-m+1}$

第二步循环：

> 从左向右（从旧到新）上一步的残余梯度和初始矩阵$\mathbf{B}_{k}^{0}$，开始恢复海森矩阵的逆的信息。
> 初始化：$\mathbf{r}_{k-m+1}=\mathbf{B}_{k}^{0} \mathbf{q}_{k-m+1}$
> 循环：计算当前近似在梯度方向上的投影$\beta_{i}=\rho_{i} \mathbf{y}_{i}^{\top} \mathbf{r}_{i}$。利用割线残差修正并合成新的搜索方向$\mathbf{r}_{i+1}=\mathbf{r}_{i} + \mathbf{s}_{i}(\alpha_{i}-\beta_{i})$
> 输出：$\mathbf{r}_{k+1}=\mathbf{B}_{k+1} \mathbf{g}_{k+1}$

## MPPI(Model Predict Path Integral)

* **基于信息论/概率**推理的MPPI算法。传统的模型预测控制（MPC）通常需要求解复杂的非线性优化问题，依赖目标函数和系统动力学的梯度信息（**需要可微**）。而MPPI最优控制问题转化为**概率推理**（probabilistic inference）问题，通过在**GPU**上模拟成千上万轨迹，并对轨迹加权平均合成最优控制信号。
* MPPI的核心思想是：为了克服一个MPC中**非凸、不可导**的复杂地形系统扰动，不寻找一个确定性的控制输入$u$，而是寻找**一个动作的概率分布$q(u)$**，使得在这个分布下的**期望代价最低，且分布不能失去控制**。
* 定义最优分布优化问题：

$$
\underset{q(u)}{\min}\ \mathbb{E}_{u \sim q(u)} \left[S(u)\right] + \lambda D_{KL}(q(u) \| p(u)) \\
\text{s.t.} \int q(u) d u = 1
$$

第一项是目标函数的期望。输入控制变成了分布，则无法单独看某个动作的代价，而是转变为评估这个分布下，机器人动作的平均表现，因此第一项就是系统在动作分布$q(u)$下总代价$S(u)$的数学期望。
第二项中，$p(u)$是先验分布，通常设定为上一帧的最优动作序列。KL散度$D_{KL}(q \| p)$衡量新计算的控制分布$q(u)$偏离上一帧$p(u)$的距离。KL散度作为惩罚项，**确保前后两帧的控制信号平滑过度**。
逆温度参数$\lambda$的意义：当$\lambda$很大时，算法更在乎KL散度，保持分布像先验一样大，系统倾向于大范围的随机探索。当$\lambda$很小时，算法期望降低目标代价，系统会强烈地收敛到当前采样中最优的单条轨迹上。

* MPPI是带约束的最优化问题，通过拉格朗日乘子法变为无约束的最优化问题：

第一项展开数学期望的定义：

$$
\mathbb{E}_{u \sim q} \left[S(u)\right] = \int q(u) S(u) d u
$$

第二项和第三项，展开KL散度：

$$
\lambda D_{KL}(q(u)\|p(u)) = \lambda \int q(u) \ln \frac{q(u)}{p(u)} d u = \lambda \int q(u) \ln q(u) d u - \lambda \int q(u) \ln p(u) d u
$$

第四项引入拉格朗日约束，$\gamma$是拉格朗日乘子：

$$
\gamma (\int q(u) d u - 1)
$$

综合得到拉格朗日函数为：

$$
\mathcal{L}(q(u)) = \int q(u) S(u) d u + \lambda \int q(u) \ln q(u) d u - \lambda \int q(u) \ln p(u) d u + \gamma (\int q(u) d u - 1)
$$

对$q(u)$求导数$\frac{\partial \mathcal{L}}{\partial q(u)}=0$：

$$
S(u) + \lambda(\ln q(u) + 1) - \lambda \ln p(u) + \gamma = 0
$$

整理：

$$
\ln q(u) = \ln p(u) - \frac{1}{\lambda} S(u) - \frac{\lambda + \gamma}{\lambda}
$$

两边t同时取指数$\exp$：

$$
q^{\star}(u) = p(u) \cdot \exp(-\frac{1}{\lambda} S(u)) \cdot \exp(-\frac{\lambda + \gamma}{\lambda})
$$

其中最后一项与$u$无关，是常数系数，**为保证$\int q(u) du=1$充当了**归一化分母（配分函数$Z$）。

则理论最优分布（玻尔兹曼对偶分布）：

$$
q^{\star}(u) = \frac{1}{Z} p(u) \exp(-\frac{1}{\lambda} S(u)),\ Z = \int p(u) \exp(-\frac{1}{\lambda} S(u)) du
$$

* 蒙特卡洛和重要性采样：
  根据先验高斯分布$p(u) \sim \mathcal{N}(\mu, \sigma)$，抽取M个独立的**高斯白噪声扰动$\epsilon^{m}$**：

$$
\epsilon^{m} \sim \mathcal{N}(0, \sigma), m=1,2,\cdots,M
$$

则对应的第m个候选控制为：

$$
u^{m}= \mu + \epsilon^{m}
$$

为了利用最优动作概率分布$q^{\star}(u)$计算出我们最终的控制指令$u^{\star}$，需要计算其整个连续空间上的数学期望：

$$
u^{\star} = \int u q^{\star} du = \mathbb{E}_{u \sim q^{\star}} \left[u\right]
$$

理论上，通过**蒙特卡洛采样**$u \sim q^{\star}$，计算其算数平均值$\frac{1}{M} \Sigma_{m} u^{m}$，即可得到$u^{\star}$。
但是，目标函数$S(u)$包含现实世界中的各种限制（非凸，不可导等），很难采样得到满足$u \sim q^{\star}$的分布。因此需要**重要性采样**。
把已知的平滑的高斯分布$p(u)$作为采样建议分布，严格按照$p(u)$的概率分布，在中间中随机采样M个点，得到$u^{m}$。每个点$u^{m}$处，不在乎$S(u)$可导与否，直接代入算出具体的目标函数值$S(u^{m})$

$$
\begin{aligned}
u^{\star} &= \int u q^{\star}(u) du \\
&= \int u \frac{q^{\star}(u)}{p(u)} p(u) du \\
\end{aligned}
$$

根据重要性采样的定义，**修正权重（似然比）**就是$W(u)=\frac{q^{\star}(u)}{p(u)}$，得到：

$$
W(u) = \frac{\frac{1}{Z} p(u) \exp(-\frac{1}{\lambda} S(u))}{p(u)} = \frac{1}{Z} \exp(-\frac{1}{\lambda} S(u))
$$

将修正权重$W(u)$放回重要性采样公式中：

$$
\begin{aligned}
u^{\star} &= \int u W(u) p(u) du \\
&= \int u \left[\frac{1}{Z} \exp(-\frac{1}{\lambda} S(u))\right] p(u) du \\
&= \frac{\int u \exp(-\frac{1}{\lambda} S(u)) p(u) du}{\int \exp(-\frac{1}{\lambda} S(u)) p(u) du}
\end{aligned}
$$

对上式离散化，利用$\mathbb{E}_{u \sim p(u)} \left[g(u)\right]$：

$$
\begin{aligned}
u^{\star} &= \frac{\int u \exp(-\frac{1}{\lambda} S(u)) p(u) du}{\int \exp(-\frac{1}{\lambda} S(u)) p(u) du} \\
&= \frac{\mathbb{E}_{u \sim p(u)} \left[u \exp(-\frac{1}{\lambda} S(u))\right]}{\mathbb{E}_{u \sim p(u)} \left[\exp(-\frac{1}{\lambda} S(u))\right]} \\
&\approx \frac{\sum_{m=1}^{M} u^{m} \exp(\frac{1}{\lambda} S(u^{m}))}{\sum_{j=1}^{M} \exp(\frac{1}{\lambda} S(u^{j}))}
\end{aligned}
$$

这一步中，从连续积分变成离散累加，$p(u)$（物理权重）消失了，因为样本的密度本身就代表了$p(u)$的权重。

* 虽然直接将$q^{\star}(u)$表达式代入$u^{\star}=\int u q(u) du$也能得到最终表达式的结果。但是从**连续积分 -> 蒙特卡洛离散求和 -> 重要性采样 -> 离散蒙特卡罗求和**才能从概率论的角度阐述背后的物理含义。
* 推导softmax权重：

$$
\begin{aligned}
u^{\star} &\approx \frac{\sum_{m=1}^{M} u^{m} \exp(\frac{1}{\lambda} S(u^{m}))}{\sum_{j=1}^{M} \exp(\frac{1}{\lambda} S(u^{j}))} \\
&= \sum_{m=1}^{M} u^{m} \left[\frac{\exp(-\frac{1}{\lambda} S(u^{m}))}{\sum_{j=1}^{M} \exp(-\frac{1}{\lambda} S(u^{j}))} \right] \\
&= \sum_{m=1}^{j} u^{m} w(u^{m})
\end{aligned}
$$

其中系数$\lambda$用于控制平滑度，当$\lambda \to 0$（温度极低）时，$-\frac{S(u)}{\lambda}$之间的差距会被无限放大，softmax会退化为hardmax，最大给到完美路径。当$\lambda \to \infin$（温度极高）时，$-\frac{S(u)}{\lambda}$趋近于0，所有路径的权重都接近1，softmax退化为均匀分布，机器人失去分辨能力，开始乱探索。
