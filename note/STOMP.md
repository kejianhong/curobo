# STOMP: Stochastic Trajectory Optimization for Motion Planning

* STOMP是一种基于抽样的轨迹优化算法，不需要计算复杂的环境障碍物信息（gradient-free），通过在当前轨迹进行随机采样（加噪），并利用目标函数（cost function）对样本进行概率加权，从而迭代逼近一条既平滑又避障的（局部）最优轨迹。

假设轨迹包含N个时间步，机器人自由度为n，在任意时刻t，机器人的关节角可以表示为$\mathbf{\theta}_t \in \mathbb{R}^{n}$，将所有时间步拼接，得到一个大向量：

$$
\mathbf{\Theta} = \begin{bmatrix}
    \mathbf{\theta}_1 \\
    \mathbf{\theta}_2 \\
    \cdots \\
    \mathbf{\theta}_N \\
\end{bmatrix} \in \mathbb{R}^{N \cdot n}
$$

相当于在$N \cdot n$维的构型空间中寻找一个点，该点代表了机器人的路径信息。

* 轨迹采样（加噪）。在第k次迭代时，当前的平均轨迹为$\mathbf{\Theta}^k$，在其上叠加一个满足多元正态分布的随机噪声$\mathbf{\epsilon}_m$来生成M条候选的轨迹（样本）：

$$
\mathbf{\Theta}_m = \mathbf{\Theta}^k + \mathbf{\epsilon}_{m}, \mathbf{\epsilon}_m \sim \mathcal{N}(\mathbf{0}, \mathbf{\Sigma})
$$

其中$\mathbf{\Sigma}$是协方差矩阵，其是STOMP的精髓。

如果直接在轨迹上加上高斯白噪声（即$\mathbf{\Sigma}=\sigma \mathbf{I}$），由于其是随机的，因此轨迹会有毛刺，违背了机器人的物理可行性（kinematic constraint）。**STOMP的精髓在于将“轨迹平滑性约束”直接编码进噪声的协方差矩阵中**。以加速度为例，引入一个有限差分矩阵$\mathbf{K}$，则机器人的加速度向量可以表示为：

$$
\ddot{\mathbf{\Theta}} = \mathbf{K} \mathbf{\Theta}
$$

轨迹的平滑代价可以写成二次型：

$$
\frac{1}{2} \ddot{\mathbf{\Theta}}^{\top} \ddot{\mathbf{\Theta}} = \frac{1}{2} \mathbf{\Theta}^{\top} (\mathbf{K}^{\top} \mathbf{K}) \mathbf{\Theta} = \frac{1}{2} \mathbf{\Theta}^{\top}  \mathbf{R}\mathbf{\Theta}, \mathbf{R} = \mathbf{K}^{\top} \mathbf{K}
$$

速度的具体形式为：

$$
\dot{\theta}_t = \frac{\theta_t - \theta_{t-1}}{\Delta t}
$$

加速度的具体形式为：

$$
\ddot{\theta}_t = \frac{\dot{\theta}_t - \dot{\theta}_{t-1}}{\Delta t} = \frac{1}{\Delta t^2}(\theta_{t-1} -2 \theta_t + \theta_{t+1})
$$

对于轨迹起点和终点，一般情况下默认其为0（固定），则有：

$$
\begin{aligned}
&t=1, \ddot{\theta}_{1} = 0 \cdot \theta_{1} + 0 \cdot \theta_{2} \\
&t=2, \ddot{\theta}_{2} = 1 \cdot \theta_{1} - 2 \cdot \theta_{2} + 1 \cdot \theta_{3} \\
&t=3, \ddot{\theta}_{3} = 1 \cdot \theta_{2} - 2 \cdot \theta_{3} + 1 \cdot \theta_{4} \\
&\cdots \\

&t=N-1, \ddot{\theta}_{N-1} = 1 \cdot \theta_{N-2} - 2 \cdot \theta_{N-1} + 1 \cdot \theta_{N} \\
&t=N, \ddot{\theta} = 0
\end{aligned}
$$

则有限差分矩阵$\mathbf{K}$的具体形式为：

$$
\begin{bmatrix}
\ddot{\theta}_{1} \\ \ddot{\theta}_{2} \\ \ddot{\theta}_{3} \\ \cdots \\ \ddot{\theta}_{N-2} \\ \ddot{\theta}_{N-1} \\ \ddot{\theta}_{N} \\
\end{bmatrix} =
\begin{bmatrix}
0 & 0 & 0 & \cdots & 0 & 0 & 0 \\
1 & -2 & 1 & \cdots & 0 & 0 & 0 \\
0 & 1 & -2 & \cdots & 0 & 0 & 0 \\
\cdots & \cdots & \cdots & \ddots & \cdots & \cdots & \cdots \\
0 & 0 & 0 & \cdots & -2 & 1 & 0 \\
0 & 0 & 0 & \cdots & 1 & -2 & 1 \\
0 & 0 & 0 & \cdots & 0 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
\theta_{1} \\ \theta_{2} \\ \theta_{3} \\ \cdots \\ \theta_{N-2} \\ \theta_{N-1} \\ \theta_{N}
\end{bmatrix}
$$

事实上对于三点法（three point）表达加速度，$\mathbf{K}$矩阵是一个三角矩阵。计算$\mathbf{R}$：

$$
\mathbf{R} = \begin{bmatrix}
1 & -2 & 1 & \cdots & \cdots & \cdots & 0 & 0 & 0 \\
-2 & 5 & 4 & 1 & \cdots & \cdots & 0 & 0 & 0 \\
1 & -4 & 6 & -4 & 1 & \cdots & 0 & 0 & 0 \\
\cdots & \cdots & \cdots & \ddots & \ddots & \ddots & \cdots & \cdots & \cdots \\
0 & 0 & 0 & \cdots & 1 & -4 & 6 & -4 & 1 \\
0 & 0 & 0 & \cdots & \cdots & 1 & -4 & 5 & -2 \\
0 & 0 & 0 & \cdots & \cdots & \cdots & 1 & -2 & 1
\end{bmatrix}
$$

其是一个五对角矩阵，内部典型的核心行为：

$$
\begin{bmatrix}
\cdots & 0 & 1 & -4 & 6 & -4 & 1 & 0 & \cdots
\end{bmatrix}
$$

在数值分析中，上式恰好是**四阶有限差分（加加速度的导数）**的系数，因此最小化加速度的平方和，在能量优化和物理效果上，等价于在轨迹上施加一个四阶阻尼器。以二阶差分（加速度）为例，当输入为低频（极其平缓）时，比如输入为一条直线$(1,2,3)$，则输出为$1-4+3=0$，差分把低频信号给抹杀掉了。当输入为高频信号（剧烈毛刺）时，比如输入出现跳变$(0,10,0)$，则输出为$0-20+0=-20$，差分把突变信号成倍放大了。因此$\mathbf{R}$矩阵可以看成一个高通滤波器，则其逆矩阵$\mathbf{\Sigma}=\mathbf{R}^{-1}$可以看成一个低通滤波器。

另一个角度，上式可以一个“滑动卷积”，亦或是“移动加权平均过滤器”。

* 上述推导加速度采用的是三点法，也即通过$\theta_{t-1}$，$\theta_{t}$和$\theta_{t+1}$推导得到加速度$\ddot{\theta}$，还存在使用五点法和七点法等计算加速度。

* 在计算机中，无法直接生成复杂的噪声，只能生成独立的高斯噪声$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$，利用高斯噪声的线性变换性质，假设$\mathbf{\epsilon}=\mathbf{L} \mathbf{z}$，则$\mathbf{\epsilon}$的协方差矩阵为$\mathbf{\Sigma}_{\mathbf{\epsilon}}=\mathbf{L} \mathbf{\Sigma}_{\mathbf{z}} \mathbf{L}^{\top} = \mathbf{L} \mathbf{L}^{\top}$。则目标就是寻找矩阵$\mathbf{L}$，然后通过$\mathbf{\epsilon} = \mathbf{L} \mathbf{z}$就能凑个标准的高斯分布计算得到STOMP要求的平滑噪声。由于协方差矩阵$\mathbf{\Sigma}$是对称正定矩阵，则必然可以通过Cholesky分解$\mathbf{L} \mathbf{L}^{\top} = \mathbf{\Sigma}_{\mathbf{\epsilon}} = \mathbf{R}^{-1}$，计算得到$\mathbf{L}$。

* [STOMP示例](./script/STOMP.py)
