# DeepGPR 离散正演、伴随与模型梯度

本文档说明程序实际执行的离散方程。梯度必须对这些离散方程求导，不能用连续公式的时间积分直接替代。

## 1. `tide-GPR` 的 2D TM 更新

令绝对介电常数 `epsilon = epsilon0 * epsilon_r`，绝对磁导率 `mu = mu0 * mu_r`，则

\[
c_a=\frac{2\epsilon-\sigma\Delta t}{2\epsilon+\sigma\Delta t},\qquad
c_b=\frac{2\Delta t}{2\epsilon+\sigma\Delta t},\qquad
c_q=\frac{\Delta t}{\mu}.
\]

以 `Ey, Hx, Hz` 表示 TM 场，正演一步为

\[
H_x^{n+1/2}=H_x^{n-1/2}-c_qD_z^+E_y^n,
\]

\[
H_z^{n+1/2}=H_z^{n-1/2}+c_qD_x^+E_y^n,
\]

\[
E_y^{n+1}=c_aE_y^n+c_b(D_x^-H_z^{n+1/2}-D_z^-H_x^{n+1/2})+f^n.
\]

`tide-GPR` 在 Python 中先用源点的 `cb` 缩放 `f`，原生反传返回 `grad_ca` 和 `grad_cb`，再由 PyTorch 把它们映射到 `epsilon_r` 和 `sigma`。其离散系数梯度为

\[
\frac{\partial J}{\partial c_a}=\sum_n\lambda_E^{n+1}E^n,\qquad
\frac{\partial J}{\partial c_b}=\sum_n\lambda_E^{n+1}\operatorname{curl}H^{n+1/2}.
\]

这正是本次修复采用的核心思路。DeepGPR 没有在 Python 中预乘源系数，因此改为保存包含源项和 CPML 项的完整离散右端项。

## 2. DeepGPR 的 3D 正演

DeepGPR 中 `ca = uE0`、`cb = uE4`、`uE1 = cb / dx`、`cq = uH4`、`uH1 = cq / dx`。不写 CPML 时：

\[
\begin{aligned}
H_x'&=H_x+c_q(D_z^+E_y-D_y^+E_z),\\
H_y'&=H_y+c_q(D_x^+E_z-D_z^+E_x),\\
H_z'&=H_z+c_q(D_y^+E_x-D_x^+E_y),
\end{aligned}
\]

\[
\begin{aligned}
E_x'&=c_aE_x+c_b(D_y^-H_z'-D_z^-H_y'),\\
E_y'&=c_aE_y+c_b(D_z^-H_x'-D_x^-H_z'),\\
E_z'&=c_aE_z+c_b(D_x^-H_y'-D_y^-H_x').
\end{aligned}
\]

电偶极源沿所选分量注入：

\[
E_p^{n+1}\mathrel{-}=c_b\frac{s^n}{\Delta x^2}.
\]

接收数据在完成场更新和源注入后采样，因此反传时首先在对应时刻、位置和分量注入数据梯度。

## 3. 2/4/8 阶空间差分

交错网格差分写成

\[
D^-f_i=\frac{1}{\Delta x}\sum_{r=1}^{R}a_r(f_{i+r-1}-f_{i-r}),
\]

\[
D^+f_i=\frac{1}{\Delta x}\sum_{r=1}^{R}a_r(f_{i+r}-f_{i-r+1}).
\]

系数为：

| 精度 | `R` | `a_r` |
|---|---:|---|
| 2 阶 | 1 | `1` |
| 4 阶 | 2 | `9/8, -1/24` |
| 8 阶 | 4 | `1225/1024, -245/3072, 49/5120, -5/7168` |

边界附近会自动缩小模板半径。伴随传播使用同一个局部模板的显式转置 `D^-T`、`D^+T`，而不是再次调用正演差分。

## 4. CPML 及其精确转置

每个 CPML 导数项可写为

\[
F' = F+s\,c\left[(R_A-1)d+R_B\phi\right],\qquad
\phi'=R_E\phi-R_Fd,
\]

其中 `s` 为该旋度项的正负号，`c` 为 `cb` 或 `cq`。给定 `lambda_F'` 和 `lambda_phi'`，精确转置为

\[
\lambda_d=s\,c(R_A-1)\lambda_F'-R_F\lambda_\phi',
\]

\[
\lambda_\phi=s\,cR_B\lambda_F'+R_E\lambda_\phi'.
\]

随后通过对应的 `D^T` 把 `lambda_d` 散射回产生该导数的场。CPU 和 CUDA 使用相同公式。

## 5. 离散伴随时间顺序

正演一步的顺序是：

1. 保存 `E^n`。
2. 更新 H，再更新 H-CPML。
3. 更新 E，再更新 E-CPML。
4. 注入源，形成 `E^{n+1}`。
5. 保存接收数据和右端项 `R^n`。

其中

\[
R^n=\frac{E^{n+1}-c_aE^n}{c_b}.
\]

它包含旋度、CPML 和源注入。反演从最后一个时间步开始，严格按相反顺序执行：

1. 注入接收数据梯度。
2. 累积 `grad_ca`、`grad_cb`。
3. 执行 E-CPML 转置和 E 基础更新转置。
4. 执行 H-CPML 转置和 H 基础更新转置。

当 `model_gradient_sampling_interval=S>1` 时，只每隔 `S` 步成像并乘 `S`，这是节省显存的时间积分近似；严格梯度检查必须使用 `S=1`。

## 6. 从离散系数映射到模型参数

对每个电场分量和炮次累积

\[
g_{c_a}=\sum_n\lambda_E^{n+1}E^n,\qquad
g_{c_b}=\sum_n\lambda_E^{n+1}R^n.
\]

链式法则为

\[
\frac{\partial c_a}{\partial\epsilon_r}
=\epsilon_0\frac{(1-c_a)c_b}{\Delta t},\qquad
\frac{\partial c_b}{\partial\epsilon_r}
=-\epsilon_0\frac{c_b^2}{\Delta t},
\]

\[
\frac{\partial c_a}{\partial\sigma}
=-\frac{1}{2}(1+c_a)c_b,\qquad
\frac{\partial c_b}{\partial\sigma}
=-\frac{1}{2}c_b^2.
\]

最终

\[
g_{\epsilon_r}=g_{c_a}\frac{\partial c_a}{\partial\epsilon_r}
+g_{c_b}\frac{\partial c_b}{\partial\epsilon_r},
\]

\[
g_\sigma=g_{c_a}\frac{\partial c_a}{\partial\sigma}
+g_{c_b}\frac{\partial c_b}{\partial\sigma}.
\]

旧实现中的 `Delta E / Delta t` 与 `E * Delta t` 成像条件既不是上述离散链式法则，也没有包含源点 `cb` 导数，因此会产生错误的大小，部分模型中还会产生错误方向。

## 7. 验证准则

验证 notebook 使用随机方向 `v` 比较

\[
g^Tv
\quad\text{和}\quad
\frac{J(m+hv)-J(m-hv)}{2h}.
\]

float32 下过小的 `h` 会发生相消，过大的 `h` 会包含非线性误差。因此应观察多个 `h`，确认中间区间误差下降且符号一致。论文计算前建议至少满足：`epsilon_r` 与 `sigma` 的最佳相对误差均小于 `0.15`，并且 `S=1`。

当前 CPML 剖面系数由边界参考介质生成，并在一次原生正演/反传中作为固定系数处理；模型反演和梯度检查都不应更新 CPML 区域。示例 notebook 因此只在 CPML 内侧构造方向扰动。
