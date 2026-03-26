# PINN 求解 Lorenz 系统

> **声明**：这是一个 toy project，仅出于兴趣和学习目的创建。这是我尝试将物理信息神经网络（PINN）应用于混沌 Lorenz 系统的一个简单有趣的探索。我并未进行详尽的文献调研来确认是否已存在类似方法。本项目仅供参考。

## 关于本项目

这是一个用深度学习求解微分方程的轻松探索。Lorenz 系统作为混沌动力学的经典案例，对基于神经网络的求解器提出了有趣的挑战。虽然传统数值方法（如 Runge-Kutta）对于这个问题更加高效和准确，但实现 PINN 能够带来宝贵的见解：

- 神经网络如何融入物理定律
- 在混沌系统上训练的挑战
- 约束条件的创造性实现方式（硬约束 vs 软约束）

## Lorenz 系统

Lorenz 系统是一个以混沌行为著称的常微分方程组：

```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz
```

标准参数：σ=10, ρ=28, β=8/3

## 项目结构

```
PINN-for-Lorenz/
├── lorenz_pinn.py      # PINN 主程序
├── requirements.txt    # Python 依赖
├── README.md           # 英文说明
├── README_CN.md        # 中文说明（本文件）
└── results/            # 输出目录（运行后生成）
    ├── lorenz_time_series.png
    ├── lorenz_attractor_3d.png
    ├── prediction_error.png
    ├── training_history.png
    └── lorenz_pinn_model.pth
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python lorenz_pinn.py
```

## 关键技术

本实现包含多项改进混沌系统训练的技术：

### 1. 初始条件硬约束

不使用软约束（惩罚项），而是精确满足初始条件：

```python
x(t) = x₀ + (1 - exp(-t)) × NN(t)
```

这自动保证 `x(0) = x₀`。

### 2. 数据辅助训练

使用少量数值解数据引导训练，防止网络收敛到平凡解（平衡点）。

### 3. 残差网络架构

ResNet 风格的跳跃连接有助于深层网络的梯度流动。

### 损失函数

```
总损失 = λ₁ × 物理损失 + λ₂ × 数据损失
```

- **物理损失**：Lorenz 方程的残差
- **数据损失**：预测值与数值解样本的均方误差

## 结果

程序生成：

1. **时间序列图**：PINN 预测与数值解的对比
2. **3D 吸引子图**：著名的 Lorenz 吸引子
3. **预测误差图**：随时间变化的误差分析
4. **训练历史**：训练过程中的损失曲线

## 局限性

- **预测时间有限**：由于 Lorenz 系统的混沌特性，约 3 秒后预测变得不准确
- **需要数据辅助**：不是纯粹的物理信息方法
- **计算成本高**：比传统数值方法慢得多
- **超参数敏感**：结果依赖于损失权重和网络架构

## 为什么用 PINN 求 Lorenz？

老实说，对于求解 Lorenz 系统，也许传统数值方法如 RK45：
- 更快
- 更准确
- 更可靠

然而，这个项目可以作为：
- PINN 实现的学习练习
- 神经网络约束的探索
- 一个有趣的周末项目，结合物理和深度学习

## 参考文献

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686-707.

2. Lorenz, E. N. (1963). Deterministic nonperiodic flow. *Journal of Atmospheric Sciences*, 20(2), 130-141.

## 许可证

MIT License

## 作者

yunxichu

---

*本项目仅用于娱乐和教育目的。欢迎探索、修改和学习！*
