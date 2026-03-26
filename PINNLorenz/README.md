# PINN for Lorenz System

> **Disclaimer**: This is a toy project created for fun and learning purposes. It represents a simple and interesting attempt to apply Physics-Informed Neural Networks (PINN) to the chaotic Lorenz system. I have not conducted a thorough literature review to check whether similar approaches already exist. This project is for reference only.

## About This Project

This is a casual exploration of using deep learning to solve differential equations. The Lorenz system, being a classic example of chaotic dynamics, presents an interesting challenge for neural network-based solvers. While traditional numerical methods (like Runge-Kutta) are far more efficient and accurate for this problem, implementing PINN provides valuable insights into:

- How neural networks can incorporate physical laws
- The challenges of training on chaotic systems
- Creative ways to enforce constraints (hard constraints vs soft constraints)

## Lorenz System

The Lorenz system is a system of ordinary differential equations known for its chaotic behavior:

```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz
```

Standard parameters: σ=10, ρ=28, β=8/3

## Project Structure

```
PINN-for-Lorenz/
├── lorenz_pinn.py      # Main PINN implementation
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── results/            # Output directory (created after running)
    ├── lorenz_time_series.png
    ├── lorenz_attractor_3d.png
    ├── prediction_error.png
    ├── training_history.png
    └── lorenz_pinn_model.pth
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python lorenz_pinn.py
```

## Key Techniques Used

This implementation includes several techniques to improve training on chaotic systems:

### 1. Hard Constraint for Initial Conditions

Instead of using soft constraints (penalty terms), the initial conditions are enforced exactly:

```python
x(t) = x₀ + (1 - exp(-t)) × NN(t)
```

This guarantees `x(0) = x₀` automatically.

### 2. Data-Assisted Training

A small amount of numerical solution data is used to guide the training, preventing the network from converging to trivial solutions (equilibrium points).

### 3. Residual Network Architecture

ResNet-style skip connections help with gradient flow in deeper networks.

### Loss Function

```
Total Loss = λ₁ × Physics Loss + λ₂ × Data Loss
```

- **Physics Loss**: Residual of the Lorenz equations
- **Data Loss**: MSE between predictions and numerical solution samples

## Results

The program generates:

1. **Time Series Plot**: Comparison of PINN prediction vs numerical solution
2. **3D Attractor Plot**: The famous Lorenz attractor
3. **Prediction Error**: Error analysis over time
4. **Training History**: Loss curves during training

## Limitations

- **Short prediction horizon**: Due to the chaotic nature of the Lorenz system, predictions become inaccurate after ~3 seconds
- **Requires data assistance**: Not a purely physics-informed approach
- **Computational cost**: Much slower than traditional numerical methods
- **Hyperparameter sensitivity**: Results depend on loss weights and network architecture

## Why PINN for Lorenz?

Honestly, for solving the Lorenz system, traditional numerical methods like RK45 are maybe:
- Faster
- More accurate
- More reliable

However, this project serves as:
- A learning exercise for PINN implementation
- An exploration of neural network constraints
- A fun weekend project combining physics and deep learning

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686-707.

2. Lorenz, E. N. (1963). Deterministic nonperiodic flow. *Journal of Atmospheric Sciences*, 20(2), 130-141.

## License

MIT License

## Author

yunxichu

---

*This project was created for fun and educational purposes. Feel free to explore, modify, and learn from it!*
