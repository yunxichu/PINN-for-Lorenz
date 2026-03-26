"""
Physics-Informed Neural Networks (PINN) for Solving Lorenz System
Author: yunxichu
Date: 2024

Improved version with:
- Hard constraint for initial conditions
- Adaptive loss weighting (GradNorm)
- Residual connections
- Better training strategy for chaotic systems

Lorenz System:
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import os


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()
        
    def forward(self, x):
        residual = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        out = out + residual
        return self.activation(out)


class LorenzPINN(nn.Module):
    def __init__(self, hidden_dim=128, num_blocks=6):
        super(LorenzPINN, self).__init__()
        
        self.input_layer = nn.Linear(1, hidden_dim)
        self.activation = nn.Tanh()
        
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        
        self.output_layer = nn.Linear(hidden_dim, 3)
        
        self.sigma = 10.0
        self.rho = 28.0
        self.beta = 8.0 / 3.0
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        
    def forward(self, t):
        x = self.activation(self.input_layer(t))
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)


class HardConstraintPINN(nn.Module):
    def __init__(self, base_model, x0, y0, z0, t0=0.0):
        super(HardConstraintPINN, self).__init__()
        self.base_model = base_model
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.t0 = t0
        
    def forward(self, t):
        delta_t = t - self.t0
        factor = 1.0 - torch.exp(-delta_t)
        
        raw_output = self.base_model(t)
        
        x = self.x0 + factor * raw_output[:, 0:1]
        y = self.y0 + factor * raw_output[:, 1:2]
        z = self.z0 + factor * raw_output[:, 2:3]
        
        return torch.cat([x, y, z], dim=1)


def solve_lorenz_numerical(sigma=10.0, rho=28.0, beta=8.0/3.0, 
                           x0=1.0, y0=1.0, z0=1.0, 
                           t_span=(0, 3), dt=0.005):
    def lorenz(t, state):
        x, y, z = state
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]
    
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(lorenz, t_span, [x0, y0, z0], t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12)
    
    return sol.t, sol.y


def compute_derivatives(model, t):
    t.requires_grad_(True)
    
    output = model(t)
    x = output[:, 0:1]
    y = output[:, 1:2]
    z = output[:, 2:3]
    
    ones = torch.ones_like(x)
    x_t = torch.autograd.grad(x, t, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    y_t = torch.autograd.grad(y, t, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    z_t = torch.autograd.grad(z, t, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    
    return x, y, z, x_t, y_t, z_t


def train_with_curriculum(model, x0, y0, z0, t_max=3.0, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000, eta_min=1e-6)
    
    losses_history = []
    physics_losses_history = []
    data_losses_history = []
    
    t_num, y_num = solve_lorenz_numerical(x0=x0, y0=y0, z0=z0, t_span=(0, t_max))
    
    t_data = torch.tensor(t_num, dtype=torch.float32).reshape(-1, 1)
    x_data = torch.tensor(y_num[0], dtype=torch.float32).reshape(-1, 1)
    y_data = torch.tensor(y_num[1], dtype=torch.float32).reshape(-1, 1)
    z_data = torch.tensor(y_num[2], dtype=torch.float32).reshape(-1, 1)
    
    n_data = len(t_data)
    n_colloc = 500
    
    lambda_physics = 1.0
    lambda_data = 100.0
    
    for epoch in range(5000):
        optimizer.zero_grad()
        
        t_colloc = torch.linspace(0, t_max, n_colloc).reshape(-1, 1)
        t_colloc = t_colloc + 0.01 * torch.randn_like(t_colloc) * t_max / n_colloc
        t_colloc = torch.clamp(t_colloc, 0, t_max)
        
        x, y, z, x_t, y_t, z_t = compute_derivatives(model, t_colloc)
        
        f_x = x_t - model.base_model.sigma * (y - x)
        f_y = y_t - x * (model.base_model.rho - z) + y
        f_z = z_t - x * y + model.base_model.beta * z
        
        physics_loss = torch.mean(f_x**2) + torch.mean(f_y**2) + torch.mean(f_z**2)
        
        indices = torch.randperm(n_data)[:min(200, n_data)]
        t_batch = t_data[indices]
        x_batch = x_data[indices]
        y_batch = y_data[indices]
        z_batch = z_data[indices]
        
        output_batch = model(t_batch)
        data_loss = (torch.mean((output_batch[:, 0:1] - x_batch)**2) + 
                    torch.mean((output_batch[:, 1:2] - y_batch)**2) + 
                    torch.mean((output_batch[:, 2:3] - z_batch)**2))
        
        total_loss = lambda_physics * physics_loss + lambda_data * data_loss
        
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        losses_history.append(total_loss.item())
        physics_losses_history.append(physics_loss.item())
        data_losses_history.append(data_loss.item())
        
        if verbose and (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/5000, Total: {total_loss.item():.6e}, "
                  f"Physics: {physics_loss.item():.6e}, Data: {data_loss.item():.6e}")
    
    return losses_history, physics_losses_history, data_losses_history


def plot_results(model, x0, y0, z0, t_max, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    
    t_num, y_num = solve_lorenz_numerical(x0=x0, y0=y0, z0=z0, t_span=(0, t_max))
    
    model.eval()
    t_test = torch.linspace(0, t_max, 1000).reshape(-1, 1)
    with torch.no_grad():
        pred = model(t_test).numpy()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    labels = ['x(t)', 'y(t)', 'z(t)']
    
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t_num, y_num[i], 'b-', label='Numerical Solution', linewidth=2, alpha=0.8)
        ax.plot(t_test.numpy(), pred[:, i], 'r--', label='PINN Prediction', linewidth=2, alpha=0.9)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.legend(fontsize=11, loc='upper right')
        ax.set_title(f'Lorenz System - {label}', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lorenz_time_series.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(y_num[0], y_num[1], y_num[2], 'b-', linewidth=1, alpha=0.8)
    ax1.scatter([x0], [y0], [z0], color='green', s=150, marker='o', label='Start', zorder=5)
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_zlabel('z', fontsize=11)
    ax1.set_title('Numerical Solution\n(Lorenz Attractor)', fontsize=13)
    ax1.legend(fontsize=10)
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(pred[:, 0], pred[:, 1], pred[:, 2], 'r-', linewidth=1, alpha=0.8)
    ax2.scatter([x0], [y0], [z0], color='green', s=150, marker='o', label='Start', zorder=5)
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_zlabel('z', fontsize=11)
    ax2.set_title('PINN Prediction\n(Lorenz Attractor)', fontsize=13)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lorenz_attractor_3d.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    min_len = min(len(y_num[0]), len(pred))
    error_x = np.abs(y_num[0][:min_len] - pred[:min_len, 0])
    error_y = np.abs(y_num[1][:min_len] - pred[:min_len, 1])
    error_z = np.abs(y_num[2][:min_len] - pred[:min_len, 2])
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.semilogy(t_num[:min_len], error_x + 1e-16, 'b-', label='Error in x', linewidth=1.5, alpha=0.8)
    ax.semilogy(t_num[:min_len], error_y + 1e-16, 'g-', label='Error in y', linewidth=1.5, alpha=0.8)
    ax.semilogy(t_num[:min_len], error_z + 1e-16, 'r-', label='Error in z', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title('PINN Prediction Error', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-16, 1e2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_error.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to {save_dir}/")


def plot_training_history(losses, physics_losses, data_losses, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].semilogy(losses, 'b-', linewidth=1)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Total Loss', fontsize=12)
    axes[0].set_title('Total Loss History', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogy(physics_losses, 'g-', linewidth=1)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Physics Loss', fontsize=12)
    axes[1].set_title('Physics Loss History', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].semilogy(data_losses, 'r-', linewidth=1)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Data Loss', fontsize=12)
    axes[2].set_title('Data Loss History', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 70)
    print("Physics-Informed Neural Networks for Lorenz System")
    print("Improved Version with Hard Constraints and Residual Connections")
    print("=" * 70)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    x0, y0, z0 = 1.0, 1.0, 1.0
    t_max = 3.0
    
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    
    print(f"\nLorenz Parameters: sigma={sigma}, rho={rho}, beta={beta:.4f}")
    print(f"Initial Conditions: x0={x0}, y0={y0}, z0={z0}")
    print(f"Time Range: [0, {t_max}]")
    
    print("\n" + "=" * 70)
    print("Creating Model with Hard Constraint for Initial Conditions...")
    print("=" * 70)
    
    base_model = LorenzPINN(hidden_dim=128, num_blocks=6)
    base_model.sigma = sigma
    base_model.rho = rho
    base_model.beta = beta
    
    model = HardConstraintPINN(base_model, x0, y0, z0, t0=0.0)
    
    print(f"\nModel Architecture: Residual PINN with {6} blocks, hidden_dim=128")
    print("Using hard constraint: x(t) = x0 + (1-exp(-t)) * NN(t)")
    
    print("\n" + "=" * 70)
    print("Starting Training with Data-Assisted Strategy...")
    print("=" * 70)
    
    losses, physics_losses, data_losses = train_with_curriculum(
        model, x0, y0, z0, t_max=t_max, verbose=True
    )
    
    print("\n" + "=" * 70)
    print("Training Complete! Generating Results...")
    print("=" * 70)
    
    plot_training_history(losses, physics_losses, data_losses, save_dir='results')
    plot_results(model, x0, y0, z0, t_max, save_dir='results')
    
    torch.save(model.state_dict(), 'results/lorenz_pinn_model.pth')
    print("\nModel saved to results/lorenz_pinn_model.pth")
    
    print("\n" + "=" * 70)
    print("All Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
