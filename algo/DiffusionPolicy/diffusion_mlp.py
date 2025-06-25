#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Diffusion Policy implementation for MLP state representation with Advantage-guided learning
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from algo.DiffusionPolicy.net_diffusion import DiffusionUNet, DiffusionMLP
from algo.DiffusionPolicy.diffusion_utils import NoiseScheduler
from config import opt
from typing import Optional, Dict


class ValueNetwork(nn.Module):
    """Value network for computing state values and advantages"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # State value function V(s)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # State-action value function Q(s,a)
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action=None):
        state_features = self.state_encoder(state)
        
        # State value
        value = self.value_head(state_features)
        
        if action is not None:
            # State-action value
            q_input = torch.cat([state_features, action], dim=-1)
            q_value = self.q_head(q_input)
            return value, q_value
        
        return value


class DiffusionPolicy_MLP:
    """
    Diffusion Policy algorithm for robotic manipulation with Advantage-guided learning
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_bound: float,
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        num_diffusion_steps: int = 100,
        num_inference_steps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",
        ema_decay: float = 0.995,
        clip_sample: bool = True,
        network_type: str = "unet",  # "unet" or "mlp"
        device: str = "cpu",
        horizon_steps: int = 16,  # 预测时域长度
        action_horizon: int = 8,   # 实际执行动作数量
        advantage_weight: float = 1.0,  # 优势函数权重
        gamma: float = 0.99,  # 折扣因子
        tau: float = 0.005   # 目标网络软更新参数
    ):
        """
        Initialize Diffusion Policy with Advantage guidance
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_bound: Bound for action values
            hidden_dim: Hidden dimension for networks
            actor_lr: Learning rate for diffusion model
            critic_lr: Learning rate for value networks
            num_diffusion_steps: Number of diffusion steps for training
            num_inference_steps: Number of denoising steps for inference
            beta_schedule: Beta schedule type for noise scheduler
            prediction_type: Whether to predict "epsilon" (noise) or "sample" (clean action)
            ema_decay: Exponential moving average decay for target network
            clip_sample: Whether to clip predicted samples
            network_type: Type of network architecture ("unet" or "mlp")
            device: Device to run on
            advantage_weight: Weight for advantage function guidance
            gamma: Discount factor for advantage computation
            tau: Soft update parameter for target networks
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.device = torch.device(device)
        self.num_inference_steps = num_inference_steps
        self.ema_decay = ema_decay
        self.advantage_weight = advantage_weight
        self.gamma = gamma
        self.tau = tau
        
        # 滚动时域优化参数
        self.horizon_steps = horizon_steps
        self.action_horizon = action_horizon
        self.current_action_sequence = None  # 存储当前的动作序列
        
        # Initialize noise scheduler
        self.noise_scheduler = NoiseScheduler(
            num_diffusion_steps=num_diffusion_steps,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
            prediction_type=prediction_type
        )
        
        # Initialize diffusion model
        # 动作序列的总维度 = 单步动作维度 * 时域长度
        action_sequence_dim = action_dim * horizon_steps
        
        if network_type == "unet":
            self.model = DiffusionUNet(
                state_dim=state_dim,
                action_dim=action_sequence_dim,
                hidden_dim=hidden_dim,
                prediction_type=prediction_type
            ).to(self.device)
        else:  # mlp
            self.model = DiffusionMLP(
                state_dim=state_dim,
                action_dim=action_sequence_dim,
                hidden_dim=hidden_dim,
                prediction_type=prediction_type
            ).to(self.device)
            
        # Initialize value networks for advantage computation
        self.value_network = ValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,  # 单步动作维度
            hidden_dim=hidden_dim
        ).to(self.device)
        
        self.target_value_network = copy.deepcopy(self.value_network).to(self.device)
        self.target_value_network.requires_grad_(False)
            
        # EMA model for stable inference
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        
        # Optimizers
        self.diffusion_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=actor_lr,
            weight_decay=1e-6
        )
        
        self.value_optimizer = torch.optim.AdamW(
            self.value_network.parameters(),
            lr=critic_lr,
            weight_decay=1e-6
        )
        
        # Action normalization parameters (will be updated during training)
        self.register_buffer('action_mean', torch.zeros(action_sequence_dim))
        self.register_buffer('action_std', torch.ones(action_sequence_dim))
        
        self.training_steps = 0
        
    def register_buffer(self, name, tensor):
        """Register a buffer that will be part of model state"""
        setattr(self, name, tensor.to(self.device))
        
    def normalize_action(self, action):
        """Normalize action to [-1, 1]"""
        return (action - self.action_mean) / (self.action_std + 1e-8)
        
    def denormalize_action(self, action):
        """Denormalize action from [-1, 1] to original scale"""
        return action * (self.action_std + 1e-8) + self.action_mean
        
    def update_action_stats(self, actions):
        """Update action normalization statistics"""
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        self.action_mean = actions.mean(dim=0)
        self.action_std = actions.std(dim=0)
        
    @torch.no_grad()
    def update_ema(self):
        """Update EMA model parameters"""
        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
            
    @torch.no_grad()
    def soft_update_target(self):
        """Soft update target value network"""
        for param, target_param in zip(self.value_network.parameters(), self.target_value_network.parameters()):
            target_param.data.mul_(1 - self.tau).add_(param.data, alpha=self.tau)
            
    def compute_advantage(self, states, actions, rewards, next_states, dones):
        """
        Compute advantage function: A(s,a) = Q(s,a) - V(s)
        
        Args:
            states: Current states
            actions: Actions taken
            rewards: Immediate rewards
            next_states: Next states
            dones: Done flags
            
        Returns:
            advantages: Computed advantages
        """
        with torch.no_grad():
            # Compute target Q-values using Bellman equation
            next_values = self.target_value_network(next_states)
            target_q_values = rewards + (1 - dones) * self.gamma * next_values
            
        # Compute current state values and Q-values
        current_values, current_q_values = self.value_network(states, actions)
        
        # Advantage = Q(s,a) - V(s)
        advantages = target_q_values - current_values
        
        return advantages, current_values, current_q_values, target_q_values
            
    def take_action(self, state: np.ndarray, use_ema: bool = True, replan: bool = True) -> np.ndarray:
        """
        Generate action using diffusion process with receding horizon
        
        Args:
            state: Current state observation
            use_ema: Whether to use EMA model for inference
            replan: Whether to replan the action sequence
            
        Returns:
            Single action to execute
        """
        # 如果需要重新规划或者没有当前动作序列
        if replan or self.current_action_sequence is None or len(self.current_action_sequence) == 0:
            self.current_action_sequence = self._generate_action_sequence(state, use_ema)
        
        # 取出第一个动作执行
        current_action = self.current_action_sequence[0]
        
        # 移除已执行的动作，为下次调用做准备
        self.current_action_sequence = self.current_action_sequence[1:]
        
        return current_action
    
    def _generate_action_sequence(self, state: np.ndarray, use_ema: bool = True) -> np.ndarray:
        """
        Generate action sequence using diffusion process
        
        Args:
            state: Current state observation
            use_ema: Whether to use EMA model for inference
            
        Returns:
            Predicted action sequence [action_horizon, action_dim]
        """
        model = self.ema_model if use_ema else self.model
        model.eval()
        
        with torch.no_grad():
            # Convert state to tensor
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Start from random noise for the entire action sequence
            action_sequence_dim = self.action_dim * self.horizon_steps
            noisy_action_sequence = torch.randn((1, action_sequence_dim), device=self.device)
            
            # Denoising loop
            for i, t in enumerate(reversed(range(self.num_inference_steps))):
                # Create timestep tensor
                timesteps = torch.full((1,), t, device=self.device, dtype=torch.long)
                
                # Predict noise
                noise_pred = model(noisy_action_sequence, timesteps, state)
                
                # Denoise step
                noisy_action_sequence = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=noisy_action_sequence
                )
                
            # Denormalize and clip action sequence
            action_sequence = self.denormalize_action(noisy_action_sequence)
            
            # Reshape to [horizon_steps, action_dim] and take only action_horizon steps
            action_sequence = action_sequence.view(self.horizon_steps, self.action_dim)
            action_sequence = action_sequence[:self.action_horizon]  # 只执行前action_horizon步
            
            # Clip actions
            action_sequence = torch.clamp(action_sequence, -self.action_bound, self.action_bound)
            
        return action_sequence.cpu().numpy()
        
    def train(self, transition_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Train diffusion model with advantage guidance
        
        Args:
            transition_dict: Dictionary containing training data
            
        Returns:
            Dictionary containing training losses
        """
        self.model.train()
        self.value_network.train()
        
        # Extract data
        states = torch.tensor(transition_dict['states'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1, 1).to(self.device)
        
        # Convert single actions to action sequences by repeating
        # For training, we repeat the current action for the entire horizon
        batch_size = actions.shape[0]
        action_sequences = actions.unsqueeze(1).repeat(1, self.horizon_steps, 1)  # [B, H, A]
        action_sequences = action_sequences.reshape(batch_size, -1)  # [B, H*A]
        
        # Update action statistics if needed
        if self.training_steps % 1000 == 0:
            self.update_action_stats(action_sequences.cpu().numpy())
            
        # === Train Value Network ===
        advantages, current_values, current_q_values, target_q_values = self.compute_advantage(
            states, actions, rewards, next_states, dones
        )
        
        # Value loss (MSE between current V(s) and target Q-values)
        value_loss = F.mse_loss(current_values, target_q_values.detach())
        
        # Q-value loss (MSE between current Q(s,a) and target Q-values)
        q_loss = F.mse_loss(current_q_values, target_q_values.detach())
        
        # Total value loss
        total_value_loss = value_loss + q_loss
        
        # Update value network
        self.value_optimizer.zero_grad()
        total_value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1.0)
        self.value_optimizer.step()
        
        # === Train Diffusion Model with Advantage Guidance ===
        # Normalize actions
        action_sequences = self.normalize_action(action_sequences)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.num_diffusion_steps, (batch_size,),
            device=self.device, dtype=torch.long
        )
        
        # Add noise to action sequences
        noise = torch.randn_like(action_sequences)
        noisy_action_sequences = self.noise_scheduler.add_noise(action_sequences, noise, timesteps)
        
        # Predict noise
        noise_pred = self.model(noisy_action_sequences, timesteps, states)
        
        # Calculate base diffusion loss
        if self.noise_scheduler.prediction_type == "epsilon":
            target = noise
        else:  # prediction_type == "sample"
            target = action_sequences
            
        base_diffusion_loss = F.mse_loss(noise_pred, target, reduction='none')
        
        # Weight the loss by advantages (detached to avoid gradient flow)
        # Positive advantages -> lower loss weight (good actions)
        # Negative advantages -> higher loss weight (bad actions)
        advantage_weights = torch.exp(-self.advantage_weight * advantages.detach())
        
        # Apply advantage weighting to diffusion loss
        # Reshape advantage weights to match diffusion loss shape
        advantage_weights = advantage_weights.expand_as(base_diffusion_loss)
        weighted_diffusion_loss = (base_diffusion_loss * advantage_weights).mean()
        
        # Update diffusion model
        self.diffusion_optimizer.zero_grad()
        weighted_diffusion_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.diffusion_optimizer.step()
        
        # Update EMA and target networks
        self.update_ema()
        self.soft_update_target()
        
        self.training_steps += 1
        
        return {
            'diffusion_loss': weighted_diffusion_loss.item(),
            'value_loss': value_loss.item(),
            'q_loss': q_loss.item(),
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item()
        }
        
    def save(self, filename: str):
        """Save model checkpoints"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'value_network_state_dict': self.value_network.state_dict(),
            'target_value_network_state_dict': self.target_value_network.state_dict(),
            'diffusion_optimizer_state_dict': self.diffusion_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'action_mean': self.action_mean,
            'action_std': self.action_std,
            'training_steps': self.training_steps
        }, filename + "_diffusion_policy_advantage.pt")
        
    def load(self, filename: str):
        """Load model checkpoints"""
        checkpoint = torch.load(filename + "_diffusion_policy_advantage.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
        self.target_value_network.load_state_dict(checkpoint['target_value_network_state_dict'])
        self.diffusion_optimizer.load_state_dict(checkpoint['diffusion_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.action_mean = checkpoint['action_mean']
        self.action_std = checkpoint['action_std']
        self.training_steps = checkpoint['training_steps'] 