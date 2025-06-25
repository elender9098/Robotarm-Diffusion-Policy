#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Diffusion Policy implementation for CNN state representation with Advantage-guided learning
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from algo.DiffusionPolicy.net_diffusion import DiffusionUNet, DiffusionMLP
from algo.DiffusionPolicy.diffusion_utils import NoiseScheduler
from config import opt
from typing import Optional, Dict, Tuple


class CNNEncoder(nn.Module):
    """CNN encoder for image observations"""
    def __init__(self, input_shape: Tuple[int, int, int], output_dim: int = 256):
        super().__init__()
        c, h, w = input_shape
        
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate output dimension after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            conv_output_dim = self.conv(dummy_input).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(conv_output_dim, output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class CNNValueNetwork(nn.Module):
    """Value network for CNN-based states to compute state values and advantages"""
    def __init__(self, state_dim: Tuple[int, int, int], action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Use the same CNN encoder as the main model
        self.cnn_encoder = CNNEncoder(state_dim, hidden_dim)
        
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
        # Encode visual state
        state_features = self.cnn_encoder(state)
        
        # State value
        value = self.value_head(state_features)
        
        if action is not None:
            # State-action value
            q_input = torch.cat([state_features, action], dim=-1)
            q_value = self.q_head(q_input)
            return value, q_value
        
        return value


class DiffusionPolicy_CNN:
    """
    Diffusion Policy algorithm for vision-based robotic manipulation with Advantage-guided learning
    """
    def __init__(
        self,
        state_dim: Tuple[int, int, int],  # (channels, height, width)
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
        network_type: str = "unet",
        device: str = "cpu",
        horizon_steps: int = 16,
        action_horizon: int = 8,
        advantage_weight: float = 1.0,  # 优势函数权重
        gamma: float = 0.99,  # 折扣因子
        tau: float = 0.005   # 目标网络软更新参数
    ):
        """
        Initialize Diffusion Policy with CNN encoder and Advantage guidance
        
        Args:
            state_dim: Dimension of state space (C, H, W) for images
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
        
        # Initialize CNN encoder
        self.cnn_encoder = CNNEncoder(state_dim, hidden_dim).to(self.device)
        
        # Initialize noise scheduler
        self.noise_scheduler = NoiseScheduler(
            num_diffusion_steps=num_diffusion_steps,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
            prediction_type=prediction_type
        )
        
        # Initialize diffusion model with encoded state dimension
        if network_type == "unet":
            self.model = DiffusionUNet(
                state_dim=hidden_dim,  # Use CNN output dimension
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                prediction_type=prediction_type
            ).to(self.device)
        else:  # mlp
            self.model = DiffusionMLP(
                state_dim=hidden_dim,  # Use CNN output dimension
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                prediction_type=prediction_type
            ).to(self.device)
            
        # Initialize value networks for advantage computation
        self.value_network = CNNValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        self.target_value_network = copy.deepcopy(self.value_network).to(self.device)
        self.target_value_network.requires_grad_(False)
            
        # EMA model for stable inference
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_cnn_encoder = copy.deepcopy(self.cnn_encoder)
        self.ema_cnn_encoder.requires_grad_(False)
        
        # Optimizers for diffusion model and CNN encoder
        self.diffusion_optimizer = torch.optim.AdamW(
            list(self.cnn_encoder.parameters()) + list(self.model.parameters()),
            lr=actor_lr,
            weight_decay=1e-6
        )
        
        # Optimizer for value network
        self.value_optimizer = torch.optim.AdamW(
            self.value_network.parameters(),
            lr=critic_lr,
            weight_decay=1e-6
        )
        
        # Action normalization parameters
        self.register_buffer('action_mean', torch.zeros(action_dim))
        self.register_buffer('action_std', torch.ones(action_dim))
        
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
        # Update diffusion model EMA
        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
        # Update CNN encoder EMA
        for param, ema_param in zip(self.cnn_encoder.parameters(), self.ema_cnn_encoder.parameters()):
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
            states: Current states (images)
            actions: Actions taken
            rewards: Immediate rewards
            next_states: Next states (images)
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
            
    def take_action(self, state: np.ndarray, use_ema: bool = True) -> np.ndarray:
        """
        Generate action using diffusion process
        
        Args:
            state: Current state observation (image)
            use_ema: Whether to use EMA model for inference
            
        Returns:
            Predicted action
        """
        model = self.ema_model if use_ema else self.model
        cnn_encoder = self.ema_cnn_encoder if use_ema else self.cnn_encoder
        model.eval()
        cnn_encoder.eval()
        
        with torch.no_grad():
            # Convert state to tensor and encode
            if isinstance(state, (list, tuple)):
                state = np.array(state)
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            
            # Handle different input shapes for CNN
            if len(state.shape) == 3:  # (C, H, W)
                state = state.unsqueeze(0)  # Add batch dimension -> (1, C, H, W)
            elif len(state.shape) == 4:  # (1, C, H, W) or (B, C, H, W)
                pass  # Already correct
            elif len(state.shape) == 5:  # (1, 1, C, H, W) - remove extra dimension
                state = state.squeeze(1)  # -> (1, C, H, W)
            
            # Encode image state
            encoded_state = cnn_encoder(state)
            
            # Start from random noise
            noisy_action = torch.randn((1, self.action_dim), device=self.device)
            
            # Denoising loop
            for t in reversed(range(self.num_inference_steps)):
                # Create timestep tensor
                timesteps = torch.full((1,), t, device=self.device, dtype=torch.long)
                
                # Predict noise
                noise_pred = model(noisy_action, timesteps, encoded_state)
                
                # Denoise
                noisy_action = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=noisy_action
                )
                
            # Denormalize and clip action
            action = self.denormalize_action(noisy_action)
            action = torch.clamp(action, -self.action_bound, self.action_bound)
            
        return action.cpu().numpy()[0]
        
    def train(self, transition_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Train diffusion model with CNN encoder and advantage guidance
        
        Args:
            transition_dict: Dictionary containing training data
            
        Returns:
            Dictionary containing training losses
        """
        self.model.train()
        self.cnn_encoder.train()
        self.value_network.train()
        
        # Extract data
        states = torch.tensor(transition_dict['states'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1, 1).to(self.device)
        
        # Handle different input shapes for CNN states
        if len(states.shape) == 5:  # (B, 1, C, H, W) - remove extra dimension
            states = states.squeeze(1)  # -> (B, C, H, W)
        if len(next_states.shape) == 5:  # (B, 1, C, H, W) - remove extra dimension
            next_states = next_states.squeeze(1)  # -> (B, C, H, W)
        
        # Update action statistics if needed
        if self.training_steps % 1000 == 0:
            self.update_action_stats(transition_dict['actions'])
            
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
        # Encode states
        encoded_states = self.cnn_encoder(states)
        
        # Normalize actions
        actions = self.normalize_action(actions)
        
        # Sample random timesteps
        batch_size = actions.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.num_diffusion_steps, (batch_size,),
            device=self.device, dtype=torch.long
        )
        
        # Add noise to actions
        noise = torch.randn_like(actions)
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        
        # Predict noise
        noise_pred = self.model(noisy_actions, timesteps, encoded_states)
        
        # Calculate base diffusion loss
        if self.noise_scheduler.prediction_type == "epsilon":
            target = noise
        else:  # prediction_type == "sample"
            target = actions
            
        base_diffusion_loss = F.mse_loss(noise_pred, target, reduction='none')
        
        # Weight the loss by advantages (detached to avoid gradient flow)
        # Positive advantages -> lower loss weight (good actions)
        # Negative advantages -> higher loss weight (bad actions)
        advantage_weights = torch.exp(-self.advantage_weight * advantages.detach())
        
        # Apply advantage weighting to diffusion loss
        # Reshape advantage weights to match diffusion loss shape
        advantage_weights = advantage_weights.expand_as(base_diffusion_loss)
        weighted_diffusion_loss = (base_diffusion_loss * advantage_weights).mean()
        
        # Update diffusion model and CNN encoder
        self.diffusion_optimizer.zero_grad()
        weighted_diffusion_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.cnn_encoder.parameters()), 
            1.0
        )
        
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
            'cnn_encoder_state_dict': self.cnn_encoder.state_dict(),
            'ema_cnn_encoder_state_dict': self.ema_cnn_encoder.state_dict(),
            'value_network_state_dict': self.value_network.state_dict(),
            'target_value_network_state_dict': self.target_value_network.state_dict(),
            'diffusion_optimizer_state_dict': self.diffusion_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'action_mean': self.action_mean,
            'action_std': self.action_std,
            'training_steps': self.training_steps
        }, filename + "_diffusion_policy_cnn_advantage.pt")
        
    def load(self, filename: str):
        """Load model checkpoints"""
        checkpoint = torch.load(filename + "_diffusion_policy_cnn_advantage.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        self.cnn_encoder.load_state_dict(checkpoint['cnn_encoder_state_dict'])
        self.ema_cnn_encoder.load_state_dict(checkpoint['ema_cnn_encoder_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
        self.target_value_network.load_state_dict(checkpoint['target_value_network_state_dict'])
        self.diffusion_optimizer.load_state_dict(checkpoint['diffusion_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.action_mean = checkpoint['action_mean']
        self.action_std = checkpoint['action_std']
        self.training_steps = checkpoint['training_steps'] 