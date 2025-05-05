import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class GaussianPolicy(nn.Module):
    """
    ガウス政策ネットワーク (連続行動空間)
    出力は平均と対数分散、sample() で tanh 補正後の行動と対数確率を返す。
    """
    def __init__(self, state_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state, eps=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        # tanh 補正の対数確率
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + eps)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

class QNetwork(nn.Module):
    """
    2 つの Q 関数を持つネットワーク
    """
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2

class SACAgent:
    """
    Soft Actor-Critic エージェント
    （固定α → 自動調整α 版）
    """
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 alpha_lr: float = 3e-4,      
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 device: str = 'cpu'):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau

        # 自動エントロピー調整パラメータ
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # ネットワーク
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # オプティマイザ
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, evaluate: bool = False):
        state = state.to(self.device).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                mean, _ = self.policy(state)
                action = torch.tanh(mean)
            else:
                action, _ = self.policy.sample(state)
        return action.cpu().numpy().flatten()

    def update(self, replay_buffer, batch_size: int):
        # バッファからサンプル取得
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # --- Critic ターゲット値計算 ---
        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(next_state)
            q1_next, q2_next = self.critic_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            q_target = reward + (1 - done) * self.gamma * q_next

        # --- Critic 更新 ---
        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor 更新 ---
        new_action, log_prob_new = self.policy.sample(state)
        q1_new, q2_new = self.critic(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob_new - q_new).mean()
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()

        # --- α（エントロピー重み）更新 ---
        # 目標エントロピーとの差を損失に
        alpha_loss = -(self.log_alpha * (log_prob_new + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- ターゲットネットソフトアップデート ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha_value': self.alpha.item()
        }
    
    def save(self, filepath: str):
        """モデルとαの保存"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict()
        }, filepath)

    def load(self, filepath: str):
        """モデルとαの読み込み"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha'].to(self.device)
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])

