from omegaconf import DictConfig

from src.agents.sac import SACAgent
def get_agent(agent_cfg: DictConfig, device: str):
    """
    エージェントの初期化
    """
    agent_name = agent_cfg.name
    if agent_name == "sac":
        
        state_dim = agent_cfg.state_dim
        action_dim = agent_cfg.action_dim
        hidden_dim = agent_cfg.hidden_dim
        actor_lr = agent_cfg.actor_lr
        critic_lr = agent_cfg.critic_lr
        alpha_lr = agent_cfg.alpha_lr
        gamma = agent_cfg.gamma
        tau = agent_cfg.tau

        return SACAgent(state_dim, action_dim, hidden_dim, actor_lr, critic_lr, alpha_lr, gamma, tau, device)
    else:
        raise ValueError(f"Unknown agent name: {agent_name}")
    