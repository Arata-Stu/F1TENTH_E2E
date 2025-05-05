from omegaconf import DictConfig

from .off_policy import ReplayBuffer

def get_buffer(buffer_cfg: DictConfig, device: str = "cpu"):
    
    name = buffer_cfg.name

    if name == "off_policy":
        buffer = ReplayBuffer(
            capacity=buffer_cfg.capacity,
            state_shape=buffer_cfg.state_shape,
            action_dim=buffer_cfg.action_dim,
            device=device
        )
    else:
        raise ValueError(f"Unknown buffer name: {name}")
    
    return buffer
    