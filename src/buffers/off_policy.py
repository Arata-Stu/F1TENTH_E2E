import numpy as np
import torch

class ReplayBuffer:
    """
    Off-policy RL 向け汎用リプレイバッファ。
    add に渡されたデータはすべて NumPy 配列に変換して格納し、
    sample では PyTorch Tensor で返却します。
    """
    def __init__(self, capacity: int, state_shape: tuple, action_dim: int=2, device: str = "cpu"):
        """
        Args:
            capacity (int): バッファの最大サイズ
            state_shape (tuple): 状態ベクトルの形状 (例: (k * num_beams, ))
            action_dim (int): 行動ベクトルの次元 (例: 2 for [steer, speed])
            device (str): サンプリング時に返す torch.Tensor のデバイス
        """
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.full = False

        # データ格納用に NumPy 配列を確保
        self.states      = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions     = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards     = np.zeros((capacity, 1), dtype=np.float32)
        self.dones       = np.zeros((capacity, 1), dtype=np.float32)

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        """
        入力を必ず np.ndarray（dtype=float32）に変換する。
        - torch.Tensor → .detach().cpu().numpy()
        - list や tuple → np.array
        - np.ndarray → float32 にキャスト
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        elif not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float32)
        return x.astype(np.float32)

    def add(self, state, action, reward, next_state, done):
        """
        1 タイムステップ分のデータをバッファに追加。
        どの型で渡されても NumPy 配列として格納される。
        """
        idx = self.pos

        # 強制的に NumPy 化
        self.states[idx]      = self._to_numpy(state)
        self.actions[idx]     = self._to_numpy(action)
        self.rewards[idx, 0]  = float(self._to_numpy(reward))
        self.next_states[idx] = self._to_numpy(next_state)
        self.dones[idx, 0]    = float(self._to_numpy(done))

        # ポインタ更新
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size: int):
        """
        ミニバッチをランダムサンプリングし，torch.Tensor で返却。
        """
        max_i = self.capacity if self.full else self.pos
        idxs = np.random.choice(max_i, size=batch_size, replace=False)

        # NumPy → Torch Tensor に変換し，指定デバイスへ
        states      = torch.from_numpy(self.states[idxs]).to(self.device)
        actions     = torch.from_numpy(self.actions[idxs]).to(self.device)
        rewards     = torch.from_numpy(self.rewards[idxs]).to(self.device)
        next_states = torch.from_numpy(self.next_states[idxs]).to(self.device)
        dones       = torch.from_numpy(self.dones[idxs]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.capacity if self.full else self.pos
