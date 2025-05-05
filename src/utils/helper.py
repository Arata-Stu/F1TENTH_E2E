import numpy as np
from collections import deque
import torch

def convert_action(action, steer_range: float=0.4, speed_range: float=10.0):
    
    steer = action[0] * steer_range
    speed = (action[1] + 1) / 2 * speed_range
    action = [steer, speed]
    return action

def convert_scan(scans, scan_range: float=30.0):
    scans = scans / scan_range
    scans = np.clip(scans, 0, 1)
    return scans

class ScanBuffer:
    def __init__(self, frame_size: int=1080, num_scan: int=2):
        """
        frame_size: 1フレームあたりのデータ長（例: 1080）
        k: バッファするフレーム数
        """
        self.frame_size = frame_size
        self.num_scan = num_scan
        self.scan_window = deque(maxlen=num_scan)

    def add_scan(self, scan: np.ndarray):
        """新しいスキャンデータを追加"""
        if scan.shape[0] != self.frame_size:
            raise ValueError(f"scan の長さが {self.frame_size} ではありません: got {scan.shape[0]}")
        self.scan_window.append(scan)

    def is_full(self) -> bool:
        """num_scan フレーム分たまっているか"""
        return len(self.scan_window) == self.num_scan

    def _pad_frames(self, frames: list) -> list:
        """
        フレーム数が num_scan 未満なら、最後のフレームを繰り返して長さ num_scan にする
        （len==1 のときは t=1 のフレームを num_scan 枚に；num_scan=2 なら 2 枚連結）
        """
        if not frames:
            raise ValueError("バッファにフレームが存在しません")
        if len(frames) < self.num_scan:
            last = frames[-1]
            pad = [last] * (self.num_scan - len(frames))
            frames = frames + pad
        return frames

    def get_concatenated_numpy(self) -> np.ndarray:
        """
        num_scan フレーム分を連結して返す（NumPy 配列）
        - バッファに 1 フレームしかなければ、最後のフレームを繰り返して num_scan 枚分にパディング
        - バッファが空なら例外
        """
        frames = list(self.scan_window)
        frames = self._pad_frames(frames)
        # 1D 配列を横にくっつける
        return np.hstack(frames)

    def get_concatenated_tensor(self,
                                device: torch.device = None,
                                dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        num_scan フレーム分を連結して返す（PyTorch Tensor）
        - device, dtype を指定可能
        - バッファに 1 フレームしかなければ、最後のフレームを繰り返して num_scan 枚分にパディング
        """
        # NumPy 配列のリスト → Tensor のリスト
        frames = list(self.scan_window)
        frames = self._pad_frames(frames)
        tensors = []
        for f in frames:
            t = f if isinstance(f, torch.Tensor) else torch.from_numpy(f)
            tensors.append(t)
        out = torch.cat(tensors, dim=0)
        if device is not None:
            out = out.to(device)
        if dtype is not None:
            out = out.to(dtype)
        return out
