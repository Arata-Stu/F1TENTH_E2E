import os
import hydra
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from f1tenth_gym.maps.map_manager import MapManager, MAP_DICT
from src.planner.purePursuit import PurePursuitPlanner
from src.envs.envs import make_env
from src.rewards.reward import make_raward
from src.buffers.buffer import get_buffer
from src.agents.agent import get_agent
from src.utils.helper import convert_action, convert_scan, ScanBuffer

@hydra.main(config_path="config", config_name="eval", version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(cfg))
    print('---------------------------')

    ## 環境の作成
    map_cfg = cfg.envs.map
    map_manager = MapManager(
        map_name=MAP_DICT[0],
        map_ext=map_cfg.ext,
        speed=map_cfg.speed,
        downsample=map_cfg.downsample,
        use_dynamic_speed=map_cfg.use_dynamic_speed,
        a_lat_max=map_cfg.a_lat_max,
        smooth_sigma=map_cfg.smooth_sigma
    )
    env = make_env(cfg.envs, map_manager, cfg.vehicle)

    planner = PurePursuitPlanner(wheelbase=cfg.planner.wheelbase,
                                 map_manager=map_manager,
                                 max_reacquire=cfg.planner.max_reacquire,
                                 lookahead=cfg.planner.lookahead)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## エージェントの初期化
    agent_cfg = cfg.agent
    agent = get_agent(agent_cfg=agent_cfg, device=device)
    agent.load(cfg.ckpt_path)
    agent.eval()

    ## 報酬関数の定義
    reward_cfg = cfg.reward
    reward_manager = make_raward(reward_cfg=reward_cfg, map_manager=map_manager)

    ## スキャンバッファの初期化
    scan_buffer = ScanBuffer(
        frame_size=cfg.envs.num_beams,
        num_scan=cfg.scan_n,
        target_size=cfg.dowansample_beam
    )

    ## 学習ループ
    num_episodes = cfg.num_episodes
    num_steps = cfg.num_steps
    num_agent = cfg.envs.num_agents


    log_dir = cfg.log_dir
    log_dir = os.path.join(log_dir, cfg.agent.name)
    os.makedirs(log_dir, exist_ok=True)


    writer = SummaryWriter(log_dir=log_dir)

    for episode in range(len(MAP_DICT)):
        map = MAP_DICT[episode]
        env.update_map(map, map_ext=map_cfg.ext)
        obs, info = env.reset()
        done = False
        total_reward = 0

        scan = obs['scans'][0]
        scan = convert_scan(scan, cfg.envs.max_beam_range)
        scan_buffer.add_scan(scan)

        for step in range(num_steps):
            actions = []

            for i in range(num_agent):
                if i==0:
                    state = scan_buffer.get_concatenated_tensor()
                    nn_action = agent.select_action(state, evaluate=True)
                    action = convert_action(nn_action, steer_range=cfg.steer_range, speed_range=cfg.speed_range)
                    actions.append(action)
                    
                else:
                    steer, speed = planner.plan(obs, id=i)
                    actions.append([steer, speed])

            next_obs, reward, terminated, truncated, info = env.step(np.array(actions))
            done = terminated or truncated

            if cfg.render:
                env.render(cfg.render_mode)
            next_scan = next_obs['scans'][0]
            next_scan = convert_scan(next_scan, cfg.envs.max_beam_range)
            scan_buffer.add_scan(next_scan)

            #報酬の計算
            reward = reward_manager.get_reward(obs=next_obs, pre_obs=obs, action=actions[0])
            total_reward += reward

            if done:
                break

        # TensorBoard: 報酬の記録
        writer.add_scalar("reward/total_reward", total_reward, global_step=episode)

    writer.close()
    env.close()
    print("Evaluation finished.")

    
if __name__ == "__main__":
    main()