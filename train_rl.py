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

@hydra.main(config_path="config", config_name="train_rl", version_base="1.2")
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

    ## バッファの初期化
    buffer_cfg = cfg.buffer
    buffer = get_buffer(buffer_cfg=buffer_cfg, device=device)

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

    eval_interval = cfg.eval_interval
    eval_episodes = cfg.eval_episodes

    log_dir = cfg.log_dir
    log_dir = os.path.join(log_dir, cfg.agent.name)
    model_dir = cfg.ckpt_dir
    model_dir = os.path.join(model_dir, cfg.agent.name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    best_reward = -float("inf")

    for episode in range(num_episodes):
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
                    nn_action = agent.select_action(state, evaluate=False)
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

            # bufferに追加
            next_state = scan_buffer.get_concatenated_numpy()
            buffer.add(state, action, reward, next_state, done)

            if len(buffer) > cfg.batch_size:
                agent.update(buffer, cfg.batch_size)

            if done:
                break

        # TensorBoard: 報酬の記録
        writer.add_scalar("reward/total_reward", total_reward, global_step=episode)

        # 毎 eval_interval エピソードごとに評価
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate(env, agent, planner, scan_buffer, cfg, device)
            writer.add_scalar("reward/eval_reward", eval_reward, global_step=episode)

            if eval_reward > best_reward:
                best_reward = eval_reward
                agent.save(os.path.join(model_dir, "best_model.pth"))
                print(f"🌟 Episode {episode}: New Best Eval Model Saved! eval_reward = {best_reward:.2f}")

    writer.close()

def evaluate(env, agent, planner, scan_buffer, cfg, device):
    num_eval_episodes = cfg.eval_episodes
    num_steps = cfg.num_steps
    num_agents = cfg.envs.num_agents
    total_rewards = []

    for _ in range(num_eval_episodes):
        obs, _ = env.reset()
        scan_buffer.scan_window.clear()  # 評価前にバッファ初期化
        scan = convert_scan(obs['scans'][0], cfg.envs.max_beam_range)
        scan_buffer.add_scan(scan)

        episode_reward = 0
        for step in range(num_steps):
            actions = []

            for i in range(num_agents):
                if i == 0:
                    if not scan_buffer.is_full():
                        action = [0.0, cfg.speed_range[0]]  # バッファが未満のときの仮アクション
                    else:
                        state = scan_buffer.get_concatenated_tensor()
                        nn_action = agent.select_action(state, evaluate=True)
                        action = convert_action(nn_action, cfg.steer_range, cfg.speed_range)
                    actions.append(action)
                else:
                    steer, speed = planner.plan(obs, id=i)
                    actions.append([steer, speed])

            next_obs, _, terminated, truncated, _ = env.step(np.array(actions))
            done = terminated or truncated
            scan = convert_scan(next_obs['scans'][0], cfg.envs.max_beam_range)
            scan_buffer.add_scan(scan)

            reward = agent.reward_manager.get_reward(
                obs=next_obs, pre_obs=obs, action=actions[0]
            )
            episode_reward += reward
            obs = next_obs

            if done:
                break

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)

    
if __name__ == "__main__":
    main()