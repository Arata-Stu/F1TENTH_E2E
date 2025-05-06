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
                # update() が返す dict を受け取る
                loss_dict = agent.update(buffer, cfg.batch_size)

                # 動的にすべてのキー／値ペアを TensorBoard に記録
                # 例として "loss/critic_loss", "loss/actor_loss", ... のようにタグ付け
                for key, value in loss_dict.items():
                    writer.add_scalar(f"loss/{key}", value, global_step=episode)

            if done:
                break

        # TensorBoard: 報酬の記録
        writer.add_scalar("reward/total_reward", total_reward, global_step=episode)

        print(f"Episode {episode}: Total Reward = {total_reward:.2f}")
        # 毎 eval_interval エピソードごとに評価
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate(env, agent, planner,reward_manager, scan_buffer, cfg, device)
            writer.add_scalar("reward/eval_reward", eval_reward, global_step=episode)

            if eval_reward > best_reward:
                best_reward = eval_reward
                agent.save(os.path.join(model_dir, "best_model.pth"))
                print(f"🌟 Episode {episode}: New Best Eval Model Saved! eval_reward = {best_reward:.2f}")

    writer.close()


def evaluate(env, agent, planner, reward_manager, scan_buffer, cfg, device):
    """
    評価用のルーチン
    - torch.no_grad() で勾配計算をオフ
    - agent.eval()/agent.train() でモード切替
    - 報酬計算を一貫
    """
    agent.eval()  # モデルを評価モードに
    total_rewards = []

    # 評価エピソードループ
    for ep in range(cfg.eval_episodes):
        obs, _ = env.reset()
        scan_buffer.scan_window.clear()  # バッファ完全クリア
        # 最初のスキャン追加
        first_scan = convert_scan(obs['scans'][0], cfg.envs.max_beam_range)
        scan_buffer.add_scan(first_scan)

        episode_reward = 0.0

        # no_grad コンテキストで勾配計算オフ
        with torch.no_grad():
            for step in range(cfg.num_steps):
                actions = []
                # エージェントとプランナーでアクション取得
                for i in range(cfg.envs.num_agents):
                    if i == 0:
                        state = scan_buffer.get_concatenated_tensor().to(device)
                        nn_action = agent.select_action(state, evaluate=True)
                        action = convert_action(nn_action, cfg.steer_range, cfg.speed_range)
                        actions.append(action)
                    else:
                        steer, speed = planner.plan(obs, id=i)
                        actions.append([steer, speed])

                # 環境ステップ
                next_obs, _, terminated, truncated, _ = env.step(np.array(actions))
                done = terminated or truncated

                # 次のスキャンをバッファに追加
                next_scan = convert_scan(next_obs['scans'][0], cfg.envs.max_beam_range)
                scan_buffer.add_scan(next_scan)

                # 報酬計算（トレーニング時と同じ reward_manager を使う想定）
                r = reward_manager.get_reward(obs=next_obs, pre_obs=obs, action=actions[0])
                episode_reward += r

                obs = next_obs
                if done:
                    break

        total_rewards.append(episode_reward)

    agent.train()  # 忘れずにトレーニングモードに戻す

    avg_reward = float(np.mean(total_rewards))
    std_reward = float(np.std(total_rewards))
    print(f"[Eval] 平均報酬: {avg_reward:.2f} ± {std_reward:.2f} over {cfg.eval_episodes} episodes")
    return avg_reward

    
if __name__ == "__main__":
    main()