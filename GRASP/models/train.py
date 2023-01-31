import numpy as np
import random
from loguru import logger

from GRASP.models.utils import exp

def train(env, agent, max_episode, save_model_dir):
    random.seed(agent.seed)
    _general_goal = np.zeros(agent.goal_dim)
    agent.is_training = True
    step = episode_steps = episode_total = 0
    episode = 1
    total_reward = 0.
    s_t = None
    episode_buffer = []
    while episode < max_episode:
        while True:
            if s_t is None:
                s_t = env.reset()
                action_space_t = env.get_valid_actions(s_t)
                agent.reset()
                if not action_space_t:
                    logger.info(
                        "%s has no valid retrosynthesis reactions, skipping"%(s_t[0])
                    )
                    s_t = None
                    break
                logger.info(
                    "Starting new episode on %s"%(s_t[0])
                )

            if step <= agent.warmup:
                raw_action, action = agent.random_action(action_space_t)
            else:
                raw_action, action = agent.select_action(s_t[1], _general_goal, action_space_t)

            s_t1, r_t, done, action_space_t1 = env.step(action)

            if agent.max_episode_length and episode_steps >= agent.max_episode_length - 1:
                done = True

            # 'exp', 'state0, action, goal, reward, state1, terminate, action_space'
            episode_buffer.append(exp(s_t[1], raw_action, _general_goal, r_t, s_t1[1], done, action_space_t1))

            step += 1
            episode_steps += 1
            total_reward += r_t
            s_t = s_t1
            action_space_t = action_space_t1

            if done:  
                if episode_buffer:
                    for i, _exp in enumerate(episode_buffer):
                        # /home/yeminyu/work/retro/GRASP/GRASP/models/base_agent.py 99 row   agent.memory.append(exp)
                        # /home/yeminyu/work/retro/GRASP/GRASP/models/utils.py 89 row    memory.observations.append(observation)  (deque)
                        agent.observe(_exp)
                        # agent.memory_imitation
                        if r_t == 1:
                            agent.observe_imitation(_exp._replace(reward=agent.gamma ** \
                                                                    (len(episode_buffer)-i-1)))
                        # for what ?
                        if random.uniform(0, 1) < agent.g_prop and step > agent.warmup:
                            if episode_buffer[i:]:
                                _goal = random.choice([x.state[1] for x in episode_buffer[i:]])
                            else:
                                _goal = s_t[1]
                            agent.observe(_exp._replace(goal=_goal, reward=1))
                # warm up: filling buffer only, only the last buffer ultilized to training? 
                if step > agent.warmup and len(agent.memory_imitation) > agent.batch_size:
                    agent.update_policy(imitation=True)

                if step % agent.report_steps == 0:
                    episode_total += episode
                    logger.info(
                        "Ep:{0} | Moving Episode R:{1:.4f}".format(episode_total, total_reward/episode)
                    )
                    total_reward = 0
                    episode = 0

                s_t = None
                episode_steps =  0
                episode_buffer = []
                episode += 1

                break

        if step > agent.warmup and episode_total % agent.save_per_epochs == 0:
            agent.save_model(save_model_dir)
            logger.info("Model Saved at Ep:{0} ".format(episode_total))

def test(env, agent, model_path, test_episode):

    agent.load_weights(model_path + 'actor.pt', model_path + 'critic.pt')
    agent.is_training = False
    agent.eval()
    _general_goal = np.zeros(agent.goal_dim)
    episodes = 1
    episode_steps = 0
    episodes_reward = 0.
    s_t = None
    for i in range(test_episode):
        while True:
            if s_t is None:
                s_t = env.reset() # sample a molecule in the batch /home/yeminyu/work/retro/GRASP/GRASP/models/env.py row 35
                action_space_t = env.get_valid_actions(s_t)  # /home/yeminyu/work/retro/GRASP/GRASP/models/env.py  action space comprised of FP?
                agent.reset() # self.random_process.reset_states() why reset random process   /home/yeminyu/work/retro/GRASP/GRASP/models/base_agent.py
                if not action_space_t:
                    logger.info(
                        "%s has no valid retrosynthesis reactions, skipping"%(s_t[0])
                    )
                    s_t = None
                    break
                logger.info(
                    "Starting new episode on %s"%(s_t[0])
                )
            # s_t[1] FP of the root molecule               is _general_goal the building block?
            # /home/yeminyu/work/retro/GRASP/GRASP/models/base_agent.py 119 row why decay the proto-action and random process
            # /home/yeminyu/work/retro/GRASP/GRASP/models/base_agent.py 245 row knn here equals similarity k=1 why knn?
            # /home/yeminyu/work/retro/GRASP/GRASP/models/base_agent.py 195 row critic model is not utilized when k=1?
            # raw_action equals action[0]['action'] ?
            raw_action, action = agent.select_action(s_t[1], _general_goal, action_space_t)
            # update env.cur_state to action['_ns_mol'](SMILES), action['action'](FP)
            # /home/yeminyu/work/retro/GRASP/GRASP/models/env.py 30 row
            s_t1, r_t, done, action_space_t1 = env.step(action)

            episode_steps += 1
            episodes_reward += r_t

            if agent.max_episode_length and episode_steps >= agent.max_episode_length - 1:
                done = True

            if done:  # end of an episode
                logger.info(
                    "Ep:{0} | Moving Episode R:{1:.4f}".format(i+1, episodes_reward/episodes)
                )
                s_t = None
                episodes += 1
                episode_steps = 0
                break
            else:
                s_t = s_t1
                action_space_t = action_space_t1
