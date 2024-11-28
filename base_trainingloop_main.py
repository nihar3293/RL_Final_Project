import gymnasium as gym
import gym_trading_env
from utils import load_stock_data, plot_epsilon_and_rewards, plot_portfolio_returns
from agent import DeepQAgent

url = "https://raw.githubusercontent.com/ClementPerroud/Gym-Trading-Env/main/examples/data/BTC_USD-Hourly.csv"
df = load_stock_data(url=url)
env = gym.make("TradingEnv",
               name= "BTCUSD",
               df = df,
               positions = [ -1, 0, 1],
               trading_fees = 0.01/100,
               borrow_interest_rate= 0.0003/100,
              #  reward_function = c_reward,
    )

agent = DeepQAgent(7, 3)

episodes = 40

cum_reward_history = []
returns = []
epsilon_history = []

for i in range(episodes):
    obs, info = env.reset()
    done = False
    truncated = False
    rewards =  0
    while not done and not truncated:
        action = agent.choose_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)

        rewards += reward
        agent.store_experience(obs, action, reward, next_obs, done)

        agent.train()
        obs = next_obs
        epsilon_history.append(agent.eps)
    cum_reward_history.append(rewards)
    ret = float(env.unwrapped.get_metrics()["Portfolio Return"].strip("%"))
    returns.append(ret)
    print("Episode: ", i+1, ", Cum rewards: ", rewards, ", Agent exploration rate: ", agent.eps, ", Steps taken: ", agent.iter_cntr)
plot_epsilon_and_rewards(epsilon_history, cum_reward_history, "training_rewards.png")
plot_portfolio_returns(returns, "portfolio_returns_training.png")

