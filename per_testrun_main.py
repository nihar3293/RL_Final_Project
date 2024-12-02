import gymnasium as gym
import gym_trading_env
import pickle
from utils import plot_portfolio_returns,load_stock_data
from agent import PERAgent

url = "https://raw.githubusercontent.com/ClementPerroud/Gym-Trading-Env/main/examples/data/BTC_USD-Hourly.csv"
df = load_stock_data(url=url)
env = gym.make("TradingEnv",
               name= "BTCUSD",
               df = df,
               positions = [ -0.5, 0, 1],
               trading_fees = 0.01/100,
               borrow_interest_rate= 0.0003/100,
              #  reward_function = c_reward,
    )

agent = PERAgent(7, 3)
with open('per_agent_state_dict.p','rb') as f:
    state_dict = pickle.load(f)
agent.set_params(state_dict)
episodes = 10

returns = []

for i in range(episodes):
    obs, info = env.reset()
    done, truncated = False, False
    while not done and not truncated:
        action = agent.choose_greedy_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        obs = next_obs
    ret = float(env.unwrapped.get_metrics()["Portfolio Return"].strip("%"))
    returns.append(ret)
    print("Episode: ",i+1, ", Portfolio Return: ",ret)
plot_portfolio_returns(returns, "per_portfolio_returns_test.png")