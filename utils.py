import pandas as pd
import matplotlib.pyplot as plt

def load_stock_data(url):
    df = pd.read_csv(url, parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    df["feature_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()
    df.dropna(inplace=True)
    return df

def plot_epsilon_and_rewards(epsilon_history,cum_reward_history, filename):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(epsilon_history)
    axs[0].set_ylabel('Epsilon')
    axs[0].set_xlabel('Steps')
    axs[0].set_title('Epsilon Decay')
    axs[1].plot(cum_reward_history)
    axs[1].set_ylabel('Cumulative Rewards')
    axs[1].set_xlabel('Episode')
    axs[1].set_title('Cumulative Rewards per episode')
    plt.tight_layout()
    plt.savefig(filename, format='png')
    plt.show()

def plot_portfolio_returns(returns, filename):
    plt.plot(returns)
    plt.xlabel('Episode')
    plt.ylabel('Portfolio Returns')
    plt.title('Portfolio Returns for each episode')
    plt.tight_layout()
    plt.savefig(filename, format='png')
    plt.show()