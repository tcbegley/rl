import matplotlib.pyplot as plt

def main():
    # Read the text file
    filename = 'rlhf_valid_rewards.log'  # Replace with the actual filename
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Extract the relevant data
    rollout_rewards = []
    rollout_kls = []
    val_rewards = []
    it_rollout = []
    it_val = []
    for line in lines:
        if 'rollout_reward' in line:
            it, rollout_reward = line.strip().split('TRAIN: it=')[1].split(": rollout_reward=")
            rollout_reward, rollout_kl = rollout_reward.split(" rollout_kl=")
            it_rollout.append(int(it))
            rollout_rewards.append(float(rollout_reward))
            rollout_kls.append(float(rollout_kl))
        elif 'val_reward' in line:
            it, val_reward = line.strip().split('VALID: it=')[1].split(": val_reward=")
            it_val.append(int(it))
            val_rewards.append(float(val_reward))

    # Plot the trending lines
    plt.plot(it_rollout, rollout_rewards, label='rollout_reward')
    plt.plot(it_rollout, rollout_kls, label='rollout_kls')
    plt.plot(it_val, val_rewards, label='val_reward')
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    plt.legend()

    # Save the figure
    plt.savefig('reward_trends.png')  # Replace with the desired filename
    plt.show()

if __name__ == "__main__":
    main()