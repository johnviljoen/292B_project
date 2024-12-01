import numpy as np
from scipy.special import rel_entr

row_rewards = np.load("saved_inferences/policy_a_rewards.npy")

start = 0
end = 1
num_bins = 2

row_rewards = row_rewards[-1][-1][:,0] # a single set of rollouts for a single world using the most trained opponent we have

total_rewards = row_rewards.flatten()
marginal_hist, marginal_bin_edges = np.histogram(total_rewards, bins=num_bins, range=(start, end))
marginal_distribution = marginal_hist / sum(marginal_hist)
# KL divergence
KL_divergence = np.zeros(len(row_rewards))
for i in range(len(row_rewards)):
    hist, bin_edges = np.histogram(row_rewards[i], bins=num_bins, range=(start, end))
    distribution = hist / sum(hist)
    KL_divergence[i] = sum(rel_entr(distribution,marginal_distribution))
# index
index = sum(KL_divergence)/len(KL_divergence)
index_list.append(index)
current_seed_list.append(index)

print('fin')