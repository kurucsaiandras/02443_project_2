import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import matplotlib.ticker as mtick
import matplotlib.cm as cm

configs_to_show = ['no_f_ward', 'const_beds', 'proportional_beds', 'andras_method', 'valdemar_method']
config_names = ['No F Ward', 'Subtract evenly', 'Subtract proportionally', 'Iterative bed swapping', 'Simulated annealing']
num_wards = 6
ward_labels = [f"Ward {chr(ord('A') + i)}" for i in range(num_wards)]

arrival_means = []
transfer_means = []
transfer_cis = []

for config in configs_to_show:
    data = np.load(f'results_{config}.npz')
    arrivals = data['arrivals']
    attempted_transfers = data['transfers']
    blocks = data['blocks']
    succesful_transfers = attempted_transfers - blocks

    n_samples, n_wards = succesful_transfers.shape

    # Pad to full ward count with NaNs
    padded = np.full((n_samples, num_wards), np.nan)
    padded[:, :n_wards] = succesful_transfers

    padded_arrivals = np.full((n_samples, num_wards), np.nan)
    padded_arrivals[:, :n_wards] = arrivals

    means = np.nanmean(padded, axis=0)
    cis = 1.96 * sem(padded, axis=0, nan_policy='omit')

    arrival_mean = np.nanmean(padded_arrivals, axis=0)

    transfer_means.append(means)
    transfer_cis.append(cis)
    arrival_means.append(arrival_mean)

transfer_means = np.array(transfer_means)  # shape (num_configs, num_wards)
transfer_cis = np.array(transfer_cis)

# Plotting
x = np.arange(num_wards)
num_configs = len(configs_to_show)
width = 0.8 / num_configs  # total width of all bars per ward

#colors = cm.get_cmap('tab10', num_configs)

fig, ax = plt.subplots(figsize=(12, 6))

for i, (means, cis, arriv_m) in enumerate(zip(transfer_means, transfer_cis, arrival_means)):
    bar_pos = x - 0.4 + i * width + width / 2
    ax.bar(bar_pos, means, width, yerr=cis,
           label=config_names[i], capsize=4, zorder=2)

ax.set_ylabel('Number of patients')
ax.set_title('Successful relocations per Ward with 95% CI')
ax.set_xticks(x)
ax.set_xticklabels(ward_labels)
ax.legend()
#ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.tight_layout()
plt.show()
fig.savefig("figures/relocations.png", dpi=300)