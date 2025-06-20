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
admission_means = []
successful_relocation_means = []
admission_cis = []

for config in configs_to_show:
    data = np.load(f'results_{config}.npz')
    arrivals = data['arrivals']
    attempted_relocations = data['transfers']
    blocks = data['blocks']

    admissions = arrivals - attempted_relocations
    n_samples, n_wards = admissions.shape

    successful_relocations = attempted_relocations - blocks

    # Pad to full ward count with NaNs
    padded = np.full((n_samples, num_wards), np.nan)
    padded[:, :n_wards] = admissions

    padded_arrivals = np.full((n_samples, num_wards), np.nan)
    padded_arrivals[:, :n_wards] = arrivals

    padded_relocations = np.full((n_samples, num_wards), np.nan)
    padded_relocations[:, :n_wards] = successful_relocations

    means = np.nanmean(padded, axis=0)
    cis = 1.96 * sem(padded, axis=0, nan_policy='omit')

    arrival_mean = np.nanmean(padded_arrivals, axis=0)
    reloc_mean = np.nanmean(padded_relocations, axis=0)

    admission_means.append(means)
    admission_cis.append(cis)
    arrival_means.append(arrival_mean)
    successful_relocation_means.append(reloc_mean)

admission_means = np.array(admission_means)  # shape (num_configs, num_wards)
admission_cis = np.array(admission_cis)
arrival_means = np.array(arrival_means)
successful_relocation_means = np.array(successful_relocation_means)

# Plotting
x = np.arange(num_wards)
num_configs = len(configs_to_show)
width = 0.8 / num_configs  # total width of all bars per ward

#colors = cm.get_cmap('tab10', num_configs)

fig, ax = plt.subplots(figsize=(12, 6))

for i, (means, cis, arriv_m, reloc_m) in enumerate(zip(admission_means, admission_cis, arrival_means, successful_relocation_means)):
    bar_pos = x - 0.4 + i * width + width / 2
    ax.bar(bar_pos, arriv_m, width,
           color='lightgray', label='_nolegend_', zorder=1)
    ax.bar(bar_pos, means+reloc_m, width,
           color='darkgray', label='_nolegend_', zorder=2)
    ax.bar(bar_pos, means, width, yerr=cis,
           label=config_names[i], capsize=4, zorder=3)

ax.set_ylabel('Number of patients')
ax.set_title('Successful admissions per Ward with 95% CI')
ax.set_xticks(x)
ax.set_xticklabels(ward_labels)
ax.bar(0, 0, color='lightgray', label='Rejections')
ax.bar(0, 0, color='darkgray', label='Relocations')
ax.legend()
#ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.tight_layout()
plt.show()
fig.savefig("figures/admissions.png", dpi=300)