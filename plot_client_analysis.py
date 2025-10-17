import scipy.io as sio
import torch
import os
import scipy.io as io
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns


# Dataset configuration

colors = ["#9e97cb", "#4586ac", "#cb5a48", "#3498db", "#95a5a6", "#e74c3c"]
palette = sns.color_palette(colors[1:2])
# colors = ["#2ecc71", "#9b59b6", "#DDA0DD", "#3498db", "#87CEFA", "#95a5a6","#e74c3c", "#95a5a6", "#9b59b6", "#B8860B"]
colors1 = ["#2ecc71", "#9b59b6", "#3498db", "#95a5a6","#e74c3c", "#95a5a6", "#9b59b6", "#B8860B"]

alg = ["FedDNN+CCVR+MOON (*)", "FedDNN (*)", "FedFNN (*)"]

client_list = ['1', '2', '3', '4', '5']

# get dataset sample number (updated for wifi dataset)
load_path = f"./results/client_analysis/wifi_fed_fpnn_csm_n_smpl_r15c5p5_homo1.0_nl0.0_bce_lr0.001_e15cr10.mat"
load_data = sio.loadmat(load_path)
n_sampl_cat_tbl = torch.tensor(load_data['n_sampl_cat_tbl'])
n_clients, n_classes = n_sampl_cat_tbl.shape
n_sampl_cat_tbl = n_sampl_cat_tbl.reshape(-1).numpy()
category_list = [f'category{i+1}' for i in range(n_classes)] * n_clients
client_list_n_smpl = [f'Client{j+1}' for j in range(n_clients) for _ in range(n_classes)]

n_smpl_data = []
for i in range(n_sampl_cat_tbl.shape[0]):
    n_smpl_data.append([client_list_n_smpl[i], n_sampl_cat_tbl[i], category_list[i]])
n_smpl_data_pd = DataFrame(n_smpl_data, columns=["client", 'Sample Number', 'Category'])


# ================plot sample number per client and category====================
plt.rcParams['figure.figsize']=[8, 6]
plt.subplots_adjust(wspace=0.3, hspace=0.3)
ax = plt.subplot(111)
sns.barplot(x="client", y='Sample Number', hue="Category", data=n_smpl_data_pd)
ax.set_ylabel('Sample Number', fontsize=13)
ax.set_xlabel('Client', fontsize=13)
ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.84, 1.02), ncol=1)
plt.title('Sample Number per Client and Category (wifi)')
plt.savefig(f"./results/client_analysis_v.pdf", bbox_inches='tight')
print("Plot saved to ./results/client_analysis_v.pdf")