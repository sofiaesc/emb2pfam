import pandas as pd
import matplotlib.pyplot as plt

file_path = 'results/fulldataset_win128_out128_1e-05/train_summary.csv' 
data = pd.read_csv(file_path)

epochs = data['Ep']
dev_error = data['Dev error']
best_error = data['Best error']

plt.figure(figsize=(10, 6))
plt.plot(epochs, dev_error, label='Dev Error', marker='o')
plt.plot(epochs, best_error, label='Best Error', marker='x')

plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.grid(True)

output_path = 'error_curves.png'  
plt.savefig(output_path)
plt.close()
