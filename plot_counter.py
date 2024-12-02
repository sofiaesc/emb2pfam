import pandas as pd
import matplotlib.pyplot as plt

file_path = 'results/fulldataset_win128_out128_1e-06/train_summary.csv' 
data = pd.read_csv(file_path)

epochs = data['Ep']
counter = data['Counter']

plt.figure(figsize=(10, 6))
plt.plot(epochs, counter, label='Patience', marker='o', color='purple')

plt.xlabel('Epoch')
plt.ylabel('Patience')
plt.legend()
plt.grid(True)

output_path = 'patience_curve.png' 
plt.savefig(output_path)

plt.close()
