import pandas as pd
import matplotlib.pyplot as plt

# Legge il file CSV
df = pd.read_csv('risultati_esecuzione.csv')

# Filtra solo i file di input desiderati
filtered_df = df[df['File di Input'].isin(['data/numbers_1000.txt', 'data/numbers_10000.txt'])]

# Crea il grafico
plt.figure(figsize=(10, 6))
for file_input, group in filtered_df.groupby('File di Input'):
    plt.plot(group['Numero Thread'], group['Tempo di Esecuzione (s)'],
             marker='o', label=file_input.split('/')[-1])

plt.xlabel('Numero di Thread')
plt.ylabel('Tempo di Esecuzione (s)')
plt.title('Tempo di Esecuzione vs Numero di Thread')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
