import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

tabela = pd.read_csv("advertising.csv")
print(tabela)

sns.heatmap(tabela.corr(), annot=True, cmap="Wistia")
plt.show()

# outra forma de ver a mesma análise
# sns.pairplot(tabela)
# plt.show()