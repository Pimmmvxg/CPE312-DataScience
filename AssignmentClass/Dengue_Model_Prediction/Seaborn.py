import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd

data = 'AssignmentClass/Data/pivot_output.csv'
dataset = pd.read_csv(data)

sb.set()
sb.pairplot(dataset,hue='province',height=2)
plt.show()

