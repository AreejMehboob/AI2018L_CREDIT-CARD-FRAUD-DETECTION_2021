# /-------------------------------------------------------GRAPHS FOR CREDIT CARD FRAUD DETECTION-------------------------------------------------------/

# /------------------------------------------------------IMPORTING LIBRARIES---------------------------------------------------------------/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




# /------/
data = pd.read_csv('creditcard.csv')


# /---------------------------------------------------------------Graphs and Plots For Visual Analysis of Data------------------------------------------------/



# /Bar Graph/
difference_class = data['Class'].value_counts()
difference_class.plot(kind='bar', color=['m', 'k'], figsize=(5, 5))
plt.xticks(range(2), ['Normal  [0]', 'Fraud  [1]'], rotation=0)
for i, v in enumerate(difference_class):
    plt.text(i-0.1, v+3000, str(v))
plt.title('Class Count')
plt.show()


# /2-D Scatter /
sns.set_style("whitegrid")
sns.FacetGrid(data, hue="Class", size = 6).map(plt.scatter, "Time", "Amount").add_legend()
plt.show()

# /3-D Scatter Plot/
DataFiltered = creditcard[['Time','Amount', 'Class']]
print(DataFiltered)
print(DataFiltered.shape)
print(DataFiltered["Class"].value_counts())
plt.close();
sns.set_style("whitegrid");
sns.pairplot(FilteredData, hue="Class", size=5);
plt.show()

# /FaceGrid Plot/
print(DataFiltered)
print(DataFiltered.shape)
print(DataFiltered["Class"].value_counts())
sns.FacetGrid(DataFiltered, hue="Class", size=10).map(sns.distplot, "Time").add_legend()
plt.show()

# /Histogram/
counts, bin_edges = np.histogram(DataFiltered['Amount'], bins=10, density = True)
pdf = counts/(sum(counts))

print("pdf = ",pdf)
print("\n")
print("Counts =",counts)
print("\n")
print("Bin edges = ",bin_edges)

cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



plt.show();