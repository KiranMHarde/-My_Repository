import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
import os
import seaborn as sns
df=pd.read_excel('C:\\Users\\KIRAN\\Desktop\\Scores_of_students.xlsx')

df1=pd.DataFrame(df)















print('most of student come from:',statistics.mode(df1['CityTown']))

df['Gender'].value_counts().plot(kind='bar',title='Gender of students',figsize=(8,4))
#plt.show()

sns.histplot(df['Mathematics'])
#plt.show()
 #Size of the dataset
print("Size of the data set:",df.size)
# variance of each subject
print("Variance of Math",statistics.variance(df['Mathematics']))
print("Variance of Physics",statistics.variance(df['Physics']))
print("Variance of Chemistry",statistics.variance(df['Chemistry']))
print("Variance of Total",statistics.variance(df['Total']))

#centeral value of each subject
print("Centeral value of math:",statistics.median(df['Mathematics']))
print("Centeral value of Physics:",statistics.median(df['Physics']))
print("Centeral value of Chemistry:",statistics.median(df['Chemistry']))
print("Centeral value of Total",statistics.median(df['Total']))


# Q. Is any observation repeating more frequently than others?
m=statistics.mode((df1["Mathematics"]))
q=statistics.mode((df1["Physics"]))
l=statistics.mode((df1["Chemistry"]))
x=[m,q,l]
y=['math','physics','chemistry']
print("repeating marks more frequently in mathematics:",m)
print("repeating marks more frequently in mathematics:",q)
print("repeating marks more frequently in mathematics:",l)
plt.bar(y,x)
plt.show()



print('standard deviation Mathematics:',statistics.stdev(df1["Mathematics"]))

print('standard deviation Physics:',statistics.stdev(df1["Physics"]))


print('standard deviation Chemistry:',statistics.stdev(df1["Chemistry"]))







