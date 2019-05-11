'''E-Commerce Data Actual transactions from UK retailer'''
'''Sumber dataset https://www.kaggle.com/carrie1/ecommerce-data'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#1 load data
dataset=pd.read_csv("E:\Semester 4\data mining/data.csv",encoding = "ISO-8859-1")
dataset.shape
# print(dataset.shape)

# 2 hilangkan duplikat entri pada dataset
# print (dataset.duplicated().sum())
dataset.drop_duplicates(inplace = True)
dataset.shape
# print(dataset.shape)

#3 hilangkan missing value di kolom customer id
dataset.dropna(axis = 0, subset =['CustomerID'], inplace = True)
dataset.shape
# print(dataset.shape)

#4 cek data apakah ada yg null
# print (pd.DataFrame(dataset.isnull().sum()))

#5 menghilangkan order yang dicancel
dataset = dataset[(dataset.InvoiceNo).apply(lambda x:( 'C' not in x))]
dataset.shape
# print(dataset.shape)
df_customerid_groups=dataset.groupby("CustomerID")
# print (len((df_customerid_groups.groups)))

'''# 6 membuat dataframe baru 'Quantity','UnitPrice','CustomerID'''
df_cluster=pd.DataFrame(columns=['Quantity','UnitPrice','CustomerID'])
count=0
for k,v in (df_customerid_groups):
    df_cluster.loc[count] = [(v['Quantity'].sum()), v['UnitPrice'].sum(), k]
    count+=1
df_cluster.shape

#7 kita hanya memakai 'Quantity', 'UnitPrice' kolom untuk di kelompokan
X = df_cluster.iloc[:, [0, 1]].values #mengambil data

#8 feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X= sc_X.fit_transform(X)

#9 gunakan Elbow method untuk mencari jumlah cluster
from sklearn.cluster import KMeans
wcss=[]
'''
n_cluster=Jumlah cluster serta jumlah centroid yang dihasilkan.
k-mean++=metode inisialisasi acak untuk centroid
max_iter=Jumlah maksimum iterasi dari algoritma k-means untuk sekali jalan.
n_init=
'''
#10
for i in range(1,11): #
    kmeans = KMeans(n_clusters = i, init ='k-means++',max_iter=300,n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11) , wcss)
plt.title('The Elbow Method')
plt.xlabel('Jumlah Kelompok Customer (kelompok jenis customer)')
plt.ylabel('With in cluster sum of squers(WCSS)')
plt.show()
'''
ketemu n_cluster yang optimal yaitu n_cluster=3 
'''
#11 mefitting k-mean ke dataset
kmeans = KMeans(n_clusters = int(input("Masukan Jumlah Clusters:")), init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)

#12 mengvisualisasikan cluster
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Customer Type 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Customer Type 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 50, c = 'green', label = 'Customer Type 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100,marker='x', c = 'r', label = 'Centroids')
plt.title('Jumlah Kelompok Customer (kelompok jenis customer')
plt.xlabel('Jumlah barang yang Dibeli(Quantity)')
plt.ylabel('Harga produk per unit dalam sterling(Unit Price)')
plt.legend()
plt.show()

# x=[];y=[]
# for i in range(4339):
#     x.append(X[i][0])
#     y.append(X[i][1])
# plt.scatter(x,y)
# plt.title('Plot of training data')
# plt.xlabel('Jumlah barang yang Dibeli(Quantity)')
# plt.ylabel('Harga produk per unit dalam sterling(Unit Price)')
# plt.show()