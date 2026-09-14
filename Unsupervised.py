#step 1: Dataset without clear labels

# Libraries

import pandas as pd

# Set datas                  1        2         3      4      5      6       7        8        9         10
activity_data = {
    'participant_name' : [ 'Jay', 'Valerie', 'Dits', 'Vel', 'Ara', 'Jan', 'Sofia', 'Kylie', 'Jayson', 'Lovely',
#                        11       12         13         14       15        16        17        18         19       20
                      'Charish', 'Karla', 'Rachel', 'Jamilla', 'Jerah', 'Effamar', 'Juan', 'Francis', 'Charles', 'Ryzel'],

# study hours                   1  2  3  4  5   6   7  8  9  10 11 12 13 14 15 16  17 18 19 20 = 20 raw data
    'coding_hours_per_week' : [13, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10, 1, 4, 8, 2],

# attendance rate           1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20 = 20 raw data
    'projects_completed' : [99, 62, 64, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 71],
}

df = pd.DataFrame(activity_data)

print(df[['participant_name', 'coding_hours_per_week', 'projects_completed']])

#Step 2: Scale the feature and Define the characteristics

from sklearn.preprocessing import StandardScaler

# Formula to get X-Features
X = df[['coding_hours_per_week', 'projects_completed']]

# Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled[:5])

# OUTPUT: The X-Features found "-1" study hours, "0" attendance rate
# Poor result means no common datas
# Negative results means the data is possibly inaccurate

#Step 3: Apply KMeans Clustering

from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Formula to get X-Features
X = df[['coding_hours_per_week', 'projects_completed']]

# Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Finds 2 groups (clusters) with the same guesses
Kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

clusters = Kmeans.fit_predict(X_scaled)

df['clusters'] = clusters

print(df[['participant_name', 'coding_hours_per_week', 'projects_completed', 'clusters']])

#Step 4: Interpret Cluster and print

centers_original = scaler.inverse_transform(Kmeans.cluster_centers_)

for i, c in enumerate(centers_original):
  print(f"cluster {i}: Code Learning = {c[0]:.1f}, Developed Projects = {c[1]:.1f} ")
