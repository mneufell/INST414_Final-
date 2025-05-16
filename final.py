import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler




df = pd.read_csv('states.csv')


# Filter for female names
df_female = df[df['Sex'] == 'F']
df_female = df_female[df_female['Year'] > 2020]

# Group by state and name, then sum the occurrences
df_grouped = df_female.groupby(['State', 'Name'])['Count'].sum().reset_index()
# Sort by state and count in descending order
df_sorted = df_grouped.sort_values(by=['State', 'Count'], ascending=[True, False])
# Get the top 5 names for each state
top_5_names = df_sorted.groupby('State').head(5)


# Create bipartite graph: nodes = States + Names
g = nx.Graph()
states = top_5_names['State'].unique()
names = top_5_names['Name'].unique()
g.add_nodes_from(top_5_names['State'].unique(), bipartite=0)  # State nodes
g.add_nodes_from(top_5_names['Name'].unique(), bipartite=1)   # Name nodes

# Add edges between states and their top 5 names
edges = list(zip(top_5_names['State'], top_5_names['Name']))
g.add_edges_from(edges)

# Compute degree centrality for name nodes
# (names are in bipartite set 1)
centrality = nx.bipartite.degree_centrality(g, top_5_names['Name'].unique())

# Filter to just baby names
name_centrality = {name: centrality[name] for name in top_5_names['Name'].unique()}

# Sort by centrality descending
sorted_centrality = sorted(name_centrality.items(), key=lambda x: x[1], reverse=True)

# Display top 10 most central baby names
for name, centrality_score in sorted_centrality[:10]:
    print(f"{name}: {centrality_score:.3f}")
    
 #Project to names: name-name co-occurrence based on shared states
name_projection = nx.bipartite.weighted_projected_graph(g, names)

# Convert to adjacency matrix
adj_matrix = nx.to_pandas_adjacency(name_projection, nodelist=names)

# Normalize
scaler = StandardScaler()
scaled_matrix = scaler.fit_transform(adj_matrix)

# Elbow method: WCSS for k = 1 to 10
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_matrix)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method: K for Baby Name Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()



# KMeans clustering (k = 3)
kmeans = KMeans(n_clusters=3, random_state=50)
clusters = kmeans.fit_predict(scaled_matrix)
name_cluster_map = dict(zip(names, clusters))
top_5_names['Cluster'] = top_5_names['Name'].map(name_cluster_map)

for cluster_num in range(3):
    print(f"\nCluster {cluster_num} Baby Name:")
    print(top_5_names[top_5_names['Cluster'] == cluster_num][['Name', 'State']].head(10))