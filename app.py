import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("Customer Segmentation Interactive Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv('Customer_Segmentation_Dataset.csv')
    return df

df = load_data()

def preprocess_data(df):
    df['Income'].fillna(df['Income'].mean(), inplace=True)
    columns_to_drop = [
        'ID', 'Z_CostContact','Z_Revenue','AcceptedCmp1',
        'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
        'Response', 'Complain'
    ]
    df.drop(columns=columns_to_drop, inplace=True)
    df['Age'] = 2025 - df['Year_Birth']
    df.drop(columns=['Year_Birth'], inplace=True)
    df['Total_Children'] = df['Kidhome'] + df['Teenhome']
    df.drop(columns=['Kidhome', 'Teenhome'], inplace=True)
    mnt_columns = [col for col in df.columns if col.startswith('Mnt')]
    df['Total_Spend'] = df[mnt_columns].sum(axis=1)
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
    df['Loyalty'] = (pd.Timestamp('2025-01-01') - df['Dt_Customer']).dt.days
    df.drop(columns=['Dt_Customer'], inplace=True)
    df['Marital_Status'].replace({'Absurd': 'Single', 'YOLO': 'Single', 'Alone': 'Single'}, inplace=True)
    df['Education'] = df['Education'].replace({'2n Cycle': 'Basic'})
    df = pd.get_dummies(df, columns=['Education', 'Marital_Status'], drop_first=True)
    return df

df_proc = preprocess_data(df.copy())
features_to_scale = ['Income', 'Age', 'Loyalty', 'Total_Children', 'Total_Spend']
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_proc[features_to_scale])
df_scaled = pd.DataFrame(df_scaled, columns=features_to_scale)

# --- Sidebar Controls ---
st.sidebar.header("Clustering Controls")
n_clusters = st.sidebar.slider("Select number of clusters (segments)", min_value=2, max_value=10, value=4)

# --- EDA Section ---
st.header("Exploratory Data Analysis")

# 1. Income vs Total Spend
st.subheader("Income vs Total Spend")
fig1, ax1 = plt.subplots(figsize=(5, 4))
sns.scatterplot(data=df_proc, x='Income', y='Total_Spend', ax=ax1)
ax1.set_title('Income vs Total Spend')
ax1.set_xlabel('Income')
ax1.set_ylabel('Total Spend')
st.pyplot(fig1)

# 2. Age Distribution
st.subheader("Age Distribution")
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.histplot(df_proc['Age'], bins=20, ax=ax2)
ax2.set_title('Age Distribution of Customers')
ax2.set_xlabel('Age')
ax2.set_ylabel('Count')
st.pyplot(fig2)

# 3. Customer Loyalty vs Total Spend
st.subheader("Customer Loyalty vs Total Spend")
fig3, ax3 = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=df_proc, x='Loyalty', y='Total_Spend', ax=ax3)
ax3.set_title('Customer Loyalty vs Total Spend')
ax3.set_xlabel('Customer Loyalty (Days)')
ax3.set_ylabel('Total Spend')
st.pyplot(fig3)

# 4. Total Spend by Number of Children
st.subheader("Total Spend by Number of Children")
fig4, ax4 = plt.subplots(figsize=(6, 4))
sns.stripplot(data=df_proc, x='Total_Children', y='Total_Spend', jitter=True, ax=ax4)
ax4.set_title('Total Spend by Number of Children')
ax4.set_xlabel('Total Children')
ax4.set_ylabel('Total Spend')
st.pyplot(fig4)

# 5. Heatmap of Correlation Matrix
st.subheader("Correlation Heatmap")
correlation_features = ['Age', 'Income', 'Loyalty', 'Total_Children', 'Total_Spend']
correlation_matrix = df_proc[correlation_features].corr()
fig5, ax5 = plt.subplots(figsize=(6, 5))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, ax=ax5)
ax5.set_title('Correlation Heatmap')
st.pyplot(fig5)

# --- KMeans Clustering Section ---
st.header("KMeans Clustering & Cluster Profiles")

km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = km.fit_predict(df_scaled)
df_proc['Cluster'] = labels

# 1. Cluster sizes
st.subheader("Cluster Sizes")
st.write(df_proc['Cluster'].value_counts().sort_index())

# 2. Cluster medians for key features
st.subheader("Cluster Median Feature Table")
cluster_table = df_proc.groupby('Cluster')[['Age', 'Income', 'Total_Spend', 'Total_Children']].median().round(1)
st.dataframe(cluster_table)

# 3. Radar chart for cluster profiles
st.subheader("Radar Chart of Cluster Profiles")
radar_df = cluster_table.copy()
radar_df_normalized = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())
labels_radar = radar_df.columns
num_vars = len(labels_radar)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig_radar, ax_radar = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for i, row in radar_df_normalized.iterrows():
    values = row.tolist() + row.tolist()[:1]
    ax_radar.plot(angles, values, label=f'Cluster {i}')
    ax_radar.fill(angles, values, alpha=0.1)
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(labels_radar)
ax_radar.set_yticklabels([])
ax_radar.set_title("Radar Chart of Clusters", size=14)
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
st.pyplot(fig_radar)

# --- Clustering Diagnostics Section ---
st.header("Clustering Diagnostics")

# Elbow Method Plot
st.subheader("Elbow Method to Determine Optimal k")
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(df_scaled)
    wcss.append(km.inertia_)
fig6, ax6 = plt.subplots()
ax6.plot(range(1, 11), wcss, marker='o')
ax6.set_xlabel('Number of Clusters (k)')
ax6.set_ylabel('WCSS')
ax6.set_title('Elbow method to determine optimal k')
ax6.grid(True)
st.pyplot(fig6)

# Silhouette Score Plot
st.subheader("Silhouette Scores for Different k")
sil_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = km.fit_predict(df_scaled)
    sil_scores.append(silhouette_score(df_scaled, labels_temp))
fig7, ax7 = plt.subplots()
ax7.plot(range(2, 11), sil_scores, marker='o')
ax7.set_xlabel('Number of clusters')
ax7.set_ylabel('Silhouette Score')
ax7.set_title('Silhouette Scores')
ax7.grid(True)
st.pyplot(fig7)

st.markdown("---")
st.markdown("**Developed by AYUSH SAINI  2025 " ) 