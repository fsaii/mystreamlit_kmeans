# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 15:27:51 2025

@author: LAB
"""

#app
import streamlit as st
import pickle
import matplotlib.pyplot as plt

#Load model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
    
#set the page config
st.set_page_config(page_title="k-means Clustering App", layout="centered")

#set title
st.title("k-means Clustering Visualizer")

#display cluster centers
st.subheader("Example Data for Visualization")
st.markdown("This demo uses example data (2D) to illustrate clustering results.")

#load from a saved dataset or generate synthetic data
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

#predict using the loaded model
y_kmeans = loaded_model.predict(X)

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
centers = loaded_model.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, label='Centroids')

ax.set_title('k-Means Clustering')
ax.legend()

st.pyplot(fig)