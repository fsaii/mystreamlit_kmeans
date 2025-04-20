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