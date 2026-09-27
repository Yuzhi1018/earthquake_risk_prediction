# Earthquake Impact Risk Analysis

A Python-based data analysis project that uses earthquake data from the USGS API to explore earthquake characteristics and classify high-impact seismic events.

## Overview

This project builds an end-to-end data analysis pipeline for earthquake impact analysis. It retrieves earthquake data from the USGS Earthquake API, performs data cleaning and feature engineering, trains statistical and machine-learning models, evaluates model performance, visualizes geographic patterns, and generates an AI-assisted risk report.

## Project Pipeline

1. Retrieve earthquake data from the USGS API
2. Clean and preprocess the dataset
3. Perform exploratory data analysis
4. Engineer features including magnitude, depth, tsunami indicator, latitude, and longitude
5. Train a logistic regression model to classify high-impact events
6. Compare the full model with a magnitude-only baseline
7. Evaluate performance using ROC-AUC and cross-validation
8. Analyze regional residual patterns
9. Visualize earthquake and predicted impact patterns using Folium heatmaps
10. Generate an AI-assisted risk report

## Data Source

Earthquake data are retrieved from the USGS Earthquake Catalog API.

The current dataset includes earthquakes with magnitude 4.0 or greater from 2024.

## High-Impact Definition

A high-impact earthquake is defined using the USGS significance score:

```python
high_impact = properties.sig >= 400