import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score

df = pd.read_csv('earthquake_data_eda.csv')
df1 = df.dropna(subset=['properties.mag','depth','properties.sig']).copy()


#properties.mag: magnitude of the earthquake is the main factor that influences significance
threshold=400
df1['tsunami_flag'] = df1['properties.tsunami'].fillna(0)
df1['high_impact'] = (df1['properties.sig'] >= threshold).astype(int)

print(df1['high_impact'].value_counts(normalize=True))

features = ['properties.mag', 'depth', 'tsunami_flag','latitude','longitude']
X = df1[features]
y = df1['high_impact']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = LogisticRegression(max_iter=2000, class_weight='balanced')
clf.fit(X_train, y_train)

y_prob = clf.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.7).astype(int)

#generate risk probability and risk level for all earthquakes
df1['risk_prob'] = clf.predict_proba(X)[:, 1]
df1['risk_level'] = pd.cut(df1['risk_prob'], bins=[0, 0.7, 0.9, 1.0], labels=['Low/Moderate', 'High', 'Very High'], 
                           include_lowest=True).astype(str)
                    
print('ROC-AUC:', roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))
print('Weights:', dict(zip(features, clf.coef_[0])))

#cross-validation
scores = cross_val_score(
    LogisticRegression(max_iter=2000, class_weight="balanced"),
    X,
    y,
    cv=5,
    scoring="roc_auc"
)
print("CV AUC:", scores.mean())

#model A: features that only contain magnitude
features_A = ['properties.mag']
X_A = df1[features_A]

X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, df1['high_impact'], test_size=0.2, random_state=42, stratify=y)

clf_A = LogisticRegression(max_iter=2000, class_weight='balanced')
clf_A.fit(X_train_A, y_train_A)

y_prob_A = clf_A.predict_proba(X_test_A)[:, 1]
print('Model A ROC-AUC:', roc_auc_score(y_test_A, y_prob_A))

#cross-validation for model A
clf_A = LogisticRegression(max_iter=2000, class_weight='balanced')
scores_A = cross_val_score(clf_A, X_A, y, cv=5, scoring='roc_auc')

print('Model A CV AUC:', scores_A.mean())

#calculate odds ratios
odds_ratio = np.exp(clf.coef_[0])
print(dict(zip(features, odds_ratio)))

#Regional Vulnerability Analysis
#only using physical factors
base_features = ['properties.mag', 'depth','tsunami_flag']
dfv = df1.dropna(subset=base_features + ['properties.sig','latitude', 'longitude']).copy()

reg = LinearRegression()
reg.fit(dfv[base_features], dfv['properties.sig'])

dfv['sig_pred'] = reg.predict(dfv[base_features])
dfv['sig_resid'] = dfv['properties.sig'] - dfv['sig_pred']
print(dfv['sig_resid'].describe())

#divide the world into regions
grid = 2.0
dfv['lat_bin'] = (np.floor(dfv['latitude'] / grid) * grid).astype(float)
dfv['lon_bin'] = (np.floor(dfv['longitude'] / grid) * grid).astype(float)
dfv['cell'] = dfv['lat_bin'].astype(str) + ',' + dfv['lon_bin'].astype(str)

#vulnerability index for every cell
cell_stats = (
    dfv.groupby(["lat_bin", "lon_bin"])
       .agg(
           n=("sig_resid", "size"),
           vuln_mean=("sig_resid", "mean"),
           vuln_median=("sig_resid", "median"),
           p_high=("sig_resid", lambda s: (s > 100).mean()),  # 阈值100你可以调
           avg_mag=("properties.mag", "mean")
       )
       .reset_index()
)

# Filtering small sample cells
cell_stats = cell_stats[cell_stats["n"] >= 20].copy()
print(cell_stats.sort_values("vuln_mean", ascending=False).head(10))

cell = dfv[(dfv["lat_bin"] == 40.0) & (dfv["lon_bin"] == -126.0)].copy()
print("cell n:", len(cell))

# check residual distribution in this cell
print(cell["sig_resid"].describe())

# check top 10 most significant earthquakes in this cell
print(cell[["properties.mag","depth","tsunami_flag","properties.sig","sig_pred","sig_resid"]]
      .sort_values("sig_resid", ascending=False)
      .head(10))

#risk map--predicted high-impact probability distribution
heat_df = df1.dropna(subset=['latitude', 'longitude', 'risk_prob']).copy()
heat_data = heat_df[['latitude', 'longitude', 'risk_prob']].values.tolist()
m = folium.Map(location=[0, 0], zoom_start=2)
HeatMap(heat_data, radius=10, blur=15, max_zoom=4).add_to(m)
m.save('risk_probability_heatmap.html')
print('Saved: risk_probability_heatmap.html')