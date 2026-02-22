
import pandas as pd
from ai_report import generate_risk_report

df1=pd.read_csv('earthquake_data_eda.csv')
top_event = df1.sort_values(by='risk_prob', ascending=False).iloc[0]

payload = {
    'place': top_event['properties.place'],
    'mag': top_event['properties.mag'],
    'depth_km': top_event['depth'],
    'tsunami_flag': top_event['properties.tsunami'],
    'risk_prob': top_event['risk_prob']
}  

report = generate_risk_report(payload)

print(report)