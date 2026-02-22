from openai import OpenAI
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_risk_report(payload: dict):

    prompt = f"""
    You are a risk assessment assistant. Based on the following earthquake data, generate a concise risk report for emergency response teams. The report should include:

    Event Information:
    Location: {payload['place']}
    Magnitude: {payload['mag']}
    Depth: {payload['depth_km']}km
    Tsunami:{payload['tsunami_flag']}
    Predicted High Impact Probability: {payload['risk_prob']:.2f}

    Provide:
    1.Executive Summary
    2.Risk Interpretation
    3.Recommeneded Immediate Actions
    4.Policy Recommendations
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.4
    )

    return response.choices[0].message.content

