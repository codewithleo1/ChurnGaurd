import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def get_llm_explanation(customer_id, churn_probability, shap_explanation):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    prompt = (
        f"You are a customer success analyst.\n"
        f"Customer {customer_id} has a {churn_probability:.1%} churn probability.\n"
        f"Top reasons:\n{shap_explanation}\n"
        f"Write 2-3 sentences explaining why and suggest one retention action. "
        f"Write for a CS rep, not a data scientist. No mention of SHAP or ML."
    )
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()