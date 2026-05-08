import pandas as pd
import re

df = pd.read_csv('cleaned_job_skills_technical_only.csv')

# Comprehensive mapping for all 45 roles
role_mapping = {
    "ai engineer": "python, machine learning, deep learning, pytorch, tensorflow, neural networks, nlp, computer vision, langchain, rag, llm, huggingface, docker, kubernetes, mlflow, model deployment, experiment tracking, generative ai, prompt engineering",
    "machine learning engineer": "python, machine learning, pytorch, tensorflow, scikit-learn, docker, kubernetes, mlflow, data pipelines, feature engineering",
    "data scientist": "python, sql, pandas, numpy, statistics, machine learning, seaborn, matplotlib, tableau, power bi",
    "prompt engineer": "llm, prompt engineering, langchain, rag, openai, huggingface, nlp, generative ai",
    "mlops specialist": "docker, kubernetes, ci/cd, terraform, mlflow, prometheus, grafana",
    "software developer": "python, javascript, sql, git, docker, rest apis, node.js, react, ci/cd, linux",
    "software engineer": "python, javascript, sql, git, docker, rest apis, node.js, react, ci/cd, linux",
    "backend developer": "python, java, node.js, sql, rest apis, graphql, docker, kubernetes",
    "full-stack developer": "javascript, react, node.js, python, sql, git, docker, html, css, typescript",
    "full stack developer": "javascript, react, node.js, python, sql, git, docker, html, css, typescript",
    "frontend developer": "javascript, react, html, css, typescript, git, tailwind",
    "devops engineer": "docker, kubernetes, ci/cd, terraform, ansible, linux, jenkins",
    "cloud engineer": "aws, azure, terraform, docker, kubernetes, cloud computing",
    "cloud architect": "aws, azure, terraform, kubernetes, docker, infrastructure as code",
    "cybersecurity engineer": "penetration testing, network security, linux, docker, kubernetes, cloud security, ethical hacking",
    "cybersecurity analyst": "penetration testing, network security, siem, incident response",
    "site reliability engineer": "kubernetes, docker, prometheus, grafana, linux, observability",
    "big data engineer": "spark, hadoop, kafka, python, sql, data pipelines",
    "blockchain engineer": "blockchain, solidity, web3, ethereum, smart contracts",
    "quantum computing specialist": "quantum computing, qiskit, python",
    "mobile applications developer": "react native, flutter, rest apis",
    "ui/ux designer": "figma, wireframing, prototyping, user research, responsive design",
    "ui ux designer": "figma, wireframing, prototyping, user research",
    "ux designer": "figma, wireframing, prototyping, user research",
    "ai ethics officer": "ai ethics, responsible ai, bias detection, fairness",
    "analyst": "data analysis, sql, excel, reporting",
    
    # Non-tech roles (clean, no tech skills)
    "dentist": "patient care, medical knowledge, healthcare, clinical skills",
    "registered nurse": "patient care, healthcare, medical knowledge",
    "nurse": "patient care, healthcare, medical knowledge",
    "marketing manager": "seo, google analytics, content marketing, social media marketing",
    "digital marketing specialist": "seo, google analytics, content marketing, social media marketing",
    "social media manager": "social media marketing, content creation",
    "financial analyst": "excel, sql, financial modeling, tableau, power bi",
    "business analyst": "sql, excel, tableau, power bi, requirements gathering",
    "product manager": "jira, agile, roadmapping, metrics, user stories",
    "project manager": "agile, jira, scrum, risk management",
    "human resources manager": "recruitment, talent acquisition, compliance",
    "operations manager": "process improvement, lean, six sigma, kpi tracking",
    "business development manager": "sales, lead generation, negotiation, crm",
    "sales representative": "sales, crm, negotiation",
    "salesperson": "sales, crm, negotiation",
    "content manager": "content writing, seo, copywriting",
    "market research analyst": "data analysis, market trends",
    "customer success manager": "customer service, onboarding, crm",
    "account manager": "client relations, account management, crm",
    "director of operations": "operations management, leadership, process improvement",
    "mental health practitioner": "counseling, mental health",
    "counselor": "counseling, mental health",
    "restaurant specialist": "customer service, operations"
}

def normalize(text):
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def get_clean_skills(job_role):
    norm = normalize(job_role)
    for key, skills in role_mapping.items():
        if key in norm:
            return skills
    # Final fallback based on keywords
    if any(word in norm for word in ["dentist", "nurse", "medical", "health"]):
        return "patient care, medical knowledge, healthcare"
    if any(word in norm for word in ["marketing", "social media"]):
        return "seo, google analytics, content marketing, social media marketing"
    if any(word in norm for word in ["finance", "financial"]):
        return "excel, sql, financial modeling, tableau, power bi"
    return "professional domain skills"   # Only as last resort

df['skills_text'] = df['job_role'].apply(get_clean_skills)

# Save
df.to_csv('cleaned_job_skills_technical_only.csv', index=False)

print("✅ Final strong cleaning completed!")
print("Remaining generic rows:", (df['skills_text'] == "professional domain skills").sum())

# Quick verification
print("\nAI Engineer sample:", df[df['job_role'].str.contains("AI Engineer", case=False)]['skills_text'].iloc[0] if not df[df['job_role'].str.contains("AI Engineer", case=False)].empty else "No AI Engineer")
print("Full-Stack sample:", df[df['job_role'].str.contains("Full-Stack|Full Stack", case=False)]['skills_text'].iloc[0] if not df[df['job_role'].str.contains("Full-Stack|Full Stack", case=False)].empty else "No Full-Stack")
print("Dentist sample:", df[df['job_role'].str.contains("Dentist", case=False)]['skills_text'].iloc[0] if not df[df['job_role'].str.contains("Dentist", case=False)].empty else "No Dentist")