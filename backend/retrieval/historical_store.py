"""
historical_store.py
===================
Historical Project-Risk Knowledge Retrieval Engine.

Maintains a searchable repository of past IT project case studies, observed risks,
root cause analyses, and mitigation strategies.

Integration:
- Uses ChromaDB when the `chromadb` package is installed.
- Features a zero-dependency Vector Cosine Similarity engine (TF-IDF + Cosine Distance)
  as a benchmark fallback when ChromaDB is unavailable in the environment.
"""

import time
from typing import List, Dict, Any, Optional
import numpy as np

# Curated synthetic historical IT project case studies
HISTORICAL_CASES = [
    {
        "id": "CASE-001",
        "title": "Core Banking Agile Migration",
        "domain": "Fintech / Banking",
        "description": "Legacy mainframe migration to cloud-native microservices suffered 45-day schedule slip due to unmanaged scope changes in transaction reconciliation and high staff turnover.",
        "observed_risks": ["Severe requirement volatility", "Key architect attrition", "Schedule delay exceeding 40 days"],
        "root_cause": "Underestimated integration complexity with legacy COBOL subsystems and lack of architectural documentation.",
        "mitigation_strategy": "Instituted strict scope freeze, brought in specialized external legacy consultants, and automated contract testing.",
        "tags": ["scope creep", "migration", "attrition", "banking", "schedule delay"]
    },
    {
        "id": "CASE-002",
        "title": "Hospital Management EHR Rollout",
        "domain": "Healthcare",
        "description": "Electronic Health Record deployment encountered massive budget variance (+42%) and critical defect accumulation in patient records synchronization.",
        "observed_risks": ["Critical bug count > 20", "Budget overrun", "Stakeholder dissatisfaction"],
        "root_cause": "Inadequate load and stress testing prior to UAT with real-time patient data streams.",
        "mitigation_strategy": "Halted new feature delivery for two sprints to execute defect burn-down and dedicated performance tuning.",
        "tags": ["defect", "budget overrun", "healthcare", "uat", "performance"]
    },
    {
        "id": "CASE-003",
        "title": "Omnichannel E-Commerce Scalability Overhaul",
        "domain": "Retail",
        "description": "Peak-season promotion engine rewrite experienced team burnout and 30% team attrition due to unrealistic milestone timelines.",
        "observed_risks": ["Team attrition rate > 25%", "Resource utilization > 120%", "Poor communication"],
        "root_cause": "Compressed schedule without buffer time for regression testing across distributed micro-frontends.",
        "mitigation_strategy": "De-scoped non-essential recommendation engine features; adopted canary deployment strategy.",
        "tags": ["attrition", "burnout", "resource overutilization", "retail", "scope creep"]
    },
    {
        "id": "CASE-004",
        "title": "Autonomous Fleet Telematics Gateway",
        "domain": "Automotive / IoT",
        "description": "High complexity edge processing module faced communication breakdown between hardware firmware teams and cloud ingestion engineers.",
        "observed_risks": ["High complexity", "Poor communication health", "Unresolved blocker tickets"],
        "root_cause": "Siloed team communication, disparate API definitions, and lack of mock integration servers.",
        "mitigation_strategy": "Mandated bi-weekly joint architectural syncs, adopted OpenAPI contract-first development, and established unified Slack escalation channel.",
        "tags": ["communication", "iot", "blocker", "complexity", "api contract"]
    },
    {
        "id": "CASE-005",
        "title": "Cross-Border Payment Gateway Upgrade",
        "domain": "Fintech",
        "description": "Project stayed completely on track with 98% test pass rate and proactive stakeholder communication through rigorous CI/CD pipeline.",
        "observed_risks": ["Minor regulatory requirement changes"],
        "root_cause": "Clear architectural blueprints and experienced senior leadership.",
        "mitigation_strategy": "Maintained continuous stakeholder feedback loops and daily automated regression runs.",
        "tags": ["on track", "fintech", "cicd", "high quality", "testing"]
    },
    {
        "id": "CASE-006",
        "title": "Global Logistics Dispatch Optimization",
        "domain": "Supply Chain",
        "description": "Graph optimization module experienced severe schedule delays and budget burn from third-party mapping API rate limits.",
        "observed_risks": ["Schedule delay > 50 days", "Budget variance > 30%", "Third party dependency"],
        "root_cause": "Vendor dependency failure and delayed risk escalation by middle management.",
        "mitigation_strategy": "Switched to self-hosted open-source routing engine and renegotiated service level agreements.",
        "tags": ["schedule delay", "vendor risk", "budget overrun", "logistics"]
    },
    {
        "id": "CASE-007",
        "title": "SaaS Customer Billing Platform Refactor",
        "domain": "B2B SaaS",
        "description": "Multi-tenant subscription billing refactor faced quality crisis with over 150 open bugs and declining customer satisfaction.",
        "observed_risks": ["Bug count > 150", "Test pass percentage < 70%", "Stakeholder satisfaction < 2.5"],
        "root_cause": "Rushed code merges without mandatory peer review checks and insufficient test automation.",
        "mitigation_strategy": "Enforced strict PR approval gates, mandatory unit test coverage > 85%, and daily bug triage.",
        "tags": ["defect", "bug count", "quality", "saas", "stakeholder dissatisfaction"]
    },
    {
        "id": "CASE-008",
        "title": "Mobile Banking Biometric Authentication",
        "domain": "Fintech / Mobile",
        "description": "Biometric SDK integration executed smoothly with low budget variance and high stakeholder alignment across iOS and Android.",
        "observed_risks": ["Hardware fragmentation edge cases"],
        "root_cause": "Well-scoped MVP and robust device cloud testing matrix.",
        "mitigation_strategy": "Extensive beta testing cohort and progressive feature rollout flags.",
        "tags": ["on track", "mobile", "fintech", "biometrics", "mvp"]
    },
    {
        "id": "CASE-009",
        "title": "Enterprise Data Lakehouse Migration",
        "domain": "Data Engineering",
        "description": "Migration of 500TB analytical pipelines struggled with 35% cost overrun due to unoptimized distributed Spark queries.",
        "observed_risks": ["Budget utilization > 140%", "Budget variance > 40%", "Critical data loss bugs"],
        "root_cause": "Lack of FinOps governance and absence of automated query partition pruning.",
        "mitigation_strategy": "Implemented cluster auto-termination policies, query cost alerts, and data tiering.",
        "tags": ["budget overrun", "cloud cost", "data lake", "finops", "spark"]
    },
    {
        "id": "CASE-010",
        "title": "In-Flight Entertainment Wireless Streaming",
        "domain": "Aviation / Media",
        "description": "Severe certification delay and hardware vendor supply chain constraints delayed airborne testing by 4 months.",
        "observed_risks": ["Schedule delay > 90 days", "Regulatory compliance blockers", "Critical issues"],
        "root_cause": "Late discovery of FAA certification requirement amendments.",
        "mitigation_strategy": "Assigned full-time compliance liaison and decoupled non-safety software streams.",
        "tags": ["schedule delay", "compliance", "hardware", "aviation", "blocker"]
    },
    {
        "id": "CASE-011",
        "title": "AI Conversational Support Bot Implementation",
        "domain": "Customer Support",
        "description": "LLM agent integration experienced hallucination risks and high response latency, triggering user complaints.",
        "observed_risks": ["Stakeholder satisfaction < 3.0", "Scope changes", "Quality drift"],
        "root_cause": "Absence of retrieval ground truth verification and prompt injection safeguards.",
        "mitigation_strategy": "Introduced guardrail filters, human-in-the-loop fallback for low-confidence queries, and semantic caching.",
        "tags": ["ai", "nlp", "stakeholder dissatisfaction", "quality", "hallucination"]
    },
    {
        "id": "CASE-012",
        "title": "Insurance Claims Fraud Detection Engine",
        "domain": "Insurtech",
        "description": "Machine learning fraud scoring pipeline delivered within timeline, meeting all accuracy KPIs through structured sprint cadence.",
        "observed_risks": ["Data drift monitoring requirements"],
        "root_cause": "Clear specification and modular microservice design.",
        "mitigation_strategy": "Automated model retraining pipelines and established weekly cross-functional governance.",
        "tags": ["on track", "machine learning", "insurance", "fraud detection"]
    },
    {
        "id": "CASE-013",
        "title": "Telecommunications Billing Mediation System",
        "domain": "Telecom",
        "description": "Legacy CDR mediation replacement suffered 60-day schedule delay and 25% engineer attrition under high stress.",
        "observed_risks": ["Schedule delay > 60 days", "Team attrition", "Critical blocker count > 10"],
        "root_cause": "Poor management communication and aggressive unrealistic delivery commitments made to executives.",
        "mitigation_strategy": "Re-baselined project schedule, instituted mental health days, and redistributed workload.",
        "tags": ["schedule delay", "attrition", "communication", "telecom", "management"]
    },
    {
        "id": "CASE-014",
        "title": "Cybersecurity SOC Threat Intelligence Mesh",
        "domain": "Cybersecurity",
        "description": "Threat mesh integration succeeded through agile two-week iterations and disciplined backlog grooming.",
        "observed_risks": ["Fast-evolving adversary techniques"],
        "root_cause": "Strong engineering expertise and automated threat simulation suites.",
        "mitigation_strategy": "Continuous automated integration and tabletop incident exercises.",
        "tags": ["on track", "cybersecurity", "soc", "agile"]
    },
    {
        "id": "CASE-015",
        "title": "EdTech Interactive Virtual Classroom",
        "domain": "Education",
        "description": "WebRTC video streaming experienced severe stuttering and test pass failures under simulated concurrent classroom load.",
        "observed_risks": ["Test pass rate < 65%", "Critical bugs", "Budget variance > 20%"],
        "root_cause": "Insufficient infrastructure capacity planning for WebRTC SFU media servers.",
        "mitigation_strategy": "Migrated to managed WebRTC cloud PaaS to offload media scaling complexity.",
        "tags": ["webrtc", "defect", "quality", "scalability", "infrastructure"]
    }
]


class HistoricalRiskKnowledgeStore:
    """
    Searchable historical risk repository with ChromaDB or Cosine Vector fallback.
    """
    def __init__(self, use_chromadb_if_available: bool = True):
        self.chroma_client = None
        self.collection = None
        self.engine_type = "Vector Cosine Similarity (TF-IDF Space)"
        self.cases = HISTORICAL_CASES
        self._init_vector_engine()

        if use_chromadb_if_available:
            try:
                import chromadb
                self.chroma_client = chromadb.Client()
                self.collection = self.chroma_client.get_or_create_collection(name="it_project_risks")
                
                # Populate ChromaDB collection
                ids = [c["id"] for c in self.cases]
                docs = [f"{c['title']} - {c['description']} Risks: {', '.join(c['observed_risks'])}" for c in self.cases]
                metadatas = [{"title": c["title"], "domain": c["domain"], "root_cause": c["root_cause"]} for c in self.cases]
                self.collection.add(ids=ids, documents=docs, metadatas=metadatas)
                self.engine_type = "ChromaDB Vector Store"
            except Exception:
                # Graceful fallback: chromadb not installed or initialization failed
                self.chroma_client = None
                self.collection = None

    def _init_vector_engine(self):
        """
        Build fallback vector representation using Scikit-learn TF-IDF.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self.corpus = [
            f"{c['title']} {c['description']} {' '.join(c['observed_risks'])} {' '.join(c['tags'])} {c['mitigation_strategy']}"
            for c in self.cases
        ]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def query(self, query_text: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieve top-k relevant historical project cases.

        Returns
        -------
        dict containing:
            - matches: List of dicts (case details + similarity score)
            - latency_ms: float
            - engine: str
        """
        start_t = time.perf_counter()

        if self.collection is not None:
            try:
                res = self.collection.query(query_texts=[query_text], n_results=top_k)
                latency_ms = (time.perf_counter() - start_t) * 1000.0
                matches = []
                for i in range(len(res['ids'][0])):
                    case_id = res['ids'][0][i]
                    orig_case = next(c for c in self.cases if c["id"] == case_id)
                    matches.append({
                        "id": case_id,
                        "title": orig_case["title"],
                        "description": orig_case["description"],
                        "observed_risks": orig_case["observed_risks"],
                        "mitigation_strategy": orig_case["mitigation_strategy"],
                        "similarity_score": round(1.0 - float(res['distances'][0][i]) if 'distances' in res else 0.85, 3)
                    })
                return {
                    "matches": matches,
                    "latency_ms": round(latency_ms, 3),
                    "engine": self.engine_type
                }
            except Exception:
                pass

        # Vector Cosine Similarity Fallback
        from sklearn.metrics.pairwise import cosine_similarity
        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = np.argsort(sims)[::-1][:top_k]
        latency_ms = (time.perf_counter() - start_t) * 1000.0

        matches = []
        for idx in top_indices:
            orig_case = self.cases[idx]
            matches.append({
                "id": orig_case["id"],
                "title": orig_case["title"],
                "description": orig_case["description"],
                "observed_risks": orig_case["observed_risks"],
                "mitigation_strategy": orig_case["mitigation_strategy"],
                "similarity_score": round(float(sims[idx]), 3)
            })

        return {
            "matches": matches,
            "latency_ms": round(float(latency_ms), 3),
            "engine": self.engine_type
        }
