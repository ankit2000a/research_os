import streamlit as st
import arxiv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from Bio import Entrez
import json
import asyncio
import pandas as pd
import sqlite3

# --- PAGE CONFIGURATION & AESTHETICS ---
st.set_page_config(
    page_title="Research OS",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# [CSS Styles Redacted For Brevity]
st.markdown("""
<style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    .main .block-container { padding: 2rem; }
    table { border: none; width: 100%; }
    table thead th { border-bottom: 2px solid #303640; font-size: 1rem; font-weight: 600; color: #CED4DA; text-align: left !important; }
    table tbody tr { border-bottom: 1px solid #303640; }
    table tbody td { padding: 0.75rem 0.5rem; vertical-align: top; text-align: left !important; }
    h1, h2, h3 { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- API KEY CONFIGURATION ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("🚨 Google API Key not found. Please add it to your Streamlit secrets.")
    st.stop()

# --- DATABASE FUNCTIONS ---
# [Database functions are unchanged and redacted for brevity]
DB_FILE = "research_os.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS briefs (id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, query TEXT NOT NULL, brief_data TEXT NOT NULL, papers_data TEXT NOT NULL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (project_id) REFERENCES projects (id))''')
    conn.commit()
    conn.close()
def add_project(name):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO projects (name) VALUES (?)", (name,))
        conn.commit()
    except sqlite3.IntegrityError:
        st.error(f"A project named '{name}' already exists.")
    finally:
        conn.close()
def rename_project(project_id, new_name):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE projects SET name = ? WHERE id = ?", (new_name, project_id))
    conn.commit()
    conn.close()
def get_projects():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, name FROM projects ORDER BY timestamp DESC")
    projects = c.fetchall()
    conn.close()
    return projects
def save_brief(project_id, query, brief_data, papers_data):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO briefs (project_id, query, brief_data, papers_data) VALUES (?, ?, ?, ?)",
              (project_id, query, json.dumps(brief_data), json.dumps(papers_data)))
    conn.commit()
    conn.close()
def get_briefs_for_project(project_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, query FROM briefs WHERE project_id = ? ORDER BY timestamp DESC", (project_id,))
    briefs = c.fetchall()
    conn.close()
    return briefs
def load_specific_brief(brief_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT query, brief_data, papers_data FROM briefs WHERE id = ?", (brief_id,))
    brief = c.fetchone()
    conn.close()
    if brief:
        query, brief_data_json, papers_data_json = brief
        return {
            "query": query, "brief_data": json.loads(brief_data_json), "papers_data": json.loads(papers_data_json)
        }
    return None

# --- DATA FETCHING & AI FUNCTIONS ---
# [fetch_data and generate_research_brief functions are unchanged and redacted]
async def fetch_data(source, query, max_results):
    if source == 'arXiv':
        return await fetch_arxiv_data(query, max_results)
    else:
        return await fetch_pubmed_data(query, max_results)
async def fetch_arxiv_data(query, max_results):
    loop = asyncio.get_running_loop()
    search = await loop.run_in_executor(None, lambda: arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance))
    results = await loop.run_in_executor(None, list, search.results())
    return [{"title": p.title, "summary": p.summary, "url": p.entry_id} for p in results if p.summary]
async def fetch_pubmed_data(query, max_results):
    loop = asyncio.get_running_loop()
    def search_and_fetch():
        Entrez.email = "your.email@example.com"
        handle = Entrez.esearch(db="pubmed", term=query, retmax=str(max_results), sort="relevance")
        record = Entrez.read(handle); handle.close()
        id_list = record["IdList"]
        if not id_list: return []
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
        records = Entrez.read(handle); handle.close()
        papers_data = []
        for i, article in enumerate(records.get('PubmedArticle', [])):
            try:
                papers_data.append({
                    "title": article['MedlineCitation']['Article']['ArticleTitle'],
                    "summary": article['MedlineCitation']['Article']['Abstract']['AbstractText'][0],
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{id_list[i]}/"
                })
            except (KeyError, IndexError): continue
        return papers_data
    return await loop.run_in_executor(None, search_and_fetch)
async def generate_research_brief(text, query):
    json_schema = {"type": "object", "properties": { "executive_summary": {"type": "string"}, "key_hypotheses_and_findings": {"type": "array", "items": {"type": "object", "properties": {"paper_title": {"type": "string"},"hypothesis": {"type": "string"},"finding": {"type": "string"}}}},"methodology_comparison": {"type": "array", "items": {"type": "object", "properties": {"paper_title": {"type": "string"},"methodology": {"type": "string"}}}}, "contradictions_and_gaps": {"type": "array", "items": {"type": "string"}}},"required": ["executive_summary", "key_hypotheses_and_findings", "methodology_comparison", "contradictions_and_gaps"]}
    generation_config = GenerationConfig(response_mime_type="application/json", response_schema=json_schema)
    model = genai.GenerativeModel('gemini-2.5-pro', generation_config=generation_config)
    prompt = f"Act as a world-class research analyst. Analyze the following abstracts on '{query}' and populate the provided JSON schema with a detailed intelligence brief.\n\nAbstracts:\n---\n{text}\n---"
    response = await model.generate_content_async(prompt)
    try:
        if not response.parts: raise ValueError("Model returned empty response.")
        return json.loads(response.text)
    except (ValueError, json.JSONDecodeError, AttributeError) as e:
        st.error(f"Error: AI response was not valid JSON. {e}")
        st.code(f"Raw Model Response:\n{response.text}", language="text")
        raise


# --- UI RENDERING ---
def display_research_brief(brief_data):
    """Renders the structured research brief."""
    query = brief_data["query"]
    st.header(f"Dynamic Research Brief: {query}")
    # [Table rendering logic is unchanged and redacted]
    st.subheader("Executive Summary")
    st.markdown(brief_data["brief_data"].get("executive_summary", "Not available."))
    st.subheader("Key Hypotheses & Findings")
    hypotheses_data = brief_data["brief_data"].get("key_hypotheses_and_findings", [])
    if hypotheses_data:
        df = pd.DataFrame(hypotheses_data)
        df.columns = ["Paper Title", "Hypothesis", "Finding"]
        st.markdown(df.to_html(index=False), unsafe_allow_html=True)

    st.subheader("Methodology Comparison")
    methods_data = brief_data["brief_data"].get("methodology_comparison", [])
    if methods_data:
        df = pd.DataFrame(methods_data)
        df.columns = ["Paper Title", "Methodology"]
        st.markdown(df.to_html(index=False), unsafe_allow_html=True)
    
    st.subheader("Identified Contradictions & Research Gaps")
    gaps_data = brief_data["brief_data"].get("contradictions_and_gaps", [])
    if gaps_data:
        for item in gaps_data: st.markdown(f"- {item}")
    
    with st.expander("View Source Papers & Raw Abstracts"):
        for paper in brief_data["papers_data"]:
            st.markdown(f"**{paper['title']}** ([Link]({paper.get('url', '#')}))")
            st.markdown(f"_{paper['summary']}_")

# --- MAIN APP LOGIC ---
init_db()
if 'current_brief' not in st.session_state:
    st.session_state.current_brief = None

# --- REVISED SIDEBAR ---
# The sidebar is now for configuration and the knowledge base only.
with st.sidebar:
    st.image("https://i.imgur.com/rLoaV0k.png", width=50)
    st.title("Research OS")
    st.markdown("The Insight Engine for Modern Research.")
    st.markdown("---")
    
    st.header("⚙️ Configuration")
    data_source = st.selectbox("Data Source", ["arXiv", "PubMed"])
    num_papers = st.slider("Number of Papers", min_value=2, max_value=5, value=3)

    st.markdown("---")
    st.header("🧠 Knowledge Base")
    # [Knowledge Base UI is unchanged and redacted]
    new_project_name = st.text_input("New Project Name", placeholder="e.g., Cancer Research Grant")
    if st.button("Create New Project", use_container_width=True):
        if new_project_name:
            add_project(new_project_name)
            st.rerun()
        else:
            st.warning("Please enter a name for the new project.")
    projects = get_projects()
    if not projects:
        st.info("Your projects will appear here.")
    for project_id, project_name in projects:
        with st.expander(project_name):
            col1, col2 = st.columns([4, 1])
            with col1:
                new_name = st.text_input("Rename", value=project_name, key=f"rename_{project_id}")
            with col2:
                if st.button("✔️", key=f"save_rename_{project_id}"):
                    rename_project(project_id, new_name)
                    st.rerun()
            briefs_in_project = get_briefs_for_project(project_id)
            if not briefs_in_project:
                st.write("_No briefs in this project yet._")
            for brief_id, brief_query in briefs_in_project:
                if st.button(brief_query, key=f"load_{brief_id}", use_container_width=True):
                    st.session_state.current_brief = load_specific_brief(brief_id)

# --- MAIN PAGE DISPLAY ---
st.title("🔬 Research OS")

# Display existing brief if one is in session state
if st.session_state.current_brief:
    display_research_brief(st.session_state.current_brief)
    # [Save to Project UI is unchanged and redacted]
    projects_for_saving = get_projects()
    if projects_for_saving:
        project_names = {name: id for id, name in projects_for_saving}
        selected_project_name = st.selectbox("Select project to save brief:", options=project_names.keys())
        if st.button("💾 Save to Project", key="save_to_project"):
            selected_project_id = project_names[selected_project_name]
            save_brief(selected_project_id, st.session_state.current_brief['query'], st.session_state.current_brief['brief_data'], st.session_state.current_brief['papers_data'])
            st.toast(f"Saved brief to '{selected_project_name}'!")
            st.rerun()
    else:
        st.warning("Please create a project in the sidebar to save this brief.")

# --- NEW CHAT INPUT ---
# This is the primary interaction point for the user now.
if prompt := st.chat_input("Enter your research topic..."):
    with st.spinner(f"Building brief for '{prompt}'..."):
        try:
            papers_data = asyncio.run(fetch_data(data_source, prompt, num_papers))
            if not papers_data:
                st.warning(f"No academic papers found for '{prompt}' on {data_source}.")
            else:
                combined_abstracts = "\n\n".join([f"**Paper:** {p['title']}\n{p['summary']}" for p in papers_data])
                brief_data = asyncio.run(generate_research_brief(combined_abstracts, prompt))
                st.session_state.current_brief = {
                    "query": prompt, "brief_data": brief_data, "papers_data": papers_data
                }
                # Rerun the script to display the new brief and the save options
                st.rerun()
        except Exception:
            st.error("Failed to generate the research brief. The model may have refused to answer.")