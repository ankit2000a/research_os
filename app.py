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
    .main .block-container { padding: 2rem; padding-bottom: 5rem; }
    table { border: none; width: 100%; }
    table thead th { border-bottom: 2px solid #303640; font-size: 1rem; font-weight: 600; color: #CED4DA; text-align: left !important; }
    table tbody tr { border-bottom: 1px solid #303640; }
    table tbody td { padding: 0.75rem 0.5rem; vertical-align: top; text-align: left !important; }
    h1, h2, h3 { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- API KEY & DATABASE SETUP ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("🚨 Google API Key not found. Please add it to your Streamlit secrets.")
    st.stop()

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
    try: c.execute("INSERT INTO projects (name) VALUES (?)", (name,)); conn.commit()
    except sqlite3.IntegrityError: st.error(f"A project named '{name}' already exists.")
    finally: conn.close()
def get_projects():
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("SELECT id, name FROM projects ORDER BY timestamp DESC"); projects = c.fetchall(); conn.close()
    return projects
def save_brief(project_id, query, brief_data, papers_data):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("INSERT INTO briefs (project_id, query, brief_data, papers_data) VALUES (?, ?, ?, ?)", (project_id, query, json.dumps(brief_data), json.dumps(papers_data))); conn.commit(); conn.close()
def get_briefs_for_project(project_id):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("SELECT id, query FROM briefs WHERE project_id = ? ORDER BY timestamp DESC", (project_id,)); briefs = c.fetchall(); conn.close()
    return briefs
def load_specific_brief(brief_id):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("SELECT query, brief_data, papers_data FROM briefs WHERE id = ?", (brief_id,)); brief = c.fetchone(); conn.close()
    if brief:
        query, brief_data_json, papers_data_json = brief
        return {"query": query, "brief_data": json.loads(brief_data_json), "papers_data": json.loads(papers_data_json)}
    return None

# --- DATA FETCHING & AI FUNCTIONS ---
# [fetch_data and generate_research_brief functions are unchanged and redacted]
async def fetch_data(source, query, max_results):
    if source == 'arXiv': return await fetch_arxiv_data(query, max_results)
    else: return await fetch_pubmed_data(query, max_results)
async def fetch_arxiv_data(query, max_results):
    loop = asyncio.get_running_loop(); search = await loop.run_in_executor(None, lambda: arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)); results = await loop.run_in_executor(None, list, search.results())
    return [{"title": p.title, "summary": p.summary, "url": p.entry_id} for p in results if p.summary]
async def fetch_pubmed_data(query, max_results):
    loop = asyncio.get_running_loop()
    def search_and_fetch():
        Entrez.email = "your.email@example.com"; handle = Entrez.esearch(db="pubmed", term=query, retmax=str(max_results), sort="relevance"); record = Entrez.read(handle); handle.close(); id_list = record["IdList"]
        if not id_list: return []
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml"); records = Entrez.read(handle); handle.close(); papers_data = []
        for i, article in enumerate(records.get('PubmedArticle', [])):
            try: papers_data.append({"title": article['MedlineCitation']['Article']['ArticleTitle'], "summary": article['MedlineCitation']['Article']['Abstract']['AbstractText'][0], "url": f"https://pubmed.ncbi.nlm.nih.gov/{id_list[i]}/"})
            except (KeyError, IndexError): continue
        return papers_data
    return await loop.run_in_executor(None, search_and_fetch)
async def generate_research_brief(text, query):
    json_schema = {"type": "object", "properties": { "executive_summary": {"type": "string"}, "key_hypotheses_and_findings": {"type": "array", "items": {"type": "object", "properties": {"paper_title": {"type": "string"},"hypothesis": {"type": "string"},"finding": {"type": "string"}}}},"methodology_comparison": {"type": "array", "items": {"type": "object", "properties": {"paper_title": {"type": "string"},"methodology": {"type": "string"}}}}, "contradictions_and_gaps": {"type": "array", "items": {"type": "string"}}},"required": ["executive_summary", "key_hypotheses_and_findings", "methodology_comparison", "contradictions_and_gaps"]}
    generation_config = GenerationConfig(response_mime_type="application/json", response_schema=json_schema)
    model = genai.GenerativeModel('gemini-2.5-pro', generation_config=generation_config); prompt = f"Act as a world-class research analyst. Analyze the following abstracts on '{query}' and populate the provided JSON schema.\n\nAbstracts:\n---\n{text}\n---"
    response = await model.generate_content_async(prompt)
    try:
        if not response.parts: raise ValueError("Model returned empty response."); return json.loads(response.text)
    except (ValueError, json.JSONDecodeError, AttributeError) as e:
        st.error(f"Error: AI response was not valid JSON. {e}"); st.code(f"Raw Model Response:\n{response.text}", language="text"); raise

# --- UI RENDERING ---
def display_research_brief(brief_data):
    query = brief_data["query"]
    st.header(f"Dynamic Research Brief: {query}")
    # [Table rendering logic is unchanged and redacted]
    st.subheader("Executive Summary"); st.markdown(brief_data["brief_data"].get("executive_summary", "Not available."))
    st.subheader("Key Hypotheses & Findings")
    if data := brief_data["brief_data"].get("key_hypotheses_and_findings", []):
        df = pd.DataFrame(data); df.columns = ["Paper Title", "Hypothesis", "Finding"]; st.markdown(df.to_html(index=False), unsafe_allow_html=True)
    st.subheader("Methodology Comparison")
    if data := brief_data["brief_data"].get("methodology_comparison", []):
        df = pd.DataFrame(data); df.columns = ["Paper Title", "Methodology"]; st.markdown(df.to_html(index=False), unsafe_allow_html=True)
    st.subheader("Identified Contradictions & Research Gaps")
    if data := brief_data["brief_data"].get("contradictions_and_gaps", []):
        for item in data: st.markdown(f"- {item}")
    with st.expander("View Source Papers & Raw Abstracts"):
        for paper in brief_data["papers_data"]:
            st.markdown(f"**{paper['title']}** ([Link]({paper.get('url', '#')}))"); st.markdown(f"_{paper['summary']}_")


# --- MAIN APP LOGIC ---
init_db()

# Initialize session state for the new workflow
if 'current_brief' not in st.session_state: st.session_state.current_brief = None
if 'search_results' not in st.session_state: st.session_state.search_results = []
if 'paper_cart' not in st.session_state: st.session_state.paper_cart = []

# --- SIDEBAR UI ---
with st.sidebar:
    st.image("https://i.imgur.com/rLoaV0k.png", width=50)
    st.title("Research OS")
    st.markdown("---")
    
    # NEW: Briefing Cart
    st.header("🛒 Briefing Cart")
    if not st.session_state.paper_cart:
        st.info("Add papers from the search results to create a brief.")
    else:
        for paper in st.session_state.paper_cart:
            st.markdown(f"- _{paper['title'][:50]}..._")
    
    if st.session_state.paper_cart:
        if st.button(f"Generate Brief from {len(st.session_state.paper_cart)} Papers", type="primary", use_container_width=True):
            with st.spinner("Synthesizing your curated brief..."):
                try:
                    # Use the search query from the first paper in the cart as the topic
                    topic_query = st.session_state.search_query
                    papers_to_analyze = st.session_state.paper_cart
                    
                    combined_abstracts = "\n\n".join([f"**Paper:** {p['title']}\n{p['summary']}" for p in papers_to_analyze])
                    brief_data = asyncio.run(generate_research_brief(combined_abstracts, topic_query))
                    
                    st.session_state.current_brief = {"query": topic_query, "brief_data": brief_data, "papers_data": papers_to_analyze}
                    # Clear cart and results after generating
                    st.session_state.paper_cart = []
                    st.session_state.search_results = []
                    st.rerun()
                except Exception:
                    st.error("Failed to generate brief.")

    st.markdown("---")
    st.header("🧠 Knowledge Base")
    # [Knowledge Base UI is unchanged and redacted]
    new_project_name = st.text_input("New Project Name", placeholder="e.g., Cancer Research Grant")
    if st.button("Create New Project", use_container_width=True):
        if new_project_name: add_project(new_project_name); st.rerun()
    projects = get_projects()
    if not projects: st.info("Your projects will appear here.")
    for project_id, project_name in projects:
        with st.expander(project_name):
            briefs_in_project = get_briefs_for_project(project_id)
            if not briefs_in_project: st.write("_No briefs yet._")
            for brief_id, brief_query in briefs_in_project:
                if st.button(brief_query, key=f"load_{brief_id}", use_container_width=True):
                    st.session_state.current_brief = load_specific_brief(brief_id)

# --- MAIN PAGE DISPLAY ---
st.title("🔬 Research OS")

# Display the generated brief if it exists
if st.session_state.current_brief:
    display_research_brief(st.session_state.current_brief)
    # [Save to Project UI is unchanged and redacted]
    projects_for_saving = get_projects()
    if projects_for_saving:
        project_names = {name: id for id, name in projects_for_saving}; selected_project_name = st.selectbox("Select project to save brief:", options=project_names.keys())
        if st.button("💾 Save to Project", key="save_to_project"):
            selected_project_id = project_names[selected_project_name]; save_brief(selected_project_id, st.session_state.current_brief['query'], st.session_state.current_brief['brief_data'], st.session_state.current_brief['papers_data']); st.toast(f"Saved brief to '{selected_project_name}'!"); st.rerun()
    else: st.warning("Please create a project to save this brief.")

# Display search results if they exist
elif st.session_state.search_results:
    st.header("Search Results")
    for i, paper in enumerate(st.session_state.search_results):
        st.subheader(paper['title'])
        st.markdown(f"_{paper['summary'][:250]}..._ [Link]({paper['url']})")
        if st.button("➕ Add to Brief", key=f"add_{i}"):
            st.session_state.paper_cart.append(paper)
            # Optional: remove from search results to prevent duplicates
            st.session_state.search_results.pop(i)
            st.rerun()
else:
    st.info("Enter a topic in the chat bar below to search for papers.")


# --- BOTTOM CONTROLS & CHAT INPUT ---
with st.container():
    st.markdown('<div class="bottom-controls">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1: data_source = st.selectbox("Data Source", ["arXiv", "PubMed"], key="data_source_bottom")
    with col2: num_papers = st.slider("Number of Papers to Find", min_value=5, max_value=20, value=10, key="num_papers_bottom")
    st.markdown('</div>', unsafe_allow_html=True)

if prompt := st.chat_input("Enter a topic to search for papers..."):
    ds = st.session_state.data_source_bottom
    np = st.session_state.num_papers_bottom
    st.session_state.search_query = prompt # Save the query for later
    
    with st.spinner(f"Searching {ds} for '{prompt}'..."):
        try:
            results = asyncio.run(fetch_data(ds, prompt, np))
            if not results:
                st.warning(f"No papers found for '{prompt}' on {ds}.")
            else:
                st.session_state.search_results = results
                st.session_state.current_brief = None # Clear any old brief
                st.rerun()
        except Exception as e:
            st.error(f"Failed to search: {e}")
