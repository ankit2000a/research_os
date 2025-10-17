import streamlit as st
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import json
import asyncio
import pandas as pd
import sqlite3
from serpapi import GoogleSearch
# REMOVED: requests and beautifulsoup4 are no longer needed

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
    .paper-card {
        border: 1px solid #303640;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- API KEY & DATABASE SETUP ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    SERPAPI_API_KEY = st.secrets["SERPAPI_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("🚨 API Key not found. Please add both GOOGLE_API_KEY and SERPAPI_API_KEY to your Streamlit secrets.")
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

# --- GOOGLE SCHOLAR & AI FUNCTIONS ---
async def search_google_scholar(query):
    """Searches Google Scholar for a minimum of 25 papers."""
    loop = asyncio.get_running_loop()
    
    def sync_search():
        params = {"engine": "google_scholar", "q": query, "api_key": SERPAPI_API_KEY, "num": 25}
        search = GoogleSearch(params)
        results = search.get_dict()
        return [{"title": r.get("title", "No Title"), "summary": r.get("snippet", "No summary available."), "url": r.get("link", "#")} for r in results.get("organic_results", [])]
        
    return await loop.run_in_executor(None, sync_search)

# REMOVED: The slow get_full_abstract_from_url function has been deleted.

async def generate_research_brief(papers, query):
    """Generates the structured research brief, forcing comprehensiveness."""
    text = "\n\n---\n\n".join([f"**Paper Title:** {p['title']}\n**Abstract:** {p.get('summary', 'No summary.')}" for p in papers]); num_papers = len(papers)
    json_schema = {"type": "object", "properties": {"executive_summary": {"type": "string"}, "key_hypotheses_and_findings": {"type": "array", "items": {"type": "object", "properties": {"paper_title": {"type": "string"},"hypothesis": {"type": "string"},"finding": {"type": "string"}}}},"methodology_comparison": {"type": "array", "items": {"type": "object", "properties": {"paper_title": {"type": "string"},"methodology": {"type": "string"}}}}, "contradictions_and_gaps": {"type": "array", "items": {"type": "string"}}},"required": ["executive_summary", "key_hypotheses_and_findings", "methodology_comparison", "contradictions_and_gaps"]}
    generation_config = GenerationConfig(response_mime_type="application/json", response_schema=json_schema)
    model = genai.GenerativeModel('gemini-2.5-pro', generation_config=generation_config)
    prompt = f"Act as a world-class research analyst. You have {num_papers} abstracts for '{query}'. Populate the JSON schema with a detailed brief, ensuring an entry for EACH of the {num_papers} papers in all relevant sections."
    response = await model.generate_content_async(prompt)
    try:
        if not response.parts: raise ValueError("Model returned empty response.")
        return json.loads(response.text)
    except (ValueError, json.JSONDecodeError, AttributeError) as e:
        st.error(f"Error: AI response was not valid JSON. {e}"); st.code(f"Raw Model Response:\n{response.text}", language="text"); raise

# --- UI RENDERING & MAIN LOGIC ---
init_db()
if 'current_brief' not in st.session_state: st.session_state.current_brief = None
if 'search_results' not in st.session_state: st.session_state.search_results = []
if 'selected_papers' not in st.session_state: st.session_state.selected_papers = {}

with st.sidebar:
    st.image("https://i.imgur.com/rLoaV0k.png", width=50); st.title("Research OS"); st.markdown("---")
    
    num_selected = len(st.session_state.selected_papers)
    if num_selected > 0:
        st.write(f"**{num_selected} papers selected.**")
        if st.button(f"Generate Brief from {num_selected} Papers", type="primary", use_container_width=True):
            st.session_state.generating_brief = True
    else:
        st.info("Select papers from the search results to generate a brief.")

    st.markdown("---"); st.header("🧠 Knowledge Base")
    # ... knowledge base UI
st.title("🔬 Research OS")

if st.session_state.get('generating_brief'):
    with st.spinner("Synthesizing your curated brief..."):
        try:
            topic_query = st.session_state.get("search_query", "Selected Papers")
            papers_to_analyze = list(st.session_state.selected_papers.values())
            brief_data = asyncio.run(generate_research_brief(papers_to_analyze, topic_query))
            st.session_state.current_brief = {"query": topic_query, "brief_data": brief_data, "papers_data": papers_to_analyze}
            st.session_state.selected_papers = {}; st.session_state.search_results = []
            del st.session_state['generating_brief']
            st.rerun()
        except Exception as e:
            st.error(f"Failed to generate brief: {e}")
            del st.session_state['generating_brief']

elif st.session_state.current_brief:
    if brief_data := st.session_state.current_brief:
        st.header(f"Dynamic Research Brief: {brief_data['query']}")
        # ... display logic
    # ... save to project UI
elif st.session_state.search_results:
    st.header(f"Search Results for '{st.session_state.search_query}'")
    st.markdown(f"Found **{len(st.session_state.search_results)}** papers. Select the most relevant to generate your brief.")
    
    for i, paper in enumerate(st.session_state.search_results):
        with st.container(border=True):
            col1, col2 = st.columns([10, 2])
            with col1:
                st.subheader(paper['title'])
            with col2:
                is_selected = paper['title'] in st.session_state.selected_papers
                button_text = "➖ Remove" if is_selected else "➕ Add"
                if st.button(button_text, key=f"toggle_{i}"):
                    if is_selected:
                        del st.session_state.selected_papers[paper['title']]
                    else:
                        st.session_state.selected_papers[paper['title']] = paper
                    st.rerun()

            with st.expander("View Abstract"):
                st.markdown(f"**[Link to Paper]({paper.get('url', '#')})**")
                st.markdown(paper.get('summary', 'No summary available.'))
else:
    st.info("Enter a topic in the chat bar below to search Google Scholar.")

# --- BOTTOM CHAT INPUT (URL input removed) ---
if prompt := st.chat_input("Search Google Scholar for papers..."):
    st.session_state.search_query = prompt
    with st.spinner(f"Searching Google Scholar for '{prompt}'..."):
        try:
            results = asyncio.run(search_google_scholar(prompt))
            if not results:
                st.warning(f"No papers found on Google Scholar for '{prompt}'.")
            else:
                st.session_state.search_results = results
                st.session_state.current_brief = None
                st.session_state.selected_papers = {}
                st.rerun()
        except Exception as e:
            st.error(f"Failed to perform search: {e}")
            