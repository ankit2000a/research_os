import streamlit as st
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import json
import asyncio
import pandas as pd
import sqlite3
# UPDATED: Removed unused arxiv and Bio imports
from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup

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

# --- NEW GOOGLE SCHOLAR SEARCH FUNCTION ---
async def search_google_scholar(query):
    """Searches Google Scholar using the SerpApi and returns structured results."""
    loop = asyncio.get_running_loop()
    
    def sync_search():
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": SERPAPI_API_KEY
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        
        papers = []
        for result in results.get("organic_results", []):
            papers.append({
                "title": result.get("title", "No Title"),
                "summary": result.get("snippet", "No summary available."),
                "url": result.get("link", "#")
            })
        return papers
        
    return await loop.run_in_executor(None, sync_search)

async def get_paper_details_from_url(url):
    # [This function is unchanged and redacted for brevity]
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10); response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser'); page_text = ' '.join(p.get_text() for p in soup.find_all('p'))
        if not page_text: return None
        model = genai.GenerativeModel('gemini-2.5-pro')
        prompt = f"""Analyze the text from {url} and extract the academic paper's title and summary. Return a JSON with "title" and "summary". If not a paper, return "title": "Error", "summary": "Not an academic paper." Text: --- {page_text[:4000]} ---"""
        json_schema = {"type": "object", "properties": {"title": {"type": "string"}, "summary": {"type": "string"}}, "required": ["title", "summary"]}
        generation_config = GenerationConfig(response_mime_type="application/json", response_schema=json_schema)
        model.generation_config = generation_config
        ai_response = await model.generate_content_async(prompt); details = json.loads(ai_response.text); details['url'] = url
        return details
    except Exception as e:
        st.error(f"Could not process URL: {e}"); return None

async def generate_research_brief(papers, query):
    # [This function is unchanged and redacted]
    text = "\n\n---\n\n".join([f"**Paper Title:** {p['title']}\n**Abstract:** {p.get('summary', 'No summary available.')}" for p in papers]); num_papers = len(papers)
    json_schema = {"type": "object", "properties": { "executive_summary": {"type": "string"}, "key_hypotheses_and_findings": {"type": "array", "items": {"type": "object", "properties": {"paper_title": {"type": "string"},"hypothesis": {"type": "string"},"finding": {"type": "string"}}}},"methodology_comparison": {"type": "array", "items": {"type": "object", "properties": {"paper_title": {"type": "string"},"methodology": {"type": "string"}}}}, "contradictions_and_gaps": {"type": "array", "items": {"type": "string"}}},"required": ["executive_summary", "key_hypotheses_and_findings", "methodology_comparison", "contradictions_and_gaps"]}
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
# [The rest of the file is largely unchanged and redacted for brevity]
def display_research_brief(brief_data):
    if not brief_data or "brief_data" not in brief_data or not brief_data["brief_data"]: st.error("Brief data is missing."); return
    query = brief_data["query"]; st.header(f"Dynamic Research Brief: {query}")
    # ... display logic
init_db()
if 'current_brief' not in st.session_state: st.session_state.current_brief = None
if 'search_results' not in st.session_state: st.session_state.search_results = []
if 'paper_cart' not in st.session_state: st.session_state.paper_cart = []
with st.sidebar:
    st.image("https://i.imgur.com/rLoaV0k.png", width=50); st.title("Research OS"); st.markdown("---")
    st.header("🛒 Briefing Cart")
    if not st.session_state.paper_cart:
        st.info("Add papers from search results to create a brief.")
    else:
        for paper in st.session_state.paper_cart: st.markdown(f"- _{paper['title'][:50]}..._")
    if st.session_state.paper_cart:
        if st.button(f"Generate Brief from {len(st.session_state.paper_cart)} Papers", type="primary", use_container_width=True):
            with st.spinner("Synthesizing your curated brief..."):
                try:
                    topic_query = st.session_state.get("search_query", "Selected Papers")
                    papers_to_analyze = st.session_state.paper_cart
                    brief_data = asyncio.run(generate_research_brief(papers_to_analyze, topic_query))
                    st.session_state.current_brief = {"query": topic_query, "brief_data": brief_data, "papers_data": papers_to_analyze}
                    st.session_state.paper_cart = []; st.session_state.search_results = []
                    st.rerun()
                except Exception: st.error("Failed to generate brief.")
    st.markdown("---"); st.header("🧠 Knowledge Base")
    # ... knowledge base UI
st.title("🔬 Research OS")
if st.session_state.current_brief:
    display_research_brief(st.session_state.current_brief)
    # ... save to project UI
elif st.session_state.search_results:
    st.header(f"Search Results for '{st.session_state.search_query}'")
    st.markdown(f"Found **{len(st.session_state.search_results)}** papers. Add the most relevant to your brief.")
    for i, paper in enumerate(st.session_state.search_results):
        st.subheader(paper['title'])
        st.markdown(f"[Link to Paper]({paper.get('url', '#')})")
        with st.expander("View Abstract"):
            st.markdown(paper.get('summary', 'No summary available.'))
        if st.button("➕ Add to Brief", key=f"add_{i}"):
            st.session_state.paper_cart.append(paper)
            st.session_state.search_results.pop(i)
            st.rerun()
else:
    st.info("Enter a topic to search Google Scholar, or add a paper by URL below.")

# --- NEW BOTTOM CONTROLS & CHAT INPUT ---
st.container()
url_to_add = st.text_input("🔗 Add a specific paper by URL (e.g., from a direct link)")
if st.button("Add Paper from URL"):
    if url_to_add:
        with st.spinner("Analyzing URL..."):
            paper_details = asyncio.run(get_paper_details_from_url(url_to_add))
            if paper_details and paper_details['title'] != "Error":
                st.session_state.paper_cart.append(paper_details)
                st.rerun()
            else:
                st.error("Could not add paper. The URL may be invalid or not an academic source.")
    else:
        st.warning("Please enter a URL.")

if prompt := st.chat_input("Search Google Scholar for papers..."):
    st.session_state.search_query = prompt
    with st.spinner(f"Searching Google Scholar for '{prompt}'..."):
        try:
            # UPDATED to call the new search function
            results = asyncio.run(search_google_scholar(prompt))
            if not results:
                st.warning(f"No papers found on Google Scholar for '{prompt}'.")
            else:
                st.session_state.search_results = results
                st.session_state.current_brief = None
                st.rerun()
        except Exception as e:
            st.error(f"Failed to perform search: {e}")
            