import streamlit as st
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import json
import asyncio
import pandas as pd
import sqlite3
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
    c.execute('''CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)''')
    c.execute('''CREATE TABLE IF NOT EXISTS briefs (id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, query TEXT NOT NULL, brief_data TEXT NOT NULL, papers_data TEXT NOT NULL, FOREIGN KEY (project_id) REFERENCES projects (id))''')
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
    c.execute("SELECT id, name FROM projects ORDER BY name"); projects = c.fetchall(); conn.close()
    return projects
def save_brief(project_id, query, brief_data, papers_data):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("INSERT INTO briefs (project_id, query, brief_data, papers_data) VALUES (?, ?, ?, ?)", (project_id, query, json.dumps(brief_data), json.dumps(papers_data))); conn.commit(); conn.close()
def get_briefs_for_project(project_id):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("SELECT id, query FROM briefs WHERE project_id = ? ORDER BY query", (project_id,)); briefs = c.fetchall(); conn.close()
    return briefs
def load_specific_brief(brief_id):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("SELECT query, brief_data, papers_data FROM briefs WHERE id = ?", (brief_id,)); brief = c.fetchone(); conn.close()
    if brief:
        query, brief_data_json, papers_data_json = brief
        return {"query": query, "brief_data": json.loads(brief_data_json), "papers_data": json.loads(papers_data_json)}
    return None

# --- SEARCH & AI FUNCTIONS ---

async def search_google_scholar(query):
    """Searches Google Scholar using SerpApi."""
    loop = asyncio.get_running_loop()
    def sync_search():
        params = {"engine": "google_scholar", "q": query, "api_key": SERPAPI_API_KEY, "num": 25}
        search = GoogleSearch(params)
        results = search.get_dict()
        return [{"title": r.get("title", "N/A"), "summary": r.get("snippet", "N/A"), "url": r.get("link", "#")} for r in results.get("organic_results", [])]
    return await loop.run_in_executor(None, sync_search)

async def select_best_papers_with_ai(papers, query, num_to_select=10):
    """
    NEW: Uses an AI model to select the most relevant and diverse papers from a list.
    """
    model = genai.GenerativeModel('gemini-2.5-pro')
    prompt_papers = [{"title": p["title"], "summary": p["summary"]} for p in papers]
    
    prompt = f"""
    Act as an expert research analyst. I have a list of {len(papers)} papers for the research query: "{query}".
    Your task is to select the {num_to_select} most relevant and diverse papers from this list. A good selection should cover the key sub-themes and avoid redundancy.

    Return a JSON object containing a single key "selected_titles" which is an array of the exact titles of the {num_to_select} papers you have chosen.

    CRITICAL: Only return the JSON object. Do not include any other text or explanation.

    Here is the JSON array of papers to choose from:
    {json.dumps(prompt_papers, indent=2)}
    """
    
    response = await model.generate_content_async(prompt)
    try:
        selected_data = json.loads(response.text)
        selected_titles = selected_data.get("selected_titles", [])
        
        # Reconstruct the full paper objects from the selected titles
        original_papers_by_title = {p['title'].lower(): p for p in papers}
        selected_full_papers = [original_papers_by_title[title.lower()] for title in selected_titles if title.lower() in original_papers_by_title]
        return selected_full_papers
    except (json.JSONDecodeError, ValueError):
        st.warning("AI paper selection failed. Using the top 10 results instead.")
        return papers[:num_to_select]

async def get_full_abstract_from_url(url):
    """Uses AI to extract a full, clean abstract from a URL."""
    # [This function is unchanged and redacted for brevity]
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        page_text = ' '.join(p.get_text() for p in soup.find_all('p'))[:8000]
        if not page_text: return "Could not extract text."

        model = genai.GenerativeModel('gemini-2.5-pro')
        prompt = f"Analyze the text from {url}. Return ONLY the full, clean abstract of the academic paper. Text:\n---\n{page_text}"
        ai_response = await model.generate_content_async(prompt, request_options={"timeout": 60})
        return ai_response.text
    except Exception as e:
        return f"Could not process URL: {e}"

async def generate_research_brief(papers, query):
    """Generates the structured research brief with a timeout."""
    # [This function is unchanged and redacted for brevity]
    text = "\n\n---\n\n".join([f"**Paper Title:** {p['title']}\n**Abstract:** {p.get('summary', 'N/A')}" for p in papers])
    num_papers = len(papers)
    json_schema = { "type": "object", "properties": { "executive_summary": {"type": "string"}, "key_hypotheses_and_findings": {"type": "array", "items": {"type": "object", "properties": {"paper_title": {"type": "string"}, "hypothesis": {"type": "string"}, "finding": {"type": "string"}}}}, "methodology_comparison": {"type": "array", "items": {"type": "object", "properties": {"paper_title": {"type": "string"}, "methodology": {"type": "string"}}}}, "contradictions_and_gaps": {"type": "array", "items": {"type": "string"}} }, "required": ["executive_summary", "key_hypotheses_and_findings", "methodology_comparison", "contradictions_and_gaps"] }
    generation_config = GenerationConfig(response_mime_type="application/json", response_schema=json_schema)
    model = genai.GenerativeModel('gemini-2.5-pro', generation_config=generation_config)
    prompt = f"Act as a world-class research analyst. You have {num_papers} abstracts for '{query}'. Populate the JSON schema with a detailed brief, ensuring an entry for EACH of the {num_papers} papers. Titles must be concise."
    response = await model.generate_content_async(prompt, request_options={"timeout": 120})
    try:
        if not response.parts: raise ValueError("Model returned empty response.")
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Error parsing AI response: {e}"); st.code(f"Raw Response:\n{response.text}"); raise


# --- UI & APP LOGIC ---
init_db()
st.session_state.setdefault('current_view', 'search')
st.session_state.setdefault('search_results', [])
st.session_state.setdefault('selected_papers', {})
st.session_state.setdefault('search_query', '')

with st.sidebar:
    st.image("https://i.imgur.com/rLoaV0k.png", width=50); st.title("Research OS"); st.markdown("---")
    num_selected = len(st.session_state.selected_papers)
    if num_selected > 0:
        st.write(f"**{num_selected} papers selected.**")
        if st.button(f"Generate Brief from {num_selected} Papers", type="primary", use_container_width=True):
            st.session_state.current_view = 'generating_manual'
            st.rerun()
    else:
        st.info("Select papers from search results to generate a brief.")
    st.markdown("---"); st.header("🧠 Knowledge Base")
    # ... [Knowledge Base UI redacted] ...
st.title("🔬 Research OS")

def handle_brief_generation(papers, query):
    """Helper function to run the brief generation and update state."""
    brief_data = asyncio.run(generate_research_brief(papers, query))
    st.session_state.current_brief_data = {"query": query, "brief_data": brief_data, "papers_data": papers}
    st.session_state.selected_papers = {}; st.session_state.search_results = []
    st.session_state.current_view = 'brief'

# --- Main Page Controller ---
if st.session_state.current_view == 'generating_manual':
    with st.spinner("Synthesizing your curated brief..."):
        try:
            papers_to_analyze = list(st.session_state.selected_papers.values())
            handle_brief_generation(papers_to_analyze, st.session_state.search_query)
            st.rerun()
        except Exception as e:
            st.error(f"Failed to generate brief: {e}"); st.session_state.current_view = 'search'

elif st.session_state.current_view == 'generating_auto':
    with st.spinner("AI is selecting the best papers and synthesizing your brief..."):
        try:
            papers_to_analyze = asyncio.run(select_best_papers_with_ai(st.session_state.search_results, st.session_state.search_query))
            handle_brief_generation(papers_to_analyze, st.session_state.search_query)
            st.rerun()
        except Exception as e:
            st.error(f"Failed to generate auto-brief: {e}"); st.session_state.current_view = 'search'

elif st.session_state.current_view == 'brief':
    if brief_data := st.session_state.get('current_brief_data'):
        st.header(f"Dynamic Research Brief: {brief_data['query']}")
        # ... [Display logic redacted] ...
    else:
        st.info("No brief to display. Please start a new search.")

elif st.session_state.current_view == 'search':
    if st.session_state.search_results:
        st.header(f"Search Results for '{st.session_state.search_query}'")
        
        # NEW: Auto-Brief Button
        if st.button("⚡ Generate Auto-Brief from Top 10 Papers", type="primary"):
            st.session_state.current_view = 'generating_auto'
            st.rerun()

        st.markdown("---")
        
        for i, paper in enumerate(st.session_state.search_results):
            with st.container(border=True):
                col1, col2 = st.columns([10, 2])
                with col1:
                    st.subheader(paper['title'])
                with col2:
                    is_selected = paper['title'] in st.session_state.selected_papers
                    button_label = "➖ Remove" if is_selected else "➕ Add"
                    if st.button(button_label, key=f"toggle_{i}"):
                        if is_selected:
                            del st.session_state.selected_papers[paper['title']]
                        else:
                            st.session_state.selected_papers[paper['title']] = paper
                        st.rerun()
                with st.expander("View Abstract"):
                    st.markdown(f"**[Link to Paper]({paper.get('url', '#')})**")
                    abstract_placeholder = st.empty()
                    abstract_placeholder.markdown(paper.get('summary', 'N/A'))
    else:
        st.info("Enter a topic below to search Google Scholar.")

if prompt := st.chat_input("Search Google Scholar for papers..."):
    st.session_state.search_query = prompt
    st.session_state.current_view = 'search'
    with st.spinner(f"Searching Google Scholar for '{prompt}'..."):
        results = asyncio.run(search_google_scholar(prompt))
        st.session_state.search_results = results
        st.session_state.current_brief_data = None
        st.session_state.selected_papers = {}
        st.rerun()