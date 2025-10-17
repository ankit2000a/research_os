import streamlit as st
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
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

# --- AI-POWERED SEARCH FUNCTION ---
async def search_and_find_papers(query):
    """
    Uses Gemini with Google Search grounding to find relevant papers from the entire web.
    """
    # --- THIS IS THE FIX ---
    # The 'tools' parameter now uses the correct dictionary format to enable Google Search.
    model = genai.GenerativeModel(
        'gemini-2.5-pro',
        tools=[{"google_search": {}}]
    )
    
    prompt = f"""
    You are an expert academic research librarian. Your task is to use Google Search to find the top 7 most relevant and recent **academic papers, preprints, and scholarly articles** from across the entire web for the topic: "{query}".

    **CRITICAL INSTRUCTIONS:**
    1.  **Prioritize Authoritative Sources:** Focus your search on academic databases (like Google Scholar, Semantic Scholar), university repositories, and well-known preprint servers (e.g., arXiv, bioRxiv) and established journals.
    2.  **Verify Scholarly Nature:** Ensure each result is a genuine research paper, article, or preprint with an identifiable author or research institution.
    3.  **AVOID NON-ACADEMIC CONTENT:** You MUST exclude blogs, news articles, marketing content, and general web pages. If you cannot find a link to the actual paper (e.g., a PDF or an abstract page on an academic site), do not include it.

    For each verified paper you find, you must provide the title, a valid URL to the paper itself, and a concise one-sentence summary.
    
    CRITICAL: Your entire response must be a single, valid JSON object, structured as an array of papers. Do not include any other text.
    Example format:
    [
      {{
        "title": "Example Paper Title",
        "url": "https://arxiv.org/abs/1234.5678",
        "summary": "This paper investigates..."
      }}
    ]
    """
    
    response = await model.generate_content_async(prompt)
    
    try:
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
        if not cleaned_response:
             raise ValueError("The AI model returned an empty response.")
        return json.loads(cleaned_response)
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        st.error("The AI failed to find and format papers correctly. This can happen with very niche topics or if the AI's safety filters were triggered.")
        st.code(f"Raw model response:\n{response.text}", language="text")
        return []

async def generate_research_brief(text, query):
    """Generates the structured research brief from the user-selected papers."""
    json_schema = {"type": "object", "properties": { "executive_summary": {"type": "string"}, "key_hypotheses_and_findings": {"type": "array", "items": {"type": "object", "properties": {"paper_title": {"type": "string"},"hypothesis": {"type": "string"},"finding": {"type": "string"}}}},"methodology_comparison": {"type": "array", "items": {"type": "object", "properties": {"paper_title": {"type": "string"},"methodology": {"type": "string"}}}}, "contradictions_and_gaps": {"type": "array", "items": {"type": "string"}}},"required": ["executive_summary", "key_hypotheses_and_findings", "methodology_comparison", "contradictions_and_gaps"]}
    generation_config = GenerationConfig(response_mime_type="application/json", response_schema=json_schema)
    model = genai.GenerativeModel('gemini-2.5-pro', generation_config=generation_config)
    prompt = f"Act as a world-class research analyst. Analyze the following abstracts on '{query}' and populate the provided JSON schema.\n\nAbstracts:\n---\n{text}\n---"
    response = await model.generate_content_async(prompt)
    try:
        if not response.parts: raise ValueError("Model returned empty response.")
        return json.loads(response.text)
    except (ValueError, json.JSONDecodeError, AttributeError) as e:
        st.error(f"Error: AI response was not valid JSON. {e}")
        st.code(f"Raw Model Response:\n{response.text}", language="text")
        raise

# --- UI RENDERING ---
# [display_research_brief function is unchanged and redacted]
def display_research_brief(brief_data):
    if not brief_data or "brief_data" not in brief_data or not brief_data["brief_data"]: st.error("Brief data is missing."); return
    query = brief_data["query"]; st.header(f"Dynamic Research Brief: {query}")
    st.subheader("Executive Summary"); st.markdown(brief_data["brief_data"].get("executive_summary", "Not available."))
    if data := brief_data["brief_data"].get("key_hypotheses_and_findings", []):
        df = pd.DataFrame(data); df.columns = ["Paper Title", "Hypothesis", "Finding"]; st.markdown(df.to_html(index=False), unsafe_allow_html=True)
    if data := brief_data["brief_data"].get("methodology_comparison", []):
        st.subheader("Methodology Comparison"); df = pd.DataFrame(data); df.columns = ["Paper Title", "Methodology"]; st.markdown(df.to_html(index=False), unsafe_allow_html=True)
    if data := brief_data["brief_data"].get("contradictions_and_gaps", []):
        st.subheader("Identified Contradictions & Research Gaps");
        for item in data: st.markdown(f"- {item}")
    with st.expander("View Source Papers & Raw Abstracts"):
        for paper in brief_data["papers_data"]:
            st.markdown(f"**{paper['title']}** ([Link]({paper.get('url', '#')}))"); st.markdown(f"_{paper.get('summary', 'No summary provided.')}_")

# --- MAIN APP LOGIC ---
init_db()

# Initialize session state for the new workflow
if 'current_brief' not in st.session_state: st.session_state.current_brief = None
if 'search_results' not in st.session_state: st.session_state.search_results = []
if 'paper_cart' not in st.session_state: st.session_state.paper_cart = []

# --- SIDEBAR UI ---
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
                    combined_abstracts = "\n\n".join([f"**Paper:** {p['title']}\n{p.get('summary', 'No summary available.')}" for p in papers_to_analyze])
                    brief_data = asyncio.run(generate_research_brief(combined_abstracts, topic_query))
                    st.session_state.current_brief = {"query": topic_query, "brief_data": brief_data, "papers_data": papers_to_analyze}
                    st.session_state.paper_cart = []; st.session_state.search_results = []
                    st.rerun()
                except Exception: st.error("Failed to generate brief.")

    st.markdown("---"); st.header("🧠 Knowledge Base")
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

if st.session_state.current_brief:
    display_research_brief(st.session_state.current_brief)
    # [Save to Project UI is unchanged and redacted]
    projects_for_saving = get_projects()
    if projects_for_saving:
        project_names = {name: id for id, name in projects_for_saving}; selected_project_name = st.selectbox("Select project to save brief:", options=project_names.keys())
        if st.button("💾 Save to Project", key="save_to_project"):
            selected_project_id = project_names[selected_project_name]; save_brief(selected_project_id, st.session_state.current_brief['query'], st.session_state.current_brief['brief_data'], st.session_state.current_brief['papers_data']); st.toast(f"Saved brief to '{selected_project_name}'!"); st.rerun()
    else: st.warning("Please create a project to save this brief.")

elif st.session_state.search_results:
    st.header(f"Search Results for '{st.session_state.search_query}'")
    st.markdown(f"Found **{len(st.session_state.search_results)}** relevant papers. Add the most relevant ones to your brief.")
    for i, paper in enumerate(st.session_state.search_results):
        st.subheader(paper['title'])
        st.markdown(f"_{paper.get('summary', 'No summary available.')}_ [Link]({paper.get('url', '#')})")
        if st.button("➕ Add to Brief", key=f"add_{i}"):
            st.session_state.paper_cart.append(paper)
            st.session_state.search_results.pop(i)
            st.rerun()
else:
    st.info("Enter a topic in the chat bar below to search the web for academic papers.")

# --- BOTTOM CHAT INPUT ---
if prompt := st.chat_input("Search the web for academic papers..."):
    st.session_state.search_query = prompt
    with st.spinner(f"Using AI to search the web for '{prompt}'..."):
        try:
            results = asyncio.run(search_and_find_papers(prompt))
            if not results:
                st.warning(f"The AI search did not find any papers for '{prompt}'.")
            else:
                st.session_state.search_results = results
                st.session_state.current_brief = None
                st.rerun()
        except Exception as e:
            st.error(f"Failed to perform AI search: {e}")
