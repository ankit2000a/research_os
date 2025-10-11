import streamlit as st
import arxiv
import google.generativeai as genai
# NEW: Import the necessary type for structured output
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
# [fetch_data functions are unchanged and redacted for brevity]
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
    """
    UPGRADED: Generates a structured JSON research brief using the SDK's native JSON mode.
    This is the most reliable method.
    """
    
    # 1. Define the JSON schema for the desired output
    json_schema = {
        "type": "object",
        "properties": {
            "executive_summary": {
                "type": "string",
                "description": "A detailed, high-level synthesis of the key findings. It should connect the papers and explain their collective importance. Minimum 3-4 sentences."
            },
            "key_hypotheses_and_findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "paper_title": {"type": "string", "description": "The full title of the paper."},
                        "hypothesis": {"type": "string", "description": "The core hypothesis or research question of the paper, stated clearly."},
                        "finding": {"type": "string", "description": "A detailed explanation of the primary finding or conclusion of the paper, including key supporting points from the abstract."}
                    },
                    "required": ["paper_title", "hypothesis", "finding"]
                }
            },
            "methodology_comparison": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "paper_title": {"type": "string", "description": "The full title of the paper."},
                        "methodology": {"type": "string", "description": "A clear description of the methodology used (e.g., 'Systematic Review of 50 papers', 'Randomized Control Trial with 250 participants')."}
                    },
                    "required": ["paper_title", "methodology"]
                }
            },
            "contradictions_and_gaps": {
                "type": "array",
                "items": {"type": "string", "description": "A bullet point describing a specific contradiction or research gap."}
            }
        },
        "required": ["executive_summary", "key_hypotheses_and_findings", "methodology_comparison", "contradictions_and_gaps"]
    }

    # 2. Configure the model to use JSON mode with the defined schema
    generation_config = GenerationConfig(
        response_mime_type="application/json",
        response_schema=json_schema
    )
    
    model = genai.GenerativeModel(
        'gemini-2.5-pro',
        generation_config=generation_config
    )
    
    # 3. The prompt can now be simpler, as it doesn't need to specify the JSON structure
    prompt = f"""
    Act as a world-class research analyst. Analyze the following research paper abstracts on the topic of '{query}'.
    Your task is to populate the provided JSON schema with a detailed, substantial, and accurate intelligence brief based on the abstracts.

    Abstracts:
    ---
    {text}
    ---
    """
    
    response = await model.generate_content_async(prompt)

    # 4. Error handling is now much cleaner
    try:
        # The SDK guarantees the output is parseable JSON, so we can access it directly
        return json.loads(response.text)
    except (ValueError, json.JSONDecodeError, AttributeError) as e:
        st.error("Error: The AI model failed to generate a valid response, possibly due to safety filters or an internal error.")
        st.code(f"Model Response Parts:\n{response.parts}", language="text")
        raise

# --- UI RENDERING ---
# [display_research_brief function is unchanged and redacted for brevity]
def display_research_brief(brief_data):
    query = brief_data["query"]
    st.header(f"Dynamic Research Brief: {query}")
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

# Sidebar UI
with st.sidebar:
    # [Sidebar code is unchanged and redacted for brevity]
    st.image("https://i.imgur.com/rLoaV0k.png", width=50)
    st.title("Research OS")
    st.markdown("The Insight Engine for Modern Research.")
    st.markdown("---")
    
    st.header("Controls")
    data_source = st.selectbox("Data Source", ["arXiv", "PubMed"])
    search_query = st.text_input("Research Topic", placeholder="e.g., CRISPR-Cas9")
    num_papers = st.slider("Number of Papers", min_value=2, max_value=5, value=3)
    
    if st.button("Generate Research Brief", type="primary", use_container_width=True):
        if not search_query:
            st.warning("Please enter a research topic.")
        else:
            with st.spinner(f"Building brief for '{search_query}'..."):
                try:
                    papers_data = asyncio.run(fetch_data(data_source, search_query, num_papers))
                    
                    if not papers_data:
                        st.warning(f"No academic papers found for '{search_query}' on {data_source}. Please try a more specific research topic.")
                        st.stop()

                    combined_abstracts = "\n\n".join([f"**Paper:** {p['title']}\n{p['summary']}" for p in papers_data])
                    brief_data = asyncio.run(generate_research_brief(combined_abstracts, search_query))
                    st.session_state.current_brief = {
                        "query": search_query, "brief_data": brief_data, "papers_data": papers_data
                    }

                except Exception:
                    st.error("Failed to generate the research brief. The model may have refused to answer due to safety filters.")

    # [Knowledge Base UI code is unchanged and redacted for brevity]
    st.markdown("---")
    st.header("🧠 Knowledge Base")
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

# Main Page Display
st.title("🔬 Research OS")

if st.session_state.current_brief:
    display_research_brief(st.session_state.current_brief)
    
    projects_for_saving = get_projects()
    if projects_for_saving:
        project_names = {name: id for id, name in projects_for_saving}
        selected_project_name = st.selectbox("Select project to save brief:", options=project_names.keys())
        
        if st.button("💾 Save to Project", key="save_to_project"):
            selected_project_id = project_names[selected_project_name]
            save_brief(
                selected_project_id,
                st.session_state.current_brief['query'],
                st.session_state.current_brief['brief_data'],
                st.session_state.current_brief['papers_data']
            )
            st.toast(f"Saved brief to '{selected_project_name}'!")
            st.rerun()
    else:
        st.warning("Please create a project in the sidebar to save this brief.")
else:
    st.info("Enter a topic in the sidebar and click 'Generate Research Brief' to begin.")
