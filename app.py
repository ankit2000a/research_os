import streamlit as st
import arxiv
import google.generativeai as genai
from Bio import Entrez
import pandas as pd
import plotly.graph_objects as go
import json

st.set_page_config(layout="wide")

# --- API KEY CONFIGURATION ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("Google API Key not found. Please add it to your Streamlit Cloud secrets.")
    st.stop()

# --- DATA FETCHING FUNCTIONS ---
def fetch_arxiv_data(query, max_results):
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    results = list(search.results())
    return [{"title": p.title, "summary": p.summary, "url": p.entry_id} for p in results]

def fetch_pubmed_data(query, max_results):
    Entrez.email = "your.email@example.com"
    handle = Entrez.esearch(db="pubmed", term=query, retmax=str(max_results), sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    id_list = record["IdList"]
    if not id_list: return []
    
    handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
    records = Entrez.read(handle)
    handle.close()
    
    papers_data = []
    for i, article in enumerate(records['PubmedArticle']):
        try:
            title = article['MedlineCitation']['Article']['ArticleTitle']
            abstract = article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
            pubmed_id = id_list[i]
            papers_data.append({"title": title, "summary": abstract, "url": f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"})
        except (KeyError, IndexError):
            continue
    return papers_data

# --- AI & GRAPHING FUNCTIONS (with updates) ---
def get_detailed_synthesis(text, query):
    model = genai.GenerativeModel('models/gemini-pro-latest')
    prompt = f"Provide a detailed, in-depth synthesized analysis of the following abstracts on '{query}'. Begin directly, identify the core theme, explain each paper's contribution, and highlight their relationships. The analysis should be at least 400 words."
    response = model.generate_content([prompt, "Abstracts:", text])
    return response.text

def get_mindmap_data(text, query):
    model = genai.GenerativeModel('models/gemini-pro-latest')
    prompt = f"""
    Read the abstracts on '{query}' and generate a hierarchical mind map structure as a JSON object with "id", "label", and "parent". The root node's parent is "". Create a 3-level hierarchy: Main Topic -> Key Themes -> Specific Concepts/Papers.

    Abstracts: ---
    {text}
    ---
    JSON Output:
    ```json
    """
    response = model.generate_content(prompt)
    cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
    return json.loads(cleaned_response)

# NEW: "Hypothesize" feature function
def get_hypotheses(text, query):
    model = genai.GenerativeModel('models/gemini-pro-latest')
    prompt = f"""
    You are a world-class research scientist. Based on the following research abstracts on '{query}', your task is to identify potential future research directions.

    Generate 3 to 5 novel research questions, unexplored hypotheses, or potential next steps. Frame them as clear, actionable questions that a PhD student could investigate.

    Abstracts:
    ---
    {text}
    ---
    
    Novel Research Hypotheses:
    """
    response = model.generate_content(prompt)
    return response.text

def draw_mindmap(data):
    df = pd.DataFrame(data)
    fig = go.Figure(go.Sunburst(ids=df['id'], labels=df['label'], parents=df['parent'], insidetextorientation='radial'))
    fig.update_layout(margin=dict(t=10, l=10, r=10, b=10), paper_bgcolor="#1E1E1E", font_color="white")
    return fig

# --- UI AND APP LOGIC ---
with st.sidebar:
    st.header("Controls")
    data_source = st.selectbox("Choose a data source:", ["arXiv", "PubMed"])
    search_query = st.text_input("Enter a research topic", placeholder="e.g., CRISPR-Cas9")
    num_papers = st.slider("Number of papers to synthesize", min_value=3, max_value=5, value=3)
    start_button = st.button("Discover & Synthesize")

st.title("🔬 Research OS")
st.write("Welcome to the Operating System for Research.")
st.markdown("---")

if start_button:
    if search_query:
        with st.spinner(f"Searching {data_source} and building analysis..."):
            try:
                if data_source == 'arXiv':
                    papers_data = fetch_arxiv_data(search_query, num_papers)
                else: # PubMed
                    papers_data = fetch_pubmed_data(search_query, num_papers)
                
                if not papers_data:
                    st.warning("No papers with abstracts found for this topic.")
                else:
                    combined_abstracts = "\n\n---\n\n".join([f"Paper Title: {p['title']}\nAbstract: {p['summary']}" for p in papers_data])
                    
                    # NEW: Three tabs for our workflow
                    tab1, tab2, tab3 = st.tabs(["Synthesized Analysis", "Mind Map", "Suggested Hypotheses"])
                    
                    with tab1:
                        st.subheader("Detailed Synthesized Analysis")
                        analysis_text = get_detailed_synthesis(combined_abstracts, search_query)
                        st.write(analysis_text)
                        
                        st.subheader(f"Papers Included")
                        for paper in papers_data:
                            st.markdown(f"- **{paper['title']}** ([Link]({paper['url']}))")

                    with tab2:
                        st.subheader("Interactive Mind Map")
                        mindmap_data = get_mindmap_data(combined_abstracts, search_query)
                        mindmap_fig = draw_mindmap(mindmap_data)
                        st.plotly_chart(mindmap_fig, use_container_width=True)

                    # NEW: Hypothesize feature in its own tab
                    with tab3:
                        st.subheader("Novel Research Hypotheses")
                        hypotheses_text = get_hypotheses(combined_abstracts, search_query)
                        st.markdown(hypotheses_text)
                    
                    st.success("Analysis complete!")

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a research topic.")