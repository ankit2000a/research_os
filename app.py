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

# --- Functions for AI and Graphing ---

# NEW: Function for the detailed text analysis
def get_detailed_synthesis(combined_abstracts, search_query):
    model = genai.GenerativeModel('models/gemini-pro-latest')
    prompt = f"""
    You are a world-class research analyst. Your task is to provide a detailed, in-depth, multi-paragraph synthesized analysis of the following scientific papers on the topic of '{search_query}'.

    Your analysis must:
    1. Begin directly, without any introductory phrases.
    2. Start with a high-level summary of the core theme that connects all the papers.
    3. Dedicate a separate, detailed paragraph to each paper, explaining its specific contributions, methods, and findings.
    4. Conclude with a final paragraph that highlights the relationships, contradictions, or overall progression of ideas between the papers.
    5. The entire analysis should be at least 400 words.

    Here is the text from the papers:
    ---
    {combined_abstracts}
    ---

    Detailed Synthesized Analysis:
    """
    response = model.generate_content(prompt)
    return response.text

# NEW: Function to get data for the mind map
def get_mindmap_data_from_ai(combined_abstracts, search_query):
    model = genai.GenerativeModel('models/gemini-pro-latest')
    prompt = f"""
    You are a research analyst. Your task is to read the abstracts on '{search_query}' and generate a hierarchical mind map structure as a JSON object.

    The structure must have a list of objects, each with "id", "label", and "parent".
    - The root node's parent should be an empty string "".
    - Create a 3-level hierarchy: Main Topic -> Key Themes -> Specific Concepts/Papers.

    Example:
    [
      {{"id": "CRISPR", "label": "CRISPR", "parent": ""}},
      {{"id": "Therapeutic Applications", "label": "Therapeutic Apps", "parent": "CRISPR"}},
      {{"id": "Gene Editing", "label": "Gene Editing", "parent": "Therapeutic Applications"}}
    ]

    Abstracts:
    ---
    {combined_abstracts}
    ---

    JSON Output:
    ```json
    """
    response = model.generate_content(prompt)
    cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
    return json.loads(cleaned_response)

# NEW: Function to draw the mind map
def draw_mindmap(mindmap_data):
    df = pd.DataFrame(mindmap_data)
    fig = go.Figure(go.Sunburst(
        ids=df['id'],
        labels=df['label'],
        parents=df['parent'],
        insidetextorientation='radial'
    ))
    fig.update_layout(
        margin=dict(t=10, l=10, r=10, b=10),
        paper_bgcolor="#1E1E1E",
        font_color="white"
    )
    return fig

# --- Sidebar & Main Page (Mostly the same) ---
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
                # --- Paper fetching logic is the same ---
                papers_data = []
                if data_source == 'arXiv':
                    search = arxiv.Search(query=search_query, max_results=num_papers, sort_by=arxiv.SortCriterion.Relevance)
                    results = list(search.results()); papers_data = [{"title": p.title, "summary": p.summary, "url": p.entry_id} for p in results]
                elif data_source == 'PubMed':
                    Entrez.email = "your.email@example.com"
                    handle = Entrez.esearch(db="pubmed", term=search_query, retmax=str(num_papers), sort="relevance")
                    record = Entrez.read(handle); handle.close()
                    id_list = record["IdList"]
                    if id_list:
                        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
                        records = Entrez.read(handle); handle.close()
                        for i, article in enumerate(records['PubmedArticle']):
                            try: papers_data.append({"title": article['MedlineCitation']['Article']['ArticleTitle'], "summary": article['MedlineCitation']['Article']['Abstract']['AbstractText'][0], "url": f"[https://pubmed.ncbi.nlm.nih.gov/](https://pubmed.ncbi.nlm.nih.gov/){id_list[i]}/"})
                            except: continue
                
                if not papers_data:
                    st.warning("No papers with abstracts found for this topic.")
                else:
                    combined_abstracts = "\n\n---\n\n".join([f"Paper Title: {p['title']}\nAbstract: {p['summary']}" for p in papers_data])
                    
                    # --- Generate and display results in tabs ---
                    tab1, tab2 = st.tabs(["Synthesized Analysis", "Mind Map"])
                    with tab1:
                        st.subheader("Detailed Synthesized Analysis")
                        analysis_text = get_detailed_synthesis(combined_abstracts, search_query)
                        st.write(analysis_text)
                        
                        st.subheader(f"Papers Included")
                        for paper in papers_data:
                            st.markdown(f"- **{paper['title']}** ([Link]({paper['url']}))")

                    with tab2:
                        st.subheader("Interactive Mind Map")
                        mindmap_data = get_mindmap_data_from_ai(combined_abstracts, search_query)
                        mindmap_fig = draw_mindmap(mindmap_data)
                        st.plotly_chart(mindmap_fig, use_container_width=True)
                    
                    st.success("Analysis complete!")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a research topic.")