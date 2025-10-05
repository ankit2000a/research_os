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
def get_detailed_synthesis(text, query):
    model = genai.GenerativeModel('models/gemini-pro-latest')
    prompt = f"Provide a detailed, in-depth synthesized analysis of the abstracts on '{query}'. Begin directly, identify the core theme, explain each paper's contribution, and highlight relationships. The analysis should be at least 400 words."
    response = model.generate_content([prompt, "Abstracts:", text])
    return response.text

def get_mindmap_data(text, query):
    model = genai.GenerativeModel('models/gemini-pro-latest')
    prompt = f"""
    Read the abstracts on '{query}' and generate a hierarchical structure as a JSON object with "id", "label", and "parent". The root node's parent is "". Create a 3-level hierarchy: Main Topic -> Key Themes -> Specific Concepts/Papers.

    Abstracts: ---
    {text}
    ---
    JSON Output:
    ```json
    """
    response = model.generate_content(prompt)
    cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
    return json.loads(cleaned_response)

# NEW: Upgraded "Hypothesize" function
def get_hypotheses(text, query):
    model = genai.GenerativeModel('models/gemini-pro-latest')
    prompt = f"""
    You are a world-class research scientist. Based on the following research abstracts on '{query}', your task is to identify potential future research directions.

    Generate 3 to 5 novel research questions. For each question, provide:
    1.  A clear, bolded **Hypothesis**.
    2.  A brief **Rationale** explaining why this is a valuable question based on the provided text.
    3.  A suggested **First Experiment** to begin testing the hypothesis.

    Format the entire output using Markdown.

    Abstracts:
    ---
    {text}
    ---
    
    Novel Research Hypotheses:
    """
    response = model.generate_content(prompt)
    return response.text

# NEW: "NotebookLM-Style" Icicle Chart function
def draw_mindmap(data):
    df = pd.DataFrame(data)
    # Add a dummy value for sizing if not present
    if 'value' not in df.columns:
        df['value'] = 1

    fig = go.Figure(go.Icicle(
        ids=df['id'],
        labels=df['label'],
        parents=df['parent'],
        root_color="lightgrey"
    ))
    fig.update_layout(
        margin=dict(t=20, l=10, r=10, b=10),
        paper_bgcolor="#1E1E1E",
        font_color="white"
    )
    return fig

# --- Sidebar & Main App Logic ---
# (This part remains the same as before)
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
                    
                    # --- Display results in tabs ---
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

                    with tab3:
                        st.subheader("Novel Research Hypotheses")
                        hypotheses_text = get_hypotheses(combined_abstracts, search_query)
                        st.markdown(hypotheses_text)
                    
                    st.success("Analysis complete!")

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a research topic.")