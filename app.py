import streamlit as st
import arxiv
import google.generativeai as genai
from Bio import Entrez
from pyvis.network import Network
import streamlit.components.v1 as components
import json

st.set_page_config(layout="wide")

# --- API KEY CONFIGURATION ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("Google API Key not found. Please add it to your Streamlit Cloud secrets.")
    st.stop()

# --- DATA FETCHING FUNCTIONS (Corrected) ---
def fetch_arxiv_data(query, max_results):
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    results = list(search.results())
    return [{"title": p.title, "summary": p.summary, "url": p.entry_id} for p in results if p.summary]

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

# --- AI & GRAPHING FUNCTIONS (New and Improved) ---
def get_detailed_synthesis(text, query):
    model = genai.GenerativeModel('models/gemini-pro-latest')
    prompt = f"Provide a detailed, in-depth synthesized analysis of the abstracts on '{query}'. Begin directly, identify the core theme, explain each paper's contribution, and highlight relationships. The analysis should be at least 400 words."
    response = model.generate_content([prompt, "Abstracts:", text])
    return response.text

def get_mindmap_data(text, query):
    model = genai.GenerativeModel('models/gemini-pro-latest')
    prompt = f"""
    You are a research analyst. Read the abstracts on '{query}' and generate a graph data structure as a JSON object with "nodes" and "edges".

    Rules:
    1. The graph should be hierarchical, starting from a single root node representing the main topic.
    2. Nodes: "id", "label", "group" (one of: 'Root', 'Theme', 'Paper', 'Concept').
    3. Edges: "from", "to", "label" (relationship like 'explores', 'introduces').

    Abstracts: ---
    {text}
    ---
    JSON Output:
    ```json
    """
    response = model.generate_content(prompt)
    cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
    return json.loads(cleaned_response)

# NEW: "NotebookLM-Style" Mind Map function
def draw_mindmap(graph_data):
    net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white', notebook=True, cdn_resources='in_line')

    for node in graph_data['nodes']:
        net.add_node(node['id'], label=node['label'], group=node.get('group', 'default'), title=node['id'])
    for edge in graph_data['edges']:
        net.add_edge(edge['from'], edge['to'])

    net.set_options("""
    var options = {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed"
        }
      },
      "physics": {
        "enabled": false
      },
      "interaction": {
        "navigationButtons": true
      }
    }
    """)

    net.show('mindmap.html')
    with open('mindmap.html', 'r', encoding='utf-8') as f:
        html_code = f.read()
    return html_code

# --- UI AND APP LOGIC ---
with st.sidebar:
    st.header("Controls")
    data_source = st.selectbox("Choose a data source:", ["arXiv", "PubMed"])
    search_query = st.text_input("Enter a research topic", placeholder="e.g., CRISPR-Cas9")
    num_papers = st.slider("Number of papers to synthesize", min_value=2, max_value=4, value=3)
    start_button = st.button("Discover & Synthesize")

st.title("🔬 Research OS")

if start_button:
    if search_query:
        with st.spinner(f"Building analysis for '{search_query}'..."):
            try:
                if data_source == 'arXiv':
                    papers_data = fetch_arxiv_data(search_query, num_papers)
                else:
                    papers_data = fetch_pubmed_data(search_query, num_papers)
                
                if not papers_data:
                    st.warning("No papers with abstracts found for this topic.")
                else:
                    combined_abstracts = "\n\n---\n\n".join([f"Paper Title: {p['title']}\nAbstract: {p['summary']}" for p in papers_data])
                    
                    tab1, tab2 = st.tabs(["Synthesized Analysis", "Mind Map"])
                    
                    with tab1:
                        st.subheader("Detailed Synthesized Analysis")
                        analysis_text = get_detailed_synthesis(combined_abstracts, search_query)
                        st.markdown(analysis_text)
                        
                        st.subheader(f"Papers Included")
                        for paper in papers_data:
                            st.markdown(f"- **{paper['title']}** ([Link]({paper.get('url', '#')}))")

                    with tab2:
                        st.subheader("Interactive Mind Map")
                        mindmap_data = get_mindmap_data(combined_abstracts, search_query)
                        mindmap_html = draw_mindmap(mindmap_data)
                        components.html(mindmap_html, height=800, scrolling=True)
                    
                    st.success("Analysis complete!")

            except Exception as e:
                st.error(f"An error occurred. Please check the logs.")
                st.exception(e) # Show the full error for debugging
    else:
        st.warning("Please enter a research topic.")