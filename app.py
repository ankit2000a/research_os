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

# --- Functions for AI and Graphing ---
def get_graph_from_ai(combined_abstracts, search_query):
    model = genai.GenerativeModel('models/gemini-pro-latest')
    prompt = f"""
    You are a research analyst. Your task is to read the following scientific abstracts on '{search_query}' and extract key concepts and their relationships as a structured graph.

    Output MUST BE a valid JSON object with "nodes" and "edges" keys.
    - Nodes: "id", "label", "group" (one of: 'Core Technology', 'Problem', 'Application', 'Method', 'Finding').
    - Edges: "from", "to", "label" (relationship like 'solves', 'improves on').

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

# NEW: Hierarchical "Mind Map" style graph function
def draw_graph(graph_data):
    net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white', notebook=True, cdn_resources='in_line')

    for node in graph_data['nodes']:
        net.add_node(node['id'], label=node['label'], group=node.get('group', 'default'), title=node['id'])
    for edge in graph_data['edges']:
        net.add_edge(edge['from'], edge['to'], label=edge.get('label', ''))

    net.set_options("""
    var options = {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "levelSeparation": 250,
          "nodeSpacing": 150,
          "treeSpacing": 200,
          "direction": "UD", 
          "sortMethod": "directed" 
        }
      },
      "physics": {
        "enabled": false 
      },
      "interaction": {
        "navigationButtons": true,
        "keyboard": { "enabled": true }
      }
    }
    """)

    net.show('knowledge_graph.html')
    with open('knowledge_graph.html', 'r', encoding='utf-8') as f:
        html_code = f.read()
    return html_code

# --- Sidebar ---
with st.sidebar:
    st.header("Controls")
    data_source = st.selectbox("Choose a data source:", ["arXiv", "PubMed"])
    search_query = st.text_input("Enter a research topic", placeholder="e.g., CRISPR-Cas9")
    num_papers = st.slider("Number of papers to synthesize", min_value=3, max_value=5, value=3)
    start_button = st.button("Discover & Synthesize")

# --- Main Page ---
st.title("🔬 Research OS")
st.write("Welcome to the Operating System for Research.")
st.markdown("---")

if start_button:
    if search_query:
        with st.spinner(f"Searching {data_source} and building knowledge graph..."):
            try:
                papers_data = []
                if data_source == 'arXiv':
                    search = arxiv.Search(query=search_query, max_results=num_papers, sort_by=arxiv.SortCriterion.Relevance)
                    results = list(search.results())
                    if not results: st.warning("No papers found on arXiv."); st.stop()
                    for paper in results: papers_data.append({"title": paper.title, "summary": paper.summary, "url": paper.entry_id})
                elif data_source == 'PubMed':
                    Entrez.email = "your.email@example.com"
                    handle = Entrez.esearch(db="pubmed", term=search_query, retmax=str(num_papers), sort="relevance")
                    record = Entrez.read(handle); handle.close()
                    id_list = record["IdList"]
                    if not id_list: st.warning("No papers found on PubMed."); st.stop()
                    handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
                    records = Entrez.read(handle); handle.close()
                    for i, pubmed_article in enumerate(records['PubmedArticle']):
                        try:
                            title = pubmed_article['MedlineCitation']['Article']['ArticleTitle']
                            abstract = pubmed_article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
                            pubmed_id = id_list[i]
                            papers_data.append({"title": title, "summary": abstract, "url": f"[https://pubmed.ncbi.nlm.nih.gov/](https://pubmed.ncbi.nlm.nih.gov/){pubmed_id}/"})
                        except (KeyError, IndexError): continue
                
                if not papers_data:
                    st.warning("Could not extract valid data.")
                else:
                    combined_abstracts = "\n\n---\n\n".join([f"Paper Title: {p['title']}\nAbstract: {p['summary']}" for p in papers_data])
                    
                    tab1, tab2 = st.tabs(["Synthesized Analysis", "Knowledge Graph"])
                    with tab1:
                        st.subheader("Synthesized Analysis")
                        model_text = genai.GenerativeModel('models/gemini-pro-latest')
                        prompt_text = f"Provide a synthesized analysis of the following abstracts on '{search_query}'. Begin directly..."
                        response_text = model_text.generate_content([prompt_text, "Here are the abstracts:", combined_abstracts])
                        st.write(response_text.text)

                        # NEW: Moved the "Top Papers" section inside this tab
                        st.subheader(f"Papers Included in this Synthesis")
                        for paper in papers_data:
                            st.markdown(f"- **{paper['title']}** ([Link]({paper['url']}))")

                    with tab2:
                        st.subheader("Interactive Knowledge Graph")
                        graph_data = get_graph_from_ai(combined_abstracts, search_query)
                        graph_html = draw_graph(graph_data)
                        components.html(graph_html, height=750, scrolling=True)
                    
                    st.success("Analysis complete!")

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a research topic.")