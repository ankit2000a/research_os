import streamlit as st
import arxiv
import google.generativeai as genai
from Bio import Entrez
import pandas as pd
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

# --- NEW: Function to get graph data from AI ---
def get_graph_from_ai(combined_abstracts, search_query):
    model = genai.GenerativeModel('models/gemini-pro-latest')
    prompt = f"""
    You are a research analyst. Your task is to read the following scientific abstracts on the topic of '{search_query}' and extract the key concepts and their relationships as a graph.

    The output must be a JSON object with two keys: "nodes" and "edges".
    - "nodes" should be a list of objects, each with an "id" (the concept name), "label" (the concept name), and "group" (a category like 'Method', 'Problem', 'Data').
    - "edges" should be a list of objects, each with a "from" (source node id), "to" (target node id), and "label" (the relationship, e.g., 'solves', 'improves', 'challenges').

    Example Output:
    {{
      "nodes": [
        {{"id": "Attention Mechanism", "label": "Attention Mechanism", "group": "Method"}},
        {{"id": "Recurrent Neural Networks", "label": "Recurrent Neural Networks", "group": "Method"}}
      ],
      "edges": [
        {{"from": "Attention Mechanism", "to": "Recurrent Neural Networks", "label": "improves on"}}
      ]
    }}

    Here are the abstracts:
    ---
    {combined_abstracts}
    ---

    JSON Output:
    ```json
    """
    
    response = model.generate_content(prompt)
    # Clean up the response to extract only the JSON part
    cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
    return json.loads(cleaned_response)

# --- NEW: Function to get graph data from AI ---
def get_graph_from_ai(combined_abstracts, search_query):
    model = genai.GenerativeModel('models/gemini-pro-latest')
    prompt = f"""
    You are a research analyst. Your task is to read the following scientific abstracts on the topic of '{search_query}' and extract the key concepts and their relationships as a structured graph.

    The output MUST BE a valid JSON object following this exact structure:
    {{
      "nodes": [
        {{"id": "Concept Name", "label": "Concept Name", "group": "category"}}
      ],
      "edges": [
        {{"from": "Source Concept", "to": "Target Concept", "label": "relationship"}}
      ]
    }}

    Rules for the graph:
    1.  **Nodes:** Identify 10-15 of the most important, high-level concepts. Do not include generic terms.
    2.  **Groups:** Categorize each node into one of the following groups: 'Core Technology', 'Problem', 'Application', 'Method', or 'Finding'.
    3.  **Edges:** Define the relationship between concepts with clear, simple labels like 'solves', 'improves on', 'enables', 'challenges', or 'is a type of'.

    Here are the abstracts:
    ---
    {combined_abstracts}
    ---

    JSON Output:
    ```json
    """
    
    response = model.generate_content(prompt)
    cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
    return json.loads(cleaned_response)

# --- Sidebar ---
with st.sidebar:
    st.header("Controls")
    data_source = st.selectbox("Choose a data source:", ["arXiv", "PubMed"])
    search_query = st.text_input("Enter a research topic", placeholder="e.g., CRISPR-Cas9")
    num_papers = st.slider("Number of papers to synthesize", min_value=3, max_value=5, value=3)
    start_button = st.button("Discover & Synthesize")

# --- Main Page ---
st.title("🔬 Research OS")
# ... (rest of the UI is similar to before)

if start_button:
    if search_query:
        with st.spinner(f"Searching {data_source} and building knowledge graph..."):
            try:
                # ... (Paper fetching logic is the same as before) ...
                papers_data = [] # Assume this gets filled with paper data

                if data_source == 'arXiv':
                    search = arxiv.Search(query=search_query, max_results=num_papers, sort_by=arxiv.SortCriterion.Relevance)
                    results = list(search.results())
                    if not results:
                        st.warning("No papers found on arXiv.")
                        st.stop()
                    for paper in results: papers_data.append({"title": paper.title, "summary": paper.summary, "url": paper.entry_id})
                elif data_source == 'PubMed':
                    Entrez.email = "your.email@example.com"
                    handle = Entrez.esearch(db="pubmed", term=search_query, retmax=str(num_papers), sort="relevance")
                    record = Entrez.read(handle)
                    handle.close()
                    id_list = record["IdList"]
                    if not id_list:
                        st.warning("No papers found on PubMed.")
                        st.stop()
                    handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
                    records = Entrez.read(handle)
                    handle.close()
                    for i, pubmed_article in enumerate(records['PubmedArticle']):
                        try:
                            title = pubmed_article['MedlineCitation']['Article']['ArticleTitle']
                            abstract = pubmed_article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
                            pubmed_id = id_list[i]
                            papers_data.append({"title": title, "summary": abstract, "url": f"[https://pubmed.ncbi.nlm.nih.gov/](https://pubmed.ncbi.nlm.nih.gov/){pubmed_id}/"})
                        except (KeyError, IndexError): continue
                
                if not papers_data:
                    st.warning("Could not extract valid data from the found papers.")
                else:
                    combined_abstracts = "\n\n---\n\n".join([f"Paper Title: {p['title']}\nAbstract: {p['summary']}" for p in papers_data])

                    # --- Generate and display results in tabs ---
                    tab1, tab2 = st.tabs(["Synthesized Analysis", "Knowledge Graph"])

                    with tab1:
                        # (The text synthesis logic is now here)
                        model_text = genai.GenerativeModel('models/gemini-pro-latest')
                        prompt_text = f"Provide a synthesized analysis of the following abstracts on '{search_query}'. Begin directly..." # Simplified for brevity
                        response_text = model_text.generate_content(prompt_text)
                        st.write(response_text.text)

                    with tab2:
                        # --- NEW: Call the functions to generate and draw the graph ---
                        st.subheader("Interactive Knowledge Graph")
                        graph_data = get_graph_from_ai(combined_abstracts, search_query)
                        graph_html = draw_graph(graph_data)
                        components.html(graph_html, height=620)
                    
                    st.subheader(f"Top {len(papers_data)} Papers Included")
                    for paper in papers_data:
                        st.markdown(f"- **{paper['title']}** ([Link]({paper['url']}))")
                    st.success("Analysis complete!")

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a research topic in the sidebar.")