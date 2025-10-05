import streamlit as st
import arxiv
import google.generativeai as genai
from Bio import Entrez

st.set_page_config(layout="wide")

# --- API KEY CONFIGURATION ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("Google API Key not found. Please add it to your secrets.")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("Controls")
    
    # NEW: Dropdown to select the data source
    data_source = st.selectbox("Choose a data source:", ["arXiv", "PubMed"])
    
    search_query = st.text_input("Enter a research topic", placeholder="e.g., CRISPR-Cas9")
    num_papers = st.slider("Number of papers to synthesize", min_value=3, max_value=10, value=3)
    start_button = st.button("Discover & Synthesize")

# --- Main Page ---
st.title("🔬 Research OS")
st.write("Welcome to the Operating System for Research.")
st.markdown("---")

# --- Results Section ---
st.header("Results")

if start_button:
    if search_query:
        with st.spinner(f"Searching {data_source} and synthesizing papers..."):
            try:
                all_abstracts = []
                all_titles = []

                # --- LOGIC TO CHOOSE BETWEEN ARXIV AND PUBMED ---
                if data_source == 'arXiv':
                    search = arxiv.Search(query=search_query, max_results=num_papers, sort_by=arxiv.SortCriterion.Relevance)
                    results = list(search.results())
                    if not results:
                        st.warning("No papers found on arXiv for this topic.")
                        st.stop()
                    
                    for paper in results:
                        all_titles.append(paper.title)
                        all_abstracts.append(f"Paper Title: {paper.title}\nAbstract: {paper.summary}")

                elif data_source == 'PubMed':
                    Entrez.email = "your.email@example.com"
                    handle = Entrez.esearch(db="pubmed", term=search_query, retmax=str(num_papers))
                    record = Entrez.read(handle)
                    handle.close()
                    id_list = record["IdList"]
                    
                    if not id_list:
                        st.warning("No papers found on PubMed for this topic.")
                        st.stop()

                    handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
                    records = Entrez.read(handle)
                    handle.close()
                    
                    for pubmed_article in records['PubmedArticle']:
                        try:
                            title = pubmed_article['MedlineCitation']['Article']['ArticleTitle']
                            abstract = pubmed_article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
                            all_titles.append(title)
                            all_abstracts.append(f"Paper Title: {title}\nAbstract: {abstract}")
                        except (KeyError, IndexError):
                            # Skip articles that don't have a proper title or abstract
                            continue

                # --- AI SYNTHESIS (This part stays the same) ---
                if not all_abstracts:
                    st.warning("Could not extract valid abstracts from the found papers.")
                else:
                    combined_abstracts = "\n\n---\n\n".join(all_abstracts)
                    model = genai.GenerativeModel('models/gemini-pro-latest')
                    prompt = f"Synthesize the following abstracts on '{search_query}':\n\n{combined_abstracts}" # Simplified prompt
                    
                    response = model.generate_content(prompt)
                    synthesized_text = response.text

                    st.subheader("Synthesized Analysis")
                    st.write(synthesized_text)
                    
                    st.subheader(f"Top Papers Included in this Synthesis")
                    for title in all_titles:
                        st.markdown(f"- {title}")

                    st.success("Synthesis complete!")

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a research topic in the sidebar.")