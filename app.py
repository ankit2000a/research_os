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
    # This handles the case where the app is run locally without secrets.
    # It allows you to fall back to a local .env file if you have one.
    st.error("Google API Key not found. Please add it to your Streamlit Cloud secrets.")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("Controls")
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
                papers_data = [] # Use a list of dictionaries to hold all paper info

                if data_source == 'arXiv':
                    search = arxiv.Search(query=search_query, max_results=num_papers, sort_by=arxiv.SortCriterion.Relevance)
                    results = list(search.results())
                    if not results:
                        st.warning("No papers found on arXiv for this topic.")
                        st.stop()
                    
                    for paper in results:
                        papers_data.append({
                            "title": paper.title,
                            "summary": paper.summary,
                            "url": paper.entry_id
                        })

                elif data_source == 'PubMed':
                    Entrez.email = "your.email@example.com" # NCBI requires an email address
                    handle = Entrez.esearch(db="pubmed", term=search_query, retmax=str(num_papers), sort="relevance")
                    record = Entrez.read(handle)
                    handle.close()
                    id_list = record["IdList"]
                    
                    if not id_list:
                        st.warning("No papers found on PubMed for this topic.")
                        st.stop()

                    handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
                    records = Entrez.read(handle)
                    handle.close()
                    
                    for i, pubmed_article in enumerate(records['PubmedArticle']):
                        try:
                            title = pubmed_article['MedlineCitation']['Article']['ArticleTitle']
                            abstract = pubmed_article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
                            pubmed_id = id_list[i]
                            papers_data.append({
                                "title": title,
                                "summary": abstract,
                                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"
                            })
                        except (KeyError, IndexError):
                            # Skip articles that are missing a title or abstract
                            continue
                
                if not papers_data:
                    st.warning("Could not extract valid data from the found papers.")
                else:
                    # --- AI SYNTHESIS ---
                    combined_abstracts = "\n\n---\n\n".join([f"Paper Title: {p['title']}\nAbstract: {p['summary']}" for p in papers_data])
                    model = genai.GenerativeModel('models/gemini-pro-latest')
                    
                    prompt = f"""
                    You are a world-class research analyst. Your task is to provide a synthesized analysis of the following scientific papers on the topic of '{search_query}'.

                    Your analysis should:
                    1. Begin directly, without any introductory phrases like "Based on the provided abstracts."
                    2. Identify the core, common theme or problem that connects all the papers.
                    3. Briefly explain the key contribution or finding of each paper.
                    4. Highlight any relationships, contradictions, or progressions of ideas between the papers.

                    Here is the text from the papers:
                    ---
                    {combined_abstracts}
                    ---

                    Synthesized Analysis:
                    """
                    
                    response = model.generate_content(prompt)
                    synthesized_text = response.text

                    st.subheader("Synthesized Analysis")
                    st.write(synthesized_text)
                    
                    st.subheader(f"Top {len(papers_data)} Papers Included in this Synthesis")
                    for paper in papers_data:
                        st.markdown(f"- **{paper['title']}** ([Link]({paper['url']}))")

                    st.success("Synthesis complete!")

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a research topic in the sidebar.")