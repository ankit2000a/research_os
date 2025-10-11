import streamlit as st
import arxiv
import google.generativeai as genai
from Bio import Entrez
import streamlit.components.v1 as components
import json
import asyncio
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Research OS",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API KEY CONFIGURATION ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("🚨 Google API Key not found. Please add it to your Streamlit secrets.")
    st.stop()

# --- DATA FETCHING FUNCTIONS (Modularized) ---

async def fetch_arxiv_data(query, max_results):
    """Fetches paper data from the arXiv API."""
    loop = asyncio.get_running_loop()
    search = await loop.run_in_executor(None, lambda: arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    ))
    results = await loop.run_in_executor(None, list, search.results())
    return [{"title": p.title, "summary": p.summary, "url": p.entry_id} for p in results if p.summary]

async def fetch_pubmed_data(query, max_results):
    """Fetches paper data from the PubMed API."""
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
        for i, article in enumerate(records['PubmedArticle']):
            try:
                papers_data.append({
                    "title": article['MedlineCitation']['Article']['ArticleTitle'],
                    "summary": article['MedlineCitation']['Article']['Abstract']['AbstractText'][0],
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{id_list[i]}/"
                })
            except (KeyError, IndexError): continue
        return papers_data
    return await loop.run_in_executor(None, search_and_fetch)

# --- AI PROCESSING FUNCTIONS (Modularized) ---

async def generate_research_brief(text, query):
    """
    NEW: Generates a structured JSON research brief using the powerful Gemini 2.5 Pro model.
    """
    # CORRECTED: Using the specific, stable model name based on the latest documentation.
    model = genai.GenerativeModel('gemini-2.5-pro')
    
    prompt = f"""
    Act as a world-class research analyst with extreme attention to detail.
    Your task is to analyze the following research paper abstracts on the topic of '{query}' and generate a structured JSON intelligence brief.

    The JSON output MUST conform to the following schema:
    {{
      "executive_summary": "A concise, high-level synthesis of the key findings and their collective importance. Should be a single paragraph.",
      "key_hypotheses_and_findings": [
        {{
          "paper_title": "The title of the paper.",
          "hypothesis": "The core hypothesis or research question of the paper.",
          "finding": "The primary finding or conclusion of the paper."
        }}
      ],
      "methodology_comparison": [
        {{
          "paper_title": "The title of the paper.",
          "methodology": "A brief description of the methodology used (e.g., 'Systematic Review', 'Randomized Control Trial with 250 participants', 'Computational Model')."
        }}
      ],
      "contradictions_and_gaps": [
        "A bullet point describing a contradiction between two or more papers.",
        "A bullet point identifying a clear research gap that is not addressed by any of the papers."
      ]
    }}

    Analyze the provided abstracts and return ONLY the raw JSON object, without any surrounding text, explanations, or markdown formatting.

    Abstracts:
    ---
    {text}
    ---
    """
    
    response = await model.generate_content_async(prompt)
    cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
    return json.loads(cleaned_response)

# --- UI & APP LOGIC ---

with st.sidebar:
    st.image("https://i.imgur.com/rLoaV0k.png", width=50)
    st.title("Research OS")
    st.markdown("The Insight Engine for Modern Research.")
    st.markdown("---")
    st.header("Controls")
    data_source = st.selectbox("Data Source", ["arXiv", "PubMed"])
    search_query = st.text_input("Research Topic", placeholder="e.g., CRISPR-Cas9 Gene Editing")
    num_papers = st.slider("Number of Papers", min_value=2, max_value=5, value=3)
    start_button = st.button("Generate Research Brief", type="primary", use_container_width=True)

st.title("🔬 Research OS")

async def main():
    if start_button:
        if not search_query:
            st.warning("Please enter a research topic to start.")
            return

        with st.spinner(f"Fetching papers and building your Dynamic Research Brief for '{search_query}'..."):
            try:
                # 1. Fetch data
                papers_data = await (fetch_arxiv_data(search_query, num_papers) if data_source == 'arXiv' else fetch_pubmed_data(search_query, num_papers))
                if not papers_data:
                    st.warning("No papers with abstracts found. Please try a different query.")
                    return

                combined_abstracts = "\n\n---\n\n".join([f"**Paper: {p['title']}**\n{p['summary']}" for p in papers_data])
                
                # 2. Generate the structured research brief
                research_brief_data = await generate_research_brief(combined_abstracts, search_query)

                # 3. Display the NEW "Dynamic Research Brief"
                st.header(f"Dynamic Research Brief: {search_query}")
                
                st.subheader("Executive Summary")
                st.markdown(research_brief_data.get("executive_summary", "No summary available."))

                st.subheader("Key Hypotheses & Findings")
                hypotheses_df = pd.DataFrame(research_brief_data.get("key_hypotheses_and_findings", []))
                st.table(hypotheses_df)

                st.subheader("Methodology Comparison")
                methods_df = pd.DataFrame(research_brief_data.get("methodology_comparison", []))
                st.table(methods_df)
                
                st.subheader("Identified Contradictions & Research Gaps")
                for item in research_brief_data.get("contradictions_and_gaps", []):
                    st.markdown(f"- {item}")
                
                with st.expander("View Source Papers & Raw Abstracts"):
                    for paper in papers_data:
                        st.markdown(f"**{paper['title']}** ([Link]({paper.get('url', '#')}))")
                        st.markdown(f"_{paper['summary']}_")

                st.success("Research Brief complete!")

            except json.JSONDecodeError:
                st.error("The AI model returned a malformed response. This can happen with complex topics. Please try again.")
            except Exception as e:
                st.error("An error occurred during analysis.")
                st.exception(e)
    else:
        st.info("Enter a topic in the sidebar and click 'Generate Research Brief' to begin.")

if __name__ == "__main__":
    asyncio.run(main())

