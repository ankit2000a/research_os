import streamlit as st
import arxiv
import google.generativeai as genai

st.set_page_config(layout="wide")

# --- API KEY CONFIGURATION ---
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

# --- Sidebar ---
with st.sidebar:
    st.header("Controls")
    paper_ids_input = st.text_area("Enter arXiv Paper IDs (one per line)", height=150, placeholder="1706.03762\n2305.15334\n2303.11366")
    start_button = st.button("Synthesize Papers")

# --- Main Page ---
st.title("🔬 Research OS")
st.write("Welcome to the Operating System for Research.")
st.markdown("---")

# --- Results Section ---
st.header("Results")

if start_button:
    if paper_ids_input:
        # CHANGED: Using standard .strip() instead of the essentials library
        paper_ids = [s.strip() for s in paper_ids_input.strip().split('\n')]
        
        with st.spinner(f"Fetching and synthesizing {len(paper_ids)} papers..."):
            try:
                all_abstracts = []
                all_titles = []
                for paper_id in paper_ids:
                    if not paper_id: # Skip empty lines
                        continue
                    search = arxiv.Search(id_list=[paper_id])
                    paper = next(search.results())
                    all_titles.append(paper.title)
                    all_abstracts.append(f"Paper Title: {paper.title}\nAbstract: {paper.summary}")
                
                if not all_abstracts:
                    st.warning("Please enter at least one valid paper ID.")
                else:
                    combined_abstracts = "\n\n---\n\n".join(all_abstracts)
                    model = genai.GenerativeModel('models/gemini-pro-latest')
                    prompt = f"""
                    You are a world-class research analyst. Your task is to read the abstracts of the following scientific papers and produce a single, synthesized summary.

                    Do not just summarize each paper individually. Your summary should:
                    1. Identify the core, common theme or problem that connects all the papers.
                    2. Briefly explain the key contribution or finding of each paper.
                    3. Highlight any relationships, contradictions, or progressions of ideas between the papers.

                    Here are the papers:
                    ---
                    {combined_abstracts}
                    ---

                    Synthesized Analysis:
                    """
                    
                    response = model.generate_content(prompt)
                    synthesized_text = response.text

                    st.subheader("Synthesized Analysis")
                    st.write(synthesized_text)
                    
                    st.subheader("Papers Included in this Synthesis")
                    for title in all_titles:
                        st.text(f"- {title}")

                    st.success("Synthesis complete!")

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter at least one paper ID in the sidebar.")