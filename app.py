import streamlit as st
import arxiv
import google.generativeai as genai

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
    search_query = st.text_input("Enter a research topic", placeholder="e.g., Large Language Models for Code Generation")
    num_papers = st.slider("Number of papers to synthesize", min_value=3, max_value=10, value=5)
    start_button = st.button("Discover & Synthesize")

# --- Main Page ---
st.title("🔬 Research OS")
st.write("Welcome to the Operating System for Research.")
st.markdown("---")


# --- Results Section ---
st.header("Results")

if start_button:
    if search_query:
        with st.spinner(f"Searching for and synthesizing the top {num_papers} papers on '{search_query}'..."):
            try:
                # --- 1. DISCOVER: Search for papers by keyword ---
                search = arxiv.Search(
                    query=search_query,
                    max_results=num_papers,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                
                results = list(search.results())
                if not results:
                    st.warning("No papers found for this topic. Please try a different query.")
                    st.stop()

                all_abstracts = []
                all_titles = []
                for paper in results:
                    all_titles.append(paper.title)
                    all_abstracts.append(f"Paper Title: {paper.title}\nAbstract: {paper.summary}")
                
                combined_abstracts = "\n\n---\n\n".join(all_abstracts)

                # --- 2. THE AI MAGIC (SYNTHESIZE) ---
                model = genai.GenerativeModel('models/gemini-pro-latest')
                prompt = f"""
                You are a world-class research analyst. Your task is to read the abstracts of the following scientific papers on the topic of '{search_query}' and produce a single, synthesized analysis.

                Your summary should:
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

                # --- 3. DISPLAY THE RESULTS ---
                st.subheader("Synthesized Analysis")
                st.write(synthesized_text)
                
                st.subheader(f"Top {len(results)} Papers Included in this Synthesis")
                for paper in results:
                    st.markdown(f"- **{paper.title}** ([Link]({paper.entry_id}))")

                st.success("Synthesis complete!")

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a research topic in the sidebar.")