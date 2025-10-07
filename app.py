import streamlit as st
import arxiv
import google.generativeai as genai
from Bio import Entrez
import streamlit.components.v1 as components
import json
import asyncio
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Research OS",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API KEY CONFIGURATION ---
try:
    # Get the API key from Streamlit secrets
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("🚨 Google API Key not found. Please add it to your Streamlit secrets.")
    st.stop()

# --- ASYNCHRONOUS DATA FETCHING & AI FUNCTIONS ---

async def fetch_arxiv_data(query, max_results):
    """
    Asynchronously fetches paper data from the arXiv API.
    """
    loop = asyncio.get_running_loop()
    # Use run_in_executor to avoid blocking the asyncio event loop with synchronous library calls
    search = await loop.run_in_executor(None, lambda: arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    ))
    results = await loop.run_in_executor(None, list, search.results())

    # Return a list of dictionaries with paper details
    return [{"title": p.title, "summary": p.summary, "url": p.entry_id} for p in results if p.summary]

async def fetch_pubmed_data(query, max_results):
    """
    Asynchronously fetches paper data from the PubMed API using Biopython.
    """
    loop = asyncio.get_running_loop()

    def search_and_fetch():
        """Synchronous helper function to perform the Entrez API calls."""
        Entrez.email = "your.email@example.com"  # As per NCBI guidelines
        handle = Entrez.esearch(db="pubmed", term=query, retmax=str(max_results), sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        id_list = record["IdList"]

        if not id_list:
            return []

        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        papers_data = []
        for i, article in enumerate(records['PubmedArticle']):
            try:
                # Extract relevant details, handling potential missing data
                papers_data.append({
                    "title": article['MedlineCitation']['Article']['ArticleTitle'],
                    "summary": article['MedlineCitation']['Article']['Abstract']['AbstractText'][0],
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{id_list[i]}/"
                })
            except (KeyError, IndexError):
                # Skip paper if title or abstract is missing
                continue
        return papers_data

    # Run the synchronous function in a separate thread
    return await loop.run_in_executor(None, search_and_fetch)

async def get_detailed_synthesis(text, query):
    """
    Generates a detailed synthesized analysis using the Gemini API.
    """
    model = genai.GenerativeModel('models/gemini-pro')
    prompt = f"""
    Act as a world-class research analyst. Your task is to provide a comprehensive, graduate-level synthesized analysis of the following research paper abstracts on the topic of '{query}'.

    The analysis must be at least 600 words and structured with the following sections:
    1.  **Executive Summary:** A concise overview of the key findings and their collective importance.
    2.  **Detailed Paper Analysis:** A breakdown of each paper's main contributions, methodology, and findings.
    3.  **Synthesis and Future Outlook:** An integrated view of how the papers relate to each other, identifying common themes, contradictions, research gaps, and potential future research directions.

    Begin the analysis directly with the "Executive Summary" heading. Do not include any conversational introduction.

    Abstracts:
    {text}
    """
    response = await model.generate_content_async(prompt)
    return response.text

async def get_mindmap_data(text, query):
    """
    Generates hierarchical JSON data for a mind map using the Gemini API.
    """
    model = genai.GenerativeModel('models/gemini-pro')
    prompt = f"""
    Read the provided research paper abstracts on '{query}'. Based on the content, generate a nested, hierarchical JSON object suitable for rendering as a D3.js collapsible tree graph.

    The JSON must have a root object with a "name" key (the main research topic) and a "children" key (an array of child nodes).
    The structure should be at least 3 levels deep:
    - Level 1 (Root): The main research topic.
    - Level 2 (Themes): Key themes, concepts, or categories found across the papers.
    - Level 3 (Details): Specific papers, findings, or sub-concepts related to each theme.

    Return only the raw JSON object, without any explanatory text or markdown formatting.

    Abstracts:
    {text}
    """
    response = await model.generate_content_async(prompt)
    # Clean the response to ensure it's valid JSON
    cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
    return json.loads(cleaned_response)

async def get_hypotheses(text, query):
    """
    Generates novel research hypotheses using the Gemini API.
    """
    model = genai.GenerativeModel('models/gemini-pro')
    prompt = f"""
    Act as a world-class research scientist. Based on the provided abstracts on '{query}', your task is to identify research gaps and generate 3-5 novel, testable research questions.

    For each question, provide the following in Markdown format:
    - **Hypothesis:** A clear, bolded statement of the proposed relationship.
    - **Rationale:** A brief explanation of why this hypothesis is important and based on the provided texts.
    - **First Experiment:** A concise description of a feasible initial experiment to test the hypothesis.

    Begin directly with the first hypothesis. Do not include any conversational introduction.

    Abstracts:
    {text}
    """
    response = await model.generate_content_async(prompt)
    return response.text

def draw_mindmap(data):
    """
    Generates a self-contained HTML string to render an interactive D3.js mind map.
    """
    # This HTML template uses a placeholder for JSON data to be injected safely.
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body { background-color: #0E1117; margin: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }
        .node circle { fill: #999; stroke: steelblue; stroke-width: 2px; }
        .node text { font: 13px sans-serif; fill: #fafafa; }
        .link { fill: none; stroke: #555; stroke-opacity: 0.6; stroke-width: 1.5px; }
      </style>
    </head>
    <body>
      <svg width="100%" height="800"></svg>
      <script src="https://d3js.org/d3.v7.min.js"></script>
      <script>
        const data = {data_json};
        const width = 1200;

        // Compute the layout.
        const root = d3.hierarchy(data);
        const dx = 30;
        const dy = width / (root.height + 1);
        const tree = d3.tree().nodeSize([dx, dy]);

        root.x0 = dy / 2;
        root.y0 = 0;
        root.descendants().forEach((d, i) => {{
          d.id = i;
          d._children = d.children;
        }});

        const svg = d3.select("svg")
            .attr("viewBox", [-250, -400, width, 800])
            .style("font", "13px sans-serif")
            .style("user-select", "none");

        const gLink = svg.append("g")
            .attr("fill", "none")
            .attr("stroke", "#555")
            .attr("stroke-opacity", 0.6)
            .attr("stroke-width", 1.5);

        const gNode = svg.append("g")
            .attr("cursor", "pointer")
            .attr("pointer-events", "all");

        function update(source) {{
          const duration = 250;
          const nodes = root.descendants().reverse();
          const links = root.links();

          tree(root);

          let left = root;
          let right = root;
          root.eachBefore(node => {{
            if (node.x < left.x) left = node;
            if (node.x > right.x) right = node;
          }});

          const transition = svg.transition().duration(duration);

          // Update the nodes…
          const node = gNode.selectAll("g").data(nodes, d => d.id);
          const nodeEnter = node.enter().append("g")
              .attr("transform", d => `translate(${source.y0},${source.x0})`)
              .attr("fill-opacity", 0)
              .attr("stroke-opacity", 0)
              .on("click", (event, d) => {{
                d.children = d.children ? null : d._children;
                update(d);
              }});

          nodeEnter.append("circle")
              .attr("r", 6)
              .attr("fill", d => d._children ? "#555" : "#999")
              .attr("stroke-width", 10);

          nodeEnter.append("text")
              .attr("dy", "0.31em")
              .attr("x", d => d._children ? -12 : 12)
              .attr("text-anchor", d => d._children ? "end" : "start")
              .text(d => d.data.name)
              .clone(true).lower()
              .attr("stroke-linejoin", "round")
              .attr("stroke-width", 3)
              .attr("stroke", "#0E1117");

          // Transition nodes to their new position.
          node.merge(nodeEnter).transition(transition)
              .attr("transform", d => `translate(${d.y},${d.x})`)
              .attr("fill-opacity", 1)
              .attr("stroke-opacity", 1);

          node.exit().transition(transition).remove()
              .attr("transform", d => `translate(${source.y},${source.x})`)
              .attr("fill-opacity", 0)
              .attr("stroke-opacity", 0);

          // Update the links…
          const link = gLink.selectAll("path").data(links, d => d.target.id);
          const linkEnter = link.enter().append("path")
              .attr("d", d => {{
                const o = {{x: source.x0, y: source.y0}};
                return d3.linkHorizontal().x(d => o.y).y(d => o.x)({{source: o, target: o}});
              }});

          link.merge(linkEnter).transition(transition)
              .attr("d", d3.linkHorizontal().x(d => d.y).y(d => d.x));

          link.exit().transition(transition).remove()
              .attr("d", d => {{
                const o = {{x: source.x, y: source.y}};
                return d3.linkHorizontal().x(d => o.y).y(d => o.x)({{source: o, target: o}});
              }});

          root.eachBefore(d => {{
            d.x0 = d.x;
            d.y0 = d.y;
          }});
        }}

        update(root);
      </script>
    </body>
    </html>
    """
    # Safely embed the JSON data into the HTML template
    return html_template.format(data_json=json.dumps(data))

# --- UI AND APP LOGIC ---

# Sidebar for user controls
with st.sidebar:
    st.image("https://i.imgur.com/rLoaV0k.png", width=50) # Placeholder logo
    st.title("Research OS")
    st.markdown("Your AI-powered research dashboard.")
    st.markdown("---")
    st.header("Controls")
    data_source = st.selectbox("Data Source", ["arXiv", "PubMed"], help="Select the database to search for papers.")
    search_query = st.text_input("Research Topic", placeholder="e.g., Quantum Entanglement", help="Enter the topic you want to research.")
    num_papers = st.slider("Number of papers", min_value=2, max_value=4, value=3, help="Select how many recent papers to analyze.")
    start_button = st.button("Discover & Synthesize", type="primary", use_container_width=True)

st.title("🔬 Research OS")
st.markdown("### Welcome! Enter a topic in the sidebar to begin your analysis.")

# --- MAIN EXECUTION ---

async def main():
    if start_button:
        # Validate input
        if not search_query:
            st.warning("Please enter a research topic to start.")
            return

        # Main processing block with spinner and error handling
        with st.spinner(f"Fetching papers and building AI analysis for '{search_query}'... This may take a moment. ⏳"):
            try:
                # 1. Fetch data from the selected source
                if data_source == 'arXiv':
                    papers_data = await fetch_arxiv_data(search_query, num_papers)
                else:
                    papers_data = await fetch_pubmed_data(search_query, num_papers)

                if not papers_data:
                    st.warning("No papers with abstracts were found for this topic. Please try a different query.")
                    return

                # Combine abstracts into a single text block for AI processing
                combined_abstracts = "\n\n---\n\n".join(
                    [f"**Paper Title: {p['title']}**\n{p['summary']}" for p in papers_data]
                )

                # 2. <<< MODIFIED SECTION >>>
                # Run all AI tasks sequentially to respect the free tier rate limit
                # We've replaced the fast 'asyncio.gather' with slower, one-by-one calls.
                sub_spinner = st.empty()

                sub_spinner.info("Step 1/3: Generating detailed analysis...")
                analysis_text = await get_detailed_synthesis(combined_abstracts, search_query)
                await asyncio.sleep(1) # Small delay to ensure user sees the message

                sub_spinner.info("Step 2/3: Creating interactive mind map...")
                mindmap_data = await get_mindmap_data(combined_abstracts, search_query)
                await asyncio.sleep(1)

                sub_spinner.info("Step 3/3: Suggesting novel hypotheses...")
                hypotheses_text = await get_hypotheses(combined_abstracts, search_query)

                sub_spinner.empty() # Clear the status message

                # 3. Display the results in tabs
                st.header(f"Analysis for: *{search_query}*")
                tab1, tab2, tab3 = st.tabs(["📄 **Synthesized Analysis**", "🧠 **Interactive Mind Map**", "💡 **Suggested Hypotheses**"])

                with tab1:
                    st.markdown(analysis_text)
                    with st.expander("View Source Papers"):
                        for paper in papers_data:
                            st.markdown(f"- **{paper['title']}** ([Link]({paper.get('url', '#')}))")

                with tab2:
                    st.info("Click on nodes to expand or collapse them.")
                    mindmap_html = draw_mindmap(mindmap_data)
                    components.html(mindmap_html, height=800, scrolling=False)

                with tab3:
                    st.markdown(hypotheses_text)

                st.success("Analysis complete! 🎉")

            except Exception as e:
                st.error("A critical error occurred. This could be due to API rate limits or an issue with the fetched data.")
                st.exception(e)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
    