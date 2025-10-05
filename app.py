import streamlit as st
import arxiv
import google.generativeai as genai
from Bio import Entrez
import streamlit.components.v1 as components
import json
import asyncio

st.set_page_config(layout="wide")

# --- API KEY CONFIGURATION ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("Google API Key not found. Please add it to your Streamlit Cloud secrets.")
    st.stop()

# --- ASYNCHRONOUS DATA FETCHING & AI FUNCTIONS ---
# These functions are now "async" to run concurrently

async def fetch_arxiv_data(query, max_results):
    loop = asyncio.get_running_loop()
    # arxiv library is not async, so we run it in a thread executor
    search = await loop.run_in_executor(None, lambda: arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance))
    results = await loop.run_in_executor(None, list, search.results())
    return [{"title": p.title, "summary": p.summary, "url": p.entry_id} for p in results if p.summary]

async def fetch_pubmed_data(query, max_results):
    loop = asyncio.get_running_loop()
    # Biopython is not async, so we run it in a thread executor
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

async def get_detailed_synthesis(text, query):
    model = genai.GenerativeModel('models/gemini-pro-latest')
    prompt = f"Provide a detailed, in-depth synthesized analysis of the abstracts on '{query}'. Begin directly. The analysis should be at least 400 words."
    response = await model.generate_content_async([prompt, "Abstracts:", text])
    return response.text

async def get_mindmap_data(text, query):
    model = genai.GenerativeModel('models/gemini-pro-latest')
    prompt = f'Read the abstracts on \'{query}\' and generate a nested, hierarchical JSON object for a D3.js collapsible tree (root "name" and "children"). Go 3 levels deep: Root -> Theme -> Paper/Concept.'
    response = await model.generate_content_async([prompt, "Abstracts: ---", text, "--- JSON Output: ```json"])
    cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
    return json.loads(cleaned_response)

async def get_hypotheses(text, query):
    model = genai.GenerativeModel('models/gemini-pro-latest')
    prompt = f"Based on the abstracts on '{query}', generate 3-5 novel research questions. For each, provide: a bolded **Hypothesis**, a **Rationale**, and a **First Experiment**. Format using Markdown."
    response = await model.generate_content_async([prompt, "Abstracts: ---", text, "--- Novel Research Hypotheses:"])
    return response.text

def draw_mindmap(data):
    # This function doesn't do network I/O, so it doesn't need to be async
    html_template = f"""
    <!DOCTYPE html>
    <html><head><meta charset="utf-8">
    <style>
        body {{ background-color: #1E1E1E; margin: 0; }}
        .node circle {{ fill: #999; stroke: steelblue; stroke-width: 2px; }}
        .node text {{ font: 14px sans-serif; fill: #ccc; }}
        .link {{ fill: none; stroke: #555; stroke-opacity: 0.6; stroke-width: 1.5px; }}
    </style></head>
    <body><svg width="100%" height="800"></svg>
    <script src="[https://d3js.org/d3.v7.min.js](https://d3js.org/d3.v7.min.js)"></script>
    <script>
        const data = {json.dumps(data)};
        const width = 1200;
        const root = d3.hierarchy(data);
        const dx = 30; const dy = width / (root.height + 1);
        const tree = d3.tree().nodeSize([dx, dy]);
        root.x0 = dy / 2; root.y0 = 0;
        root.descendants().forEach((d, i) => {{ d.id = i; d._children = d.children; }});
        const svg = d3.select("svg").attr("viewBox", [-200, -400, width, 800]);
        const gLink = svg.append("g").attr("fill", "none").attr("stroke", "#555").attr("stroke-opacity", 0.6).attr("stroke-width", 1.5);
        const gNode = svg.append("g").attr("cursor", "pointer").attr("pointer-events", "all");
        function update(source) {{
            const duration = 250;
            const nodes = root.descendants().reverse(); const links = root.links();
            tree(root);
            let left = root; let right = root;
            root.eachBefore(node => {{
                if (node.x < left.x) left = node;
                if (node.x > right.x) right = node;
            }});
            const transition = svg.transition().duration(duration);
            const node = gNode.selectAll("g").data(nodes, d => d.id);
            const nodeEnter = node.enter().append("g").attr("transform", d => `translate(${source.y0},${source.x0})`).attr("fill-opacity", 0).attr("stroke-opacity", 0).on("click", (event, d) => {{ d.children = d.children ? null : d._children; update(d); }});
            nodeEnter.append("circle").attr("r", 6).attr("fill", d => d._children ? "#555" : "#999");
            nodeEnter.append("text").attr("dy", "0.31em").attr("x", d => d._children ? -12 : 12).attr("text-anchor", d => d._children ? "end" : "start").text(d => d.data.name).clone(true).lower().attr("stroke-linejoin", "round").attr("stroke-width", 3).attr("stroke", "#1E1E1E");
            node.merge(nodeEnter).transition(transition).attr("transform", d => `translate(${d.y},${d.x})`).attr("fill-opacity", 1).attr("stroke-opacity", 1);
            node.exit().transition(transition).remove().attr("transform", d => `translate(${source.y},${source.x})`).attr("fill-opacity", 0).attr("stroke-opacity", 0);
            const link = gLink.selectAll("path").data(links, d => d.target.id);
            const linkEnter = link.enter().append("path").attr("d", d3.linkHorizontal().x(d => source.y0).y(d => source.x0));
            link.merge(linkEnter).transition(transition).attr("d", d3.linkHorizontal().x(d => d.y).y(d => d.x));
            link.exit().transition(transition).remove().attr("d", d3.linkHorizontal().x(d => source.y).y(d => source.x));
            root.eachBefore(d => {{ d.x0 = d.x; d.y0 = d.y; }});
        }}
        update(root);
    </script></body></html>
    """
    return html_template

# --- UI AND APP LOGIC ---
with st.sidebar:
    st.header("Controls")
    data_source = st.selectbox("Choose a data source:", ["arXiv", "PubMed"])
    search_query = st.text_input("Enter a research topic", placeholder="e.g., CRISPR-Cas9")
    num_papers = st.slider("Number of papers to synthesize", min_value=2, max_value=4, value=3)
    start_button = st.button("Discover & Synthesize")

st.title("🔬 Research OS")

# This is the main async function that will run our app logic
async def main():
    if start_button:
        if search_query:
            with st.spinner(f"Searching {data_source} and building analysis..."):
                try:
                    if data_source == 'arXiv':
                        papers_data = await fetch_arxiv_data(search_query, num_papers)
                    else:
                        papers_data = await fetch_pubmed_data(search_query, num_papers)
                    
                    if not papers_data:
                        st.warning("No papers with abstracts found for this topic.")
                    else:
                        combined_abstracts = "\n\n---\n\n".join([f"Paper Title: {p['title']}\nAbstract: {p['summary']}" for p in papers_data])
                        
                        # --- Run all AI tasks at the same time ---
                        synthesis_task = get_detailed_synthesis(combined_abstracts, search_query)
                        mindmap_task = get_mindmap_data(combined_abstracts, search_query)
                        hypotheses_task = get_hypotheses(combined_abstracts, search_query)
                        
                        # Wait for all three tasks to complete
                        results = await asyncio.gather(synthesis_task, mindmap_task, hypotheses_task)
                        
                        analysis_text = results[0]
                        mindmap_data = results[1]
                        hypotheses_text = results[2]
                        
                        # --- Display results in tabs ---
                        tab1, tab2, tab3 = st.tabs(["Synthesized Analysis", "Mind Map", "Suggested Hypotheses"])
                        
                        with tab1:
                            st.subheader("Detailed Synthesized Analysis")
                            st.markdown(analysis_text)
                            st.subheader(f"Papers Included")
                            for paper in papers_data: st.markdown(f"- **{paper['title']}** ([Link]({paper.get('url', '#')}))")

                        with tab2:
                            st.subheader("Interactive Mind Map")
                            mindmap_html = draw_mindmap(mindmap_data)
                            components.html(mindmap_html, height=800, scrolling=True)

                        with tab3:
                            st.subheader("Novel Research Hypotheses")
                            st.markdown(hypotheses_text)
                        
                        st.success("Analysis complete!")

                except Exception as e:
                    st.error(f"An error occurred.")
                    st.exception(e) # Show the full error for debugging
        else:
            st.warning("Please enter a research topic.")

# This runs the main async function
if __name__ == "__main__":
    asyncio.run(main())