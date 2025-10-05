import streamlit as st
import arxiv
import google.generativeai as genai
from Bio import Entrez
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

# --- DATA FETCHING & AI FUNCTIONS ---
# (These functions remain the same as the last version)
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

def get_detailed_synthesis(text, query):
    model = genai.GenerativeModel('models/gemini-pro-latest')
    prompt = f"Provide a detailed, in-depth synthesized analysis of the abstracts on '{query}'. Begin directly, identify the core theme, explain each paper's contribution, and highlight relationships. The analysis should be at least 400 words."
    response = model.generate_content([prompt, "Abstracts:", text])
    return response.text

# NEW PROMPT: Asks for a nested JSON structure perfect for a tree
def get_mindmap_data(text, query):
    model = genai.GenerativeModel('models/gemini-pro-latest')
    prompt = f"""
    You are a research analyst. Read the abstracts on '{query}' and generate a nested, hierarchical JSON object representing a mind map.

    The structure must have a root object with "name" and "children". Each child can have its own "children". Go at least 3 levels deep.

    Example:
    {{
      "name": "Main Topic",
      "children": [
        {{
          "name": "Theme 1",
          "children": [
            {{"name": "Concept A from Paper 1"}},
            {{"name": "Concept B from Paper 2"}}
          ]
        }}
      ]
    }}

    Abstracts: ---
    {text}
    ---
    JSON Output:
    ```json
    """
    response = model.generate_content(prompt)
    cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
    return json.loads(cleaned_response)

def get_hypotheses(text, query):
    model = genai.GenerativeModel('models/gemini-pro-latest')
    prompt = f"""
    You are a world-class research scientist. Based on the following research abstracts on '{query}', your task is to identify potential future research directions.
    Generate 3 to 5 novel research questions. For each question, provide:
    1. A clear, bolded **Hypothesis**.
    2. A brief **Rationale**.
    3. A suggested **First Experiment**.
    Format the entire output using Markdown.
    Abstracts: ---
    {text}
    ---
    Novel Research Hypotheses:
    """
    response = model.generate_content(prompt)
    return response.text

# NEW: "NotebookLM-Style" Mind Map function using pure HTML/JS
def draw_mindmap(data):
    # D3.js based collapsible tree inside an HTML template
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{
          background-color: #1E1E1E;
          color: white;
          font-family: sans-serif;
        }}
        .node circle {{
          fill: #999;
          stroke: steelblue;
          stroke-width: 3px;
        }}
        .node text {{
          font: 14px sans-serif;
          fill: #ccc;
        }}
        .link {{
          fill: none;
          stroke: #555;
          stroke-opacity: 0.4;
          stroke-width: 1.5px;
        }}
      </style>
    </head>
    <body>
      <div id="mindmap"></div>
      <script src="[https://d3js.org/d3.v7.min.js](https://d3js.org/d3.v7.min.js)"></script>
      <script>
        const data = {json.dumps(data)};
        
        const width = 1200;
        const margin = {{top: 20, right: 120, bottom: 20, left: 120}};
        
        const tree = d3.tree().size([width, width]);
        const root = d3.hierarchy(data);
        
        root.x0 = 0;
        root.y0 = 0;
        root.descendants().forEach((d, i) => {{
          d.id = i;
          d._children = d.children;
        }});
        
        const svg = d3.create("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", 800) // Fixed height
            .attr("viewBox", [-margin.left, -margin.top, width + margin.left + margin.right, 800])
            .style("font", "14px sans-serif")
            .style("user-select", "none");

        const gLink = svg.append("g")
            .attr("fill", "none")
            .attr("stroke", "#555")
            .attr("stroke-opacity", 0.4)
            .attr("stroke-width", 1.5);

        const gNode = svg.append("g")
            .attr("cursor", "pointer")
            .attr("pointer-events", "all");
            
        function update(source) {{
          const duration = d3.event && d3.event.altKey ? 2500 : 250;
          const nodes = root.descendants().reverse();
          const links = root.links();
          
          tree(root);

          let left = root;
          let right = root;
          root.eachBefore(node => {{
            if (node.x < left.x) left = node;
            if (node.x > right.x) right = node;
          }});

          const height = right.x - left.x + margin.top + margin.bottom;

          const transition = svg.transition()
              .duration(duration)
              .attr("height", height)
              .attr("viewBox", [-margin.left, left.x - margin.top, width, height])
              .tween("resize", window.ResizeObserver ? null : () => () => svg.dispatch("toggle"));
              
          const node = gNode.selectAll("g")
            .data(nodes, d => d.id);
            
          const nodeEnter = node.enter().append("g")
              .attr("transform", d => `translate(${source.y0},${source.x0})`)
              .attr("fill-opacity", 0)
              .attr("stroke-opacity", 0)
              .on("click", (event, d) => {{
                d.children = d.children ? null : d._children;
                update(d);
              }});
              
          nodeEnter.append("circle")
              .attr("r", 5)
              .attr("fill", d => d._children ? "#555" : "#999")
              .attr("stroke-width", 10);
              
          nodeEnter.append("text")
              .attr("dy", "0.31em")
              .attr("x", d => d._children ? -10 : 10)
              .attr("text-anchor", d => d._children ? "end" : "start")
              .text(d => d.data.name)
              .clone(true).lower()
              .attr("stroke-linejoin", "round")
              .attr("stroke-width", 3)
              .attr("stroke", "#1E1E1E");
              
          const nodeUpdate = node.merge(nodeEnter).transition(transition)
              .attr("transform", d => `translate(${d.y},${d.x})`)
              .attr("fill-opacity", 1)
              .attr("stroke-opacity", 1);
              
          const nodeExit = node.exit().transition(transition).remove()
              .attr("transform", d => `translate(${source.y},${source.x})`)
              .attr("fill-opacity", 0)
              .attr("stroke-opacity", 0);
              
          const link = gLink.selectAll("path")
            .data(links, d => d.target.id);
            
          const linkEnter = link.enter().append("path")
              .attr("d", d => {{
                const o = {{x: source.x0, y: source.y0}};
                return d3.linkHorizontal()({{source: o, target: o}});
              }});
              
          link.merge(linkEnter).transition(transition)
              .attr("d", d3.linkHorizontal().x(d => d.y).y(d => d.x));
              
          link.exit().transition(transition).remove()
              .attr("d", d => {{
                const o = {{x: source.x, y: source.y}};
                return d3.linkHorizontal()({{source: o, target: o}});
              }});
              
          root.eachBefore(d => {{
            d.x0 = d.x;
            d.y0 = d.y;
          }});
        }}
        
        update(root);
        
        document.getElementById("mindmap").append(svg.node());
      </script>
    </body>
    </html>
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

if start_button:
    if search_query:
        with st.spinner(f"Building analysis for '{search_query}'..."):
            try:
                # ... (Paper fetching logic is the same) ...
                papers_data = []
                if data_source == 'arXiv':
                    papers_data = fetch_arxiv_data(search_query, num_papers)
                else:
                    papers_data = fetch_pubmed_data(search_query, num_papers)
                
                if not papers_data:
                    st.warning("No papers with abstracts found for this topic.")
                else:
                    combined_abstracts = "\n\n---\n\n".join([f"Paper Title: {p['title']}\nAbstract: {p['summary']}" for p in papers_data])
                    
                    tab1, tab2, tab3 = st.tabs(["Synthesized Analysis", "Mind Map", "Suggested Hypotheses"])
                    
                    with tab1:
                        # ... (Same as before) ...
                        st.subheader("Detailed Synthesized Analysis")
                        analysis_text = get_detailed_synthesis(combined_abstracts, search_query)
                        st.markdown(analysis_text)
                        st.subheader(f"Papers Included")
                        for paper in papers_data: st.markdown(f"- **{paper['title']}** ([Link]({paper.get('url', '#')}))")

                    with tab2:
                        st.subheader("Interactive Mind Map")
                        mindmap_data = get_mindmap_data(combined_abstracts, search_query)
                        mindmap_html = draw_mindmap(mindmap_data)
                        components.html(mindmap_html, height=800, scrolling=True)

                    with tab3:
                        # ... (Same as before) ...
                        st.subheader("Novel Research Hypotheses")
                        hypotheses_text = get_hypotheses(combined_abstracts, search_query)
                        st.markdown(hypotheses_text)
                    
                    st.success("Analysis complete!")
            except Exception as e:
                st.error(f"An error occurred. Please check the logs.")
                st.exception(e)
    else:
        st.warning("Please enter a research topic.")