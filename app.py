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

def fetch_arxiv_data(query, max_results):
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    results = list(search.results())
    return [{"title": p.title, "summary": p.summary, "url": p.entry_id} for p in results if p.summary]

def fetch_pubmed_data(query, max_results):
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

def get_detailed_synthesis(text, query):
    model = genai.GenerativeModel('models/gemini-pro-latest')
    prompt = f"Provide a detailed, in-depth synthesized analysis of the abstracts on '{query}'. Begin directly. The analysis should be at least 400 words."
    response = model.generate_content([prompt, "Abstracts:", text])
    return response.text

def get_mindmap_data(text, query):
    model = genai.GenerativeModel('models/gemini-pro-latest')
    prompt = f"""
    Read the abstracts on '{query}' and generate a nested, hierarchical JSON object for a D3.js collapsible tree. The structure must have a root object with "name" and "children". Each child can have its own "children". Go 3 levels deep: Root -> Theme -> Paper/Concept.

    Example:
    {{
      "name": "CRISPR",
      "children": [
        {{ "name": "Therapeutic Applications", "children": [ {{"name": "Gene Editing"}}, {{"name": "Cancer Therapy"}} ] }},
        {{ "name": "Core Technology", "children": [ {{"name": "Cas9 Nuclease"}}, {{"name": "Guide RNA"}} ] }}
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
    Based on the abstracts on '{query}', generate 3-5 novel research questions. For each, provide: a bolded **Hypothesis**, a **Rationale**, and a **First Experiment**. Format using Markdown.
    Abstracts: ---
    {text}
    ---
    Novel Research Hypotheses:
    """
    response = model.generate_content(prompt)
    return response.text

# --- NEW: FINAL MIND MAP FUNCTION ---
def draw_mindmap(data):
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ background-color: #1E1E1E; }}
        .node circle {{ fill: #999; }}
        .node text {{ font: 14px sans-serif; fill: #ccc; }}
        .node--internal circle {{ fill: #555; }}
        .link {{ fill: none; stroke: #555; stroke-opacity: 0.6; stroke-width: 1.5px; }}
      </style>
    </head>
    <body>
      <svg width="100%" height="800"></svg>
      <script src="[https://d3js.org/d3.v7.min.js](https://d3js.org/d3.v7.min.js)"></script>
      <script>
        const data = {json.dumps(data)};
        const width = 1200;

        const tree = d3.tree().nodeSize([25, 300]);
        const root = d3.hierarchy(data);

        root.x0 = 0;
        root.y0 = 0;
        root.descendants().forEach((d, i) => {{
          d.id = i;
          d._children = d.children;
        }});

        const svg = d3.select("svg")
            .attr("viewBox", [-300, -400, width, 800]);

        const g = svg.append("g")
            .attr("font-family", "sans-serif")
            .attr("font-size", 14);

        const gLink = g.append("g")
            .attr("fill", "none")
            .attr("stroke", "#555")
            .attr("stroke-opacity", 0.6)
            .attr("stroke-width", 1.5);

        const gNode = g.append("g")
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
              .attr("stroke", "#1E1E1E");
              
          node.merge(nodeEnter).transition(transition)
              .attr("transform", d => `translate(${d.y},${d.x})`)
              .attr("fill-opacity", 1)
              .attr("stroke-opacity", 1);
              
          node.exit().transition(transition).remove()
              .attr("transform", d => `translate(${source.y},${source.x})`)
              .attr("fill-opacity", 0)
              .attr("stroke-opacity", 0);
              
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
              
          root.eachBefore(d => {{ d.x0 = d.x; d.y0 = d.y; }});
        }}
        
        update(root);
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
                # --- Paper fetching and processing ---
                papers_data = fetch_arxiv_data(search_query, num_papers) if data_source == 'arXiv' else fetch_pubmed_data(search_query, num_papers)
                
                if not papers_data:
                    st.warning("No papers with abstracts found for this topic.")
                else:
                    combined_abstracts = "\n\n---\n\n".join([f"Paper Title: {p['title']}\nAbstract: {p['summary']}" for p in papers_data])
                    
                    # --- Display results in tabs ---
                    tab1, tab2, tab3 = st.tabs(["Synthesized Analysis", "Mind Map", "Suggested Hypotheses"])
                    
                    with tab1:
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
                        st.subheader("Novel Research Hypotheses")
                        hypotheses_text = get_hypotheses(combined_abstracts, search_query)
                        st.markdown(hypotheses_text)
                    
                    st.success("Analysis complete!")
            except Exception as e:
                st.error(f"An error occurred. Please check the logs.")
                st.exception(e)
    else:
        st.warning("Please enter a research topic.")