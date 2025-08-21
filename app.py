import time
import streamlit as st
import json
import re
import os
import pandas as pd
from io import BytesIO
from typing import Dict, List, TypedDict
from groq import Groq
import asyncio
from playwright.async_api import async_playwright
import random
from requests.adapters import HTTPAdapter, Retry
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

class CompanyState(TypedDict):
    company_url: str
    extracted_content: str
    company_name: str
    company_links: List[Dict[str, str]]
    growth_initiatives: List[Dict[str, str]]
    it_issues: List[str]
    industry_pain_points: str
    company_pain_points: str
    products_services: str
    pitch: str
    analysis_complete: bool

# ----------------- PAGE SETUP -----------------
def setup_page():
    st.set_page_config(page_title="Sharp SSDI Company Analysis", layout="wide")
    st.title("Sharp SSDI Company Analysis Agent")
    st.markdown("""
    This tool analyzes company websites to extract key business insights and identify opportunities for Sharp SSDI's document management solutions.
    """)

# ----------------- PLAYWRIGHT SCRAPER -----------------
async def scrape_with_playwright(url: str) -> (str, str, List[Dict[str, str]]):
    try:
        if not url.startswith("http"):
            url = "https://" + url

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117 Safari/537.36"
            ))
            page = await context.new_page()
            await page.goto(url, timeout=60000, wait_until="networkidle")
            await asyncio.sleep(3)

            text_content = await page.inner_text("body")
            company_name = await page.title() or "Unknown"
            
            # Extract links with Playwright
            links = await page.evaluate("""() => {
                const links = [];
                document.querySelectorAll('a[href]').forEach(a => {
                    if (a.href && a.textContent.trim()) {
                        links.push({
                            text: a.textContent.trim(),
                            url: a.href
                        });
                    }
                });
                return links;
            }""")

            await browser.close()

        if len(text_content.strip()) < 500:
            raise Exception("Content too short or page might be JS-heavy")

        content = f"COMPANY NAME: {company_name}\n\n{text_content}"[:15000]
        return content, company_name, links
    except Exception as e:
        print(f"Playwright error: {e}")
        return "", "", []

# ----------------- FALLBACK SCRAPER (Requests + BS4) -----------------
def scrape_with_requests(url: str):
    try:
        if not url.startswith("http"):
            url = "https://" + url

        headers = {
            "User-Agent": random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.6 Safari/605.1.15'
            ])
        }

        retry_strategy = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        response = session.get(url, headers=headers, timeout=20)
        if response.status_code != 200:
            return "", "", []

        soup = BeautifulSoup(response.content, "lxml")
        company_name = soup.title.text.split("|")[0].split("-")[0].strip() if soup.title else "Unknown"

        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()

        main_content = " ".join([e.get_text(" ", strip=True) for e in soup.select("main, article, div[class*='content'], div[class*='main']")])
        if not main_content:
            main_content = soup.get_text(separator=' ', strip=True)

        if len(main_content) < 500:
            return "", "", []

        content = f"COMPANY NAME: {company_name}\n\n{main_content}"[:15000]
        links = extract_links_and_text(soup, url)
        return content, company_name, links
    except Exception as e:
        print(f"Requests error: {e}")
        return "", "", []

def normalize_url(url: str, base_url: str = "") -> str:
    if not url:
        return ""
    url = url.strip()
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("/"):
        return urljoin(base_url, url)  
    if url.startswith("#") or url.startswith("javascript:"):
        return ""
    return urljoin(base_url, "/" + url.lstrip("/"))

# ----------------- MAIN EXTRACTION FUNCTION -----------------
async def extract_website_content(url: str):
    content, name, links = await scrape_with_playwright(url)
    if not content:
        content, name, links = scrape_with_requests(url)
    return content, name, links

# ----------------- GROQ ANALYSIS -----------------
def analyze_with_groq(prompt: str, model: str = "llama-3.3-70B-Versatile") -> str:
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a business analyst expert at identifying document management and contract lifecycle opportunities."},
                {"role": "user", "content": prompt}
            ],
            model=model,
            temperature=0.1,
            max_tokens=4000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Groq API error: {str(e)}")
        return ""

def analyze_company_content(state: CompanyState, model: str) -> CompanyState:
    prompt = f"""
    Analyze the following company information and provide structured outputs:

    COMPANY CONTENT:
    {state['extracted_content']}

    AVAILABLE LINKS:
    {json.dumps(state['company_links'], indent=2)}

    REQUIRED OUTPUTS:

    1. Growth/Transformation Initiatives (last 6 months):
    - Identify top 3 priority initiatives
    - Summarize each in one crisp line
    - For each growth initiative, set "source" to the most relevant URL from AVAILABLE LINKS.
    - Only use URLs from AVAILABLE LINKS (do NOT make up links).
    - If none match, leave "source" as an empty string.

    2. Top 3 IT-Related Issues (last 6 months):
    - Summarize any reported IT problems or incidents

    3. Industry Pain Points (Technical Challenges):
    - Summarize in maximum 2 lines

    4. Company Pain Points:
    - What specific document/contract management challenges does this company face?
    - Focus on manual processes, compliance issues, and efficiency gaps

    5. Products/Services to Pitch:
    - Which Sharp SSDI solutions could address these pain points?

    6. Pitch Recommendation:
    - Create a compelling pitch that addresses the company's specific needs

    Format your response as JSON with the following structure:
    {{
        "growth_initiatives": [
            {{"initiative": "summary text", "source": "must be one of the urls from AVAILABLE LINKS"}}
        ],
        "it_issues": ["issue 1", "issue 2", "issue 3"],
        "industry_pain_points": "text here",
        "company_pain_points": "text here",
        "products_services": "text here",
        "pitch": "text here"
    }}
    """
    try:
        response = analyze_with_groq(prompt, model)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group())
            used_links = set()

            # First pass: validate and use the sources provided by the model
            for gi in analysis.get("growth_initiatives", []):
                source = gi.get("source", "")
                if source and source in [link["url"] for link in state["company_links"]]:
                    used_links.add(source)
                else:
                    gi["source"] = ""  # Clear invalid sources

            # Second pass: find best matches for initiatives without sources
            initiatives_without_sources = [gi for gi in analysis.get("growth_initiatives", []) if not gi.get("source")]
            
            for gi in initiatives_without_sources:
                best_match = None
                best_score = 0
                initiative_text = gi["initiative"].lower()
                
                # Extract keywords from initiative text
                initiative_keywords = extract_keywords(initiative_text)
                
                for link in state["company_links"]:
                    if link["url"] in used_links:
                        continue                
                        # Calculate relevance score based on text similarity
                    link_text = link["text"].lower()
                    score = calculate_relevance_score(initiative_text, initiative_keywords, link_text, link["url"])
                    
                    if score > best_score:
                        best_score = score
                        best_match = link["url"]
                        
                 if best_match and best_score > 2:  # Only use if score is above threshold
                    gi["source"] = best_match
                    used_links.add(best_match)

            return {
                "growth_initiatives": analysis.get("growth_initiatives", []),
                "it_issues": analysis.get("it_issues", []),
                "industry_pain_points": analysis.get("industry_pain_points", ""),
                "company_pain_points": analysis.get("company_pain_points", ""),
                "products_services": analysis.get("products_services", ""),
                "pitch": analysis.get("pitch", ""),
                "analysis_complete": True
            }

        else:
            return {**state, "analysis_complete": False}
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return {**state, "analysis_complete": False}

def extract_keywords(text):
    """Extract important keywords from text"""
    # Remove common stop words
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
    words = text.split()
    return [word for word in words if word not in stop_words and len(word) > 3]

def calculate_relevance_score(initiative_text, initiative_keywords, link_text, link_url):
    """Calculate how relevant a link is to an initiative"""
    score = 0
    
    # Check for exact matches of important words
    for word in initiative_keywords:
        if word in link_text:
            score += 3
        if word in link_url:
            score += 2
            
    # Check for shared words
    shared_words = set(initiative_text.split()) & set(link_text.split())
    score += len(shared_words)
    
    # Check if URL path contains relevant sections
    parsed_url = urlparse(link_url)
    path = parsed_url.path.lower()
    
    # Bonus for relevant URL sections
    relevant_sections = ["news", "blog", "press", "media", "insights", "updates", "announcements"]
    for section in relevant_sections:
        if section in path:
            score += 2
    
    # Penalize irrelevant URL sections
    irrelevant_sections = ["image", "img", "photo", "picture", "css", "js", "static", "assets"]
    for section in irrelevant_sections:
        if section in path:
            score -= 5
    
    # Penalize very short link text
    if len(link_text) < 5:
        score -= 3
        
    return score

# ----------------- DISPLAY RESULTS -----------------
def display_results(state: CompanyState):
    st.header(f"Analysis Results for {state.get('company_name', 'the company')}")
    tab1, tab2, tab3, tab4 = st.tabs(["Growth Initiatives", "IT Issues & Pain Points", "Opportunities", "Pitch Recommendation"])

    with tab1:
        st.subheader("Growth/Transformation Initiatives")
        for i, initiative in enumerate(state.get('growth_initiatives', []), 1):
            text = initiative.get('initiative', 'N/A')
            source = initiative.get('source', '')
            if source.startswith('http'):
                st.markdown(f"**{i}. {text}**")
                st.markdown(f"Source: [{source}]({source})")
                st.markdown("---")
            else:
                st.write(f"**{i}. {text}** (No relevant source found)")
                st.markdown("---")
    with tab2:
        st.subheader("IT-Related Issues")
        for i, issue in enumerate(state.get('it_issues', []), 1):
            st.write(f"{i}. {issue}")
        st.subheader("Industry Pain Points")
        st.info(state.get('industry_pain_points', 'No pain points identified'))

    with tab3:
        st.subheader("Company Pain Points")
        st.error(state.get('company_pain_points', 'No specific pain points identified'))
        st.subheader("Recommended Solutions")
        st.success(state.get('products_services', 'No specific solutions identified'))

    with tab4:
        st.subheader("Pitch Recommendation")
        st.write(state.get('pitch', 'No pitch recommendation generated'))

def extract_links_and_text(soup, base_url):
    links = []
    for a in soup.find_all('a', href=True):
        text = a.get_text(" ", strip=True)
        href = a['href']
        
        # Filter out irrelevant links
        if (href and text and len(text) > 3 and 
            not href.startswith(('javascript:', 'mailto:', 'tel:')) and
            not any(ext in href.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.css', '.js'])):
            
            normalized_url = normalize_url(href, base_url)
            if normalized_url and normalized_url.startswith("http"):
                links.append({"text": text, "url": normalized_url})
    
    return links

# ----------------- BULK ANALYSIS -----------------
async def bulk_analysis(model_option: str):
    start_time = time.time()
    st.subheader("Upload CSV with Website URLs")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "website" not in df.columns:
            st.error("CSV must contain a column named 'website'")
            return

        if st.button("Start Bulk Analysis"):
            results = []
            progress_bar = st.progress(0)
            total = len(df)
            status_text = st.empty()

            for idx, row in df.iterrows():
                url = row["website"]
                status_text.text(f"Processing {idx+1}/{total}: {url}")
                
                extracted_content, company_name, links = await extract_website_content(url)

                if not extracted_content:
                    results.append({"Website URL": url, "Company Name": "", "Pitch": ""})
                    continue

                state = CompanyState(
                    company_url=url, 
                    extracted_content=extracted_content, 
                    company_name=company_name,
                    company_links=links, 
                    growth_initiatives=[], 
                    it_issues=[], 
                    industry_pain_points="",
                    company_pain_points="", 
                    products_services="", 
                    pitch="", 
                    analysis_complete=False
                )

                result = analyze_company_content(state, model_option)
                result["company_name"] = company_name

                # Prepare sources for Excel output
                sources = []
                for i, gi in enumerate(result.get("growth_initiatives", [])):
                    source = gi.get("source", "")
                    if source:
                        sources.append(f'=HYPERLINK("{source}", "Source {i+1}")')
                    else:
                        sources.append("No source found")

                results.append({
                    "Website URL": url,
                    "Company Name": company_name,
                    "Growth Initiatives": "; ".join([gi.get("initiative", "") for gi in result.get("growth_initiatives", [])]),
                    "IT Issues": "; ".join(result.get("it_issues", [])),
                    "Industry Pain Points": result.get("industry_pain_points", ""),
                    "Company Pain Points": result.get("company_pain_points", ""),
                    "Products/Services": result.get("products_services", ""),
                    "Pitch": result.get("pitch", ""),
                    "Source URLs": "; ".join(sources)
                })
                
                progress_bar.progress((idx + 1) / total)

            progress_bar.empty()
            status_text.empty()
            
            result_df = pd.DataFrame(results)

            st.subheader("Analysis Results")
            st.dataframe(result_df)

            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                result_df.to_excel(writer, index=False, sheet_name="Analysis Results")
            output.seek(0)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes, seconds = divmod(elapsed_time, 60)
            st.success(f"Analysis completed in {int(minutes)} min {int(seconds)} sec")

            st.download_button(
                label="Download Excel", 
                data=output,
                file_name="company_analysis_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# ----------------- MAIN -----------------
def main():
    setup_page()
    mode = st.radio("Choose Mode:", ["Single URL", "Bulk CSV"])

    model_option = st.selectbox("Select Groq Model:",
                                 ["llama-3.3-70B-Versatile", "llama2-70b-4096", "gemma-7b-it", "llama3-8b-8192", "llama3-70b-8192"],
                                 index=0)

    if mode == "Single URL":
        with st.form("company_analysis_form"):
            company_url = st.text_input("Enter Company URL:", placeholder="https://example.com")
            submitted = st.form_submit_button("Analyze Company")

        if submitted and company_url:
            with st.spinner("Analyzing company information..."):
                extracted_content, company_name, links = asyncio.run(extract_website_content(company_url))

                if not extracted_content:
                    st.error("Failed to extract content from the website.")
                    return

                initial_state = CompanyState(
                    company_url=company_url, 
                    extracted_content=extracted_content,
                    company_name=company_name, 
                    company_links=links,
                    growth_initiatives=[], 
                    it_issues=[],
                    industry_pain_points="", 
                    company_pain_points="", 
                    products_services="",
                    pitch="", 
                    analysis_complete=False
                )

                result = analyze_company_content(initial_state, model_option)
                result["company_name"] = company_name

                if result["analysis_complete"]:
                    display_results(result)
                else:
                    st.error("Analysis failed. Try another URL.")
    else:
        asyncio.run(bulk_analysis(model_option))

if __name__ == "__main__":
    main()
