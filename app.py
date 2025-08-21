import streamlit as st
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List, TypedDict
import os
from dotenv import load_dotenv
import json
import re
import groq
import pandas as pd
from io import BytesIO

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

class CompanyState(TypedDict):
    company_url: str
    extracted_content: str
    company_name: str
    growth_initiatives: List[Dict[str, str]]
    it_issues: List[str]
    industry_pain_points: str
    company_pain_points: str
    products_services: str
    pitch: str
    analysis_complete: bool

def setup_page():
    st.set_page_config(
        page_title="Sharp SSDI Company Analysis",
        layout="wide"
    )
    st.title(" Sharp SSDI Company Analysis Agent")
    st.markdown("""
    This tool analyzes company websites to extract key business insights and identify opportunities for Sharp SSDI's document management solutions.
    """)

def extract_website_content(url: str) -> str:
    """Extract text content from a company website with improved parsing"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Extract company name from title or h1
        company_name = ""
        if soup.title:
            company_name = soup.title.text.split('|')[0].split('-')[0].strip()
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
            
        # Prioritize main content areas
        main_content = ""
        for tag in ['main', 'article', 'div[class*="content"]', 'div[class*="main"]']:
            elements = soup.select(tag)
            for element in elements:
                main_content += element.get_text(separator=' ', strip=True) + " "
        
        # If no main content found, use body
        if not main_content:
            main_content = soup.get_text(separator=' ', strip=True)
            
        # Limit to first 15k characters to avoid token limits but preserve more content
        content = f"COMPANY NAME: {company_name}\n\n{main_content}"[:15000]
        return content, company_name
        
    except Exception as e:
        st.error(f"Error extracting content: {str(e)}")
        return "", ""

def analyze_with_groq(prompt: str, model: str = "llama-3.3-70B-Versatile") -> str:
    """Use Groq API for analysis"""
    try:
        
        client = Groq(api_key=GROQ_API_KEY)

        
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a business analyst expert at identifying document management and contract lifecycle opportunities."
                },
                {
                    "role": "user",
                    "content": prompt
                }
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
    """Use Groq to analyze company content and extract required information"""
    
    prompt = f"""
    Analyze the following company information and provide structured outputs:

    COMPANY CONTENT:
    {state['extracted_content']}

    REQUIRED OUTPUTS:

    1. Growth/Transformation Initiatives (last 6 months):
    - Identify top 3 priority initiatives
    - Summarize each in one crisp line
    - Include a hyperlink to the source (if available in content)

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
            {{"initiative": "summary text", "source": "url or source reference"}},
            ... (3 items)
        ],
        "it_issues": ["issue 1", "issue 2", "issue 3"],
        "industry_pain_points": "text here",
        "company_pain_points": "text here",
        "products_services": "text here",
        "pitch": "text here"
    }}
    
    Focus particularly on identifying opportunities for Document Management Systems (DMS) and 
    Contract Lifecycle Management (CLM) solutions based on the company's pain points.
    """
    
    try:
        response = analyze_with_groq(prompt, model)
        
        if not response:
            return {
                "growth_initiatives": [],
                "it_issues": [],
                "industry_pain_points": "",
                "company_pain_points": "",
                "products_services": "",
                "pitch": "",
                "analysis_complete": False
            }
        
        # Clean the response to extract JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        
        if json_match:
            analysis = json.loads(json_match.group())
        else:
            st.error("Failed to parse analysis results. Please try again.")
            return {
                "growth_initiatives": [],
                "it_issues": [],
                "industry_pain_points": "",
                "company_pain_points": "",
                "products_services": "",
                "pitch": "",
                "analysis_complete": False
            }
        
        return {
            "growth_initiatives": analysis.get("growth_initiatives", []),
            "it_issues": analysis.get("it_issues", []),
            "industry_pain_points": analysis.get("industry_pain_points", ""),
            "company_pain_points": analysis.get("company_pain_points", ""),
            "products_services": analysis.get("products_services", ""),
            "pitch": analysis.get("pitch", ""),
            "analysis_complete": True
        }
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return {
            "growth_initiatives": [],
            "it_issues": [],
            "industry_pain_points": "",
            "company_pain_points": "",
            "products_services": "",
            "pitch": "",
            "analysis_complete": False
        }

def display_results(state: CompanyState):
    """Display the analysis results in Streamlit"""
    st.header(f"Analysis Results for {state.get('company_name', 'the company')}")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Growth Initiatives", "IT Issues & Pain Points", "Opportunities", "Pitch Recommendation"])
    
    with tab1:
        st.subheader(" Growth/Transformation Initiatives")
        for i, initiative in enumerate(state.get('growth_initiatives', []), 1):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**{i}. {initiative.get('initiative', 'N/A')}**")
            with col2:
                source = initiative.get('source', '')
                if source and source.startswith('http'):
                    st.markdown(f"[Source]({source})", unsafe_allow_html=True)
                elif source:
                    st.caption(f"Source: {source}")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(" IT-Related Issues")
            for i, issue in enumerate(state.get('it_issues', []), 1):
                st.write(f"{i}. {issue}")
                
        with col2:
            st.subheader("Industry Pain Points")
            st.info(state.get('industry_pain_points', 'No pain points identified'))
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(" Company Pain Points")
            st.error(state.get('company_pain_points', 'No specific pain points identified'))
            
        with col2:
            st.subheader(" Recommended Solutions")
            st.success(state.get('products_services', 'No specific solutions identified'))
    
    with tab4:
        st.subheader("Pitch Recommendation")
        st.markdown("---")
        st.write(state.get('pitch', 'No pitch recommendation generated'))
        st.markdown("---")
        
        if st.button(" Copy Pitch to Clipboard"):
            st.code(state.get('pitch', ''), language=None)
            st.success("Pitch copied to clipboard!")

def bulk_analysis():
    st.subheader("Upload CSV with Website URLs")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    model_option = st.selectbox(
        "Select Groq Model:",
        ["llama-3.3-70B-Versatile", "llama2-70b-4096", "gemma-7b-it", "llama3-8b-8192", "llama3-70b-8192"],
        index=0
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "website_url" not in df.columns:
            st.error("CSV must contain a column named 'website_url'")
            return

        if st.button("Start Bulk Analysis"):
            results = []
            progress_bar = st.progress(0)
            total = len(df)

            for idx, row in df.iterrows():
                url = row["website_url"]
                if not url.startswith("http"):
                    url = "https://" + url
                
                extracted_content, company_name = extract_website_content(url)
                
                if not extracted_content:
                    results.append({
                        "Website URL": url,
                        "Company Name": "",
                        "Growth Initiatives": "",
                        "IT Issues": "",
                        "Industry Pain Points": "",
                        "Company Pain Points": "",
                        "Products/Services": "",
                        "Pitch": "",
                        "Source URL(s)": ""
                    })
                    continue
                
                state = CompanyState(
                    company_url=url,
                    extracted_content=extracted_content,
                    company_name=company_name,
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

                growth_text = "; ".join([gi.get("initiative", "") for gi in result.get("growth_initiatives", [])])
                sources_text = "; ".join([gi.get("source", "") for gi in result.get("growth_initiatives", [])])

                results.append({
                    "Website URL": url,
                    "Company Name": company_name,
                    "Growth Initiatives": growth_text,
                    "IT Issues": "; ".join(result.get("it_issues", [])),
                    "Industry Pain Points": result.get("industry_pain_points", ""),
                    "Company Pain Points": result.get("company_pain_points", ""),
                    "Products/Services": result.get("products_services", ""),
                    "Pitch": result.get("pitch", ""),
                    "Source URL(s)": sources_text
                })

                progress_bar.progress((idx + 1) / total)
            
            progress_bar.empty()
            result_df = pd.DataFrame(results)

            st.subheader("Analysis Results")
            st.dataframe(result_df)

            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                result_df.to_excel(writer, index=False, sheet_name="Analysis Results")
            output.seek(0)

            st.download_button(
                label=" Download Excel",
                data=output,
                file_name="company_analysis_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

def main():
    setup_page()
    
    if not os.getenv("GROQ_API_KEY"):
        st.warning("Please set the GROQ_API_KEY environment variable in your Streamlit Cloud settings.")
    
    mode = st.radio("Choose Mode:", ["Single URL", "Bulk CSV"])
    
    if mode == "Single URL":
        with st.form("company_analysis_form"):
            company_url = st.text_input(
                "Enter Company URL:",
                placeholder="https://example.com",
                help="Enter the full URL of the company website you want to analyze"
            )
            
            model_option = st.selectbox(
                "Select Groq Model:",
                ["llama-3.3-70B-Versatile", "llama2-70b-4096", "gemma-7b-it", "llama3-8b-8192", "llama3-70b-8192"],
                index=0
            )
            
            submitted = st.form_submit_button("Analyze Company")
        
        if submitted and company_url:
            if not company_url.startswith('http'):
                st.error("Please enter a valid URL including http:// or https://")
                return
                
            with st.spinner("Analyzing company information. This may take a minute..."):
                extracted_content, company_name = extract_website_content(company_url)
                
                if not extracted_content:
                    st.error("Failed to extract content from the website. Please check the URL and try again.")
                    return
                
                initial_state = CompanyState(
                    company_url=company_url,
                    extracted_content=extracted_content,
                    company_name=company_name,
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
                    st.error("Analysis failed. Please try again with a different URL.")
    
    else:
        bulk_analysis()

if __name__ == "__main__":
    main()
