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
import pyperclip
from urllib.parse import urljoin, urlparse

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
    st.title("Sharp SSDI Company Analysis Agent")
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

def analyze_with_groq(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
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
            # Fallback if JSON parsing fails
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

def display_results_table(state: CompanyState):
    """Display the analysis results in a table format"""
    st.header(f"Analysis Results for {state.get('company_name', 'the company')}")
    
    # Create a table for Growth Initiatives
    st.subheader(" Growth/Transformation Initiatives")
    if state.get('growth_initiatives'):
        growth_data = []
        for i, initiative in enumerate(state.get('growth_initiatives', []), 1):
            source = initiative.get('source', '')
            if source and source.startswith('http'):
                source_link = f"[Source]({source})"
            else:
                source_link = source if source else "Not available"
            
            growth_data.append({
                "#": i,
                "Initiative": initiative.get('initiative', 'N/A'),
                "Source": source_link
            })
        
        df_growth = pd.DataFrame(growth_data)
        st.markdown(df_growth.to_markdown(index=False), unsafe_allow_html=True)
        
        # Add copy button for growth initiatives
        if st.button("Copy Growth Initiatives", key="copy_growth"):
            growth_text = "\n".join([f"{i}. {item['Initiative']} - Source: {item.get('source', 'N/A')}" 
                                   for i, item in enumerate(state.get('growth_initiatives', []), 1)])
            pyperclip.copy(growth_text)
            st.success("Growth initiatives copied to clipboard!")
    else:
        st.info("No growth initiatives found.")
    
    st.markdown("---")
    
    # Create a table for IT Issues
    st.subheader(" IT-Related Issues")
    if state.get('it_issues'):
        it_data = []
        for i, issue in enumerate(state.get('it_issues', []), 1):
            it_data.append({
                "#": i,
                "Issue": issue
            })
        
        df_it = pd.DataFrame(it_data)
        st.table(df_it)
        
        # Add copy button for IT issues
        if st.button(" Copy IT Issues", key="copy_it"):
            it_text = "\n".join([f"{i}. {issue}" for i, issue in enumerate(state.get('it_issues', []), 1)])
            pyperclip.copy(it_text)
            st.success("IT issues copied to clipboard!")
    else:
        st.info("No IT issues found.")
    
    st.markdown("---")
    
    # Create a table for Pain Points
    st.subheader(" Pain Points")
    pain_points_data = [
        {"Type": "Industry Pain Points", "Description": state.get('industry_pain_points', 'Not identified')},
        {"Type": "Company Pain Points", "Description": state.get('company_pain_points', 'Not identified')}
    ]
    
    df_pain = pd.DataFrame(pain_points_data)
    st.table(df_pain)
    
    # Add copy button for pain points
    if st.button(" Copy Pain Points", key="copy_pain"):
        pain_text = f"Industry Pain Points: {state.get('industry_pain_points', '')}\nCompany Pain Points: {state.get('company_pain_points', '')}"
        pyperclip.copy(pain_text)
        st.success("Pain points copied to clipboard!")
    
    st.markdown("---")
    
    # Create a table for Opportunities
    st.subheader(" Opportunities for Sharp SSDI")
    opportunities_data = [
        {"Aspect": "Recommended Solutions", "Details": state.get('products_services', 'Not identified')},
        {"Aspect": "Pitch Recommendation", "Details": state.get('pitch', 'Not identified')}
    ]
    
    df_opp = pd.DataFrame(opportunities_data)
    st.table(df_opp)
    
    # Add copy button for opportunities
    if st.button(" Copy Opportunities", key="copy_opp"):
        opp_text = f"Recommended Solutions: {state.get('products_services', '')}\nPitch Recommendation: {state.get('pitch', '')}"
        pyperclip.copy(opp_text)
        st.success("Opportunities copied to clipboard!")
    
    st.markdown("---")
    
    # Add full report copy button
    if st.button(" Copy Full Report", key="copy_full"):
        full_text = f"""
Analysis Report for {state.get('company_name', 'the company')}

GROWTH INITIATIVES:
{chr(10).join([f"{i}. {item['initiative']} - Source: {item.get('source', 'N/A')}" for i, item in enumerate(state.get('growth_initiatives', []), 1)])}

IT ISSUES:
{chr(10).join([f"{i}. {issue}" for i, issue in enumerate(state.get('it_issues', []), 1)])}

PAIN POINTS:
Industry: {state.get('industry_pain_points', '')}
Company: {state.get('company_pain_points', '')}

OPPORTUNITIES:
Recommended Solutions: {state.get('products_services', '')}
Pitch Recommendation: {state.get('pitch', '')}
"""
        pyperclip.copy(full_text)
        st.success("Full report copied to clipboard!")

def main():
    setup_page()
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        st.warning("Please set the GROQ_API_KEY environment variable in your Streamlit Cloud settings.")
    
    # Input section
    with st.form("company_analysis_form"):
        company_url = st.text_input(
            "Enter Company URL:",
            placeholder="https://example.com",
            help="Enter the full URL of the company website you want to analyze"
        )
        
        # Model selection - updated with correct Groq model names
        model_option = st.selectbox(
            "Select Groq Model:",
            [
                "mixtral-8x7b-32768", 
                "llama2-70b-4096", 
                "gemma-7b-it",
                "llama3-8b-8192",
                "llama3-70b-8192"
            ],
            index=0
        )
        
        submitted = st.form_submit_button("Analyze Company")
    
    if submitted and company_url:
        if not company_url.startswith('http'):
            st.error("Please enter a valid URL including http:// or https://")
            return
            
        with st.spinner("Analyzing company information. This may take a minute..."):
            # Extract content
            extracted_content, company_name = extract_website_content(company_url)
            
            if not extracted_content:
                st.error("Failed to extract content from the website. Please check the URL and try again.")
                return
            
            # Initialize state
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
            
            # Run analysis
            result = analyze_company_content(initial_state, model_option)
            result["company_name"] = company_name
            
            # Display results
            if result["analysis_complete"]:
                display_results_table(result)
                
                # Add feedback section
                st.markdown("---")
                st.subheader("Feedback")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(" Useful Analysis"):
                        st.success("Thanks for your feedback!")
                with col2:
                    if st.button(" Needs Improvement"):
                        st.error("We'll work to improve our analysis.")
                with col3:
                    if st.button(" Analyze Another Company"):
                        st.experimental_rerun()
            else:
                st.error("Analysis failed. Please try again with a different URL.")
    
    # Add sidebar with information
    with st.sidebar:
        st.header("About")
        st.info("""
        This tool analyzes company websites to identify opportunities for Sharp SSDI's solutions:
        
        - Document Management Systems (DMS)
        - Contract Lifecycle Management (CLM)
        - Process Automation
        
        Enter a company URL to get started.
        """)
        
        st.header("Example URLs")
        st.code("https://www.ab-inbev.com\nhttps://www.jindalaluminium.com\nhttps://www.lntvalves.com")
        
        st.header("Settings")
        if st.button("Clear Analysis"):
            st.experimental_rerun()

if __name__ == "__main__":
    main()
