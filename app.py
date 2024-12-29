import streamlit as st
import requests
from typing import Dict, Optional
from collections import defaultdict

st.set_page_config(page_title="Product NER Analysis", layout="wide")

def analyze_product(title: str) -> Optional[Dict]:
    """Send request to FastAPI endpoint"""
    try:
        response = requests.post(
            "http://localhost:8000/analyze",
            json={"title": title, "attributes": {}}  
        )
        response.raise_for_status()  
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None
    
def group_entities(entities: list) -> Dict[str, list]:
    """Group entities by their labels"""
    grouped = defaultdict(list)
    for entity in entities:
        grouped[entity['label']].append(entity['text'])
    return dict(grouped)

def main():
    st.title("Product Name Entity Recognition")
    st.write("Analyze product titles to identify and extract entities")

    # Input section
    st.header("Input")
    title = st.text_input("Enter Product Title")

    # Analysis section
    if st.button("Generate") and title:
        if not title.strip():  
            st.error("Please enter a valid product title")
            return

        with st.spinner("Analyzing..."):  
            result = analyze_product(title)
            
            if result:
                st.header("Identified Entities")

                if result["entities"]:
                    grouped_entities = group_entities(result["entities"])
                    for label, texts in grouped_entities.items():
                        if len(texts) == 1:
                            st.markdown(f"**{label}**: {texts[0]}")
                        else:
                            st.markdown(f"**{label}**: {', '.join(texts)}")
                else:
                    st.info("No entities found in the provided text")
                    
                    
if __name__ == "__main__":
    main()