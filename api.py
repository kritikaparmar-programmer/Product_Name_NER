from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import spacy
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Product NER API")

# Load the trained model
try:
    nlp = spacy.load("models/product_ner_model")
    logger.info("Successfully loaded NER model")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError("Model not found. Please ensure the model is trained and saved in models/product_ner_model")

def preprocess_text(text: str) -> str:
    """Preprocess text similar to what we did while training"""
    return ' '.join(text.lower().strip().split())

class ProductInput(BaseModel):
    title: str = Field(..., min_length=1, description="The product title to analyze")
    attributes: Dict[str, List[str]] = Field(default_factory=dict, description="Additional product attributes")

class EntityResponse(BaseModel):
    text: str
    label: str
    start: int
    end: int

class NERResponse(BaseModel):
    entities: List[EntityResponse]
    processed_text: str

@app.post("/analyze", response_model=NERResponse)
async def analyze_text(product: ProductInput):
    """
    Analyze product title using NER model
    """
    try:
        logger.info(f"Processing title: {product.title}")
        
        # Preprocess the text
        processed_text = preprocess_text(product.title)
        
        # Process with spaCy model
        doc = nlp(processed_text)
        
        # Extract entities
        entities = [
            EntityResponse(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char
            )
            for ent in doc.ents
        ]
        
        logger.info(f"Found {len(entities)} entities")
        
        return NERResponse(
            entities=entities,
            processed_text=processed_text
        )
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Check if the service is healthy and model is loaded"""
    return {
        "status": "healthy",
        "model_loaded": nlp is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")