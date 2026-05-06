import os
from typing import List
from dotenv import load_dotenv
import instructor
from groq import Groq
from pydantic import BaseModel, Field

# Load your Groq API key from the .env file
load_dotenv()

# Initialize the Groq client and wrap it with Instructor for JSON mode
# This guarantees the AI will ONLY output our Pydantic schema
client = instructor.from_groq(Groq(api_key=os.getenv("GROQ_API_KEY")))

# --- 1. Define the Strict JSON Schema ---
class ActionItem(BaseModel):
    compliance_action: str = Field(
        description="The specific action required by the judgment (e.g., 'File a compliance report', 'Pay compensation')."
    )
    responsible_department: str = Field(
        description="The specific government department or authority responsible for this action. If not explicitly stated, infer based on context or write 'Unspecified'."
    )
    timeline_days: str = Field(
        description="The timeline or deadline. Extract the exact timeframe (e.g., '60 days', '4 weeks'). If none is given, write 'Statutory Period'."
    )
    confidence_score: int = Field(
        description="A score from 0 to 100 representing how explicitly this information was stated in the text.",
        ge=0, le=100
    )
    verbatim_source_quote: str = Field(
        description="CRITICAL: Provide the EXACT, verbatim quote from the text that proves this action item. Do not paraphrase. This is used for UI highlighting."
    )

class ActionPlan(BaseModel):
    case_summary: str = Field(description="A brief 2-sentence summary of what the judgment is about.")
    action_items: List[ActionItem] = Field(
        description="A list of all required actions extracted from the judgment."
    )

# --- 2. The Extraction Engine ---
def extract_action_plan(retrieved_context: str) -> ActionPlan:
    """
    Sends the concentrated text to Groq Llama 3 and forces it into the ActionPlan JSON schema.
    """
    print("\nSending context to Groq API (Llama 3 8B) for JSON Extraction...")
    
    # We use Llama-3-8b because it is lightning fast on Groq
    action_plan = client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        response_model=ActionPlan,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert Indian Legal AI Assistant working for the government. "
                    "Your job is to read extracts from court judgments and output strict, factual action plans. "
                    "Never hallucinate. If a detail is missing, state 'Not explicitly mentioned'. "
                    "You MUST extract exact verbatim quotes for every action item."
                )
            },
            {
                "role": "user",
                "content": f"Extract the action plan from the following retrieved context:\n\n{retrieved_context}"
            }
        ],
        temperature=0.1, # Low temperature ensures factual, robotic accuracy
    )
    
    return action_plan

# --- Quick Test Block ---
if __name__ == "__main__":
    from document_processor import extract_text_from_pdf, chunk_text
    from rag_engine import build_vector_store, multi_query_search
    
    if os.path.exists("test.pdf"):
        print("--- FULL PIPELINE TEST ---")
        # 1. Process
        raw_text = extract_text_from_pdf("test.pdf")
        my_chunks = chunk_text(raw_text)
        
        # 2. Search
        my_collection = build_vector_store(my_chunks)
        final_context = multi_query_search(my_collection)
        
        # 3. Extract (The new step!)
        try:
            result = extract_action_plan(final_context)
            
            print("\n✅ SUCCESS! HERE IS THE GUARANTEED JSON OUTPUT:")
            print("=" * 50)
            # .model_dump_json(indent=2) formats it beautifully for the console
            print(result.model_dump_json(indent=2))
            
        except Exception as e:
            print(f"\n❌ API Error: {e}")
            print("Make sure your GROQ_API_KEY is correct in the .env file!")
    else:
        print("test.pdf not found.")