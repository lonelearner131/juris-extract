from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import os
import json
import uuid

# Import our pipeline functions
from document_processor import extract_text_from_pdf, chunk_text
from rag_engine import build_vector_store, multi_query_search
from llm_extractor import extract_action_plan

app = FastAPI(title="JurisExtract API")

# Allow the frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Setup (SQLite) ---
DB_FILE = "juris_extract.db"

def init_db():
    """Creates the necessary tables if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cases (
            id TEXT PRIMARY KEY,
            filename TEXT,
            summary TEXT,
            status TEXT DEFAULT 'pending_review'
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS action_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT,
            compliance_action TEXT,
            responsible_department TEXT,
            timeline_days TEXT,
            confidence_score INTEGER,
            verbatim_source_quote TEXT,
            status TEXT DEFAULT 'pending_review',
            FOREIGN KEY(case_id) REFERENCES cases(id)
        )
    ''')
    conn.commit()
    conn.close()

# Run initialization on startup
init_db()

# --- The Core API Endpoint ---
@app.post("/upload-judgment/")
async def upload_judgment(file: UploadFile = File(...)):
    """
    1. Receives the PDF from the frontend.
    2. Runs the RAG + Extraction Pipeline.
    3. Saves everything to the database.
    4. Returns the JSON structure to the frontend.
    """
    print(f"Received file: {file.filename}")
    
    # Save the uploaded file temporarily
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # Step 1: Process
        print("Extracting text and chunking...")
        raw_text = extract_text_from_pdf(temp_file_path)
        my_chunks = chunk_text(raw_text, chunk_size_words=200, overlap_words=30)
        
        # Step 2: RAG Search
        print("Building local Vector DB and searching...")
        
        # Generate our unique case_id HERE instead of Step 4
        case_id = str(uuid.uuid4())
        
        # Create a guaranteed safe, unique collection name (e.g., 'case_123e4567...')
        safe_collection_name = f"case_{case_id.replace('-', '')}"
        
        my_collection = build_vector_store(my_chunks, collection_name=safe_collection_name)
        final_context = multi_query_search(my_collection, top_k=3)
        
        # Step 3: Extract with Groq
        print("Generating Action Plan via Groq...")
        action_plan_json = extract_action_plan(final_context)
        
        # Step 4: Save to Database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Save the main case
        cursor.execute("INSERT INTO cases (id, filename, summary) VALUES (?, ?, ?)", 
                       (case_id, file.filename, action_plan_json.case_summary))
        
        # Save the action items
        for item in action_plan_json.action_items:
            cursor.execute('''
                INSERT INTO action_items 
                (case_id, compliance_action, responsible_department, timeline_days, confidence_score, verbatim_source_quote) 
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                case_id, 
                item.compliance_action, 
                item.responsible_department, 
                item.timeline_days, 
                item.confidence_score, 
                item.verbatim_source_quote
            ))
            
        conn.commit()
        conn.close()

        print(f"Successfully processed {file.filename} and saved to DB.")

        # Return success to the frontend
        return {
            "status": "success",
            "case_id": case_id,
            "filename": file.filename,
            "data": action_plan_json.model_dump()
        }

    except Exception as e:
        print(f"Error during processing: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/verify-case/{case_id}")
async def verify_case(case_id: str):
    """Marks a case and all its action items as verified by the Nodal Officer."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        # Update both the case and its action items
        cursor.execute("UPDATE cases SET status = 'verified' WHERE id = ?", (case_id,))
        cursor.execute("UPDATE action_items SET status = 'verified' WHERE case_id = ?", (case_id,))
        conn.commit()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- Dashboard Fetch Endpoint ---
@app.get("/api/dashboard-data/")
def get_dashboard_data():
    """Fetches all verified action items for the Executive Dashboard."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Fetch verified items and join with cases to get the filename
        cursor.execute('''
            SELECT 
                a.id, a.compliance_action, a.responsible_department, 
                a.timeline_days, a.confidence_score, c.filename 
            FROM action_items a
            JOIN cases c ON a.case_id = c.id
            WHERE a.status = 'verified'
        ''')
        rows = cursor.fetchall()
        conn.close()

        # Format the SQL rows into a clean flat JSON list for React
        items = []
        for row in rows:
            items.append({
                "id": row[0],
                "compliance_action": row[1],
                "responsible_department": row[2],
                "timeline_days": row[3],
                "confidence_score": row[4],
                "filename": row[5]
            })
            
        return {"status": "success", "data": items}
    except Exception as e:
        print(f"Dashboard Error: {e}")
        return {"status": "error", "message": str(e)}

# --- Basic Health Check Endpoint ---
@app.get("/")
def read_root():
    return {"status": "online", "message": "JurisExtract API is running!"}