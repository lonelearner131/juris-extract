import chromadb
from chromadb.utils import embedding_functions
import os

# Initialize ChromaDB - saving to a local folder named "vector_db"
# This makes it persistent, meaning your demo will be instant on the second run!
chroma_client = chromadb.PersistentClient(path="./vector_db")

# Use the fast, local MiniLM model for embeddings (runs on your CPU flawlessly)
sentence_transformer_ef = embedding_functions.DefaultEmbeddingFunction()

def build_vector_store(chunks: list[str], collection_name: str = "judgments"):
    """
    Takes the text chunks, converts them to vectors locally, and stores them in ChromaDB.
    """
    print("Initializing local Vector Database...")
    
    # Create or get the collection
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef
    )
    
    # Prepare unique IDs for each chunk (e.g., "chunk_0", "chunk_1")
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # Add data to the database
    print(f"Embedding and storing {len(chunks)} chunks. (This is 100% local)")
    collection.upsert(
        documents=chunks,
        ids=ids
    )
    print("Vector storage complete!")
    return collection

def multi_query_search(collection, top_k: int = 3):
    """
    The 'Hacker Fix': Performs 3 parallel searches to guarantee we don't miss anything.
    """
    print("\nPerforming Multi-Query Semantic Search...")
    
    # The 3 distinct queries to hunt down every single action item
    queries = [
        "Directives, orders, respondent shall, compliance, hereby directed",
        "Timeline, within days, limitation period, appeal, deadline",
        "Nodal officer, responsible department, authority, committee"
    ]
    
    all_retrieved_chunks = set() # Using a set automatically removes duplicates!
    
    for i, query in enumerate(queries):
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Extract the documents from the result
        documents = results['documents'][0]
        for doc in documents:
            all_retrieved_chunks.add(doc) # Adds chunk to set (deduplicates)
            
        print(f"Query {i+1} completed. Found {len(documents)} relevant chunks.")
        
    print(f"\nSearch complete. Total UNIQUE chunks retrieved: {len(all_retrieved_chunks)}")
    
    # Combine everything into one concentrated 'Mega-Context' string for the AI
    mega_context = "\n\n...[CONTINUED]...\n\n".join(list(all_retrieved_chunks))
    return mega_context

# --- Quick Test Block ---
if __name__ == "__main__":
    # Import our chunking tools from Step 2
    from document_processor import extract_text_from_pdf, chunk_text
    
    if os.path.exists("test.pdf"):
        # 1. Process the PDF
        raw_text = extract_text_from_pdf("test.pdf")
        my_chunks = chunk_text(raw_text)
        
        # 2. Build the database
        my_collection = build_vector_store(my_chunks)
        
        # 3. Run the Multi-Query Search
        final_context = multi_query_search(my_collection)
        
        print("\n--- PREVIEW OF HIGHLY CONCENTRATED CONTEXT (Ready for Groq API) ---")
        print(final_context[:500] + "...\n[TRUNCATED FOR DISPLAY]")
    else:
        print("test.pdf not found. Please put a test.pdf in this folder.")