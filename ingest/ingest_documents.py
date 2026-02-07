import os
import sys
import glob
print("Script started... importing modules...", flush=True)
import fitz  # PyMuPDF
from typing import List, Dict, Generator

# Add project root to sys.path to allow imports from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.embeddings import EmbeddingModel
from app.core.vectorstore import VectorStore
from app.utils.chunker import chunk_text

DOCS_DIR = "documents"
INDEX_PATH = "index/faiss.index"
METADATA_PATH = "index/metadata.pkl"
BATCH_SIZE = 32  # Keeping batch size small for safety

def yield_file_content(file_path: str) -> Generator[str, None, None]:
    """
    Yields content from a file. 
    For PDF: Yields page by page text.
    For TXT: Yields line by line or chunks of text.
    """
    try:
        if file_path.lower().endswith(".pdf"):
            doc = fitz.open(file_path)
            for page in doc:
                text = page.get_text()
                if text:
                    yield text
            doc.close()
            
        elif file_path.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                # Read in 4KB chunks for TXT
                while True:
                    data = f.read(4096)
                    if not data:
                        break
                    yield data
                    
    except Exception as e:
        print(f"Error reading {file_path}: {e}", flush=True)

def process_batch(batch_chunks: List[Dict[str, str]], embedding_model, vector_store):
    """
    Generate embeddings for a batch and add to index.
    """
    if not batch_chunks:
        return

    texts = [item["content"] for item in batch_chunks]
    
    # Generate embeddings
    embeddings = embedding_model.get_passage_embeddings(texts)
    
    # Add to index
    vector_store.add_documents(embeddings, batch_chunks)

def main():
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"Folder created: {DOCS_DIR}. Please add files.", flush=True)
        return

    # Initialize models
    print("Initializing models...", flush=True)
    try:
        embedding_model = EmbeddingModel()
        print("Embedding model loaded.", flush=True)
    except Exception as e:
        print(f"Error loading embedding model: {e}", flush=True)
        return
    
    vector_store = VectorStore(index_path=INDEX_PATH, metadata_path=METADATA_PATH)
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        print("Loading existing index...", flush=True)
        vector_store.load()
    else:
        print("Creating new index...", flush=True)
        vector_store.create_index(dimension=768)

    # Get file list
    pdf_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    txt_files = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    all_files = pdf_files + txt_files
    
    if not all_files:
        print("No documents found.", flush=True)
        return

    print(f"Found {len(all_files)} documents. Processing...", flush=True)
    
    current_batch = []
    total_processed = 0
    
    # Buffer to handle text across page boundaries/small chunks
    text_buffer = ""
    MAX_BUFFER = 5000 # Process text when buffer reaches this size

    for file_path in all_files:
        print(f"Processing file: {file_path}", flush=True)
        
        # Stream content from file
        for content_chunk in yield_file_content(file_path):
            text_buffer += content_chunk
            
            # If buffer is large enough, chunk it
            # We assume chunk_text handles overlap internally but here we are treating
            # a stream of text. `chunk_text` is designed for a static string.
            # To properly stream with overlap we need to keep the tail.
            
            if len(text_buffer) > MAX_BUFFER:
                # We chunk the buffer.
                # Important: we need to keep some overlap for the next buffer iteration
                # effectively implementing sliding window on the stream.
                
                # Let's verify chunk_text behavior. It returns chunks.
                # If we just chunk the current buffer, we lose overlap with next page.
                # We can cheat: chunk the buffer, keep the last X chars.
                
                # Or simpler:
                # Generate chunks from buffer
                generated_chunks = list(chunk_text(text_buffer))
                
                if generated_chunks:
                    # If we have chunks, add all EXCEPT the last one to the batch,
                    # and keep the last one + some context as the new buffer.
                    # Actually, `chunk_text` covers the whole string. 
                    # If we want continuous overlap, we should keep the end of the text_buffer
                    # that equals to overlap size + some safety margin.
                    
                    # Optimization:
                    # 1. Yield all complete chunks.
                    # 2. Keep the "remainder" text (last overlap size) in buffer.
                    
                    overlap_size = 400 # 100 tokens * 4 chars
                    
                    # We process everything, but we reconstruct buffer from the last part of text_buffer
                    # waiting for more content.
                    
                    # Iterating over the generator
                    batch_candidates = []
                    for chunk in chunk_text(text_buffer):
                         batch_candidates.append(chunk)

                    # Now, we put all full chunks into current_batch
                    # But we must be careful about the edge.
                    # To be safe in streaming, we can process all chunks, 
                    # and reset buffer to the last `overlap` chars of `text_buffer`.
                    
                    for chunk in batch_candidates:
                         current_batch.append({"content": chunk, "source": os.path.basename(file_path)})
                         
                         if len(current_batch) >= BATCH_SIZE:
                            process_batch(current_batch, embedding_model, vector_store)
                            total_processed += len(current_batch)
                            print(f"Processed {total_processed} chunks...", flush=True)
                            current_batch = []
                    
                    # Reset buffer to the end of the current text to ensure overlap for next page
                    if len(text_buffer) > overlap_size:
                        text_buffer = text_buffer[-overlap_size:]
                    else:
                        text_buffer = "" # Should not happen if buffer > MAX_BUFFER
                    
        # End of file: process remaining buffer for this file
        if text_buffer:
            for chunk in chunk_text(text_buffer):
                current_batch.append({"content": chunk, "source": os.path.basename(file_path)})
                if len(current_batch) >= BATCH_SIZE:
                    process_batch(current_batch, embedding_model, vector_store)
                    total_processed += len(current_batch)
                    print(f"Processed {total_processed} chunks...", flush=True)
                    current_batch = []
            text_buffer = "" # Reset for next file

    # Final batch
    if current_batch:
        process_batch(current_batch, embedding_model, vector_store)
        total_processed += len(current_batch)
    
    print(f"Total processed chunks: {total_processed}", flush=True)
    print("Saving index...", flush=True)
    vector_store.save()
    print("Completed!", flush=True)

if __name__ == "__main__":
    main()
