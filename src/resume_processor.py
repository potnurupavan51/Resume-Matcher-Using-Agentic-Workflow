import os
import json
import uuid
import traceback
from typing import List, Optional, Set, Dict, Any
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Import necessary models and functions
from src.extract_jd_text import extract_text_from_docx
from src.llm_handlers import extract_resume_parser_data_with_langchain_safe
from src.models import ResumeParser, ResumeRecord, Skill, Experience, Education



def process_and_embed_resumes(resumes_dir: str, faiss_index_save_path: str, output_json_dir: str) -> List[ResumeRecord]:
    """
    Processes DOCX resumes, extracts structured data, generates embeddings, and manages a FAISS vector index:
    - If a FAISS index exists, it loads it and adds new resumes.
    - If no index exists, it creates a new one.
    - Stores ResumeRecord data as metadata within the FAISS index documents.
    - Persists the updated/new FAISS index to the specified path.
    - Saves processed resume data as JSON files.

    Args:
        resumes_dir (str): Path to the directory containing resume DOCX files.
        faiss_index_save_path (str): Path to the directory where the FAISS index will be saved/loaded.
        output_json_dir (str): Path to the directory where processed resume JSON files will be saved.

    Returns:
        List[ResumeRecord]: A list of processed ResumeRecord objects.
    """
    load_dotenv() # Ensure .env is loaded for API key

    if not os.path.isdir(resumes_dir):
        print(f"❌ Error: Resume directory not found at '{resumes_dir}'.")
        return []

    os.makedirs(output_json_dir, exist_ok=True)
    os.makedirs(faiss_index_save_path, exist_ok=True)

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    faiss_vector_store: Optional[FAISS] = None
    processed_resume_records: List[ResumeRecord] = []
    
    # Track resumes already in the FAISS index to avoid re-processing
    existing_resume_ids_in_faiss: Set[str] = set()

    # --- Load existing FAISS index if it exists ---
    if os.path.exists(faiss_index_save_path) and len(os.listdir(faiss_index_save_path)) > 0:
        try:
            # When loading, provide a dummy embedding function. The actual embeddings are stored.
            faiss_vector_store = FAISS.load_local(faiss_index_save_path, embedding_model, allow_dangerous_deserialization=True)
            print(f"✅ Loaded existing FAISS index from '{faiss_index_save_path}'.")
            
            # Populate existing_resume_ids_in_faiss from the loaded index's metadata
            # Iterate through the index's internal store if possible, or assume all previously saved
            # are "existing". For simplicity, we'll rely on the filenames in the JSON output directory
            # as a proxy for what's already processed and embedded.
            for fname in os.listdir(output_json_dir):
                if fname.endswith('.json'):
                    try:
                        with open(os.path.join(output_json_dir, fname), 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if 'id' in data:
                                existing_resume_ids_in_faiss.add(data['id'])
                    except Exception as e:
                        print(f"⚠️ Could not read existing JSON file {fname}: {e}")

        except Exception as e:
            print(f"❌ Error loading existing FAISS index from '{faiss_index_save_path}': {e}")
            print("A new FAISS index will be created if new resumes are found.")
            faiss_vector_store = None # Ensure it's None if loading failed

    # --- Process new resumes ---
    resume_files = [f for f in os.listdir(resumes_dir) if f.endswith('.docx')]
    new_documents_for_faiss: List[Document] = []

    for filename in resume_files:
        file_path = os.path.join(resumes_dir, filename)
        resume_id = str(uuid.uuid5(uuid.NAMESPACE_URL, filename)) # Generate consistent UUID based on filename
        
        # Check if this resume (based on its derived ID) has already been processed and embedded
        if resume_id in existing_resume_ids_in_faiss:
            print(f"⏩ Skipping '{filename}': Already processed and embedded.")
            # If resume already processed, load its JSON and add to processed_resume_records
            json_filename = os.path.join(output_json_dir, f"{os.path.splitext(filename)[0]}.json")
            if os.path.exists(json_filename):
                try:
                    with open(json_filename, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        processed_resume_records.append(ResumeRecord(**data))
                except Exception as e:
                    print(f"⚠️ Error loading existing processed JSON for '{filename}': {e}")
            continue

        print(f"\n⚙️ Processing new resume: '{filename}'...")
        try:
            resume_text = extract_text_from_docx(file_path)
            if not resume_text:
                print(f"❌ Could not extract text from '{filename}'. Skipping.")
                continue

            # Extract structured data
            parsed_data: Optional[ResumeParser] = extract_resume_parser_data_with_langchain_safe(resume_text)
            if not parsed_data:
                print(f"❌ Failed to parse structured data for '{filename}'. Skipping embedding.")
                continue

            # Generate embedding for the full resume text
            # It's generally better to embed the full text or a concise summary
            # depending on whether the entire content or just key points are relevant for similarity.
            # Using the full text for more comprehensive similarity.
            resume_embedding = embedding_model.embed_query(resume_text)
            
            # Create ResumeRecord
            resume_record = ResumeRecord(
                id=resume_id,
                candidate_id=str(uuid.uuid4()), # Generate a new candidate_id for new records
                file_path=file_path,
                filename=filename,
                original_text=resume_text,
                parsed_data=parsed_data,
                embedding=resume_embedding # Store the embedding in the record
            )
            processed_resume_records.append(resume_record)

            # Prepare document for FAISS: content is the text, metadata is the ResumeRecord as dict
            # The 'page_content' of the Document should be what you want to search over.
            # We use the full text and store the ResumeRecord in metadata.
            # Ensure parsed_data is converted to dict for metadata storage
            doc_metadata = resume_record.model_dump(exclude={'embedding', 'original_text'}) # Exclude large fields
            
            # We explicitly add the resume_id to metadata so we can retrieve it later
            doc_metadata['resume_id'] = resume_record.id

            new_documents_for_faiss.append(
                Document(page_content=resume_text, metadata=doc_metadata)
            )

            # Save parsed data to JSON
            json_output_path = os.path.join(output_json_dir, f"{os.path.splitext(filename)[0]}.json")
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(resume_record.model_dump(mode='json'), f, indent=2, ensure_ascii=False)
            print(f"✅ Processed data for '{filename}' saved to '{json_output_path}'.")

        except Exception as e:
            print(f"❌ An error occurred processing '{filename}': {e}")
            traceback.print_exc()
            continue

    # --- Update FAISS index with new documents ---
    if new_documents_for_faiss:
        print(f"\nAdding {len(new_documents_for_faiss)} new resumes to FAISS index...")
        if faiss_vector_store:
            try:
                faiss_vector_store.add_documents(new_documents_for_faiss)
                print(f"✅ Added {len(new_documents_for_faiss)} new documents to existing FAISS index.")
            except Exception as e:
                print(f"❌ Error adding new documents to FAISS index: {e}")
                print("FAISS index might be corrupted or not updated correctly.")
                traceback.print_exc()
        else:
            # If no existing store, create a new one from new documents
            try:
                faiss_vector_store = FAISS.from_documents(new_documents_for_faiss, embedding_model)
                print(f"✅ Created a new FAISS index with {len(new_documents_for_faiss)} documents.")
            except Exception as e:
                print(f"❌ Error creating new FAISS index: {e}")
                print("No FAISS index was created.")
                traceback.print_exc()
                return [] # Exit if creation fails
    else:
        print("No new resumes to add to the FAISS index.")

    # --- Save the FAISS index (either updated or newly created) ---
    if faiss_vector_store:
        try:
            faiss_vector_store.save_local(faiss_index_save_path)
            print(f"✅ FAISS index saved/updated to '{faiss_index_save_path}'.")
        except Exception as e:
            print(f"❌ Error saving FAISS index: {e}")
            traceback.print_exc()
    else:
        print("⚠️ No FAISS index to save (no resumes processed or created).")

    return processed_resume_records