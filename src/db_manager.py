# import os
# from typing import List, Tuple, Optional, Dict, Any
# import numpy as np
# # from langchain_community.vectorstores import FAISS # We are using faiss directly
# from langchain_openai import OpenAIEmbeddings
# from langchain_core.documents import Document
# import logging
# import json
# import faiss # Explicitly import faiss
 
# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
 
# # Import your custom models
# from src.llm_handlers import get_embedding_dimension
# from src.models import ResumeRecord
 
# class FAISSVectorDB:
#     """A manager for the FAISS vector database with a separate document store."""
#     def __init__(self, embedding_dimension: Optional[int] = None, index_path: str = "faiss_index/index.faiss", id_map_path: str = "faiss_index/id_map.json", docstore_path: str = "faiss_index/docstore.json", embedding_model_name: str = "text-embedding-3-small"):
#         """
#         Initializes the FAISS index and document store.
#         :param embedding_dimension: The dimension of your embedding vectors.
#         :param index_path: Path to save/load the FAISS index binary.
#         :param id_map_path: Path to save/load the mapping from FAISS internal IDs to resume_ids.
#         :param docstore_path: Path to save/load the metadata for each document.
#         :param embedding_model_name: The name of the OpenAI embedding model.
#         """
#         self.index_path = index_path
#         self.id_map_path = id_map_path
#         self.docstore_path = docstore_path # *** NEW ***
 
#         # Determine embedding dimension
#         self.embedding_dimension = embedding_dimension or get_embedding_dimension()
#         if self.embedding_dimension is None or self.embedding_dimension <= 0:
#             raise ValueError("Embedding dimension could not be determined or is invalid.")
 
#         self.embedding_model = OpenAIEmbeddings(model=embedding_model_name)
#         self.index: Optional[faiss.Index] = None
       
#         # *** NEW: The document store for metadata ***
#         # Key: resume_id (str), Value: metadata dictionary
#         self.docstore: Dict[str, Dict[str, Any]] = {}
       
#         # *** CHANGED: This now maps FAISS internal ID (int) back to resume_id (str) ***
#         self.index_to_resume_id: Dict[int, str] = {}
       
#         self._load_or_create_index()
 
#     def _load_or_create_index(self):
#         """Loads an existing FAISS index, ID map, and docstore, or creates new ones."""
#         os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
 
#         if os.path.exists(self.index_path) and os.path.exists(self.id_map_path) and os.path.exists(self.docstore_path):
#             try:
#                 logger.info(f"Loading FAISS index from {self.index_path}")
#                 self.index = faiss.read_index(self.index_path)
               
#                 logger.info(f"Loading ID map from {self.id_map_path}")
#                 with open(self.id_map_path, 'r', encoding='utf-8') as f:
#                     # JSON keys are strings, so convert them back to integers
#                     self.index_to_resume_id = {int(k): v for k, v in json.load(f).items()}
 
#                 # *** NEW: Load the document store ***
#                 logger.info(f"Loading document store from {self.docstore_path}")
#                 with open(self.docstore_path, 'r', encoding='utf-8') as f:
#                     self.docstore = json.load(f)
 
#                 if self.index.d != self.embedding_dimension:
#                     raise ValueError(f"Dimension mismatch: Index dimension ({self.index.d}) doesn't match expected dimension ({self.embedding_dimension}).")
               
#                 logger.info(f"FAISS system loaded successfully. Items: {self.index.ntotal}")
#             except Exception as e:
#                 logger.error(f"Error loading FAISS system: {e}. Creating new.", exc_info=True)
#                 self._create_new_index()
#         else:
#             logger.info("FAISS index, ID map, or docstore not found. Creating new system.")
#             self._create_new_index()
 
#     def _create_new_index(self):
#         """Creates a new FAISS index, ID map, and docstore."""
#         self.index = faiss.IndexFlatL2(self.embedding_dimension)
#         self.docstore = {} # Reset docstore
#         self.index_to_resume_id = {} # Reset ID map
#         logger.info(f"New FAISS IndexFlatL2 created with dimension {self.embedding_dimension}.")
 
#     async def add_resumes(self, resumes: List[ResumeRecord]):
#         """Creates embeddings, adds them to FAISS, and stores their metadata."""
#         if not resumes:
#             return
 
#         documents_for_embedding = []
#         metadata_to_store = []
 
#         for resume in resumes:
#             content_parts = []
#             if resume.parsed_data:
#                 if resume.parsed_data.summary: content_parts.append(resume.parsed_data.summary)
#                 if resume.parsed_data.skills: content_parts.append(' '.join([s.name for s in resume.parsed_data.skills]))
#                 if resume.parsed_data.previous_experience:
#                     for exp in resume.parsed_data.previous_experience:
#                         content_parts.append(f"{exp.title} at {exp.company}")
#                         if exp.description: content_parts.append(exp.description)
           
#             content_to_embed = ". ".join(filter(None, content_parts))
#             if not content_to_embed:
#                 logger.warning(f"No content to embed for resume {resume.filename}. Skipping.")
#                 continue
 
#             doc_metadata = resume.model_dump(exclude={'original_text', 'embedding'})
#             if doc_metadata.get('parsed_data'):
#                 doc_metadata['parsed_data'] = resume.parsed_data.model_dump()
           
#             # Ensure these crucial fields are present for linking
#             doc_metadata['resume_id'] = resume.id
#             doc_metadata['resume_filename'] = resume.filename
           
#             documents_for_embedding.append(content_to_embed)
#             metadata_to_store.append(doc_metadata)
       
#         if not documents_for_embedding:
#             return
 
#         logger.info(f"Embedding and adding {len(documents_for_embedding)} resumes...")
#         try:
#             embeddings_np = np.array(await self.embedding_model.aembed_documents(documents_for_embedding)).astype('float32')
 
#             if embeddings_np.shape[1] != self.embedding_dimension:
#                 raise ValueError("Embedding dimension mismatch.")
 
#             start_internal_id = self.index.ntotal
#             self.index.add(embeddings_np)
           
#             # *** CHANGED: Populate the docstore and the new ID map ***
#             for i, metadata in enumerate(metadata_to_store):
#                 internal_faiss_id = start_internal_id + i
#                 resume_id = metadata['resume_id']
               
#                 self.index_to_resume_id[internal_faiss_id] = resume_id
#                 self.docstore[resume_id] = metadata
           
#             logger.info(f"Successfully added {len(documents_for_embedding)} documents.")
 
#         except Exception as e:
#             logger.error(f"Error adding resumes to FAISS: {e}", exc_info=True)
 
#     def save(self):
#         """Saves the FAISS index, ID map, and document store to disk."""
#         if self.index:
#             faiss.write_index(self.index, self.index_path)
#             logger.info("FAISS index saved.")
 
#         # Save the ID map
#         with open(self.id_map_path, 'w', encoding='utf-8') as f:
#             json.dump(self.index_to_resume_id, f)
#         logger.info("ID map saved.")
 
#         # *** NEW: Save the document store ***
#         with open(self.docstore_path, 'w', encoding='utf-8') as f:
#             json.dump(self.docstore, f, indent=4) # indent for readability
#         logger.info("Document store saved.")


#     from typing import Union
 
#     async def search(self, query: Union[str, np.ndarray], k: int = 5) -> List[Tuple[Document, float]]:
#         """
#         Performs a similarity search using either a raw text query or a pre-computed embedding vector.
#         :param query: The search query, either as a string or a NumPy array embedding.
#         :param k: The number of results to return.
#         :return: A list of (Document, distance_score) tuples.
#         """
#         if not self.index or self.index.ntotal == 0:
#             logger.warning("FAISS index is not initialized or is empty.")
#             return []
 
#         query_embedding: np.ndarray
 
#         # *** NEW: Smartly handle either text or a vector ***
#         if isinstance(query, str):
#             # If the query is text, embed it
#             logger.info("Query is a string. Generating embedding...")
#             query_embedding = np.array(await self.embedding_model.aembed_query(query)).astype('float32')
#         elif isinstance(query, np.ndarray):
#             # If the query is already a vector, use it directly
#             logger.info("Query is already an embedding vector. Proceeding with search.")
#             query_embedding = query.astype('float32')
#         else:
#             raise TypeError(f"Unsupported query type: {type(query)}. Must be str or np.ndarray.")
 
#         # Ensure the embedding is the correct shape for FAISS (batch_size, dimension)
#         if query_embedding.ndim == 1:
#             query_embedding = query_embedding.reshape(1, -1)
       
#         # Validate embedding dimension
#         if query_embedding.shape[1] != self.embedding_dimension:
#             raise ValueError(f"Query embedding dimension ({query_embedding.shape[1]}) mismatch with index dimension ({self.embedding_dimension}).")
 
#         try:
#             distances, indices = self.index.search(query_embedding, k)
           
#             results = []
#             for i in range(len(indices[0])):
#                 faiss_internal_id = indices[0][i]
#                 if faiss_internal_id == -1: continue
 
#                 distance = distances[0][i]
               
#                 resume_id = self.index_to_resume_id.get(faiss_internal_id)
#                 if resume_id:
#                     metadata = self.docstore.get(resume_id)
#                     if metadata:
#                         retrieved_doc = Document(page_content="", metadata=metadata)
#                         results.append((retrieved_doc, float(distance)))
#                     else:
#                         logger.warning(f"Resume ID {resume_id} found in map, but not in docstore.")
#                 else:
#                     logger.warning(f"FAISS ID {faiss_internal_id} not found in ID map.")
 
#             logger.info(f"Search found {len(results)} results.")
#             return results
#         except Exception as e:
#             logger.error(f"Error during FAISS search: {e}", exc_info=True)
#             return []






# db_manager.py

import os
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import logging
import json
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import your custom models
from src.llm_handlers import get_embedding_dimension
from src.models import ResumeRecord # Ensure ResumeRecord is imported

class FAISSVectorDB:
    """A manager for the FAISS vector database with a separate document store."""
    def __init__(self, embedding_dimension: Optional[int] = None, index_path: str = "faiss_index/index.faiss", id_map_path: str = "faiss_index/id_map.json", docstore_path: str = "faiss_index/docstore.json", embedding_model_name: str = "text-embedding-3-small"):
        """
        Initializes the FAISS index and document store.
        :param embedding_dimension: The dimension of your embedding vectors.
        :param index_path: Path to save/load the FAISS index binary.
        :param id_map_path: Path to save/load the mapping from FAISS internal IDs to resume_ids.
        :param docstore_path: Path to save/load the metadata for each document.
        :param embedding_model_name: The name of the OpenAI embedding model.
        """
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.docstore_path = docstore_path

        # Determine embedding dimension
        self.embedding_dimension = embedding_dimension or get_embedding_dimension()
        if self.embedding_dimension is None or self.embedding_dimension <= 0:
            raise ValueError("Embedding dimension could not be determined or is invalid.")

        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name)
        self.index: Optional[faiss.Index] = None
       
        # The document store for metadata
        # Key: resume_id (str), Value: metadata dictionary
        self.docstore: Dict[str, Dict[str, Any]] = {}
       
        # This now maps FAISS internal ID (int) back to resume_id (str)
        self.index_to_resume_id: Dict[int, str] = {}
       
        self._load_or_create_index()

    def _load_or_create_index(self):
        """Loads an existing FAISS index, ID map, and docstore, or creates new ones."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        if os.path.exists(self.index_path) and os.path.exists(self.id_map_path) and os.path.exists(self.docstore_path):
            try:
                logger.info(f"Loading FAISS index from {self.index_path}")
                self.index = faiss.read_index(self.index_path)
               
                logger.info(f"Loading ID map from {self.id_map_path}")
                with open(self.id_map_path, 'r', encoding='utf-8') as f:
                    # JSON keys are strings, so convert them back to integers
                    self.index_to_resume_id = {int(k): v for k, v in json.load(f).items()}

                # Load the document store
                logger.info(f"Loading document store from {self.docstore_path}")
                with open(self.docstore_path, 'r', encoding='utf-8') as f:
                    self.docstore = json.load(f)

                if self.index.d != self.embedding_dimension:
                    raise ValueError(f"Dimension mismatch: Index dimension ({self.index.d}) doesn't match expected dimension ({self.embedding_dimension}).")
               
                logger.info(f"FAISS system loaded successfully. Items: {self.index.ntotal}")
            except Exception as e:
                logger.error(f"Error loading FAISS system: {e}. Creating new.", exc_info=True)
                self._create_new_index()
        else:
            logger.info("FAISS index, ID map, or docstore not found. Creating new system.")
            self._create_new_index()

    def _create_new_index(self):
        """Creates a new FAISS index, ID map, and docstore."""
        # Using IndexFlatIP for cosine similarity with normalized vectors
        self.index = faiss.IndexFlatIP(self.embedding_dimension) 
        self.docstore = {} # Reset docstore
        self.index_to_resume_id = {} # Reset ID map
        logger.info(f"New FAISS IndexFlatIP created with dimension {self.embedding_dimension}.") # Updated log for IndexFlatIP

    async def add_resumes(self, resumes: List[ResumeRecord]):
        """Creates embeddings, adds them to FAISS, and stores their metadata."""
        if not resumes:
            return

        documents_for_embedding = []
        metadata_to_store = []

        for resume in resumes:
            content_parts = []
            if resume.parsed_data:
                if resume.parsed_data.summary: content_parts.append(resume.parsed_data.summary)
                if resume.parsed_data.skills: content_parts.append(' '.join([s.name for s in resume.parsed_data.skills]))
                if resume.parsed_data.previous_experience:
                    for exp in resume.parsed_data.previous_experience:
                        content_parts.append(f"{exp.title} at {exp.company}")
                        if exp.description: content_parts.append(exp.description)
           
            content_to_embed = ". ".join(filter(None, content_parts))
            if not content_to_embed:
                logger.warning(f"No content to embed for resume {resume.filename}. Skipping.")
                continue

            # Ensure 'filename' is included directly in the metadata for easier retrieval later
            doc_metadata = resume.model_dump(exclude={'original_text', 'embedding'})
            if doc_metadata.get('parsed_data'):
                doc_metadata['parsed_data'] = resume.parsed_data.model_dump()
           
            # Ensure these crucial fields are present for linking
            doc_metadata['resume_id'] = resume.id
            doc_metadata['resume_filename'] = resume.filename # Explicitly add filename

            documents_for_embedding.append(content_to_embed)
            metadata_to_store.append(doc_metadata)
       
        if not documents_for_embedding:
            return

        logger.info(f"Embedding and adding {len(documents_for_embedding)} resumes...")
        try:
            embeddings_np = np.array(await self.embedding_model.aembed_documents(documents_for_embedding)).astype('float32')
            
            # Normalize embeddings if using IndexFlatIP for cosine similarity
            faiss.normalize_L2(embeddings_np)

            if embeddings_np.shape[1] != self.embedding_dimension:
                raise ValueError("Embedding dimension mismatch.")

            start_internal_id = self.index.ntotal
            self.index.add(embeddings_np)
           
            # Populate the docstore and the new ID map
            for i, metadata in enumerate(metadata_to_store):
                internal_faiss_id = start_internal_id + i
                resume_id = metadata['resume_id']
               
                self.index_to_resume_id[internal_faiss_id] = resume_id
                self.docstore[resume_id] = metadata
           
            logger.info(f"Successfully added {len(documents_for_embedding)} documents.")

        except Exception as e:
            logger.error(f"Error adding resumes to FAISS: {e}", exc_info=True)

    def delete_resumes(self, resume_ids: List[str]) -> List[Dict[str, str]]:
        """
        Deletes resumes from the FAISS index and internal document store.
        Uses complete index rebuild for consistency.
        Returns a list of dictionaries, each containing the 'id' and 'filename' of the deleted resume.
        """
        if not self.index or self.index.ntotal == 0:
            logger.warning("FAISS index is empty or not initialized. No resumes to delete.")
            return []

        if not resume_ids:
            return []

        deleted_resumes_info = []
        resume_ids_set = set(resume_ids)

        # Step 1: Collect info about resumes being deleted
        for resume_id in resume_ids:
            if resume_id in self.docstore:
                metadata = self.docstore[resume_id]
                deleted_resumes_info.append({
                    'id': resume_id,
                    'filename': metadata.get('resume_filename', 'UNKNOWN_FILENAME')
                })
                logger.info(f"Prepared '{resume_id}' for deletion.")

        # Step 2: Remove from docstore first
        for resume_id in resume_ids:
            self.docstore.pop(resume_id, None)

        # Step 3: Rebuild the entire FAISS index without deleted resumes
        if self.docstore:
            logger.info("Rebuilding FAISS index after deletion...")
            self._rebuild_faiss_index_from_docstore_sync()
        else:
            logger.info("All resumes deleted. Creating empty index.")
            self._create_new_index()

        logger.info(f"Successfully deleted {len(deleted_resumes_info)} resume(s) and rebuilt index.")
        return deleted_resumes_info

    def _rebuild_faiss_index_from_docstore_sync(self):
        """
        Synchronous version that rebuilds the FAISS index from scratch using current docstore data.
        This ensures complete consistency after deletions.
        """
        # Create new empty index
        new_index = faiss.IndexFlatIP(self.embedding_dimension)
        new_index_to_resume_id = {}
        
        if not self.docstore:
            self.index = new_index
            self.index_to_resume_id = new_index_to_resume_id
            return

        # Collect all embeddings and metadata
        embeddings_to_add = []
        resume_ids_order = []
        
        for resume_id, metadata in self.docstore.items():
            # Re-create embedding from stored content
            content_parts = []
            parsed_data = metadata.get('parsed_data', {})
            
            if parsed_data.get('summary'): 
                content_parts.append(parsed_data['summary'])
            if parsed_data.get('skills'): 
                if isinstance(parsed_data['skills'], list):
                    content_parts.append(' '.join([s.get('name', '') if isinstance(s, dict) else str(s) for s in parsed_data['skills']]))
            if parsed_data.get('previous_experience'):
                for exp in parsed_data['previous_experience']:
                    if isinstance(exp, dict):
                        content_parts.append(f"{exp.get('title', '')} at {exp.get('company', '')}")
                        if exp.get('description'): 
                            content_parts.append(exp['description'])
            
            content_to_embed = ". ".join(filter(None, content_parts))
            if content_to_embed:
                embeddings_to_add.append(content_to_embed)
                resume_ids_order.append(resume_id)

        if embeddings_to_add:
            try:
                # Generate embeddings synchronously using the sync method
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    embeddings_np = np.array(loop.run_until_complete(
                        self.embedding_model.aembed_documents(embeddings_to_add)
                    )).astype('float32')
                finally:
                    loop.close()
                
                # Normalize for cosine similarity
                faiss.normalize_L2(embeddings_np)
                
                # Add to new index
                new_index.add(embeddings_np)
                
                # Rebuild ID mapping
                for i, resume_id in enumerate(resume_ids_order):
                    new_index_to_resume_id[i] = resume_id
                
                logger.info(f"Rebuilt FAISS index with {len(embeddings_to_add)} resumes.")
                
            except Exception as e:
                logger.error(f"Error rebuilding FAISS index: {e}", exc_info=True)
                # Fall back to empty index
                new_index = faiss.IndexFlatIP(self.embedding_dimension)
                new_index_to_resume_id = {}
        
        # Replace the old index
        self.index = new_index
        self.index_to_resume_id = new_index_to_resume_id

    async def delete_resumes_async(self, resume_ids: List[str]) -> List[Dict[str, str]]:
        """
        Async version of delete_resumes that properly handles embedding generation.
        Deletes resumes from the FAISS index and internal document store.
        Uses complete index rebuild for consistency.
        Returns a list of dictionaries, each containing the 'id' and 'filename' of the deleted resume.
        """
        if not self.index or self.index.ntotal == 0:
            logger.warning("FAISS index is empty or not initialized. No resumes to delete.")
            return []

        if not resume_ids:
            return []

        deleted_resumes_info = []
        resume_ids_set = set(resume_ids)

        # Step 1: Collect info about resumes being deleted
        for resume_id in resume_ids:
            if resume_id in self.docstore:
                metadata = self.docstore[resume_id]
                deleted_resumes_info.append({
                    'id': resume_id,
                    'filename': metadata.get('resume_filename', 'UNKNOWN_FILENAME')
                })
                logger.info(f"Prepared '{resume_id}' for deletion.")

        # Step 2: Remove from docstore first
        for resume_id in resume_ids:
            self.docstore.pop(resume_id, None)

        # Step 3: Rebuild the entire FAISS index without deleted resumes
        if self.docstore:
            logger.info("Rebuilding FAISS index after deletion...")
            await self._rebuild_faiss_index_from_docstore_async()
        else:
            logger.info("All resumes deleted. Creating empty index.")
            self._create_new_index()

        logger.info(f"Successfully deleted {len(deleted_resumes_info)} resume(s) and rebuilt index.")
        return deleted_resumes_info

    async def _rebuild_faiss_index_from_docstore_async(self):
        """
        Async version that rebuilds the FAISS index from scratch using current docstore data.
        This ensures complete consistency after deletions.
        """
        # Create new empty index
        new_index = faiss.IndexFlatIP(self.embedding_dimension)
        new_index_to_resume_id = {}
        
        if not self.docstore:
            self.index = new_index
            self.index_to_resume_id = new_index_to_resume_id
            return

        # Collect all embeddings and metadata
        embeddings_to_add = []
        resume_ids_order = []
        
        for resume_id, metadata in self.docstore.items():
            # Re-create embedding from stored content
            content_parts = []
            parsed_data = metadata.get('parsed_data', {})
            
            if parsed_data.get('summary'): 
                content_parts.append(parsed_data['summary'])
            if parsed_data.get('skills'): 
                if isinstance(parsed_data['skills'], list):
                    content_parts.append(' '.join([s.get('name', '') if isinstance(s, dict) else str(s) for s in parsed_data['skills']]))
            if parsed_data.get('previous_experience'):
                for exp in parsed_data['previous_experience']:
                    if isinstance(exp, dict):
                        content_parts.append(f"{exp.get('title', '')} at {exp.get('company', '')}")
                        if exp.get('description'): 
                            content_parts.append(exp['description'])
            
            content_to_embed = ". ".join(filter(None, content_parts))
            if content_to_embed:
                embeddings_to_add.append(content_to_embed)
                resume_ids_order.append(resume_id)

        if embeddings_to_add:
            try:
                # Generate embeddings asynchronously
                embeddings_np = np.array(await self.embedding_model.aembed_documents(embeddings_to_add)).astype('float32')
                
                # Normalize for cosine similarity
                faiss.normalize_L2(embeddings_np)
                
                # Add to new index
                new_index.add(embeddings_np)
                
                # Rebuild ID mapping
                for i, resume_id in enumerate(resume_ids_order):
                    new_index_to_resume_id[i] = resume_id
                
                logger.info(f"Rebuilt FAISS index with {len(embeddings_to_add)} resumes.")
                
            except Exception as e:
                logger.error(f"Error rebuilding FAISS index: {e}", exc_info=True)
                # Fall back to empty index
                new_index = faiss.IndexFlatIP(self.embedding_dimension)
                new_index_to_resume_id = {}
        
        # Replace the old index
        self.index = new_index
        self.index_to_resume_id = new_index_to_resume_id

    def validate_index_consistency(self) -> Dict[str, Any]:
        """
        Validates the consistency between FAISS index, ID mapping, and docstore.
        Returns a report with any inconsistencies found.
        """
        report = {
            "consistent": True,
            "issues": [],
            "stats": {
                "faiss_vectors": self.index.ntotal if self.index else 0,
                "id_mappings": len(self.index_to_resume_id),
                "docstore_entries": len(self.docstore)
            }
        }
        
        if not self.index:
            report["consistent"] = False
            report["issues"].append("FAISS index is not initialized")
            return report
        
        # Check if counts match
        if report["stats"]["faiss_vectors"] != report["stats"]["id_mappings"]:
            report["consistent"] = False
            report["issues"].append(f"FAISS vector count ({report['stats']['faiss_vectors']}) != ID mapping count ({report['stats']['id_mappings']})")
        
        # Check for orphaned ID mappings
        for faiss_id, resume_id in self.index_to_resume_id.items():
            if resume_id not in self.docstore:
                report["consistent"] = False
                report["issues"].append(f"ID mapping {faiss_id}->{resume_id} has no corresponding docstore entry")
        
        # Check for orphaned docstore entries
        mapped_resume_ids = set(self.index_to_resume_id.values())
        for resume_id in self.docstore:
            if resume_id not in mapped_resume_ids:
                report["consistent"] = False
                report["issues"].append(f"Docstore entry {resume_id} has no corresponding ID mapping")
        
        # Check FAISS ID range
        if self.index_to_resume_id:
            max_faiss_id = max(self.index_to_resume_id.keys())
            if max_faiss_id >= self.index.ntotal:
                report["consistent"] = False
                report["issues"].append(f"Maximum FAISS ID ({max_faiss_id}) >= total vectors ({self.index.ntotal})")
        
        return report

    async def repair_index_consistency(self) -> bool:
        """
        Attempts to repair index inconsistencies by rebuilding from docstore.
        Returns True if repair was successful, False otherwise.
        """
        try:
            logger.info("Starting index consistency repair...")
            consistency_report = self.validate_index_consistency()
            
            if consistency_report["consistent"]:
                logger.info("Index is already consistent, no repair needed.")
                return True
            
            logger.warning(f"Index inconsistencies detected: {consistency_report['issues']}")
            logger.info("Rebuilding index from docstore...")
            
            await self._rebuild_faiss_index_from_docstore_async()
            
            # Validate again
            new_report = self.validate_index_consistency()
            if new_report["consistent"]:
                logger.info("Index consistency repair completed successfully.")
                return True
            else:
                logger.error(f"Index repair failed, remaining issues: {new_report['issues']}")
                return False
                
        except Exception as e:
            logger.error(f"Error during index consistency repair: {e}", exc_info=True)
            return False

    def delete_resumes_with_validation(self, resume_ids: List[str]) -> List[Dict[str, str]]:
        """
        Deletes resumes with pre and post validation to ensure consistency.
        Synchronous version that falls back to sync embedding generation.
        """
        # Pre-deletion validation
        pre_report = self.validate_index_consistency()
        if not pre_report["consistent"]:
            logger.warning(f"Index inconsistent before deletion: {pre_report['issues']}")
        
        # Perform deletion
        result = self.delete_resumes(resume_ids)
        
        # Post-deletion validation
        post_report = self.validate_index_consistency()
        if not post_report["consistent"]:
            logger.error(f"Index inconsistent after deletion: {post_report['issues']}")
            logger.info("Attempting to repair consistency...")
            # Note: This would need async context for full repair
            
        return result

    async def delete_resumes_with_validation_async(self, resume_ids: List[str]) -> List[Dict[str, str]]:
        """
        Deletes resumes with pre and post validation to ensure consistency.
        Async version with full validation and repair capabilities.
        """
        # Pre-deletion validation
        pre_report = self.validate_index_consistency()
        if not pre_report["consistent"]:
            logger.warning(f"Index inconsistent before deletion: {pre_report['issues']}")
            logger.info("Attempting to repair consistency before deletion...")
            repair_success = await self.repair_index_consistency()
            if not repair_success:
                logger.error("Failed to repair index before deletion. Proceeding with caution.")
        
        # Perform deletion
        result = await self.delete_resumes_async(resume_ids)
        
        # Post-deletion validation
        post_report = self.validate_index_consistency()
        if not post_report["consistent"]:
            logger.error(f"Index inconsistent after deletion: {post_report['issues']}")
            logger.info("Attempting to repair consistency after deletion...")
            repair_success = await self.repair_index_consistency()
            if not repair_success:
                logger.error("Failed to repair index after deletion. Manual intervention may be required.")
        
        return result

    async def force_rebuild_index(self) -> bool:
        """
        Forces a complete rebuild of the FAISS index from the docstore.
        Use this to fix any existing inconsistencies.
        """
        try:
            logger.info("Starting forced index rebuild...")
            await self._rebuild_faiss_index_from_docstore_async()
            
            # Validate the rebuilt index
            validation_report = self.validate_index_consistency()
            if validation_report["consistent"]:
                logger.info("Forced index rebuild completed successfully.")
                return True
            else:
                logger.error(f"Index still inconsistent after forced rebuild: {validation_report['issues']}")
                return False
                
        except Exception as e:
            logger.error(f"Error during forced index rebuild: {e}", exc_info=True)
            return False

    def force_rebuild_index_sync(self) -> bool:
        """
        Forces a complete rebuild of the FAISS index from the docstore (sync version).
        Use this to fix any existing inconsistencies.
        """
        try:
            logger.info("Starting forced index rebuild (sync)...")
            self._rebuild_faiss_index_from_docstore_sync()
            
            # Validate the rebuilt index
            validation_report = self.validate_index_consistency()
            if validation_report["consistent"]:
                logger.info("Forced index rebuild completed successfully.")
                return True
            else:
                logger.error(f"Index still inconsistent after forced rebuild: {validation_report['issues']}")
                return False
                
        except Exception as e:
            logger.error(f"Error during forced index rebuild: {e}", exc_info=True)
            return False

    def save(self):
        """Saves the FAISS index, ID map, and document store to disk."""
        if self.index:
            faiss.write_index(self.index, self.index_path)
            logger.info("FAISS index saved.")
 
        # Save the ID map
        with open(self.id_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.index_to_resume_id, f)
        logger.info("ID map saved.")
 
        # Save the document store
        with open(self.docstore_path, 'w', encoding='utf-8') as f:
            json.dump(self.docstore, f, indent=4) # indent for readability
        logger.info("Document store saved.")

    async def search(self, query: Union[str, np.ndarray], k: int = 5, validate_results: bool = True) -> List[Tuple[Document, float]]:
        """
        Performs a similarity search using either a raw text query or a pre-computed embedding vector.
        :param query: The search query, either as a string or a NumPy array embedding.
        :param k: The number of results to return.
        :param validate_results: Whether to validate that returned results exist in docstore.
        :return: A list of (Document, distance_score) tuples.
        """
        if not self.index or self.index.ntotal == 0:
            logger.warning("FAISS index is not initialized or is empty.")
            return []
 
        query_embedding: np.ndarray
 
        if isinstance(query, str):
            logger.info("Query is a string. Generating embedding...")
            try:
                embedding_list = await self.embedding_model.aembed_query(query)
                if not embedding_list: # Check if the embedding list is empty
                    logger.error("Embedding model returned an empty list for the query string. Cannot perform FAISS search.")
                    return []
                query_embedding = np.array(embedding_list).astype('float32')
            except Exception as e:
                logger.error(f"Error generating embedding for query string: {e}", exc_info=True)
                return []
        elif isinstance(query, np.ndarray):
            logger.info("Query is already an embedding vector. Proceeding with search.")
            query_embedding = query.astype('float32')
        else:
            raise TypeError(f"Unsupported query type: {type(query)}. Must be str or np.ndarray.")

        # --- IMPORTANT: Add Robustness Checks for query_embedding ---
        # Check if the embedding is empty or has an unexpected shape (e.g., 0-dimensional or empty 1D)
        if query_embedding.size == 0:
            logger.error("Generated/provided query embedding is empty. Cannot perform FAISS search.")
            return []
        
        # Ensure the embedding is 2D for FAISS operations if it's 1D
        # This handles cases where a single embedding might be (dimension,) instead of (1, dimension)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Final validation of embedding dimension after reshaping
        if query_embedding.shape[1] != self.embedding_dimension:
            logger.error(f"Query embedding dimension ({query_embedding.shape[1]}) mismatch with index dimension ({self.embedding_dimension}). Cannot perform FAISS search.")
            return []

        # Now, normalize the query embedding for cosine similarity (IndexFlatIP)
        try: 
            faiss.normalize_L2(query_embedding)
        except Exception as e:
            logger.error(f"Error during FAISS L2 normalization of query embedding: {e}", exc_info=True)
            return []

        try:
            # Request more results than needed to account for potential filtering
            search_k = min(k * 2, self.index.ntotal) if validate_results else k
            distances, indices = self.index.search(query_embedding, search_k)
           
            results = []
            valid_results_count = 0
            
            for i in range(len(indices[0])):
                if valid_results_count >= k:
                    break
                    
                faiss_internal_id = indices[0][i]
                if faiss_internal_id == -1: continue # Skip if FAISS returns -1 (no match)
 
                distance = distances[0][i]
               
                resume_id = self.index_to_resume_id.get(faiss_internal_id)
                if resume_id:
                    metadata = self.docstore.get(resume_id)
                    if metadata:
                        # Reconstruct a Document object with the metadata
                        retrieved_doc = Document(page_content="", metadata=metadata)
                        results.append((retrieved_doc, float(distance)))
                        valid_results_count += 1
                    else:
                        if validate_results:
                            logger.warning(f"Resume ID {resume_id} found in map, but not in docstore. Skipping result.")
                        else:
                            logger.warning(f"Resume ID {resume_id} found in map, but not in docstore.")
                else:
                    if validate_results:
                        logger.warning(f"FAISS ID {faiss_internal_id} not found in ID map. Skipping result.")
                    else:
                        logger.warning(f"FAISS ID {faiss_internal_id} not found in ID map.")
 
            logger.info(f"Search found {len(results)} valid results out of {len([i for i in indices[0] if i != -1])} total matches.")
            
            # If validation is enabled and we got significantly fewer results than expected, log a warning
            if validate_results and len(results) < k and len(results) < len([i for i in indices[0] if i != -1]) * 0.8:
                logger.warning(f"Search returned fewer valid results ({len(results)}) than expected. This might indicate index inconsistencies.")
            
            return results
        except Exception as e:
            logger.error(f"Error during FAISS search: {e}", exc_info=True)
            return []

    def list_all_resumes(self) -> List[Dict[str, Any]]:
        """
        Lists all indexed resumes with their metadata (excluding raw text and embedding).
        """
        all_resumes_info = []
        for resume_id, metadata in self.docstore.items():
            # Create a copy to avoid modifying the original docstore entry
            resume_data = metadata.copy()
            
            # Exclude potentially large or sensitive fields if desired for listing
            resume_data.pop('raw_text', None) 
            resume_data.pop('embedding', None)
            
            # Ensure the ID is present and correct
            resume_data['id'] = resume_id 
            all_resumes_info.append(resume_data)
        
        logger.info(f"Retrieved {len(all_resumes_info)} indexed resumes.")
        return all_resumes_info