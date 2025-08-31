import asyncio
import logging
from pathlib import Path
from datetime import datetime
import uuid
import json
import numpy as np
import os # Import os for path checks

from typing import Dict, Any, List, Optional, Tuple
from src.llm_handlers import extract_jd_summary_and_skills
from src.models import JDSummaryAndSkills
import json
from src.llm_handlers import extract_jd_summary_and_skills
from src.models import JDSummaryAndSkills

# Import necessary models here where they are used, not globally at the top
# from src.models import ... # Keep imports within functions or agents


from src.extract_jd_text import extract_text_from_docx,extract_text_from_file,process_single_resume_file_for_ingestion
from src.llm_handlers import (
    extract_jd_criteria_with_langchain_safe,
    extract_resume_parser_data_with_langchain_safe,
    evaluate_candidate_with_llm,
    get_sentence_embedding,
    get_embedding_dimension
)
from src.db_manager import FAISSVectorDB

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Initialize FAISS DB Manager ---
# This is initialized once when services.py is imported.
faiss_db_manager: Optional[FAISSVectorDB] = None
try:
    embedding_dim = get_embedding_dimension()
    # Use a directory for FAISS files
    faiss_db_manager = FAISSVectorDB(embedding_dimension=embedding_dim, index_path="faiss_index/index.faiss", id_map_path="faiss_index/id_map.json")
except ValueError as e:
    logger.error(f"Error initializing FAISSVectorDB: {e}. FAISS operations will fail.", exc_info=True)
    faiss_db_manager = None
except Exception as e:
    logger.error(f"Unexpected error during FAISSVectorDB initialization: {e}. FAISS operations will fail.", exc_info=True)
    faiss_db_manager = None


# --- Helper Function for Hashing Content ---
def generate_content_hash(text: str) -> str:
    """Generates an SHA256 hash of the given text content."""
    import hashlib
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# --- Workflow Nodes (Agents) ---
from src.models import ResumeMatcherState, ResumeRecord, JDRecord, EvaluationPlan, Criterion ,JDSummaryAndSkills 
async def file_ingestion_agent(state: ResumeMatcherState) -> ResumeMatcherState:
    """Agent: Reads the Job Description file and all resume files from disk into memory."""
    # Idempotency Check: If JD text is already present AND all_raw_resumes is populated
    # (Meaning this agent has run before in this state's lifecycle)
    if state.get("workflow_status", "INITIALIZED") != "INITIALIZED" and state.get("jd_text") and state.get("all_raw_resumes") is not None and len(state.get("all_raw_resumes", [])) > 0:
         logger.info("--- Skipping: File Ingestion (JD and all raw resumes already in state) ---")
         # Ensure status reflects that files were ingested (might be useful if resuming mid-workflow)
         if state.get("workflow_status") == "INITIALIZED": # Avoid overwriting specific statuses like FAILED
             state["workflow_status"] = "FILES_INGESTED"
         return state


    logger.info("--- Starting: File Ingestion ---")

    # Initialize/clear state variables for a new *full* run if starting from INITIALIZED
    if state.get("workflow_status") == "INITIALIZED":
        state["error_log"] = [] # Reset error log
        state["processing_stats"] = {}
        state["jd_text"] = ""
        state["parsed_jd"] = None
        state["evaluation_plan"] = None
        state["jd_embedding"] = None
        state["all_raw_resumes"] = [] # Will be populated here
        state["newly_profiled_resumes"] = [] # New: Will store only newly profiled records
        state["profiled_resumes"] = [] # Will store ALL records (newly profiled + skeletal for old ones)
        state["retrieved_candidates"] = []
        state["triaged_resumes"] = []
        state["top_5_candidates"] = []
        state["user_action_required"] = False
        state["user_message"] = None

        state["processing_stats"]["start_time"] = datetime.now().isoformat()
        state["processing_stats"]["status"] = "Workflow Started: File Ingestion"


    jd_file_path = state.get("jd_file_path")
    # Read JD file
    if not state.get("jd_text"):
        if not jd_file_path:
            error_msg = "JD file path not provided in state."
            logger.error(error_msg)
            state["error_log"].append(error_msg)
            state["workflow_status"] = "FAILED"
            return state

        jd_path = Path(jd_file_path)
        if not jd_path.is_file():
            error_msg = f"JD file not found at {jd_path}"
            logger.error(error_msg)
            state["error_log"].append(error_msg)
            state["workflow_status"] = "FAILED"
            return state

        try:
            supported_extensions = {'.docx', '.pdf'}
            if jd_path.suffix.lower() in supported_extensions:
                jd_text = extract_text_from_file(str(jd_path))
            else:
                 error_msg = f"Unsupported JD file type: {jd_path.suffix}. Supported types: {', '.join(supported_extensions)}."
                 logger.error(error_msg)
                 state["error_log"].append(error_msg)
                 state["workflow_status"] = "FAILED"
                 return state

            if not jd_text or not jd_text.strip():
                error_msg = f"Could not extract text from JD file or file is empty: {jd_path}"
                logger.error(error_msg)
                state["error_log"].append(error_msg)
                state["workflow_status"] = "FAILED"
                return state
            state["jd_text"] = jd_text
            logger.info(f"Successfully read JD text from {jd_path.name}.")

        except Exception as e:
             logger.error(f"Error reading JD file {jd_path.name}: {e}", exc_info=True)
             state["error_log"].append(f"Error reading JD file {jd_path.name}: {str(e)}")
             state["workflow_status"] = "FAILED"
             return state
    else:
         logger.info("Using existing JD text from state.")


    # Read ALL resume files concurrently
    # This agent now reads ALL files, the filtering happens in the profiling agent.
    if state.get("all_raw_resumes") is None or len(state.get("all_raw_resumes", [])) == 0:
        resume_directory = state.get("resume_directory")
        if not resume_directory:
            error_msg = "Resume directory not provided in state."
            logger.error(error_msg)
            state["error_log"].append(error_msg)
            state["workflow_status"] = "FAILED"
            return state

        resume_dir_path = Path(resume_directory)

        if not resume_dir_path.exists():
            error_msg = f"Resume directory not found at {resume_dir_path}. Proceeding without resumes."
            logger.warning(error_msg)
            state["error_log"].append(error_msg)
            state["workflow_status"] = "JD_INGESTED_NO_RESUMES_DIR"
            state["processing_stats"]["total_resumes_found"] = 0 # Stat for total files found
            state["all_raw_resumes"] = [] # Ensure empty list is set
            return state

        # resume_files = list(resume_dir_path.glob("*.docx", "*.pdf"))
        resume_files = list(resume_dir_path.glob("*.docx")) + list(resume_dir_path.glob("*.pdf"))
        resume_files = [f for f in resume_files if not f.name.startswith('~$')]

        if not resume_files:
            logger.warning(f"No supported resume files found in {resume_dir_path}.")
            state["processing_stats"]["status"] = "Completed JD Ingestion, No Resume Files Found"
            state["processing_stats"]["total_resumes_found"] = 0
            state["workflow_status"] = "JD_INGESTED_NO_RESUMES_FOUND"
            state["all_raw_resumes"] = [] # Ensure empty list is set
            return state

        tasks = [process_single_resume_file_for_ingestion(file_path) for file_path in resume_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_raw_resumes_list = []
        errors_occurred = []
        for res in results:
            if isinstance(res, Exception):
                 logger.error(f"Error during resume file processing task: {res}", exc_info=True)
                 errors_occurred.append(f"Error during resume file processing task: {str(res)}")
            elif isinstance(res, str) and res.startswith("Skipping"):
                 logger.warning(res)
                 errors_occurred.append(res)
            elif res is not None:
                all_raw_resumes_list.append(res)

        state["all_raw_resumes"] = all_raw_resumes_list
        state["error_log"].extend(errors_occurred)
        state["processing_stats"]["total_resumes_found"] = len(resume_files) # Total files attempted
        state["processing_stats"]["total_resumes_ingested"] = len(all_raw_resumes_list) # Total successfully ingested
        logger.info(f"Found {len(resume_files)} potential resumes, successfully ingested {len(all_raw_resumes_list)} raw resumes.")
    else:
         logger.info("Using existing all raw resumes from state.")


    # Final status determination
    if state.get("jd_text") and state.get("all_raw_resumes"):
         state["processing_stats"]["status"] = "Completed File Ingestion"
         state["workflow_status"] = "FILES_INGESTED"
         logger.info("File ingestion complete (JD and resumes).")
    elif state.get("jd_text"): # JD ingested but no resumes found (handled by earlier checks)
         pass # Status already set to JD_INGESTED_NO_RESUMES_DIR or JD_INGESTED_NO_RESUMES_FOUND
    else: # Should be caught by earlier checks
         state["workflow_status"] = "FAILED"
         state["error_log"].append("File ingestion finished but no JD text found.")
         logger.error("File ingestion finished in an unexpected state.")


    return state



async def jd_parsing_agent(state: ResumeMatcherState) -> ResumeMatcherState:
    """Agent: Parses JD text for an EvaluationPlan and creates a specialized embedding."""
    logger.info("--- Starting: JD Parsing ---")
    state["processing_stats"]["status"] = "JD Parsing"
    
    jd_text = state.get("jd_text")
    if not jd_text:
        error_msg = "No JD text available for parsing."
        logger.error(error_msg)
        state["error_log"].append(error_msg)
        state["workflow_status"] = "FAILED"
        return state

    try:
        # Step 1: Get the full evaluation plan (this logic is unchanged and still required)
        evaluation_plan_obj = await extract_jd_criteria_with_langchain_safe(jd_text)

        if not evaluation_plan_obj or not getattr(evaluation_plan_obj, 'criteria', None):
            error_msg = "Failed to create a valid Evaluation Plan from the JD."
            logger.error(error_msg)
            state["error_log"].append(error_msg)
            state["workflow_status"] = "FAILED"
            return state
            
        # Store the plan in the state as a dictionary
        state["evaluation_plan"] = evaluation_plan_obj.model_dump()

        # --- Store the full summary and skills object in the parsed JD cache --
        jd_name = state.get("jd_name")
        summary_and_skills: JDSummaryAndSkills = await extract_jd_summary_and_skills(jd_text)
        state["parsed_jd"] = summary_and_skills.model_dump()
        # Save to cache for future use
        if jd_name:
            parsed_text_path = Path(f"cache/jd_cache/parsed_jds/{jd_name}.json")
            with open(parsed_text_path, "w", encoding="utf-8") as f:
                json.dump(summary_and_skills.model_dump(), f, indent=2)

        # --- START OF THE NEW/MODIFIED LOGIC FOR EMBEDDING ---
        # Step 2: Attempt to get the specialized summary and skills for embedding
        summary_and_skills: Optional[JDSummaryAndSkills] = await extract_jd_summary_and_skills(jd_text)
        
        embedding_text_source = ""
        if summary_and_skills and summary_and_skills.skills:
            # New, preferred method: Combine the generated summary and skills
            skills_text = ", ".join(summary_and_skills.skills)
            embedding_text_source = f"{summary_and_skills.summary} Key Skills: {skills_text}"
            logger.info("Successfully created embedding text from dedicated summary and skills.")
        else:
            # Fallback to the old method if the new one fails
            logger.warning("Could not extract dedicated summary/skills. Falling back to using evaluation plan details for embedding.")
            plan_dict = state["evaluation_plan"]
            summary = plan_dict.get('overall_summary', '').strip()
            criteria_list = plan_dict.get('criteria', [])
            criteria_text = ". ".join([f"{c.get('category', '')}: {c.get('criteria', '')}" for c in criteria_list])
            embedding_text_source = f"{summary}. Key criteria: {criteria_text}"

        if not embedding_text_source.strip():
            error_msg = "Could not generate any text content for JD embedding from any method."
            logger.error(error_msg)
            state["error_log"].append(error_msg)
            state["workflow_status"] = "FAILED"
            return state

        # Step 3: Create the embedding from the chosen text source
        jd_embedding = await get_sentence_embedding(embedding_text_source)
        
        # --- END OF THE NEW/MODIFIED LOGIC FOR EMBEDDING ---

        if jd_embedding is None or not isinstance(jd_embedding, np.ndarray) or jd_embedding.size == 0:
            error_msg = "Failed to generate JD embedding or the embedding is empty."
            logger.error(error_msg)
            state["error_log"].append(error_msg)
            state["workflow_status"] = "FAILED"
            return state

        state["jd_embedding"] = jd_embedding
        logger.info("JD embedding generated successfully.")
        
        state["processing_stats"]["status"] = "Completed JD Parsing"
        state["workflow_status"] = "JD_PARSED"

    except Exception as e:
        logger.error(f"A critical error occurred during JD parsing: {e}", exc_info=True)
        state["error_log"].append(f"Critical error during JD parsing: {str(e)}")
        state["workflow_status"] = "FAILED"

    return state




# In src/services.py

# ... (keep all other imports and functions as they are) ...

async def resume_profiling_agent(state: ResumeMatcherState) -> ResumeMatcherState:
    """
    Agent: Parses raw resume text using LLM *only if* not already indexed in FAISS.
    This agent is now robust and handles the cached JD workflow correctly.
    """
    logger.info("--- Starting: Resume Profiling ---")

    # Idempotency Check: If resumes are already profiled, and we are NOT in the
    # specific "JD_PARSED" state (which requires this agent to run), then we can skip.
    if state.get("profiled_resumes") and state.get("workflow_status") != "JD_PARSED":
        logger.info("--- Skipping: Resume Profiling (Resumes already profiled in this workflow run) ---")
        state["workflow_status"] = "RESUMES_PROFILED"
        return state

    state["processing_stats"]["status"] = "Resume Profiling"
    state["workflow_status"] = "RESUME_PROFILING"

    # --- START OF THE CRITICAL FIX FOR THE INFINITE LOOP ---
    # This block ensures that even if the file_ingestion_agent was skipped (cached JD path),
    # this agent still knows which resumes to process.

    # 1. Check if the state's list of raw resumes is empty
    if not state.get("all_raw_resumes"):
        logger.warning("`all_raw_resumes` is empty. Manually loading from directory for cached JD workflow.")
        resume_dir = state.get("resume_directory", "RR")

        if not os.path.isdir(resume_dir):
            error_msg = f"Resume directory '{resume_dir}' not found."
            logger.error(error_msg)
            state["error_log"].append(error_msg)
            state["workflow_status"] = "FAILED"
            return state

        # 2. Get the list of supported resume files from the directory (.docx and .pdf)
        supported_extensions = ('.docx', '.pdf')
        resumes_in_dir = [f for f in os.listdir(resume_dir) 
                         if f.lower().endswith(supported_extensions) and not f.startswith("~$")]
        if not resumes_in_dir:
            logger.warning(f"No supported resume files (.docx, .pdf) found in {resume_dir}.")
            state["workflow_status"] = "RESUMES_PROFILED_EMPTY"
            state["profiled_resumes"] = []
            state["newly_profiled_resumes"] = []
            return state
            
        # 3. Populate `all_raw_resumes` by reading each file
        raw_resumes_list = []
        for filename in resumes_in_dir:
            file_path = os.path.join(resume_dir, filename)
            try:
                text = extract_text_from_file(str(file_path))
                if text and text.strip():
                    raw_resumes_list.append({
                        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, filename)),
                        "file_path": file_path,
                        "filename": filename,
                        "raw_text": text,
                        "name": Path(filename).stem
                    })
                else:
                    logger.warning(f"Skipping empty or unreadable resume: {filename}")
            except Exception as e:
                logger.error(f"Error reading and processing {filename} for raw list: {e}")
        
        state["all_raw_resumes"] = raw_resumes_list
        logger.info(f"Manually populated `all_raw_resumes` with {len(raw_resumes_list)} resumes.")
    # --- END OF THE CRITICAL FIX ---


    all_raw_resumes = state.get("all_raw_resumes", [])
    if not all_raw_resumes:
        logger.warning("No raw resumes found in state for profiling.")
        state["error_log"].append("No raw resumes to profile.")
        state["workflow_status"] = "RESUMES_PROFILED_EMPTY"
        state["profiled_resumes"] = []
        state["newly_profiled_resumes"] = []
        return state

    # Import ResumeRecord model here to avoid circular dependency issues at the top level
    from src.models import ResumeRecord

    profiled_resumes_list: List[ResumeRecord] = []      # Will hold ALL records (new + old)
    newly_profiled_list: List[ResumeRecord] = []        # Will hold only NEWLY processed records
    tasks_for_llm_profiling = []                        # For concurrent LLM calls

    logger.info(f"Checking {len(all_raw_resumes)} raw resumes against FAISS index...")

    # Identify resumes that are new vs. already indexed in FAISS
    for raw_resume_data in all_raw_resumes:
        resume_id = raw_resume_data["id"]
        # Check if the resume's metadata is already in our document store
        if faiss_db_manager and resume_id in faiss_db_manager.docstore:
            logger.info(f"  - Resume '{raw_resume_data['filename']}' found in FAISS. Skipping LLM profiling.")
            # Create a "skeletal" record for this existing resume. We need its original_text
            # for the final detailed evaluation step later.
            profiled_resumes_list.append(ResumeRecord(
                id=resume_id,
                file_path=raw_resume_data["file_path"],
                filename=raw_resume_data["filename"],
                original_text=raw_resume_data["raw_text"],
                name=raw_resume_data.get("name"),
                parsed_data=None, # Not profiled in this run
                embedding=None    # Embedding is already in FAISS, not stored in the state object
            ))
        else:
            logger.info(f"  - Resume '{raw_resume_data['filename']}' is new. Scheduling for profiling.")
            # Add a task to profile this new resume using the LLM
            tasks_for_llm_profiling.append(process_single_resume_task(raw_resume_data))

    # Process all new resumes concurrently using the LLM
    if tasks_for_llm_profiling:
        logger.info(f"Profiling {len(tasks_for_llm_profiling)} new resumes with the LLM...")
        results = await asyncio.gather(*tasks_for_llm_profiling, return_exceptions=True)

        for res in results:
            if isinstance(res, Exception):
                logger.error(f"An error occurred during a resume profiling task: {res}", exc_info=True)
                state["error_log"].append(f"Error profiling resume: {str(res)}")
            elif res is not None:
                profiled_resumes_list.append(res) # Add the full new record to the main list
                newly_profiled_list.append(res)   # Add it to the list of resumes that need indexing

    # Update state with the results
    state["profiled_resumes"] = profiled_resumes_list
    state["newly_profiled_resumes"] = newly_profiled_list
    state["processing_stats"]["total_resumes_profiled_this_run"] = len(newly_profiled_list)
    state["processing_stats"]["total_resumes_available_for_matching"] = len(profiled_resumes_list)

    if not profiled_resumes_list:
        logger.warning("No resumes were successfully profiled or available.")
        state["workflow_status"] = "RESUMES_PROFILED_EMPTY"
    else:
        logger.info(f"Profiling complete. Total available: {len(profiled_resumes_list)}, Newly profiled: {len(newly_profiled_list)}.")
        state["processing_stats"]["status"] = "Completed Resume Profiling"
        state["workflow_status"] = "RESUMES_PROFILED"

    return state


async def process_single_resume_task(raw_resume_data: Dict[str, Any]) -> Optional[Any]:
    """
    Helper function for the profiling agent. It takes a single raw resume dictionary,
    parses it with the LLM, and returns a full ResumeRecord object.
    """
    # Import models here to avoid circular dependencies
    from src.models import ResumeRecord, ResumeParser
    try:
        filename = raw_resume_data["filename"]
        raw_text = raw_resume_data["raw_text"]

        # Call the LLM to parse the resume text into a structured format
        parsed_data: Optional[ResumeParser] = await extract_resume_parser_data_with_langchain_safe(raw_text)
        if not parsed_data:
            logger.warning(f"Skipping {filename}: Failed to get structured data from LLM.")
            return None

        # Create the full record for the newly profiled resume
        return ResumeRecord(
            id=raw_resume_data["id"],
            file_path=raw_resume_data["file_path"],
            filename=filename,
            original_text=raw_text,
            parsed_data=parsed_data,
            name=raw_resume_data.get("name") # Use the name from the file stem
        )
    except Exception as e:
        logger.error(f"  - LLM profiling failed for {raw_resume_data.get('filename', 'unknown')}: {e}", exc_info=True)
        # Return the exception to be handled by the gather call
        raise e


async def embedding_and_indexing_agent(state: ResumeMatcherState) -> ResumeMatcherState:
    """Agent: Creates embeddings for JD and NEWLY profiled resumes, and indexes them in FAISS."""
    if state.get("workflow_status") in ["INDEXING_COMPLETE", "INDEXING_COMPLETE_NO_NEW_RESUMES", "INDEXING_COMPLETE_NO_RESUMES", "EVAL_PLAN_APPROVED", "COMPLETED"]:
         logger.info("--- Skipping: Embedding and Indexing (already completed) ---")
         return state

    if state.get("workflow_status") == "RESUMES_PROFILED_EMPTY":
         logger.warning("--- Skipping: Resume Embedding and Indexing (no profiled resumes available) ---")
         state["processing_stats"]["status"] = "Skipped Resume Embedding (no profiled resumes)"
         state["workflow_status"] = "INDEXING_COMPLETE_NO_RESUMES"
         return state

    logger.info("--- Starting: Embedding & Indexing ---")
    state["processing_stats"]["status"] = "Embedding & Indexing"

    if faiss_db_manager is None:
        error_msg = "FAISS Vector DB Manager not initialized. Cannot perform embedding or indexing."
        logger.error(error_msg)
        state["error_log"].append(error_msg)
        state["workflow_status"] = "FAILED"
        return state

    # --- START OF THE FIX ---
    # Import the model
    from src.models import EvaluationPlan

    # Get the dictionary from the state
    evaluation_plan_dict = state.get("evaluation_plan")

    # Validate the dictionary back into a Pydantic object. This is the robust way.
    if not evaluation_plan_dict:
        error_msg = "Evaluation plan is missing from state. Cannot proceed with embedding."
        logger.error(error_msg)
        state["error_log"].append(error_msg)
        state["workflow_status"] = "FAILED"
        return state

    try:
        # This gives us a proper object to work with, preventing the AttributeError
        evaluation_plan = EvaluationPlan.model_validate(evaluation_plan_dict)
    except Exception as e:
        error_msg = f"Failed to validate evaluation plan from state: {e}"
        logger.error(error_msg)
        state["error_log"].append(error_msg)
        state["workflow_status"] = "FAILED"
        return state
    # --- END OF THE FIX ---

    # Now, the rest of the code can safely use dot notation
    if not evaluation_plan.criteria:
        error_msg = "Evaluation plan has no criteria. Cannot proceed."
        logger.error(error_msg)
        state["error_log"].append(error_msg)
        state["workflow_status"] = "FAILED"
        return state

    newly_profiled_resumes = state.get("newly_profiled_resumes", [])
    if not newly_profiled_resumes:
        logger.info("No newly profiled resumes found. Skipping resume indexing for this run.")
        state["processing_stats"]["status"] = "Completed, No NEW Resume Indexing"
        state["workflow_status"] = "INDEXING_COMPLETE_NO_NEW_RESUMES"
        
        if state.get("workflow_status") == "RESUMES_PROFILED_EMPTY":
             state["workflow_status"] = "INDEXING_COMPLETE_NO_RESUMES"

        if faiss_db_manager and faiss_db_manager.index:
            logger.info(f"Current FAISS index size: {faiss_db_manager.index.ntotal}")
        return state

    try:
        await faiss_db_manager.add_resumes(newly_profiled_resumes)
        faiss_db_manager.save()

        logger.info(f"Successfully embedded and indexed {len(newly_profiled_resumes)} NEW resumes.")
        state["processing_stats"]["status"] = "Completed Embedding & Indexing"
        state["workflow_status"] = "INDEXING_COMPLETE"

    except Exception as e:
        error_msg = f"Critical error during embedding and indexing of NEW resumes: {e}"
        logger.error(error_msg, exc_info=True)
        state["error_log"].append(error_msg)
        state["workflow_status"] = "FAILED"

    return state


    

async def matching_and_triage_agent(state: ResumeMatcherState) -> ResumeMatcherState:
    """
    Agent: Retrieves top candidates using RAG (FAISS search).
    This agent ONLY performs the retrieval step over the *entire* FAISS index.
    """
    # Idempotency Check: If retrieved_candidates is populated and status is past MATCHING_TRIAGE
    if state.get("workflow_status") in ["MATCHING_TRIAGE", "RETRIEVAL_COMPLETE", "RETRIEVAL_COMPLETE_NO_MATCHES", "TRIAGE_COMPLETE", "TRIAGE_COMPLETE_NO_RESUMES", "TRIAGE_COMPLETE_FAISS_EMPTY", "TRIAGE_COMPLETE_NO_MATCHES", "FINAL_SELECTION", "COMPLETED", "COMPLETED_EMPTY_RESULTS"]:
         logger.info("--- Skipping: RAG - Matching and Triage (Retrieval) (already completed or subsequent stage reached) ---")
         return state

     # Handle cases where Evaluation Plan was approved but there were no resumes available at all
    # Check statuses indicating no resumes were available after profiling/indexing
    if state.get("workflow_status") in ["EVAL_PLAN_APPROVED"] and not state.get("profiled_resumes"):
         logger.warning("--- Skipping: RAG - Matching and Triage (No profiled resumes available after approval) ---")
         state["processing_stats"]["status"] = "Skipped Matching & Triage (no resumes available)"
         state["workflow_status"] = "TRIAGE_COMPLETE_NO_RESUMES" # Indicate triage complete but skipped
         state["retrieved_candidates"] = [] # Ensure empty
         return state # Exit agent early


    logger.info("--- Starting: RAG - Matching and Triage (Retrieval) ---")
    state["processing_stats"]["status"] = "Matching & Triage (Retrieval)"
    state["workflow_status"] = "MATCHING_TRIAGE" # Status indicating retrieval is running


    # Import necessary types
    from langchain_core.documents import Document

    jd_embedding = state.get("jd_embedding")
    # We don't need profiled_resumes here, just the JD embedding and the FAISS index.

    if jd_embedding is None or not isinstance(jd_embedding, np.ndarray) or jd_embedding.size == 0:
        error_msg = "JD embedding not found, is not a NumPy array, or is empty. Cannot perform candidate search."
        logger.error(error_msg)
        state["error_log"].append(error_msg)
        state["workflow_status"] = "FAILED"
        state["retrieved_candidates"] = []
        return state

    if faiss_db_manager is None:
        error_msg = "FAISS Vector DB Manager not initialized. Cannot perform search."
        logger.error(error_msg)
        state["error_log"].append(error_msg)
        state["workflow_status"] = "FAILED"
        state["retrieved_candidates"] = []
        return state

    # Check if FAISS index has any items to search
    if faiss_db_manager.index is None or faiss_db_manager.index.ntotal == 0:
        error_msg = "FAISS index is empty. No resumes to match against."
        logger.warning(error_msg)
        state["error_log"].append(error_msg)
        state["processing_stats"]["status"] = "Completed Matching & Triage (FAISS empty)"
        state["workflow_status"] = "TRIAGE_COMPLETE_FAISS_EMPTY"
        state["retrieved_candidates"] = [] # Ensure empty list
        return state # Exit agent early

   
    # top_k_retrieved_count = int(state.get("top_n",5))*4  #Retrieve top 20 references from the entire index
    top_k_retrieved_count = int(state.get("top_n"))*4
    logger.info(f"Retrieving top {top_k_retrieved_count} candidate references based on JD embedding similarity from FAISS...")

    try:
        # FAISS search returns list of (Document, score) where Document metadata contains resume_id


        retrieved_candidates_docs: List[Tuple[Document, float]] = await faiss_db_manager.search(jd_embedding, k=top_k_retrieved_count)

#         # Retrieve more candidates temporarily to debug
#         debug_k = 50  # or use config.debug_k if you define it

# # Fetch all candidates
#         all_candidates: List[Tuple[Document, float]] = await faiss_db_manager.search(jd_embedding, k=debug_k)

# # Log scores for inspection
#         for i, (doc, score) in enumerate(all_candidates):
#             resume_id = doc.metadata.get("resume_id", f"resume_{i}")
#             logger.info(f"[{i+1}] Resume ID: {resume_id} | Similarity Score: {score:.4f}")

# # Optional: Filter based on similarity threshold
#         SIMILARITY_THRESHOLD = 0.75  # or get from config
#         filtered_candidates = [(doc, score) for doc, score in all_candidates if score >= SIMILARITY_THRESHOLD]
        # sorted_candidates = sorted(retrieved_candidates_docs, key=lambda x: x[1], reverse=True)

# # Finally, take top-K from filtered
        # retrieved_candidates_docs = retrieved_candidates_docs[:top_k_retrieved_count]  # Reverse order to get highest scores first

        logger.info(f"After threshold filtering, {len(retrieved_candidates_docs)} candidates remain.")
        retrieved_candidates_docs= sorted(retrieved_candidates_docs, key=lambda x: x[1], reverse=True)
        for i, (doc, score) in enumerate(retrieved_candidates_docs):
            resume_id = doc.metadata.get("resume_id", f"resume_{i}")
            # logger.info(f"[{i+1}] Resume ID: {resume_id} | Similarity Score: {score:.4f}")
            # logger.info(retrieved_candidates_docs[i])  # Log the full tuple for debugging




        state["retrieved_candidates"] = retrieved_candidates_docs # Store list of (Document, score)
        logger.info(f"Retrieved {len(retrieved_candidates_docs)} candidate references from FAISS.")

        state["processing_stats"]["status"] = "Completed Matching & Triage (Retrieval)"
        state["workflow_status"] = "RETRIEVAL_COMPLETE"

        if not retrieved_candidates_docs:
             logger.warning("FAISS search returned no candidate references.")
             state["workflow_status"] = "RETRIEVAL_COMPLETE_NO_MATCHES"

    except Exception as e:
        error_msg = f"Error during FAISS search: {e}"
        logger.error(error_msg, exc_info=True)
        state["error_log"].append(error_msg)
        state["workflow_status"] = "FAILED"
        state["retrieved_candidates"] = [] # Ensure empty list on failure
    return state


import config
async def final_selection_agent(state: ResumeMatcherState) -> ResumeMatcherState:
    """
    Agent: Takes retrieved candidates, evaluates them using LLM against the JD criteria,
    selects the top N candidates with scores above a defined threshold, and displays them.
    This agent performs the Augmentation & Generation step.
    """
    # --- IMPORTANT: Debug logging at the very start of the agent ---
    logger.debug(f"[{__name__}] - Entering final_selection_agent. Initial state keys: {list(state.keys()) if isinstance(state, dict) else 'State is NOT a dict!'}")

    # Critical early check: Ensure state is a dictionary from the start
    if not isinstance(state, dict):
        logger.critical(f"[{__name__}] - Received state is NOT a dictionary: {type(state)}. Cannot proceed.")
        return {"workflow_status": "FAILED", "error_log": ["Initial state provided to final_selection_agent was not a dictionary."]}


    if state.get("workflow_status") in ["FINAL_SELECTION", "COMPLETED", "COMPLETED_EMPTY_RESULTS"]:
        logger.info("--- Skipping: Final Selection (Evaluation & Display) (already completed) ---")
        return state
    if state.get("workflow_status") in ["RETRIEVAL_COMPLETE_NO_MATCHES", "TRIAGE_COMPLETE_NO_RESUMES", "TRIAGE_COMPLETE_FAISS_EMPTY", "TRIAGE_COMPLETE_NO_MATCHES"]:
        logger.warning("--- Skipping: Final Selection (No candidates retrieved or profiled) ---")
        print("\n" + "="*50 + "\n🏆 TOP CANDIDATES 🏆\n" + "="*50)
        print("No candidates were retrieved or successfully processed for final evaluation.")
        state.update({
            "processing_stats": {"status": "Completed Final Selection (No results)"},
            "workflow_status": "COMPLETED_EMPTY_RESULTS",
            "triaged_resumes": [],
            "top_5_candidates": []
        })
        return state

    logger.info("--- Starting: Final Selection (Evaluation & Display) ---")
    state["processing_stats"]["status"] = "Final Selection (Evaluation)"
    state["workflow_status"] = "FINAL_SELECTION"
    
    # Models are imported here in your original code; keeping it for consistency
    from src.models import EvaluationPlan, ResumeRecord, JobMatchResult 
    
    retrieved_candidates_docs: List[Tuple[Document, float]] = state.get("retrieved_candidates", [])
    evaluation_plan: EvaluationPlan = state.get("evaluation_plan")
    profiled_resumes: List[ResumeRecord] = state.get("profiled_resumes", [])

    # Add comprehensive checks for essential data
    if not evaluation_plan:
        error_msg = "Evaluation plan is missing. Cannot perform final selection."
        logger.error(error_msg)
        state["error_log"].append(error_msg)
        state["workflow_status"] = "FAILED"
        state["triaged_resumes"] = []
        state["top_5_candidates"] = []
        return state

    if not retrieved_candidates_docs:
        logger.warning("No candidates were retrieved for evaluation.")
        print("\n" + "="*50 + "\n🏆 TOP CANDIDATES 🏆\n" + "="*50)
        print("No candidates to evaluate after retrieval.")
        state.update({
            "processing_stats": {"status": "Completed Final Selection (No retrieved candidates)"},
            "workflow_status": "COMPLETED_EMPTY_RESULTS",
            "triaged_resumes": [],
            "top_5_candidates": []
        })
        return state

    profiled_resumes_map = {res.id: res for res in profiled_resumes}
    valid_retrieved_resumes_to_evaluate: List[ResumeRecord] = []

    for doc, _distance in retrieved_candidates_docs:
        resume_id = doc.metadata.get('resume_id')
        if resume_id and (resume_record := profiled_resumes_map.get(resume_id)):
            valid_retrieved_resumes_to_evaluate.append(resume_record)
        else:
            logger.warning(f"Could not find full record for retrieved resume_id: {resume_id}. Skipping for evaluation.")

    if not valid_retrieved_resumes_to_evaluate:
        logger.warning("No valid resume records found for evaluation after mapping retrieved IDs.")
        print("\n" + "="*50 + "\n🏆 TOP CANDIDATES 🏆\n" + "="*50)
        print("No valid candidates found for final evaluation after initial retrieval.")
        state.update({
            "processing_stats": {"status": "Completed Final Selection (No valid resumes to evaluate)"},
            "workflow_status": "COMPLETED_EMPTY_RESULTS",
            "triaged_resumes": [],
            "top_5_candidates": []
        })
        return state

    # ### STEP 1: PREPARE THE ASYNCHRONOUS TASKS ###
    evaluation_tasks = []
    for resume_record in valid_retrieved_resumes_to_evaluate:
        evaluation_tasks.append(evaluate_candidate_with_llm(resume_record, evaluation_plan))

     
    # async def _add_evaluation_tasks_recursively(resumes_list: list,evaluation_plan: any, tasks_collector: list,limit: int,current_index: int = 0):
 
#     # Base Cases for Recursion:
#     # 1. If we've reached or exceeded the limit
#     # 2. If there are no more resumes to process in the list
#         if current_index >= limit or current_index >= len(resumes_list):
#             return
 
#     # Recursive Step:
#     # Get the current resume record
#         resume_record = resumes_list[current_index]
 
#     # Add the evaluation task for the current resume
#         tasks_collector.append(evaluate_candidate_with_llm(resume_record, evaluation_plan))
 
#     # Make the recursive call for the next resume
#         await _add_evaluation_tasks_recursively(resumes_list,evaluation_plan,tasks_collector,limit,current_index + 1)
       
#     evaluation_tasks = []
#     num_to_evaluate = int(state.get("retrive_n", 5))
#     await _add_evaluation_tasks_recursively(
#     valid_retrieved_resumes_to_evaluate,
#     evaluation_plan,
#     evaluation_tasks,
#     num_to_evaluate
# )
 
   
    # async def add_evaluation_tasks(resumes_list: list,evaluation_plan: any, tasks_collector: list,limit: int,current_index: int = 0):
    #     if current_index >= limit or current_index >= len(resumes_list):
    #         return
    #     evaluation_tasks.append(evaluate_candidate_with_llm(resume_record[current_index], evaluation_plan))
    #     current_index+=1
   
       
    # add_evaluation_tasks(valid_retrieved_resumes_to_evaluate,evaluation_plan)
 

    # ### STEP 2: EXECUTE ALL TASKS CONCURRENTLY ###
    logger.info(f"Performing detailed LLM evaluation on {len(evaluation_tasks)} retrieved candidates...")
    evaluated_results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)

    # ### STEP 3: PROCESS THE RESULTS ###
    valid_results: List[JobMatchResult] = []
    errors_occurred = []
    for res in evaluated_results:
        if isinstance(res, Exception):
            logger.error(f"An error occurred during LLM evaluation for one candidate: {res}", exc_info=True)
            errors_occurred.append(f"Error during LLM evaluation for candidate: {str(res)}")
        elif res is not None:
            if not isinstance(res, JobMatchResult):
                logger.error(f"Expected JobMatchResult but got type: {type(res)} for result: {res}. Skipping.")
                errors_occurred.append(f"Invalid LLM evaluation result type for candidate: {type(res)}")
                continue
            if not hasattr(res, 'overall_score') or not isinstance(res.overall_score, (int, float)):
                logger.error(f"JobMatchResult for '{getattr(res, 'candidate_name', 'N/A')}' missing or invalid 'overall_score' attribute ({getattr(res, 'overall_score', 'MISSING')}). Skipping.")
                errors_occurred.append(f"JobMatchResult missing or invalid score for candidate: {getattr(res, 'candidate_name', 'N/A')}")
                continue
            
            valid_results.append(res)
            # logger.error(f"JobMatchResult for '{getattr(res, 'candidate_name', 'N/A')}' 'verall_score' attribute ({getattr(res, 'overall_score', 'MISSING')}). Skipping.") #PAVAN DEBUG
            

    state["error_log"].extend(errors_occurred)
    logger.info(f"Successfully evaluated {len(valid_results)} candidates before score filtering.")



    
    # --- START OF CRITICAL FIX/DEBUG SECTION FOR NORMALIZATION ---
    # Define the score threshold (default to 60%) - this is on a 0-1 scale
    score_display_threshold = state.get("score_display_threshold", config.score_display_threshold)
    logger.info(f"Configured score_display_threshold (0-1 scale): {score_display_threshold:.2f}")

    normalized_candidates = []
    for candidate in valid_results:
        original_score = candidate.overall_score
        normalized_score = original_score # Start with original, adjust if needed

        # --- ACTIVE Normalization Logic ---
        # Log the raw score before any normalization attempts
        logger.info(f"Candidate {candidate.candidate_name}: Raw original_score = {original_score}")

        # If the score is between 0 and 1 (exclusive), and very small (e.g., < 0.05),
        # it's highly likely it's a mis-scaled percentage (like 0.009 for 90%)
        # This is the most direct fix for "0.9% -> 90%"
        if original_score > 0.0 and original_score < 0.1: # Catch 0.009, 0.07 etc.
            # Assume it's a percentage already divided by 100. Multiply by 100 to bring it to a 0-1 range.
            normalized_score = original_score * 100 
            logger.info(f"Candidate {candidate.candidate_name}: Score {original_score:.4f} detected as very small (likely mis-scaled percentage). Normalized to {normalized_score:.4f} (0-1 scale).")
        elif original_score > 1.0: # If it's larger than 1, assume it's 0-100 scale (e.g., 75, 90)
            normalized_score = original_score / 100.0
            logger.info(f"Candidate {candidate.candidate_name}: Score {original_score:.2f} detected as >1 (assumed 0-100 scale). Normalized to {normalized_score:.4f} (0-1 scale).")
        else: # Otherwise, assume it's already in the correct 0-1 scale or zero/negative (which will be capped)
            logger.info(f"Candidate {candidate.candidate_name}: Score {original_score:.2f} assumed to be valid 0-1 scale. No normalization applied.")

        # Ensure the score is always clamped between 0 and 1 (inclusive)
        normalized_score = max(0.0, min(1.0, normalized_score))
        
        candidate.overall_score = normalized_score # Update the score on the object
        normalized_candidates.append(candidate) 

    valid_results = normalized_candidates # Use the list with potentially normalized scores

    # Log all scores AFTER normalization (and before filtering)
    if valid_results:
        logger.info("--- All Candidate Scores (After Normalization, Before Filtering) ---")
        for vr in valid_results:
            logger.info(f"Candidate: {vr.candidate_name or vr.resume_filename}, Normalized Score: {vr.overall_score:.4f} (0-1 scale)")
        logger.info("------------------------------------------------------------------")
    else:
        logger.info("No valid evaluation results after normalization.")


    # Filter candidates by the score threshold - ENHANCED WITH LOGGING
    logger.info(f"Filtering candidates with score >= {score_display_threshold:.2f} (which is {score_display_threshold*100:.0f}%)")
    candidates_above_threshold = []
    candidates_below_threshold = []
    
    for candidate in valid_results:
        if candidate.overall_score >= score_display_threshold:
            candidates_above_threshold.append(candidate)
            logger.info(f"✅ PASSED: {candidate.candidate_name or candidate.resume_filename} with score {candidate.overall_score:.4f} ({candidate.overall_score*100:.1f}%)")
        else:
            candidates_below_threshold.append(candidate)
            logger.info(f"❌ FILTERED OUT: {candidate.candidate_name or candidate.resume_filename} with score {candidate.overall_score:.4f} ({candidate.overall_score*100:.1f}%) - Below {score_display_threshold*100:.0f}% threshold")
    
    logger.info(f"Threshold filtering results: {len(candidates_above_threshold)} passed, {len(candidates_below_threshold)} filtered out")

    # Sort the filtered candidates
    candidates_above_threshold.sort(key=lambda x: x.overall_score, reverse=True)
    
    state["triaged_resumes"] = candidates_above_threshold # Update triaged_resumes with filtered list
    
    top_n = int(state.get("top_n"))
    top_candidates = candidates_above_threshold[:top_n]

    # This is the crucial check: If, after filtering, no candidates remain
    if not top_candidates:
        logger.warning(f"Detailed LLM evaluation resulted in no valid candidates above the {score_display_threshold:.2f} threshold.")
        print("\n" + "="*50 + "\n🏆 TOP CANDIDATES 🏆\n" + "="*50)
        print(f"No candidates were found with an overall score greater than or equal to {score_display_threshold*100:.0f}%.")
        state["processing_stats"]["status"] = "Completed Final Selection (No candidates above threshold)"
        state["workflow_status"] = "COMPLETED_EMPTY_RESULTS" # Indicate no qualified results
        state["top_5_candidates"] = [] # Ensure empty list
        return state # Exit here if no qualified candidates

    # If we reach here, there are candidates above the threshold to display
    print("\n" + "="*50 + f"\n🏆 TOP {len(top_candidates)} CANDIDATES (Score >= {score_display_threshold*100:.0f}%) 🏆\n" + "="*50)
    for i, candidate in enumerate(top_candidates):
        display_name = candidate.candidate_name if candidate.candidate_name else candidate.resume_filename
        
        print(f"\n{i+1}. {display_name}") 
        print(f"Overall Reasoning: {candidate.overall_reasoning}")
        # THIS IS THE KEY LINE FOR DISPLAYING PERCENTAGE:
        # We ensure overall_score is already 0-1. So, multiply by 100 for percentage.
        print(f"Overall Match Score: {candidate.overall_score*100:.2f}%") # Changed to :.0f% for "90%" instead of "90.0%"
        print(f"📊 Show Detailed Score Breakdown") 
        print(f"   File: {candidate.resume_filename or 'N/A'}")


    logger.info(f"Final selection complete. Identified and displayed top {len(top_candidates)} candidates above score threshold.")
    state["processing_stats"]["status"] = "Completed Final Selection"
    state["workflow_status"] = "COMPLETED"
    state["top_5_candidates"] = top_candidates

    return state
