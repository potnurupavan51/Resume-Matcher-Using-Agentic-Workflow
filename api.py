import numpy as np
import os
import uuid
import shutil
import json
import config  # Import config module
from src.models import UploadResumeResponse,GenericSuccessResponse
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, status, Path, Form
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pathlib import Path as PathlibPath # Use an alias to avoid conflict with fastapi.Path

# Use the canonical import for the state and models
from src.models import ResumeMatcherState, JobMatchResult, EvaluationPlan,StartJobResponse,JobActionRequest,DeleteResumesRequest, ChatRequest, ChatResponse
from src.graph import get_workflow
from src.extract_jd_text import extract_text_from_docx, extract_text_from_file

# LLM Imports for Chat API
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# In-memory dictionary to hold the state of active jobs.
JOB_STATES: Dict[str, ResumeMatcherState] = {}

# Initialize FastAPI app
app = FastAPI(
    title="Interactive Resume Matcher API",
    description="An API to generate evaluation plans from Job Descriptions, find top resume matches, request additional candidates, and chat with AI about the results.",
    version="2.2.0" 
)

def make_json_serializable(obj):
    if isinstance(obj, EvaluationPlan):
        return obj.model_dump()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Define directories as you specified
TEMP_UPLOAD_DIR = "temp_uploads"
RESUMES_DIR = os.getenv("RESUMES_DIRECTORY", "RR")

JD_CACHE_DIR = "cache/jd_cache"
EVALUATION_PLAN_CACHE_DIR = "cache/jd_cache/evaluation_plans"
EMBEDDING_CACHE_DIR = "cache/jd_cache/embeddings"
JD_FILE_CACHE_DIR = "cache/jd_cache/jd_files" 
PARSED_JD_CACHE_DIR = "cache/jd_cache/parsed_jds" 

# Ensure all directories exist
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
os.makedirs(EVALUATION_PLAN_CACHE_DIR, exist_ok=True)
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
os.makedirs(JD_FILE_CACHE_DIR, exist_ok=True)
os.makedirs(PARSED_JD_CACHE_DIR, exist_ok=True)

if not os.path.isdir(RESUMES_DIR):
    logger.error(f"FATAL: Resume directory '{RESUMES_DIR}' not found. Please create it or set RESUMES_DIRECTORY in your .env file.")


def manage_jd_cache_fifo():
    """
    Manages JD cache using FIFO (First In, First Out) approach.
    Deletes oldest cached JDs when cache size exceeds the configured maximum.
    """
    try:
        # Get all cached JD files with their creation times
        cached_files = []
        
        if not os.path.isdir(JD_FILE_CACHE_DIR):
            return
            
        for fname in os.listdir(JD_FILE_CACHE_DIR):
            if fname.endswith((".docx", ".pdf")):
                base_name, _ = os.path.splitext(fname)
                jd_file_path = os.path.join(JD_FILE_CACHE_DIR, fname)
                
                # Check if all related cache files exist (complete cache entry)
                parsed_path = os.path.join(PARSED_JD_CACHE_DIR, f"{base_name}.json")
                embedding_path = os.path.join(EMBEDDING_CACHE_DIR, f"{base_name}.npy")
                eval_plan_path = os.path.join(EVALUATION_PLAN_CACHE_DIR, f"{base_name}.json")
                
                if (os.path.exists(parsed_path) and 
                    os.path.exists(embedding_path) and 
                    os.path.exists(eval_plan_path)):
                    
                    # Use creation time for FIFO (oldest first)
                    creation_time = os.path.getctime(jd_file_path)
                    cached_files.append({
                        'base_name': base_name,
                        'file_name': fname,
                        'creation_time': creation_time,
                        'paths': {
                            'jd_file': jd_file_path,
                            'parsed': parsed_path,
                            'embedding': embedding_path,
                            'eval_plan': eval_plan_path
                        }
                    })
        
        # Check if we need to delete old entries
        if len(cached_files) > config.JD_CACHE_MAX_SIZE:
            # Sort by creation time (oldest first)
            cached_files.sort(key=lambda x: x['creation_time'])
            
            # Calculate how many files to delete
            files_to_delete = len(cached_files) - config.JD_CACHE_MAX_SIZE
            
            logger.info(f"JD cache size ({len(cached_files)}) exceeds maximum ({config.JD_CACHE_MAX_SIZE}). Deleting {files_to_delete} oldest entries.")
            
            # Delete the oldest entries
            for i in range(files_to_delete):
                file_info = cached_files[i]
                try:
                    # Delete all related cache files
                    for path_type, file_path in file_info['paths'].items():
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            logger.debug(f"Deleted {path_type} cache file: {file_path}")
                    
                    logger.info(f"Successfully deleted JD cache entry: {file_info['file_name']}")
                    
                except Exception as delete_error:
                    logger.error(f"Error deleting cache files for {file_info['file_name']}: {delete_error}")
                    
        else:
            logger.debug(f"JD cache size ({len(cached_files)}) is within limit ({config.JD_CACHE_MAX_SIZE}). No cleanup needed.")
            
    except Exception as e:
        logger.error(f"Error managing JD cache FIFO: {e}", exc_info=True)


@app.get("/jds/list", response_model=List[Dict[str, str]], tags=["Job Management"], summary="List Cached JDs")
def list_cached_jds() -> List[Dict[str, str]]:
    """Lists cached Job Descriptions for which all required artifacts (JD, parsed data, embedding, evaluation plan) exist."""
    jds = []
    # Ensure all relevant cache directories exist before proceeding
    if not os.path.isdir(JD_FILE_CACHE_DIR) or \
       not os.path.isdir(PARSED_JD_CACHE_DIR) or \
       not os.path.isdir(EMBEDDING_CACHE_DIR) or \
       not os.path.isdir(EVALUATION_PLAN_CACHE_DIR):
        logger.warning("One or more cache directories do not exist. Returning empty list.")
        return [] 
    logger.info(f"Scanning for cached JDs in '{JD_FILE_CACHE_DIR}' and checking for related artifacts.")

    for fname in os.listdir(JD_FILE_CACHE_DIR):
        # Now check for both DOCX and PDF files
        if fname.endswith((".docx", ".pdf")):
            base_name, _ = os.path.splitext(fname)
            
            parsed_jd_path = os.path.join(PARSED_JD_CACHE_DIR, f"{base_name}.json")
            embedding_path = os.path.join(EMBEDDING_CACHE_DIR, f"{base_name}.npy")
            eval_plan_path = os.path.join(EVALUATION_PLAN_CACHE_DIR, f"{base_name}.json")
            
            if os.path.exists(parsed_jd_path) and os.path.exists(embedding_path) and os.path.exists(eval_plan_path):
                logger.debug(f"Found complete cache for '{fname}': Parsed={os.path.exists(parsed_jd_path)}, Embedding={os.path.exists(embedding_path)}, EvalPlan={os.path.exists(eval_plan_path)}")
                jds.append({"cache_key": fname, "display_name": fname})
            else:
                logger.debug(f"Incomplete cache for '{fname}'. Missing files: Parsed={os.path.exists(parsed_jd_path)}, Embedding={os.path.exists(embedding_path)}, EvalPlan={os.path.exists(eval_plan_path)}")

    logger.info(f"Found {len(jds)} complete cached JDs.")
    return jds


@app.post("/jobs/start",
          response_model=StartJobResponse,
          summary="Start a New Matching Job",
          tags=["Job Management"])
async def start_job(
    jd_file: Optional[UploadFile] = File(None, description="Job Description file in .docx or .pdf format."),
    selected_jd: Optional[str] = Form(None, description="Optional JD cache key to use instead of uploading a file."),
    top_n: Optional[int] = Form(None, description="Override for the number of top candidates to select.") # NEW PARAMETER
):
    job_id = str(uuid.uuid4())
    app_state: ResumeMatcherState
    temp_jd_path = None
    EMBEDDING_CACHE_DIR = "cache/jd_cache/embeddings"
    PARSED_JD_CACHE_DIR = "cache/jd_cache/parsed_jds"
    EVALUATION_PLAN_CACHE_DIR = "cache/jd_cache/evaluation_plans"
    JD_FILE_CACHE_DIR = "cache/jd_cache/jd_files" # For the original .docx
    if selected_jd:
        clean_name, _ = os.path.splitext(selected_jd)
        logger.info(f"Attempting cache hit for: {selected_jd} (clean name: {clean_name})")

        # --- START OF THE FIX ---
        # The path to the original .docx file in the cache
        cached_jd_file_path = PathlibPath(JD_FILE_CACHE_DIR) / selected_jd
        # --- END OF THE FIX ---

        parsed_text_path = PathlibPath(PARSED_JD_CACHE_DIR) / f"{clean_name}.json"
        embedding_path = PathlibPath(EMBEDDING_CACHE_DIR) / f"{clean_name}.npy"
        eval_plan_path = PathlibPath(EVALUATION_PLAN_CACHE_DIR) / f"{clean_name}.json"

       
        if not all(p.exists() for p in [cached_jd_file_path, parsed_text_path, embedding_path, eval_plan_path]):
            raise HTTPException(status_code=404, detail=f"Cache for '{selected_jd}' is incomplete or not found.")

        try:
            jd_embedding = np.load(embedding_path)
            with open(eval_plan_path, "r", encoding="utf-8") as f:
                evaluation_plan_data = json.load(f)
            with open(parsed_text_path, "r", encoding="utf-8") as f:
                parsed_jd_summary_and_skills = json.load(f)

            validated_eval_plan = EvaluationPlan.model_validate(evaluation_plan_data)
            
            app_state = {

                "jd_file_path": str(cached_jd_file_path),
                "jd_name": clean_name,  
                "resume_directory": RESUMES_DIR,
                "workflow_status": "JD_PARSED",
                "evaluation_plan": validated_eval_plan,
                "parsed_jd": parsed_jd_summary_and_skills,  
                "jd_embedding": jd_embedding,
                "jd_text": parsed_jd_summary_and_skills.get("summary", ""),
                "all_raw_resumes": [], "newly_profiled_resumes": [], "profiled_resumes": [],
                "retrieved_candidates": [], "triaged_resumes": [], "top_5_candidates": [],
                "error_log": [], "processing_stats": {}, "user_action_required": False, "user_message": None,
                "top_n": top_n,
                "retrive_n":top_n*4 if top_n else None,
                "score_display_threshold": config.score_display_threshold  # Add threshold to state
            }
            logger.info(f"Cache hit successful. Loaded artifacts for: {selected_jd}")
        except Exception as e:
            logger.error(f"Error loading cached JD artifacts for {selected_jd}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to load or process cached JD.")
    
    elif jd_file:
        # Check file extension - now supports both DOCX and PDF
        allowed_extensions = {".docx", ".pdf"}
        file_extension = PathlibPath(jd_file.filename).suffix.lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(400, f"Only {', '.join(allowed_extensions)} files are supported for job descriptions.")

        if not jd_file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

        cached_jd_file_path = PathlibPath(JD_FILE_CACHE_DIR) / jd_file.filename
        with open(cached_jd_file_path, "wb") as buffer:
            shutil.copyfileobj(jd_file.file, buffer)
            jd_file.file.seek(0)

        # Use the unified extraction function that handles both DOCX and PDF
        jd_text = extract_text_from_file(str(cached_jd_file_path))
        if not jd_text:
            raise HTTPException(status_code=400, detail="Could not extract text from uploaded JD.")
        
        clean_name, _ = os.path.splitext(jd_file.filename)
        parsed_text_path = PathlibPath(PARSED_JD_CACHE_DIR) / f"{clean_name}.json"
        with open(parsed_text_path, "w", encoding="utf-8") as f:
            json.dump({"jd_text": jd_text}, f, indent=2)

        app_state = {
            "jd_file_path": str(cached_jd_file_path),
            "resume_directory": RESUMES_DIR, "workflow_status": "INITIALIZED",
            "jd_text": jd_text, "jd_name": clean_name,
            "evaluation_plan": None, "parsed_jd": None, "jd_embedding": None,
            "all_raw_resumes": [], "newly_profiled_resumes": [], "profiled_resumes": [],
            "retrieved_candidates": [], "triaged_resumes": [], "top_5_candidates": [],
            "error_log": [], "processing_stats": {}, "user_action_required": False, "user_message": None,
            "top_n": top_n,
            "retrive_n":top_n*4 if top_n else None,
            "score_display_threshold": config.score_display_threshold  # Add threshold to state
        }
    else:
        raise HTTPException(status_code=400, detail="You must either upload a JD file or select a cached one.")

    try:
        graph = get_workflow(ResumeMatcherState)
        async for event in graph.astream(app_state, {"recursion_limit": 20}):
            for key, value in event.items():
                if key != "__end__":
                    app_state.update(value)
        
        # if app_state.get("workflow_status") == "FAILED":
        #     raise HTTPException(status_code=500, detail=f"Job failed during processing. Errors: {app_state.get('error_log')}")

        


        if app_state.get("workflow_status") == "FAILED":
            # --- START OF CLEANUP BLOCK 1 ---
            if app_state.get("jd_name"):
                clean_name = app_state["jd_name"]
                logger.warning(f"Workflow failed for JD '{clean_name}'. Attempting to clean up cache artifacts.")
                
                try:
                    # Define cache directories (ensure these match your api.py constants)
                    EMBEDDING_CACHE_DIR = "cache/jd_cache/embeddings"
                    PARSED_JD_CACHE_DIR = "cache/jd_cache/parsed_jds"
                    EVALUATION_PLAN_CACHE_DIR = "cache/jd_cache/evaluation_plans"
                    JD_FILE_CACHE_DIR = "cache/jd_cache/jd_files" # For the original files

                    # Remove the original file (could be .docx or .pdf) if it was saved in the cache
                    # Find the actual file by checking both extensions
                    for ext in [".docx", ".pdf"]:
                        cached_jd_file_path = PathlibPath(JD_FILE_CACHE_DIR) / f"{clean_name}{ext}"
                        if cached_jd_file_path.exists():
                            cached_jd_file_path.unlink()
                            logger.info(f"Removed cached JD file: {cached_jd_file_path}")
                            break
                    
                    # Remove other related cache files
                    parsed_text_path = PathlibPath(PARSED_JD_CACHE_DIR) / f"{clean_name}.json"
                    if parsed_text_path.exists():
                        parsed_text_path.unlink()
                        logger.info(f"Removed cached parsed JD text: {parsed_text_path}")

                    embedding_path = PathlibPath(EMBEDDING_CACHE_DIR) / f"{clean_name}.npy"
                    if embedding_path.exists():
                        embedding_path.unlink()
                        logger.info(f"Removed cached JD embedding: {embedding_path}")

                    eval_plan_path = PathlibPath(EVALUATION_PLAN_CACHE_DIR) / f"{clean_name}.json"
                    if eval_plan_path.exists():
                        eval_plan_path.unlink()
                        logger.info(f"Removed cached evaluation plan: {eval_plan_path}")

                except Exception as cleanup_e:
                    logger.error(f"Error during cache cleanup after workflow failure: {cleanup_e}", exc_info=True)
            # --- END OF CLEANUP BLOCK 1 ---
            raise HTTPException(status_code=500, detail=f"Job failed during processing. Errors: {app_state.get('error_log')}")




        # if not app_state.get("evaluation_plan"):
        #     raise HTTPException(status_code=500, detail="Failed to generate an evaluation plan.")


        if not app_state.get("evaluation_plan"):
            # --- START OF CLEANUP BLOCK 2 ---
            if app_state.get("jd_name"):
                clean_name = app_state["jd_name"]
                logger.warning(f"Workflow failed to generate evaluation plan for JD '{clean_name}'. Attempting to clean up cache artifacts.")
                
                try:
                    # Define cache directories (ensure these match your api.py constants)
                    EMBEDDING_CACHE_DIR = "cache/jd_cache/embeddings"
                    PARSED_JD_CACHE_DIR = "cache/jd_cache/parsed_jds"
                    EVALUATION_PLAN_CACHE_DIR = "cache/jd_cache/evaluation_plans"
                    JD_FILE_CACHE_DIR = "cache/jd_cache/jd_files" # For the original files

                    # Remove the original file (could be .docx or .pdf) if it was saved in the cache
                    # Find the actual file by checking both extensions
                    for ext in [".docx", ".pdf"]:
                        cached_jd_file_path = PathlibPath(JD_FILE_CACHE_DIR) / f"{clean_name}{ext}"
                        if cached_jd_file_path.exists():
                            cached_jd_file_path.unlink()
                            logger.info(f"Removed cached JD file: {cached_jd_file_path}")
                            break
                    
                    # Remove other related cache files
                    parsed_text_path = PathlibPath(PARSED_JD_CACHE_DIR) / f"{clean_name}.json"
                    if parsed_text_path.exists():
                        parsed_text_path.unlink()
                        logger.info(f"Removed cached parsed JD text: {parsed_text_path}")

                    embedding_path = PathlibPath(EMBEDDING_CACHE_DIR) / f"{clean_name}.npy"
                    if embedding_path.exists():
                        embedding_path.unlink()
                        logger.info(f"Removed cached JD embedding: {embedding_path}")

                    eval_plan_path = PathlibPath(EVALUATION_PLAN_CACHE_DIR) / f"{clean_name}.json"
                    if eval_plan_path.exists():
                        eval_plan_path.unlink()
                        logger.info(f"Removed cached evaluation plan: {eval_plan_path}")

                except Exception as cleanup_e:
                    logger.error(f"Error during cache cleanup after evaluation plan generation failure: {cleanup_e}", exc_info=True)
            # --- END OF CLEANUP BLOCK 2 ---
            raise HTTPException(status_code=500, detail="Failed to generate an evaluation plan.")


        if app_state.get("jd_name"):
            clean_name = app_state["jd_name"]
            
            if plan := app_state.get("evaluation_plan"):
                eval_plan_path = PathlibPath(EVALUATION_PLAN_CACHE_DIR) / f"{clean_name}.json"
                with open(eval_plan_path, "w", encoding="utf-8") as f:
                    json.dump(plan, f, indent=2, default=make_json_serializable)
                logger.info(f"Saved new evaluation plan to cache: {eval_plan_path}")
            
            embedding = app_state.get("jd_embedding")
            if embedding is not None and embedding.size > 0:
                embedding_path = PathlibPath(EMBEDDING_CACHE_DIR) / f"{clean_name}.npy"
                np.save(embedding_path, embedding)
                logger.info(f"Saved new JD embedding to cache: {embedding_path}")
                
            # Manage cache size using FIFO when a new JD is successfully cached
            logger.info("Running FIFO cache management after successful JD caching...")
            manage_jd_cache_fifo()

        JOB_STATES[job_id] = app_state
        return StartJobResponse(
            job_id=job_id,
            message="Job started successfully. Review the plan and proceed.",
            evaluation_plan=EvaluationPlan.model_validate(app_state["evaluation_plan"])
        )
    # except Exception as e:
    #     logger.error(f"An error occurred during job {job_id} processing: {e}", exc_info=True)
    #     if temp_jd_path and os.path.exists(temp_jd_path) and TEMP_UPLOAD_DIR in str(temp_jd_path):
    #         os.remove(temp_jd_path)
    #     raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


    except Exception as e:
        # --- START OF CLEANUP BLOCK 3 ---
        # This block catches ANY exception that might occur after caching was attempted
        if app_state.get("jd_name"):
            clean_name = app_state["jd_name"]
            logger.warning(f"Workflow encountered an unexpected error for JD '{clean_name}'. Attempting to clean up cache artifacts.")
            try:
                # Define cache directories (ensure these match your api.py constants)
                EMBEDDING_CACHE_DIR = "cache/jd_cache/embeddings"
                PARSED_JD_CACHE_DIR = "cache/jd_cache/parsed_jds"
                EVALUATION_PLAN_CACHE_DIR = "cache/jd_cache/evaluation_plans"
                JD_FILE_CACHE_DIR = "cache/jd_cache/jd_files" # For the original files

                # Remove the original file (could be .docx or .pdf) if it was saved in the cache
                # Find the actual file by checking both extensions
                for ext in [".docx", ".pdf"]:
                    cached_jd_file_path = PathlibPath(JD_FILE_CACHE_DIR) / f"{clean_name}{ext}"
                    if cached_jd_file_path.exists():
                        cached_jd_file_path.unlink()
                        logger.info(f"Removed cached JD file: {cached_jd_file_path}")
                        break
                
                # Remove other related cache files
                parsed_text_path = PathlibPath(PARSED_JD_CACHE_DIR) / f"{clean_name}.json"
                if parsed_text_path.exists():
                    parsed_text_path.unlink()
                    logger.info(f"Removed cached parsed JD text: {parsed_text_path}")

                embedding_path = PathlibPath(EMBEDDING_CACHE_DIR) / f"{clean_name}.npy"
                if embedding_path.exists():
                    embedding_path.unlink()
                    logger.info(f"Removed cached JD embedding: {embedding_path}")

                eval_plan_path = PathlibPath(EVALUATION_PLAN_CACHE_DIR) / f"{clean_name}.json"
                if eval_plan_path.exists():
                    eval_plan_path.unlink()
                    logger.info(f"Removed cached evaluation plan: {eval_plan_path}")

            except Exception as cleanup_e:
                logger.error(f"Error during general cache cleanup after unexpected error: {cleanup_e}", exc_info=True)
        # --- END OF CLEANUP BLOCK 3 ---
        logger.error(f"An error occurred during job {job_id} processing: {e}", exc_info=True)
        
        # ... (The existing cleanup for temp uploads should remain as it is) ...
        temp_jd_path = app_state.get("jd_file_path") 
        if temp_jd_path and TEMP_UPLOAD_DIR in str(temp_jd_path) and os.path.exists(temp_jd_path):
            try:
                os.remove(temp_jd_path)
                logger.info(f"Cleaned up temporary upload file for job {job_id}: {temp_jd_path}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary upload file {temp_jd_path}: {e}", exc_info=True)
        
        # Re-raise the original exception after cleanup attempts
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    

@app.post("/jobs/{job_id}/action",
          response_model=List[JobMatchResult],
          summary="Approve, Modify, or Reject a Job's Plan",
          tags=["Job Management"])
async def handle_job_action(
    request: JobActionRequest,
    job_id: str = Path(..., description="The unique ID of the job to act upon.")
):
    if job_id not in JOB_STATES:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job ID '{job_id}' not found.")
    
    app_state = JOB_STATES[job_id]
    jd_path = app_state.get("jd_file_path") # This variable now holds a valid path or None

    try:
        if request.action == "reject":
            logger.info(f"Job {job_id} was rejected by the user.")
            return []

        elif request.action == "modify" or request.action == "approve":
            if request.action == "modify":
                if not request.evaluation_plan:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="An 'evaluation_plan' must be provided for the 'modify' action.")
                plan_dict = request.evaluation_plan.model_dump()
                if "criteria" in plan_dict and isinstance(plan_dict["criteria"], list):
                    plan_dict["criteria"] = [c for c in plan_dict["criteria"] if c.get("weightage", 0) > 0]
                app_state["evaluation_plan"] = plan_dict
                logger.info(f"Job {job_id}: Evaluation plan updated by user.")

                # --- START OF THE FIX: Overwrite the evaluation plan cache if this is a cached JD ---
                # Only update the cache if the JD is from cache (not a temp upload)
                jd_name = app_state.get("jd_name")
                if jd_name:
                    from pathlib import Path as PathlibPath
                    import json
                    eval_plan_path = PathlibPath("cache/jd_cache/evaluation_plans") / f"{jd_name}.json"
                    try:
                        with open(eval_plan_path, "w", encoding="utf-8") as f:
                            json.dump(plan_dict, f, indent=2)
                        logger.info(f"Evaluation plan cache updated for cached JD: {jd_name}")
                    except Exception as cache_exc:
                        logger.error(f"Failed to update evaluation plan cache for {jd_name}: {cache_exc}")
                # --- END OF THE FIX ---

            logger.info(f"Job {job_id}: Plan approved. Resuming workflow.")
            app_state["workflow_status"] = "EVAL_PLAN_APPROVED"
            
            graph = get_workflow(ResumeMatcherState)
            
            async for event in graph.astream(app_state, {"recursion_limit": 50}):
                for key, value in event.items():
                    if key != "__end__":
                        app_state.update(value)
            
            if app_state.get("workflow_status") == "COMPLETED":
                top_candidates = app_state.get("top_5_candidates", [])
                logger.info(f"Job {job_id} completed successfully. Found {len(top_candidates)} candidates.")
                return top_candidates
            else:
                logger.warning(f"Job {job_id} finished with status '{app_state.get('workflow_status')}' and no results.")
                return []

        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid action. Must be 'approve', 'modify', or 'reject'.")

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error processing action for job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    finally:
        # Don't delete job state immediately - preserve it for potential "more resumes" requests
        # Only clean up temporary files
        if jd_path and TEMP_UPLOAD_DIR in str(jd_path) and os.path.exists(jd_path):
            os.remove(jd_path)
            logger.info(f"Cleaned up temporary JD file for job {job_id}: {jd_path}")

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to the Resume Matcher API. Go to /docs to see the API documentation."}

@app.get("/jds/{cache_key}/download", response_model=bytes, tags=["Job Management"], summary="Download a Cached JD")
async def download_cached_jd(
    cache_key: str = Path(..., description="The cache key (filename) of the JD to download.")
):
    """
    Retrieves and returns the content of a cached JD file (.docx or .pdf).
    """
    # Ensure the cache directory exists
    if not os.path.exists(JD_FILE_CACHE_DIR):
        logger.error(f"JD cache directory '{JD_FILE_CACHE_DIR}' does not exist.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="JD cache directory not found.")

    # Construct the full file path using Pathlib for safety and clarity
    file_path = PathlibPath(JD_FILE_CACHE_DIR) / cache_key
    
    # Validate file existence and type - now supports both DOCX and PDF
    allowed_extensions = {".docx", ".pdf"}
    if not file_path.is_file() or file_path.suffix.lower() not in allowed_extensions:
        logger.warning(f"Cached JD file not found or not a supported file type: {file_path}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Cached JD file '{cache_key}' not found or is not a supported file type ({', '.join(allowed_extensions)}).")

    try:
        # Return the file content as bytes
        with open(file_path, "rb") as f:
            content = f.read()
        return content
    except Exception as e:
        logger.error(f"Error reading cached JD file '{file_path}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to read cached JD file: {e}")




@app.post("/resumes/upload",
          response_model=UploadResumeResponse,
          tags=["Resume Management"],
          summary="Upload a Resume and Index Immediately")
async def upload_resume(
    resume_file: UploadFile = File(..., description="The resume file (.docx or .pdf) to upload.")
):
    """
    Uploads a resume file (.docx or .pdf), saves it to the configured RESUMES_DIRECTORY,
    and immediately processes and indexes it in FAISS without requiring the graph workflow.
    """
    # Basic file validation
    if not resume_file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    # Check file extension - now supports both DOCX and PDF
    allowed_extensions = {".docx", ".pdf"} # Updated to support both DOCX and PDF
    ext = PathlibPath(resume_file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"File type '{ext}' not allowed for resume upload. Only {', '.join(allowed_extensions)} are permitted.")

    # Define the destination path within the configured RESUMES_DIR
    # FastAPI's UploadFile handles basic filename sanitization to prevent path traversal.
    destination_filepath = os.path.join(RESUMES_DIR, resume_file.filename)

    # Save the file first
    try:
        with open(destination_filepath, "wb") as buffer:
            shutil.copyfileobj(resume_file.file, buffer)
        
        logger.info(f"Resume '{resume_file.filename}' uploaded successfully to '{destination_filepath}'")
        
    except Exception as e:
        logger.error(f"Error saving uploaded resume '{resume_file.filename}' to '{destination_filepath}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded resume: {e}")

    # Now immediately process and index the resume in FAISS
    try:
        from src.extract_jd_text import extract_text_from_file
        from src.llm_handlers import extract_resume_parser_data_with_langchain_safe
        from src.models import ResumeRecord
        import uuid
        from pathlib import Path
        
        # Check if FAISS DB Manager is available
        if not faiss_db_manager:
            logger.warning("FAISS DB Manager not initialized. Resume uploaded but not indexed.")
            return UploadResumeResponse(
                filename=resume_file.filename,
                message="Resume uploaded successfully but could not be indexed (FAISS not initialized).",
                destination_path=destination_filepath
            )
        
        logger.info(f"Starting immediate processing and indexing of '{resume_file.filename}'...")
        
        # Extract text from the uploaded file
        resume_text = extract_text_from_file(destination_filepath)
        if not resume_text or not resume_text.strip():
            logger.warning(f"Could not extract text from '{resume_file.filename}'. File uploaded but not indexed.")
            return UploadResumeResponse(
                filename=resume_file.filename,
                message="Resume uploaded successfully but could not extract text for indexing.",
                destination_path=destination_filepath
            )
        
        # Generate consistent resume ID based on filename
        resume_id = str(uuid.uuid5(uuid.NAMESPACE_URL, resume_file.filename))
        
        # Check if this resume is already in FAISS
        if resume_id in faiss_db_manager.docstore:
            logger.info(f"Resume '{resume_file.filename}' already exists in FAISS index. Skipping indexing.")
            return UploadResumeResponse(
                filename=resume_file.filename,
                message="Resume uploaded successfully. Already indexed in FAISS.",
                destination_path=destination_filepath
            )
        
        # Create raw resume data structure
        raw_resume_data = {
            "id": resume_id,
            "file_path": destination_filepath,
            "filename": resume_file.filename,
            "raw_text": resume_text,
            "name": Path(resume_file.filename).stem
        }
        
        # Process the resume with LLM to get structured data
        logger.info(f"Processing resume '{resume_file.filename}' with LLM...")
        parsed_data = await extract_resume_parser_data_with_langchain_safe(resume_text)
        
        if not parsed_data:
            logger.warning(f"Failed to parse structured data for '{resume_file.filename}'. File uploaded but not indexed.")
            return UploadResumeResponse(
                filename=resume_file.filename,
                message="Resume uploaded successfully but failed to parse structured data for indexing.",
                destination_path=destination_filepath
            )
        
        # Create ResumeRecord object
        resume_record = ResumeRecord(
            id=resume_id,
            file_path=destination_filepath,
            filename=resume_file.filename,
            original_text=resume_text,
            parsed_data=parsed_data,
            name=raw_resume_data.get("name")
        )
        
        # Add to FAISS index
        logger.info(f"Adding resume '{resume_file.filename}' to FAISS index...")
        await faiss_db_manager.add_resumes([resume_record])
        
        # Save the updated FAISS index
        faiss_db_manager.save()
        
        logger.info(f"Successfully processed and indexed resume '{resume_file.filename}' in FAISS.")
        
        return UploadResumeResponse(
            filename=resume_file.filename,
            message="Resume uploaded and immediately indexed in FAISS successfully.",
            destination_path=destination_filepath
        )
        
    except Exception as e:
        logger.error(f"Error processing and indexing resume '{resume_file.filename}': {e}", exc_info=True)
        # File was uploaded successfully, but indexing failed
        return UploadResumeResponse(
            filename=resume_file.filename,
            message=f"Resume uploaded successfully but indexing failed: {str(e)}",
            destination_path=destination_filepath
        )




@app.post("/jds/cache/cleanup",
          summary="Cleanup JD Cache using FIFO",
          tags=["Job Management"])
def cleanup_jd_cache():
    """
    Manually trigger JD cache cleanup using FIFO approach.
    Useful for testing or manual cache management.
    """
    try:
        manage_jd_cache_fifo()
        return {
            "message": "JD cache cleanup completed successfully",
            "max_cache_size": config.JD_CACHE_MAX_SIZE
        }
    except Exception as e:
        logger.error(f"Error during manual cache cleanup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cleanup cache: {str(e)}")


@app.get("/config/top_n",
          summary="Get Current Top N Configuration",
          tags=["Configuration"])
async def get_top_n_config():
    """
    Retrieves the current top_n value from config.py
    """
    from src.config_manager import get_current_top_n
    try:
        current_top_n = get_current_top_n()
        return {
            "top_n": current_top_n,
            "top_k_retrieved_count": current_top_n * 4,
            "message": f"Current configuration: top_n={current_top_n}, top_k_retrieved_count={current_top_n * 4}"
        }
    except Exception as e:
        logger.error(f"Error retrieving configuration: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve configuration")


@app.post("/jobs/{job_id}/more-resumes",
          response_model=List[JobMatchResult],
          summary="Get More Resumes from Existing Job Results",
          tags=["Job Management"])
async def get_more_resumes(
    job_id: str = Path(..., description="The unique ID of the completed job."),
    additional_count: int = Form(..., description="Number of additional resumes to return (must be less than RAG retrieval count).")
):
    """
    Returns additional resumes from a completed job's results.
    The additional_count must be less than the original RAG retrieval count (top_n * 4).
    """
    if job_id not in JOB_STATES:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job ID '{job_id}' not found.")
    
    app_state = JOB_STATES[job_id]
    
    # Validate that the job is completed
    if app_state.get("workflow_status") != "COMPLETED":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job must be completed to request more resumes.")
    
    # Get the original top_n and calculate max additional count
    original_top_n = int(app_state.get("top_n", 5))
    max_additional_count = (original_top_n * 4) - original_top_n  # RAG retrieval count minus already returned top_n
    
    if additional_count <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Additional count must be greater than 0.")
    
    if additional_count > max_additional_count:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Additional count ({additional_count}) cannot exceed {max_additional_count} (RAG retrieval limit minus already returned candidates)."
        )
    
    # Get all triaged resumes (sorted candidates above threshold)
    all_triaged_resumes = app_state.get("triaged_resumes", [])
    top_candidates = app_state.get("top_5_candidates", [])
    
    if not all_triaged_resumes:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No triaged resumes found for this job.")
    
    # Calculate how many candidates we already returned
    already_returned_count = len(top_candidates)
    
    # Get additional candidates from the sorted triaged list
    start_index = already_returned_count
    end_index = start_index + additional_count
    
    additional_candidates = all_triaged_resumes[start_index:end_index]
    
    if not additional_candidates:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No additional qualified candidates available.")
    
    logger.info(f"Job {job_id}: Returning {len(additional_candidates)} additional candidates (requested: {additional_count}).")
    return additional_candidates


@app.post("/jobs/{job_id}/chat",
          response_model=ChatResponse,
          summary="Chat with AI about Job Results",
          tags=["Chat"])
async def chat_with_ai(
    chat_request: ChatRequest,
    job_id: str = Path(..., description="The unique ID of the completed job.")
):
    """
    Chat with an AI assistant about the job results and candidate resumes.
    The AI will use the job's top candidates and their full resume content as context.
    """
    # Validate job exists
    if job_id not in JOB_STATES:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job ID '{job_id}' not found.")
    
    # Validate job is completed
    app_state = JOB_STATES[job_id]
    if app_state.get("workflow_status") != "COMPLETED":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job must be completed to chat about results.")
    
    # Get top candidates for context
    top_candidates = app_state.get("top_5_candidates", [])
    if not top_candidates:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No candidates found for this job.")
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="OpenAI API key not configured on server.")
    
    try:
        # Initialize LLM client
        llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini", api_key=api_key)
        
        # Prepare context from top candidates
        formatted_context_parts = []
        resume_dir = app_state.get("resume_directory", "RR")
        
        for i, candidate in enumerate(top_candidates[:5]):  # Limit to top 5 for context
            candidate_name = getattr(candidate, 'candidate_name', None) or getattr(candidate, 'resume_filename', 'Unknown')
            overall_score = getattr(candidate, 'overall_score', 0)
            overall_reasoning = getattr(candidate, 'overall_reasoning', 'No reasoning provided')
            resume_filename = getattr(candidate, 'resume_filename', None)
            
            formatted_context_parts.append(f"--- Candidate #{i+1}: {candidate_name} ---")
            formatted_context_parts.append(f"Overall Match Score: {overall_score*100:.1f}%")
            formatted_context_parts.append(f"AI's Initial Reasoning: {overall_reasoning}")
            
            # Try to load full resume text
            if resume_filename:
                resume_path = os.path.join(resume_dir, resume_filename)
                if os.path.exists(resume_path):
                    try:
                        # Use the unified extraction function that handles both DOCX and PDF
                        full_text = extract_text_from_file(resume_path)
                        if full_text:
                            formatted_context_parts.append("\nFull Resume Content:")
                            formatted_context_parts.append(full_text)
                        else:
                            formatted_context_parts.append("\nFull resume content could not be extracted.")
                    except Exception as e:
                        logger.warning(f"Error extracting text from {resume_filename}: {e}")
                        formatted_context_parts.append("\nFull resume content could not be loaded.")
                else:
                    formatted_context_parts.append(f"\nResume file not found: {resume_filename}")
            else:
                formatted_context_parts.append("\nNo resume file available.")
            
            formatted_context_parts.append("--- End of Candidate ---")
        
        formatted_context = "\n\n".join(formatted_context_parts)
        
        # Prepare chat history for the prompt
        chat_history_messages = []
        for msg in chat_request.chat_history:
            chat_history_messages.append((msg.role, msg.content))
        
        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert HR analyst AI assistant. Your task is to answer questions about candidate resumes based on the provided context.

The context includes:
- Top candidate matches from a recent job search
- Their match scores and reasoning from the initial AI evaluation
- Full resume content for detailed analysis

Instructions:
1. Base your answers on the provided resume context
2. Be specific and reference actual information from the resumes
3. If asked to compare candidates, use the scores and content provided
4. If information is not available in the context, clearly state that
5. Provide actionable insights for hiring decisions
6. Be professional and concise in your responses"""),
            *chat_history_messages,
            ("human", f"""**Context - Top Candidate Resumes:**
{formatted_context}

**User Question:** {chat_request.query}

Please provide a helpful response based on the resume context above.""")
        ])
        
        # Get AI response
        chain = prompt | llm
        response_content = ""
        
        async for chunk in chain.astream({}):
            if hasattr(chunk, 'content') and chunk.content:
                response_content += chunk.content
        
        if not response_content:
            response_content = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
        
        logger.info(f"Chat query processed for job {job_id}. Context: {len(top_candidates)} candidates.")
        
        return ChatResponse(
            response=response_content,
            job_id=job_id,
            context_used=len(top_candidates)
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request for job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to process chat request: {str(e)}"
        )


# Resume Management Endpoints with FAISS Integration
from src.services import faiss_db_manager
from src.models import DeleteResumesRequest, ResumeSearchRequest, ResumeSearchResponse, ResumeSearchResult

@app.get("/resumes/list",
         response_model=List[Dict[str, Any]],
         tags=["Resume Management"],
         summary="List All Indexed Resumes")
def list_indexed_resumes():
    """
    Retrieves a list of all resumes currently stored in the FAISS vector database.
    Each item in the list contains metadata about the resume, including its unique ID and filename.
    """
    if not faiss_db_manager:
        raise HTTPException(status_code=500, detail="Database manager is not initialized.")
    try:
        all_resumes = faiss_db_manager.list_all_resumes()
        return all_resumes
    except Exception as e:
        logger.error(f"Error listing resumes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve resume list.")


@app.post("/resumes/delete",
          response_model=GenericSuccessResponse,
          tags=["Resume Management"],
          summary="Delete Indexed Resumes")
async def delete_indexed_resumes(request: DeleteResumesRequest):
    """
    Deletes one or more resumes from the FAISS vector database based on their IDs.
    This operation is permanent and also deletes the physical files from the RESUMES_DIR.
    Uses complete index rebuild for consistency.
    """
    if not faiss_db_manager:
        raise HTTPException(status_code=500, detail="Database manager is not initialized.")
    
    if not request.resume_ids:
        raise HTTPException(status_code=400, detail="No resume_ids provided for deletion.")

    deleted_filenames = []
    try:
        # Call the async delete_resumes from FAISS_DB_Manager with validation
        deleted_resumes_info = await faiss_db_manager.delete_resumes_with_validation_async(request.resume_ids)
        
        for resume_info in deleted_resumes_info:
            if 'filename' in resume_info:
                deleted_filenames.append(resume_info['filename'])
            else:
                logger.warning(f"Resume info from DB manager missing filename for ID: {resume_info.get('id', 'N/A')}")

        # Delete files from the RESUMES_DIR
        deleted_from_disk_count = 0
        for filename in deleted_filenames:
            file_path_on_disk = PathlibPath(RESUMES_DIR) / filename
            if file_path_on_disk.exists() and file_path_on_disk.is_file():
                try:
                    os.remove(file_path_on_disk)
                    logger.info(f"Successfully deleted resume file from disk: {file_path_on_disk}")
                    deleted_from_disk_count += 1
                except OSError as e:
                    logger.error(f"Error deleting resume file from disk {file_path_on_disk}: {e}", exc_info=True)
            else:
                logger.warning(f"Resume file not found on disk or is not a file: {file_path_on_disk}. Skipping disk deletion.")

        # Save the changes to disk after deletion from FAISS (metadata update)
        faiss_db_manager.save()

        message = f"Successfully deleted {len(deleted_resumes_info)} resume(s) from index and {deleted_from_disk_count} file(s) from disk. Index rebuilt for consistency."
        logger.info(message)
        return GenericSuccessResponse(message=message)
        
    except Exception as e:
        logger.error(f"Error deleting resumes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during resume deletion: {e}")


@app.get("/resumes/validate-index",
         response_model=Dict[str, Any],
         tags=["Resume Management"],
         summary="Validate FAISS Index Consistency")
def validate_faiss_index():
    """
    Validates the consistency between FAISS index, ID mappings, and document store.
    Returns a detailed report of any inconsistencies found.
    """
    if not faiss_db_manager:
        raise HTTPException(status_code=500, detail="Database manager is not initialized.")
    
    try:
        validation_report = faiss_db_manager.validate_index_consistency()
        return validation_report
    except Exception as e:
        logger.error(f"Error validating index consistency: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during index validation: {e}")


@app.post("/resumes/repair-index",
          response_model=GenericSuccessResponse,
          tags=["Resume Management"],
          summary="Repair FAISS Index Consistency")
async def repair_faiss_index():
    """
    Attempts to repair FAISS index inconsistencies by rebuilding from the document store.
    This operation may take some time for large datasets.
    """
    if not faiss_db_manager:
        raise HTTPException(status_code=500, detail="Database manager is not initialized.")
    
    try:
        repair_success = await faiss_db_manager.repair_index_consistency()
        
        if repair_success:
            # Save the repaired index
            faiss_db_manager.save()
            message = "Index consistency repair completed successfully."
            logger.info(message)
            return GenericSuccessResponse(message=message)
        else:
            raise HTTPException(status_code=500, detail="Index repair failed. Check logs for details.")
            
    except Exception as e:
        logger.error(f"Error repairing index consistency: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during index repair: {e}")


@app.post("/resumes/force-rebuild-index",
          response_model=GenericSuccessResponse,
          tags=["Resume Management"],
          summary="Force Rebuild FAISS Index")
async def force_rebuild_faiss_index():
    """
    Forces a complete rebuild of the FAISS index from the document store.
    Use this to fix severe inconsistencies or after problematic deletions.
    This operation may take some time for large datasets.
    """
    if not faiss_db_manager:
        raise HTTPException(status_code=500, detail="Database manager is not initialized.")
    
    try:
        rebuild_success = await faiss_db_manager.force_rebuild_index()
        
        if rebuild_success:
            # Save the rebuilt index
            faiss_db_manager.save()
            message = "Index force rebuild completed successfully. All inconsistencies should be resolved."
            logger.info(message)
            return GenericSuccessResponse(message=message)
        else:
            raise HTTPException(status_code=500, detail="Index force rebuild failed. Check logs for details.")
            
    except Exception as e:
        logger.error(f"Error during forced index rebuild: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during forced index rebuild: {e}")


@app.post("/resumes/search",
          response_model=ResumeSearchResponse,
          tags=["Resume Management"],
          summary="Search Indexed Resumes")
async def search_indexed_resumes(request: ResumeSearchRequest):
    """
    Searches indexed resumes by filename or candidate name (if extracted and stored).
    """
    if not faiss_db_manager:
        logger.error("Database manager (faiss_db_manager) is not initialized.")
        raise HTTPException(status_code=500, detail="Database manager is not initialized.")

    search_query = request.query.lower().strip()
    matching_resumes: List[ResumeSearchResult] = []

    try:
        # Get all resume metadata from the FAISS document store
        all_resumes_metadata = faiss_db_manager.list_all_resumes()

        for resume_meta in all_resumes_metadata:
            # Safely get fields, providing defaults if they don't exist
            filename = resume_meta.get("filename", "").lower()
            # Assuming 'name' (extracted candidate name) might be stored in metadata
            candidate_name = resume_meta.get("name", "").lower() 
            resume_id = resume_meta.get("id")
            file_path = resume_meta.get("file_path")  # Path to the original file

            # Perform case-insensitive search on filename and candidate name
            if search_query and (
                search_query in filename or
                (candidate_name and search_query in candidate_name)
            ):
                # Create the search result object
                matching_resumes.append(ResumeSearchResult(
                    id=resume_id,
                    filename=resume_meta.get("filename"),  # Use original case for display
                    name=resume_meta.get("name"),         # Use original case for display
                    file_path=file_path
                ))
        
        logger.info(f"Search for '{request.query}' found {len(matching_resumes)} resumes.")
        return ResumeSearchResponse(
            message=f"Found {len(matching_resumes)} resumes matching '{request.query}'.",
            results=matching_resumes,
            total_found=len(matching_resumes)
        )

    except Exception as e:
        logger.error(f"Error during resume search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to search resumes: {e}")



