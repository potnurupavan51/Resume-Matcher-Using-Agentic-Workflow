from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple, Literal,TypedDict
import uuid
import numpy as np
# Removed: import numpy as np # No longer needed here as np.ndarray is handled in main.py and services.py
# Removed: from langchain_core.documents import Document # No longer needed here

# --- Core Data Models for LLM Parsing ---

class Skill(BaseModel):
    name: str
    # years_of_experience: Optional[float] = None

class Experience(BaseModel):
    """Represents a single work experience entry from a resume."""
    title: str
    company: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None

class Education(BaseModel):
    """Represents a single education entry from a resume."""
    degree: str
    university: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class ExperienceRequirement(BaseModel):
    description: str

class EducationRequirement(BaseModel):
    description: str

class ResumeParser(BaseModel):
    """Schema for structured data parsed from a single resume."""
    name: Optional[str] = None # Will be set to None for anonymization
    summary: Optional[str] = None
    skills: List[Skill] = Field(default_factory=list)
    previous_experience: List[Experience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    years_experience: Optional[float] = None
    achievements: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)

class JDParser(BaseModel):
    """Schema for the initial, raw LLM parsing of a Job Description."""
    job_title: Optional[str] = None
    overall_summary: Optional[str] = None
    required_skills: List[Skill] = Field(default_factory=list) # This is now included for completeness
    years_of_experience: Optional[float] = None
    experience_requirements: List[ExperienceRequirement] = Field(default_factory=list)
    education_requirements: List[EducationRequirement] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)

# --- Evaluation Plan Models ---

class Criterion(BaseModel):
    """Defines a single, unified criterion for evaluation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    category: str = Field(description="e.g., 'Skill', 'Experience', 'Responsibility'")
    criteria: str = Field(description="Specific aspect to evaluate, e.g., 'Python proficiency'")
    weightage: int = Field(description="Normalized weightage (e.g., part of a sum of 200)")
    evaluation_guidelines: str = Field(description="Instructions for how to evaluate this criterion.")
    score: Optional[float] = Field(None, ge=0, description="Score assigned during evaluation (0-10 for individual criterion).")
    reason: Optional[str] = Field(None, description="Reasoning for the assigned score.")

class EvaluationPlan(BaseModel):
    """A collection of criteria used to evaluate resumes for a specific JD."""
    job_title: str = "N/A"
    overall_summary: str = "N/A"
    criteria: List[Criterion] = Field(default_factory=list)

# --- LLM Helper Models (for structuring LLM output before processing) ---
class LlmEvaluationCriterion(BaseModel):
    """Schema for a single criterion as output by the LLM."""
    category: str
    criteria: str
    weightage: Literal[1, 2, 3, 4, 5]
    evaluation_guidelines: str

class LlmJDExtractionModel(BaseModel):
    """The root model for the LLM's JD extraction output."""
    evaluation_criteria: List[LlmEvaluationCriterion]

# --- Internal Data Storage Models ---

class ResumeRecord(BaseModel):
    """Internal record for a single Resume, including all its data."""
    id: str
    file_path: str
    filename: str
    original_text: str
    parsed_data: Optional[ResumeParser] = None
    embedding: Optional[List[float]] = None # Optional: embedding can be stored elsewhere
    name: Optional[str] = None # This name can be kept (e.g., from filename) while parsed_data.name is anonymized

class JDRecord(BaseModel):
    """Internal record for a single Job Description."""
    id: str
    original_text: str
    parsed_data: JDParser
    evaluation_plan: Optional[EvaluationPlan] = None
    embedding: Optional[List[float]] = None

# --- RAG and Final Results Models ---

class JobMatchResult(BaseModel):
    """Represents the detailed evaluation outcome for a single resume against a JD."""
    resume_id: str
    resume_filename: str # Added for easier identification
    candidate_name: Optional[str] = None
    overall_score: float = Field(..., ge=0, le=100) # Overall score from 0 to 100
    overall_reasoning: str
    criterion_scores: List[Criterion] = Field(default_factory=list) # List of Criterion objects with scores/reasons

# Removed ResumeMatcherState TypedDict from here
class ResumeMatcherState(TypedDict):
    """
    Represents the state of the Resume Matcher workflow.
    This dictionary is passed between each node in the graph.
    """
    # Inputs
    top_n: Optional[int]
    jd_file_path: str
    resume_directory: str
    jd_name:str
    

    # Processed Data - JD
    jd_text: str
    parsed_jd: Optional[Any] # Using Any to avoid importing JDParser here
    evaluation_plan: Optional[Any] # Using Any to avoid importing EvaluationPlan here
    jd_embedding: Optional[np.ndarray] # Use numpy array for embeddings

    # Processed Data - Resumes
    all_raw_resumes: List[Dict[str, Any]] # New: Store ALL raw resume data here after ingestion
    newly_profiled_resumes: List[Any] # New: Store only the ResumeRecords that were newly profiled
    profiled_resumes: List[Any] # Store ALL ResumeRecords found (newly profiled + skeletal for existing ones)

    # RAG & Evaluation
    retrieved_candidates: List[Tuple[Any, float]] # List of (Document, score) from FAISS (using Any for Document)
    triaged_resumes: List[Any] # Using Any to avoid importing JobMatchResult here
    top_5_candidates: List[Any]

    # For user interaction
    user_action_required: bool # Flag to indicate main loop needs to handle user input
    user_message: Optional[str] # Message to display to the user

    

    # Workflow Management
    workflow_status: str # Current status (e.g., "INITIALIZED", "JD_PARSED", "INDEXING_COMPLETE", "INDEXING_COMPLETE_NO_NEW_RESUMES", "INDEXING_COMPLETE_NO_RESUMES", "EVAL_PLAN_PENDING_REVIEW", "EVAL_PLAN_APPROVED", "COMPLETED", "FAILED")
    error_log: List[str] # Log of errors encountered
    processing_stats: Dict[str, Any] # Dictionary for timing, counts, etc.


# In src/models.py
class StartJobResponse(BaseModel):
    job_id: str
    message: str
    evaluation_plan: Optional[EvaluationPlan]

class JobActionRequest(BaseModel):
    action: str = Field(..., description="Action to perform: 'approve', 'modify', or 'reject'.")
    evaluation_plan: Optional[EvaluationPlan] = Field(None, description="The new evaluation plan, required if action is 'modify'.")

class JDSummaryAndSkills(BaseModel):
    """A model to hold a concise summary and key skills for embedding purposes."""
    summary: str = Field(description="A concise summary of the job description, focusing on the core role and responsibilities.")
    skills: List[str] = Field(description="A list of the most important technical and soft skills mentioned in the job description.")



class UploadResumeResponse(BaseModel):
    filename: str
    message: str
    destination_path: str

class GenericSuccessResponse(BaseModel):
    message: str

# Chat-related models
class ChatMessage(BaseModel):
    """Represents a single chat message in the conversation history."""
    role: str = Field(..., description="Role of the message sender: 'user' or 'assistant'")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    """Request model for chat functionality."""
    job_id: str = Field(..., description="The job ID to get context from")
    query: str = Field(..., description="User's question or query")
    chat_history: List[ChatMessage] = Field(default=[], description="Previous conversation history")

class ChatResponse(BaseModel):
    """Response model for chat functionality."""
    response: str = Field(..., description="AI assistant's response")
    job_id: str = Field(..., description="The job ID this response is for")
    context_used: int = Field(..., description="Number of resumes used for context")

# Resume Management Models
class DeleteResumesRequest(BaseModel):
    """Request model for deleting resumes from the database."""
    resume_ids: List[str] = Field(..., description="List of resume IDs to delete")

class ResumeSearchResult(BaseModel):
    """Result model for resume search operations."""
    id: str
    filename: str
    name: Optional[str] = None  # Candidate name, if extracted and available
    file_path: Optional[str] = None  # Path to the original file

class ResumeSearchResponse(BaseModel):
    """Response model for resume search operations."""
    message: str
    results: List[ResumeSearchResult]
    total_found: int

class ResumeSearchRequest(BaseModel):
    """Request model for searching resumes."""
    query: str = Field(..., description="Search query string for resume filenames or names.")
