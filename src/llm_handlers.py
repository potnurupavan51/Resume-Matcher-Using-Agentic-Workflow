import json
import os
import traceback
import uuid
from typing import Optional, List, Dict, Any, Literal
from dotenv import load_dotenv
import logging
import numpy as np
import asyncio # Import asyncio for concurrent tasks

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError
from langchain_text_splitters import RecursiveCharacterTextSplitter # Import the splitter

# Import all necessary models from your models file
from src.models import (
    JDParser, EvaluationPlan, Criterion,
    ResumeRecord, Skill, ExperienceRequirement, EducationRequirement,
    LlmEvaluationCriterion, LlmJDExtractionModel, JobMatchResult,JDSummaryAndSkills  # Ensure JobMatchResult is imported
)
# Import normalize_weights from the new utils file
from src.utils import normalize_weights

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# --- Configuration Constants ---
LLM_MODEL = "gpt-4o-mini" # Changed to a more cost-effective model for general tasks
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
DEFAULT_TEMPERATURE = 0.1
HIGH_TEMPERATURE = 0.0 # For deterministic tasks like evaluation scoring

# Heuristic for estimating resume pages based on character count
CHARS_PER_ESTIMATED_PAGE = 2000 # Roughly estimate 2000 characters per page

# Target length for the final summary (in estimated pages)
TARGET_SUMMARY_PAGES = 4
TARGET_SUMMARY_CHARS = TARGET_SUMMARY_PAGES * CHARS_PER_ESTIMATED_PAGE


# --- LLM Initialization ---
def get_llm_model(temperature: float = DEFAULT_TEMPERATURE):
    """Initializes and returns the ChatOpenAI LLM."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return ChatOpenAI(temperature=temperature, model=LLM_MODEL, api_key=api_key)

def get_embeddings_model():
    """Initializes and returns the OpenAIEmbeddings model."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, api_key=api_key)

def get_embedding_dimension():
    """Returns the dimension of the embedding model."""
    if EMBEDDING_MODEL_NAME == "text-embedding-3-small":
        return 1536
    return 1536 # Default fallback


async def get_sentence_embedding(text: str) -> np.ndarray:
    """Creates a vector embedding for a single piece of text asynchronously."""
    if not text:
        logger.warning("Attempted to embed empty text.")
        return np.array([]) # Return empty array for empty text
    embeddings_model = get_embeddings_model()
    try:
        embedding = await embeddings_model.aembed_query(text)
        return np.array(embedding).astype('float32')
    except Exception as e:
        logger.error(f"Error getting embedding for text: {e}. Text: '{text[:100]}'...", exc_info=True)
        return np.array([])

# --- Helper for Recursive Summarization ---

async def summarize_text_chunk(text: str) -> str:
    """Uses LLM to summarize a single chunk of text."""
    llm = get_llm_model(temperature=DEFAULT_TEMPERATURE) # Use default temperature for summarization
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional resume summarizer. Summarize the following text chunk from a resume. Focus on key skills, experience highlights, and achievements. Keep the summary concise and informative."),
        ("human", "{text}")
    ])
    chain = prompt | llm # No structured output needed, just text
    try:
        response = await chain.ainvoke({"text": text})
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error summarizing text chunk: {e}", exc_info=True)
        return "" # Return empty string on failure

async def recursive_summarize_resume(text: str, target_chars: int = TARGET_SUMMARY_CHARS) -> str:
    """Recursively summarizes a long resume text until it reaches the target character length."""
    logger.info(f"Starting recursive summarization. Target chars: {target_chars}")

    # Use RecursiveCharacterTextSplitter
    # Chunk size should be large enough for context but fit in LLM window.
    # Let's use a chunk size slightly larger than an estimated page, maybe 3000, with overlap.
    # Or, use a larger chunk size suitable for LLM context (e.g., 8000-10000) for initial splits.
    # Let's try chunks designed to be summarized into a smaller size.
    # A chunk of ~4000-6000 chars might summarize well into a smaller chunk.
    # Let's use 4000 with overlap as a starting point.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, # Characters per chunk for splitting
        chunk_overlap=200, # Overlap between chunks
        length_function=len, # Use character length
        # separators=["\n\n", "\n", " ", ""] # Default separators are usually fine
    )

    current_texts = splitter.split_text(text)
    logger.info(f"Initial split into {len(current_texts)} chunks.")

    # Summarize chunks until the total length is below target or only one summary remains
    while len(current_texts) > 1 or sum(len(t) for t in current_texts) > target_chars * 1.1: # Add a buffer for overshoot
        if not current_texts:
            logger.warning("Recursive summarization process resulted in empty text list.")
            return ""

        logger.info(f"Summarizing {len(current_texts)} chunks. Combined length: {sum(len(t) for t in current_texts)}")

        # Create tasks to summarize each current text chunk concurrently
        summary_tasks = [summarize_text_chunk(chunk) for chunk in current_texts]
        summaries = await asyncio.gather(*summary_tasks)

        # Filter out empty summaries from failed calls
        summaries = [s for s in summaries if s]

        if not summaries:
            logger.error("All summarization tasks failed. Cannot proceed with recursive summarization.")
            return "" # Return empty on complete failure

        # If only one summary remains (e.g., from the last step), and it's within target, return it.
        if len(summaries) == 1 and len(summaries[0]) <= target_chars * 1.1:
            logger.info(f"Recursive summarization complete: 1 final summary within target length ({len(summaries[0])} chars).")
            return summaries[0]
            
        # Combine summaries if needed for the next recursion level, or just use the summaries directly
        # If there are multiple summaries, treat them as the input for the next splitting/summarization cycle.
        # Join them with a separator that helps the splitter/summarizer understand they are distinct.
        combined_summaries_text = "\n\n---\n\n".join(summaries) # Use a clear separator

        # Split the combined summaries for the next round of summarization
        current_texts = splitter.split_text(combined_summaries_text)


    # After the loop, current_texts should contain a single item or items
    # whose combined length is <= target_chars. Join them for the final summary.
    final_summary = " ".join(current_texts).strip() # Use space for joining final parts
    logger.info(f"Recursive summarization loop finished. Final summary length: {len(final_summary)} chars.")
    return final_summary


# --- Main JD Parsing Function (NOW ASYNC) ---
async def extract_jd_criteria_with_langchain_safe(jd_text: str) -> Optional[EvaluationPlan]:
    """Analyzes job description text asynchronously to create a structured EvaluationPlan."""
    if not jd_text or not jd_text.strip():
        logger.warning("Job description text is empty or whitespace only.")
        return None

    llm = get_llm_model(temperature=DEFAULT_TEMPERATURE)

    # Prompt for LlmJDExtractionModel
    # prompt_for_llm_extraction = ChatPromptTemplate.from_messages([
    #     ("system", """You are an elite HR strategist tasked with translating a Job Description (JD)
    #     into structured and weighted evaluation criteria for candidate selection.
         


    #              Think beyond explicit requirements and infer essential traits from responsibilities and desired outcomes.

    #     **CRITICAL INITIAL CHECK - VERY IMPORTANT:**
    #     **If the provided Job Description text is empty, consists of only a few words (e.g., less than 50 significant words), or is clearly just a title without substantial content, you MUST NOT attempt to generate any criteria.** Instead, you must return a specific JSON object indicating insufficient data.

    #     **FIRST, ANALYZE THE JOB DESCRIPTION TO DETERMINE ITS PRIMARY FOCUS:**
    #     Is it predominantly a 'MANAGEMENT' role (e.g., leadership, strategy, team development, project oversight, stakeholder management)
    #     or a 'TECHNICAL' role (e.g., specific programming languages, software tools, system architecture, data analysis, hands-on development)?
    #     This analysis will guide the emphasis of your extracted criteria.

    #     🔹 Your goal is to extract **clear, distinct, and specific** evaluation criteria across categories like:
    #     - technical skill
    #     - education
    #     - experience
    #     - soft skill
    #     - domain expertise
    #     - certification

    #     🎯 Each criterion must include:
    #     - **category**: the grouping type
    #     - **criteria**: the specific requirement
    #     - **weightage**: a number from 1–5 (5 = most critical)
    #     - **evaluation_guidelines**: a brief tip on how to evaluate this criterion

    #     🚨 Pay close attention to:
    #     - Technical or functional **skills** (don’t miss those hidden in bullets)
    #     - **Responsibilities** section, which often embeds soft skills and tools
    #     - Avoid duplication or vague criteria.
         

    #       **CRITICAL INSTRUCTIONS - ADHERE STRICTLY:**
    #     1.  **Contextual Prioritization**:
    #         * **IF 'MANAGEMENT' ROLE**: Dedicate significant focus and higher weights to criteria under 'experience' (especially leadership/project management), 'soft skill' (e.g., strategic thinking, team building, conflict resolution), and relevant 'domain expertise' related to organizational oversight. Extract nuanced management responsibilities as distinct criteria.
    #         * **IF 'TECHNICAL' ROLE**: Prioritize and assign higher weights to criteria under 'technical skill' (e.g., specific programming languages, cloud platforms, data tools) and 'experience' (e.g., hands-on coding, architecture design, system integration). Extract granular technical requirements.
    #         * For all roles, ensure all relevant categories are covered, but adjust the depth and weighting based on the primary focus.
    #     2.  **Thoroughness**: Scrutinize **every single sentence and bullet point** in the JD. Skills, tools, and traits are often embedded within responsibilities.
    #     3.  **Specificity over Generality**: Instead of "Good communication", aim for "Excellent written communication for client reports". Instead of "Software development", aim for "Proficiency in Java Spring Boot".
    #     4.  **Actionable Evaluation**: Ensure `evaluation_guidelines` are practical and suggest *how* to verify the criterion (e.g., "Portfolio review", "Technical assessment", "Interview questions").
    #     5.  **No Duplication**: Ensure each extracted criterion is distinct. Combine closely related concepts under a single, well-defined criterion if appropriate.
    #     6.  **Relevance**: Every criterion must be directly derivable from or strongly implied by the Job Description. Do not invent criteria.

    #     📊 You will return a list of **evaluation_criteria**.
    #     We will normalize the total to **100** later.

    #     ### Job Description:
    #     ```
    #     {jd_text}
    #     ```
    #     """),
    #     ("human", "Generate ONLY the JSON object containing the list of evaluation criteria under the key 'evaluation_criteria'. Ensure all fields for each criterion are thoroughly filled out.")
    # ])

    prompt_for_llm_extraction = ChatPromptTemplate.from_messages([
    ("system", """You are an elite HR strategist tasked with translating a Job Description (JD)
    into structured and weighted evaluation criteria for candidate selection.
     
    Think beyond explicit requirements and infer essential traits from responsibilities and desired outcomes.

    **CRITICAL INITIAL CHECK - VERY IMPORTANT:**
    **If the provided Job Description text is empty, consists of only a few words (e.g., less than 50 significant words), or is clearly just a title without substantial content, you MUST NOT attempt to generate any criteria.** Instead, you must return a specific JSON object indicating insufficient data.

    **FIRST, ANALYZE THE JOB DESCRIPTION TO DETERMINE ITS PRIMARY FOCUS:**
    Is it predominantly a 'MANAGEMENT' role (e.g., leadership, strategy, team development, project oversight, stakeholder management)
    or a 'TECHNICAL' role (e.g., specific programming languages, software tools, system architecture, data analysis, hands-on development)?
    This analysis will guide the emphasis of your extracted criteria.

    🔹 Your goal is to extract **clear, distinct, and specific** evaluation criteria across categories like:
    - technical skill
    - education
    - experience
    - soft skill
    - domain expertise
    - certification

    🔢 **WEIGHTAGE SCALE – Use This to Decide the Weight (1–5):**

    Assign weightage to each criterion based on its **impact on candidate success** in the role. Use the following rubric:

    - **5 – Mission Critical**: The role *cannot be performed* without this. Essential for core job performance (e.g., must-have technical skill, leadership experience, domain knowledge explicitly required).
    - **4 – Highly Important**: Strongly contributes to success; a major differentiator between average and excellent candidates.
    - **3 – Important but Not Exclusive**: Adds value but can be compensated with other strengths or trained post-hire.
    - **2 – Nice to Have**: Useful in some scenarios or team settings but not consistently required.
    - **1 – Peripheral**: Only marginally related; has minimal impact on performance in this specific role.

    💡 When assigning weightage, always consider:
    - The job's **primary focus** (Technical or Management)
    - Whether the criterion is **explicitly stated**, **strongly implied**, or **contextually inferred**
    - How critical the criterion is for the **first 90 days** of the role

    🎯 Each criterion must include:
    - **category**: the grouping type
    - **criteria**: the specific requirement
    - **weightage**: a number from 1–5 (use the rubric above)
    - **evaluation_guidelines**: a brief tip on how to evaluate this criterion

    🚨 Pay close attention to:
    - Technical or functional **skills** (don’t miss those hidden in bullets)
    - **Responsibilities** section, which often embeds soft skills and tools
    - Avoid duplication or vague criteria.
     
    **CRITICAL INSTRUCTIONS - ADHERE STRICTLY:**
    1.  **Contextual Prioritization**:
        * **IF 'MANAGEMENT' ROLE**: Dedicate significant focus and higher weights to criteria under 'experience' (especially leadership/project management), 'soft skill' (e.g., strategic thinking, team building, conflict resolution), and relevant 'domain expertise' related to organizational oversight. Extract nuanced management responsibilities as distinct criteria.
        * **IF 'TECHNICAL' ROLE**: Prioritize and assign higher weights to criteria under 'technical skill' (e.g., specific programming languages, cloud platforms, data tools) and 'experience' (e.g., hands-on coding, architecture design, system integration). Extract granular technical requirements.
        * For all roles, ensure all relevant categories are covered, but adjust the depth and weighting based on the primary focus.
    2.  **Thoroughness**: Scrutinize **every single sentence and bullet point** in the JD. Skills, tools, and traits are often embedded within responsibilities.
    3.  **Specificity over Generality**: Instead of "Good communication", aim for "Excellent written communication for client reports". Instead of "Software development", aim for "Proficiency in Java Spring Boot".
    4.  **Actionable Evaluation**: Ensure `evaluation_guidelines` are practical and suggest *how* to verify the criterion (e.g., "Portfolio review", "Technical assessment", "Interview questions").
    5.  **No Duplication**: Ensure each extracted criterion is distinct. Combine closely related concepts under a single, well-defined criterion if appropriate.
    6.  **Relevance**: Every criterion must be directly derivable from or strongly implied by the Job Description. Do not invent criteria.

    📊 You will return a list of **evaluation_criteria**.
    We will normalize the total to **100** later.

    ### Job Description:
    ```
    {jd_text}
    ```
    """),
    ("human", "Generate ONLY the JSON object containing the list of evaluation criteria under the key 'evaluation_criteria'. Ensure all fields for each criterion are thoroughly filled out.")
])














    

    # Chain for LlmJDExtractionModel
    llm_extraction_chain = prompt_for_llm_extraction | llm.with_structured_output(LlmJDExtractionModel)

    try:
        # First, extract the raw criteria using LlmJDExtractionModel
        extracted_llm_criteria: LlmJDExtractionModel = await llm_extraction_chain.ainvoke({"jd_text": jd_text})

        # Convert LlmEvaluationCriterion objects to Criterion objects for normalization and EvaluationPlan
        initial_criteria_objects = [
            Criterion(
                id=str(uuid.uuid4()), # Generate a new UUID for each Criterion
                category=c.category,
                criteria=c.criteria,
                weightage=c.weightage, # Use the relative weight initially
                evaluation_guidelines=c.evaluation_guidelines
            ) for c in extracted_llm_criteria.evaluation_criteria
        ]

        # Normalize the weights IN-PLACE using the Criterion objects
        normalize_weights(initial_criteria_objects)

        # A quick LLM call to get job_title and overall_summary for EvaluationPlan
        # Use the JDParser model which has job_title and overall_summary fields
        temp_jd_parser_llm = get_llm_model(temperature=0).with_structured_output(JDParser) # Lower temp for extraction
        temp_jd_parser_prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract only the job title and a concise overall summary from the provided job description."),
            ("human", "{jd_text}")
        ])
        temp_jd_parser_runnable = temp_jd_parser_prompt | temp_jd_parser_llm
        inferred_jd_data: Optional[JDParser] = None
        try:
            inferred_jd_data = await temp_jd_parser_runnable.ainvoke({"jd_text": jd_text})
        except Exception as e:
            logger.warning(f"Could not infer job_title/overall_summary for EvaluationPlan: {e}. Using 'N/A'.", exc_info=True)

        job_title = inferred_jd_data.job_title if inferred_jd_data and inferred_jd_data.job_title else "N/A"
        overall_summary = inferred_jd_data.overall_summary if inferred_jd_data and inferred_jd_data.overall_summary else "N/A"


        # Create and return the final EvaluationPlan
        evaluation_plan = EvaluationPlan(
            job_title=job_title,
            overall_summary=overall_summary,
            criteria=initial_criteria_objects # Use the list of Criterion objects with normalized weights
        )
        logger.info("✅ Evaluation Plan generated successfully.")
        return evaluation_plan

    except ValidationError as e:
        logger.error(f"Pydantic validation error during JD criteria extraction: {e}", exc_info=True)
        return None

    except Exception as e:
        logger.error(f"An unexpected error occurred during JD criteria extraction: {e}", exc_info=True)
        return None

# --- Function to get weight for a new criterion (NOW ASYNC) ---
async def get_llm_weight_for_new_criterion(new_criterion_data: Dict[str, Any], existing_criteria: List[Criterion]) -> Optional[int]:
    """Uses LLM to assign a relative weight (1-5) to a new criterion asynchronously."""
    llm = get_llm_model()

    class WeightAssignment(BaseModel):
        weightage: Literal[1, 2, 3, 4, 5]

    # Convert existing criteria objects to dictionaries for JSON serialization in the prompt
    existing_criteria_dicts = [c.model_dump() for c in existing_criteria]
    context_str = json.dumps(existing_criteria_dicts, indent=2) if existing_criteria_dicts else "No existing criteria."

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert HR analyst. Assign a relative importance weightage (1-5) to the new job criterion based on the existing criteria. 1=low priority, 5=critical. Return only the weightage value as a JSON object with the key 'weightage'."),
        ("human", f"**Existing Criteria:**\n```json\n{context_str}\n```\n\n**New Criterion to Weight:**\n- Category: \"{new_criterion_data.get('category')}\"\n- Criteria: \"{new_criterion_data.get('criteria')}\"\n- Guidance: \"{new_criterion_data.get('evaluation_guidelines', 'N/A')}\"") # Ensure guidance is passed
    ])
    chain = prompt | llm.with_structured_output(WeightAssignment)

    try:
        response = await chain.ainvoke({}) # Invoke with empty dict is fine here as prompt has no explicit variables
        logger.info(f"AI assigned a relative weight of: {response.weightage}")
        return response.weightage
    except Exception as e:
        logger.error(f"Error assigning weight with LLM: {e}. Defaulting to 3.", exc_info=True)
        return 3 # Default weight if LLM fails

# --- Function to parse resume text (NOW ASYNC) ---
async def extract_resume_parser_data_with_langchain_safe(resume_text: str) -> Optional[Any]: # Changed return type hint
    """
    Analyzes resume text asynchronously using an LLM to extract structured data.
    Performs recursive summarization for long resumes.
    """
    if not resume_text or not resume_text.strip():
        logger.warning("Resume text is empty or whitespace only. Skipping LLM processing.")
        return None

    llm = get_llm_model(temperature=DEFAULT_TEMPERATURE)

    # Import ResumeParser here, where it's used
    from src.models import ResumeParser

    # --- Step 1: Extract all fields (including an initial summary) from the full text ---
    # We will replace the summary later if the text is too long.
    initial_parse_prompt = f"""
    You are an expert resume parser. Analyze the following resume text and extract the information into the specified JSON schema.
    Be thorough and accurate. If a field is not present, omit it or set it to null/empty list as per the schema.

    **Instructions for JSON output:**
    - `name`: The full name of the candidate. If not clearly present, use null.
    - `summary`: A brief professional summary or career objective based on the text.
    - `skills`: Extract each skill as an object with a 'name'.
    - `previous_experience`: Extract each work experience. Each should be an object with 'title', 'company', 'start_date', 'end_date', and 'description'.
    - `education`: Extract each education entry. Each should be an object with 'degree', 'university', 'start_date', and 'end_date'.
    - `years_experience`: The total years of professional experience, if inferable.
    - `achievements`: A list of notable achievements or accomplishments.
    - `certifications`: A list of certification names.

    **Resume Text:**
    ```
    {resume_text}
    ```
    """
    initial_parse_chain = llm.with_structured_output(ResumeParser)

    try:
        logger.info("Performing initial LLM parse for resume fields and summary.")
        parsed_data: ResumeParser = await initial_parse_chain.ainvoke(initial_parse_prompt)

        # --- Step 2: Check resume length and perform recursive summarization if needed ---
        estimated_pages = len(resume_text) / CHARS_PER_ESTIMATED_PAGE

        if estimated_pages > TARGET_SUMMARY_PAGES:
            logger.info(f"Resume is long ({estimated_pages:.2f} estimated pages). Starting recursive summarization.")
            recursive_summary = await recursive_summarize_resume(resume_text, TARGET_SUMMARY_CHARS)

            # --- Step 3: Replace the summary in the parsed data ---
            parsed_data.summary = recursive_summary if recursive_summary else "Summary could not be generated due to processing error."
            logger.info(f"Replaced summary with recursive summary ({len(parsed_data.summary)} chars).")
        else:
            logger.info(f"Resume is within target length ({estimated_pages:.2f} estimated pages). Using initial summary.")
            # The initial summary generated by the first parse is kept.


        # Anonymize name after parsing if it was extracted by LLM (set to None as per requirement)
        if parsed_data.name:
             logger.info(f"Anonymizing name '{parsed_data.name}' from parsed_data.")
             parsed_data.name = None # Set to None to remove the name from parsed_data

        return parsed_data

    except ValidationError as e:
        logger.error(f"Pydantic validation error during resume parsing: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during resume processing: {e}", exc_info=True)
        return None


import logging
from typing import Any, Optional
from src.models import JobMatchResult, Criterion, EvaluationPlan, ResumeRecord
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.exceptions import ValidationError
# This is the correct import for the error
from pydantic import ValidationError


logger = logging.getLogger(__name__)

import logging
from typing import Any, Optional
from src.models import JobMatchResult, Criterion, EvaluationPlan, ResumeRecord
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

# (Keep the rest of llm_handlers.py as it is, just replace this one function)

logger = logging.getLogger(__name__)

async def evaluate_candidate_with_llm(
    resume_record: Any, # ResumeRecord object
    evaluation_plan: Any # EvaluationPlan object
) -> Optional[Any]: # JobMatchResult object
    """
    Evaluates a single candidate's resume against the detailed evaluation plan asynchronously.
    """
    # This prevents the 'AttributeError' by safely checking for the filename.
    filename = getattr(resume_record, 'filename', getattr(resume_record, 'source_filename', 'N/A'))
    logger.info(f"  - Performing detailed LLM evaluation for: {filename}")

    llm = get_llm_model(temperature=HIGH_TEMPERATURE)
    eval_chain = llm.with_structured_output(JobMatchResult)
    # Ensure evaluation_plan is a Pydantic model before calling model_dump_json
    from src.models import EvaluationPlan
    if isinstance(evaluation_plan, dict):
        evaluation_plan = EvaluationPlan(**evaluation_plan)
    evaluation_plan_json = evaluation_plan.model_dump_json(indent=2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert HR analyst. Your task is to evaluate a candidate's resume against a detailed, weighted evaluation plan for a specific job description.
        For each criterion in the plan, assign a score (0-10) based on the evidence in the resume.
        Calculate a weighted average score for each criterion based on its `weightage`.
        Provide a concise overall score (0-100) and a brief overall reasoning.
        Your reasoning must be directly related to the evidence found in the resume and the evaluation criteria.
        Return your analysis in the specified JSON format (`JobMatchResult`). Ensure all required fields are populated.
        The `criterion_scores` list in the output MUST contain an entry for EACH criterion provided in the `Evaluation Plan`,
        even if the score is 0. Ensure the `id`, `category`, `criteria`, `weightage`, and `evaluation_guidelines`
        from the original `Criterion` in the `Evaluation Plan` are copied exactly into the corresponding `Criterion` object
        within `criterion_scores`, along with the new `score` and `reason`."""),
        ("human", """
        **Evaluation Plan:**
        ```json
        {evaluation_plan_json}
        ```

        **Candidate's Resume Text:**
        ```
        {resume_text}
        ```

        Please provide your evaluation as a JSON object conforming to the JobMatchResult schema.
        """)
    ])

    chain = prompt | eval_chain

    try:
        result: JobMatchResult = await chain.ainvoke({
            "evaluation_plan_json": evaluation_plan_json,
            "resume_text": resume_record.original_text
        })

        result.resume_id = resume_record.id
        result.resume_filename = filename # Use the safe filename
        result.candidate_name = resume_record.name or (resume_record.parsed_data.name if resume_record.parsed_data else None)

        evaluated_criterion_ids = {c.id for c in result.criterion_scores}
        for plan_criterion in evaluation_plan.criteria:
            if plan_criterion.id not in evaluated_criterion_ids:
                result.criterion_scores.append(Criterion(
                    id=plan_criterion.id, category=plan_criterion.category, criteria=plan_criterion.criteria,
                    weightage=plan_criterion.weightage, evaluation_guidelines=plan_criterion.evaluation_guidelines,
                    score=0.0, reason="Not evaluated by LLM or no information found."
                ))

        return result

    except (ValidationError, Exception) as e:
        logger.error(f"LLM evaluation failed for {filename}: {e}", exc_info=True)
        # Instead of trying to access a non-existent 'state', re-raise the exception.
        # asyncio.gather in the calling agent will catch this and handle it.
        raise e
    
def get_jd_summary_for_embedding(evaluation_plan: Any) -> str:
    """Extracts a summary string from the evaluation plan for embedding."""
    # Import EvaluationPlan here, where it's used
    from src.models import EvaluationPlan
    if not evaluation_plan:
        return ""

    summary_parts = [evaluation_plan.overall_summary]
    # Add key criteria descriptions to the summary for embedding
    if evaluation_plan.criteria:
        for crit in evaluation_plan.criteria: # Take top 5 criteria as example
            summary_parts.append(f"{crit.category}: {crit.criteria}")

    return ". ".join(filter(None, summary_parts))






async def extract_jd_summary_and_skills(jd_text: str) -> Optional[JDSummaryAndSkills]:
    """
    Uses an LLM to extract a concise summary and a list of key skills from a JD.
    This is specifically for creating a high-quality embedding.
    """
    if not jd_text or not jd_text.strip():
        return None

    logger.info("Extracting dedicated summary and skills for embedding via LLM.")
    llm = get_llm_model(temperature=DEFAULT_TEMPERATURE)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert HR analyst. Your task is to process a job description and extract two key pieces of information for a semantic search embedding:
        1. A **detailed and well-written `summary` of the role, responsibilities, and ideal candidate profile.
        2. A list of the most critical `skills` (both technical and soft) required for the job.
        
        Return the output in the specified JSON format."""),
        ("human", """
        **Job Description Text:**
        ```
        {jd_text}
        ```
        
        Please provide your analysis as a JSON object conforming to the JDSummaryAndSkills schema.
        """)
    ])
    
    chain = prompt | llm.with_structured_output(JDSummaryAndSkills)

    try:
        result = await chain.ainvoke({"jd_text": jd_text})
        return result
    except Exception as e:
        logger.error(f"Failed to extract JD summary and skills for embedding: {e}", exc_info=True)
        return None