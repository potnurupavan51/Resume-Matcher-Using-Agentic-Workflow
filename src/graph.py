from langgraph.graph import StateGraph, END
from typing import Dict, Any
from src.models import ResumeMatcherState
from src.services import (
    file_ingestion_agent,
    jd_parsing_agent,
    resume_profiling_agent,
    embedding_and_indexing_agent,
    matching_and_triage_agent,
    final_selection_agent
)
import logging

logger = logging.getLogger(__name__)

def entry_router(state: ResumeMatcherState) -> str:
    """
    Entry router for the workflow. It directs the graph to the correct starting
    node based on the current workflow status.
    """
    status = state.get("workflow_status")
    logger.info(f"Entry router: Current workflow_status is '{status}'.")

    if status == "JD_PARSED":
        # This is for when a cached JD is selected from the start
        logger.info("Routing to 'resume_profiling' for cached JD.")
        return "resume_profiling"
    
    # --- FIX: Handle the case where the user has approved the plan ---
    if status == "EVAL_PLAN_APPROVED":
        # This is for resuming the workflow after user approval
        logger.info("Routing to 'matching_and_triage' after plan approval.")
        return "matching_and_triage"
    
    else:
        # Default entry point for all new jobs (status is 'INITIALIZED')
        logger.info("Routing to 'file_ingestion' for a new job.")
        return "file_ingestion"

def route_after_indexing_or_review(state: ResumeMatcherState) -> str:
    """
    Routes the workflow after resume indexing.
    This router's only job now is to pause the graph to wait for user input.
    The resumption is handled by the main entry_router.
    """
    workflow_status = state.get("workflow_status")
    logger.info(f"Routing after embedding_and_indexing. Current status: {workflow_status}")

    if workflow_status == "FAILED":
        return END

    # If indexing completed, the graph should always end here to await user action.
    if workflow_status in ["INDEXING_COMPLETE", "INDEXING_COMPLETE_NO_NEW_RESUMES", "INDEXING_COMPLETE_NO_RESUMES"]:
        logger.info(f"Indexing is complete. Routing to END to trigger user review.")
        return "request_review" # This key maps to the END node.

    # This router no longer needs to handle 'EVAL_PLAN_APPROVED' because the main
    # entry_router will handle it on the next run.
    else:
        logger.error(f"Unexpected status '{workflow_status}' for routing after embedding. Routing to END.")
        return END


def get_workflow(state_schema: type) -> StateGraph:
    """Builds and returns the compiled LangGraph workflow with an intelligent entry router."""
    workflow = StateGraph(state_schema)

    # --- Define Nodes ---
    workflow.add_node("file_ingestion", file_ingestion_agent)
    workflow.add_node("jd_parsing", jd_parsing_agent)
    workflow.add_node("resume_profiling", resume_profiling_agent)
    workflow.add_node("embedding_and_indexing", embedding_and_indexing_agent)
    workflow.add_node("matching_and_triage", matching_and_triage_agent)
    workflow.add_node("final_selection", final_selection_agent)

    # --- Set the single, intelligent conditional entry point ---
    workflow.set_conditional_entry_point(
        entry_router,
        {
            "file_ingestion": "file_ingestion",
            "resume_profiling": "resume_profiling",
            # --- FIX: Add the new routing path for approved plans ---
            "matching_and_triage": "matching_and_triage",
        }
    )

    # --- Define Edges (The flow of the graph) ---
    workflow.add_edge("file_ingestion", "jd_parsing")
    workflow.add_edge("jd_parsing", "resume_profiling")
    workflow.add_edge("resume_profiling", "embedding_and_indexing")

    # This conditional edge now only handles pausing the graph.
    workflow.add_conditional_edges(
        "embedding_and_indexing",
        route_after_indexing_or_review,
        {
            "request_review": END, # Pause for user review
            END: END
        }
    )

    # This is the second half of the workflow, which will now be correctly
    # entered after the user approves the plan.
    workflow.add_edge("matching_and_triage", "final_selection")
    workflow.add_edge("final_selection", END)

    app = workflow.compile()
    return app