import asyncio
from typing import Dict, Any, List, Optional, TypedDict, Tuple
import logging
import os
from dotenv import load_dotenv
import numpy as np
import json
import hashlib
import pickle
from src.models import ResumeMatcherState

from src.graph import get_workflow
from src.user_interface import (
    display_criteria, add_criterion, modify_criterion,
    delete_criterion, save_evaluation_plan
)
from src.llm_handlers import get_llm_model # For initial API key check

# Load environment variables
load_dotenv()

# Configure logging for main.py
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Function to handle user criteria management ---
async def handle_criteria_management(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handles the interactive user loop for managing evaluation criteria."""
    # Ensure evaluation_plan exists before starting the loop
    current_evaluation_plan = state.get("evaluation_plan")
    if not current_evaluation_plan:
        logger.error("Error: handle_criteria_management called but no evaluation plan found in state.")
        state["workflow_status"] = "FAILED" # Set state to failed
        return state # Return state to signal failure


    print(f"\n{state.get('user_message', 'Review the generated evaluation plan.')}")

    while True:
        display_criteria(current_evaluation_plan)
        print("\n--- Edit Criteria Menu ---")
        print("1. Add a new criterion")
        print("2. Modify an existing criterion")
        print("3. Delete a criterion")
        print("4. Approve and Continue Workflow")
        print("5. Save and Exit Criteria Menu (Back to Main Menu)")

        choice = input("Enter your choice (1-5): ").strip()

        if choice == '1':
            await add_criterion(current_evaluation_plan) # Await the async function
        elif choice == '2':
            await modify_criterion(current_evaluation_plan) # Await the async function
        elif choice == '3':
            delete_criterion(current_evaluation_plan)
        elif choice == '4':
            print("\n--- ▶️ Approving and Continuing Workflow ---")
            state["evaluation_plan"] = current_evaluation_plan # Ensure updated plan is in state
            state["user_action_required"] = False # Clear the user action flag
            state["user_message"] = None # Clear user message
            state["workflow_status"] = "EVAL_PLAN_APPROVED" # Set status to approved
            return state # Return the updated state
        elif choice == '5':
            save_evaluation_plan(current_evaluation_plan)
            state["user_action_required"] = False # Clear user action flag
            state["user_message"] = None # Clear user message
            state["workflow_status"] = "CRITERIA_MENU_EXITED" # Indicate exit status
            return state
        else:
            print("Invalid choice. Please enter a number from 1 to 5.")


async def main():
    """Main function to run the resume matching workflow."""
    print("--- 🚀 Resume Matcher Workflow Tool 🚀 ---")

    # Initial API key check
    try:
        get_llm_model()
        logger.info("✅ OpenAI API key appears valid.")
    except ValueError as e:
        logger.error(f"❌ API Key Error: {e}")
        logger.error("Please ensure OPENAI_API_KEY is set correctly in your .env file.")
        return

    # Define default paths
    default_jd_path = "jd/JD_SAP FICO Functional Lead_architect.docx"
    default_resumes_dir = "data/resumes"

    jd_path = input(f"Enter the path to the Job Description file (default: {default_jd_path}): ").strip()
    if not jd_path:
        jd_path = default_jd_path

    resumes_dir = input(f"Enter the path to the directory containing resumes (default: {default_resumes_dir}): ").strip()
    if not resumes_dir:
        resumes_dir = default_resumes_dir

    # Validate input paths exist (only for JD file, directory creation/check is in ingestion)
    if not os.path.exists(jd_path):
        logger.error(f"❌ Error: Job Description file not found at '{jd_path}'. Please check the path.")
        return
    # The resume directory existence check is moved into the ingestion agent

    # --- JD Caching Logic ---
    # Read JD file content and hash it
    with open(jd_path, "rb") as f:
        jd_bytes = f.read()
    jd_hash = hashlib.sha256(jd_bytes).hexdigest()
    cache_dir = "jd_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{jd_hash}.pkl")

    cached_jd = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                loaded_jd = pickle.load(f)
            # Only use cache if both jd_text and evaluation_plan are present and not None
            if loaded_jd.get("jd_text") and loaded_jd.get("evaluation_plan"):
                cached_jd = loaded_jd
                logger.info(f"Loaded JD and evaluation plan from cache: {cache_path}")
            else:
                logger.warning(f"JD cache at {cache_path} is incomplete (missing evaluation_plan or jd_text). Ignoring cache.")
        except Exception as e:
            logger.warning(f"Failed to load JD cache: {e}")

    # Initialize state dictionary for LangGraph
    app_state: ResumeMatcherState = {
        "jd_file_path": jd_path,
        "resume_directory": resumes_dir,
        "workflow_status": "INITIALIZED",
        "user_action_required": False,
        "user_message": None,
        "evaluation_plan": None,
        "parsed_jd": None,
        "jd_embedding": None,
        "all_raw_resumes": [],
        "newly_profiled_resumes": [],
        "profiled_resumes": [],
        "retrieved_candidates": [],
        "triaged_resumes": [],
        "top_5_candidates": [],
        "error_log": [],
        "processing_stats": {}
    }

    # If cache is found, populate state with cached values
    if cached_jd:
        app_state["jd_text"] = cached_jd.get("jd_text", "")
        app_state["evaluation_plan"] = cached_jd.get("evaluation_plan")
        app_state["parsed_jd"] = cached_jd.get("parsed_jd")
        logger.info("Using cached JD and evaluation plan. Skipping JD parsing.")
        app_state["workflow_status"] = "JD_PARSED"
    else:
        # Not cached, will be parsed and cached after JD parsing agent
        with open(jd_path, "r", encoding="utf-8", errors="ignore") as f:
            app_state["jd_text"] = f.read()

    # Pass ResumeMatcherState to get_workflow
    app = get_workflow(ResumeMatcherState)

    # --- Main Application Loop ---
    # The loop will run until a terminal status is reached or the user exits the criteria menu.
    while True:
        try:
            # LangGraph will run from the current state until it hits a router returning END
            # or completes its path to a final END node.
            logger.info(f"Starting/resuming graph execution from status: {app_state.get('workflow_status')}")

            # The astream iterator yields events as nodes complete.
            # The loop continues until the graph run stops (due to END or exception).
            # Reverting to default stream_mode="events" which yields dicts {node_name: state_update}
            async for event in app.astream(app_state, {"recursion_limit": 50}):
                # Update state with each event yield from the graph
                # In "events" mode, event is a dictionary {node_name: state_update} or {"__end__": final_state}
                for key, value in event.items():
                    if key != "__end__":
                        # Ensure we update the state dictionary with the changes from the agent
                        app_state.update(value)
                        # Log agent completion and new status
                        logger.info(f"--- Agent Completed: {key} --- New Status: {app_state.get('workflow_status')}")

                    # Check for immediate failure within the astream loop
                    if app_state.get("workflow_status") == "FAILED":
                        logger.error(f"Workflow failed in agent {key if key != '__end__' else 'unknown'}.")
                        break # Break the inner astream loop

                # If the inner loop broke (e.g., due to FAILED), break the outer one too
                if app_state.get("workflow_status") == "FAILED":
                     break # Break the outer while loop

                # If the graph reached its final END node (COMPLETED or COMPLETED_EMPTY_RESULTS)
                # The status check below the astream loop will handle this.
                # We just need to break the inner loop here if the graph explicitly ended.
                # The "__end__" key is the signal for termination in "events" mode.
                if "__end__" in event:
                    final_state_at_end = event["__end__"] # Access the final state from the END event value
                    app_state.update(final_state_at_end) # Update state one last time with final state from END event
                    logger.info(f"Graph run finished at __end__. Final status for this run: {app_state.get('workflow_status')}")
                    break # Break the inner astream loop


            # --- Logic AFTER the astream loop finishes ---
            # Check the final status AFTER the inner astream loop has completed its run.
            final_workflow_status = app_state.get("workflow_status")
            logger.info(f"astream loop finished its run. Evaluating final status: {final_workflow_status}")


            # Case 1: Graph stopped to request user review (INDEXING_COMPLETE statuses triggered END route)
            if final_workflow_status in ["INDEXING_COMPLETE", "INDEXING_COMPLETE_NO_NEW_RESUMES", "INDEXING_COMPLETE_NO_RESUMES"]:
                logger.info("Indexing completed (with or without new resumes). User action required to review evaluation plan.")
                app_state["user_action_required"] = True
                app_state["user_message"] = "Review and approve the generated evaluation plan before proceeding."

                updated_state_from_user = await handle_criteria_management(app_state)
                app_state.update(updated_state_from_user)

                if app_state.get("workflow_status") == "EVAL_PLAN_APPROVED":
                    logger.info("User approved evaluation plan. Outer loop will re-run astream with updated state.")
                    continue
                elif app_state.get("workflow_status") == "CRITERIA_MENU_EXITED":
                    logger.info("User exited criteria management menu. Stopping workflow.")
                    break
                elif app_state.get("workflow_status") == "FAILED":
                    logger.error("Workflow failed during user criteria management.")
                    break
                else:
                    logger.error(f"Unexpected workflow status after criteria management: {app_state.get('workflow_status')}. Stopping workflow.")
                    break

            # After JD parsing, if not cached, save to cache (only if both jd_text and evaluation_plan are present)
            if not cached_jd and app_state.get("evaluation_plan") and app_state.get("jd_text"):
                try:
                    if app_state["evaluation_plan"] is not None and app_state["jd_text"]:
                        with open(cache_path, "wb") as f:
                            pickle.dump({
                                "jd_text": app_state["jd_text"],
                                "evaluation_plan": app_state["evaluation_plan"],
                                "parsed_jd": app_state.get("parsed_jd")
                            }, f)
                        logger.info(f"Cached JD and evaluation plan to {cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to write JD cache: {e}")

            # Case 2: Graph reached a final terminal state (routed directly to END)
            elif final_workflow_status in ["COMPLETED", "COMPLETED_EMPTY_RESULTS"]:
                 print("\n--- ✅ Workflow Complete ---")
                 break # Exit the outer while True loop

            # Case 3: Graph terminated with a FAILED status (set by an agent)
            elif final_workflow_status == "FAILED":
                 logger.error("Workflow terminated with FAILED status.")
                 break # Exit the outer while True loop

             # Case 4: Graph stopped due to no resumes found at earlier stages (redundant check but safe)
            elif final_workflow_status in ["JD_INGESTED_NO_RESUMES_DIR", "JD_INGESTED_NO_RESUMES_FOUND", "RESUMES_PROFILED_EMPTY", "TRIAGE_COMPLETE_NO_RESUMES", "TRIAGE_COMPLETE_FAISS_EMPTY", "TRIAGE_COMPLETE_NO_MATCHES"]:
                 logger.warning(f"Workflow completed with status indicating no resumes processed or found: {final_workflow_status}. Exiting.")
                 # Optionally print a message about no resumes/matches found - messages are already logged in the agents
                 break # Exit the outer while True loop


            # Case 5: If the astream loop finished but the status is not one that triggers
            # user review or indicates completion/failure, something is unexpected.
            # This might happen if the graph paths don't correctly lead to END
            # or a state handled above. Add a safety break.
            else:
                 logger.warning(f"Workflow loop finished in unexpected status: {final_workflow_status}. Exiting.")
                 break

        except Exception as e:
            logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
            # Attempt to set state to FAILED before breaking
            app_state["workflow_status"] = "FAILED"
            app_state["error_log"].append(f"Critical error in main loop: {str(e)}")
            break # Exit the outer while True loop

    # --- End of Main Loop ---
    logger.info("Application finished.")
    # Optionally, print final state or results if needed
    # print("\nFinal Workflow State:")
    # print(json.dumps(app_state, indent=2, default=str)) # Use default=str for numpy arrays etc.


if __name__ == "__main__":
    asyncio.run(main())