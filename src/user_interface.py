import json
from typing import List, Dict, Any, Optional
import uuid
import asyncio # Import asyncio for await calls

# Corrected import from our unified models file
from src.models import Criterion, EvaluationPlan # Changed to src.models
# Import normalize_weights from the new utils file
from src.utils import normalize_weights
# Import the LLM handler for getting weight (needs to be async)
from src.llm_handlers import get_llm_weight_for_new_criterion

import logging
logger = logging.getLogger(__name__)

def display_criteria(evaluation_plan: EvaluationPlan):
    """Displays the list of evaluation criteria from the EvaluationPlan object."""
    criteria_list = evaluation_plan.criteria
    if not criteria_list:
        print("\nNo criteria to display yet.")
        return

    print("\n--- Current Evaluation Criteria ---")
    print(f"Job Title: {evaluation_plan.job_title}")
    print(f"Overall Summary: {evaluation_plan.overall_summary}")
    print("\nCriteria List:")
    for i, criterion in enumerate(criteria_list):
        print(f"\nCriterion {i+1}:")
        print(f"  Category: {criterion.category}")
        print(f"  Criteria: {criterion.criteria}")
        print(f"  Weightage: {criterion.weightage}")
        print(f"  Guidance: {criterion.evaluation_guidelines}")
    print("-----------------------------------")
    total_weight = sum(c.weightage for c in criteria_list)
    # --- FIX: Change the hardcoded target weight in the print statement ---
    print(f"TOTAL WEIGHTAGE: {int(total_weight)} / 100")
    # --- End Fix ---
    print("-----------------------------------\n")

def get_new_criterion_details_from_user() -> Optional[Dict[str, Any]]:
    """Gets user input for a new criterion's details (without weight)."""
    print("\n--- Enter New Criterion Details (AI will assign weight) ---")
    category = input("Enter Category (e.g., 'technical skill', 'experience'): ").strip()
    criteria_text = input("Enter Criteria text (e.g., 'Proficiency in Python'): ").strip()
    guidance = input("Enter Evaluation Guidance: ").strip()

    if not all([category, criteria_text, guidance]):
        print("⚠️ All fields are required. Aborting add.")
        return None

    return {
        "category": category,
        "criteria": criteria_text,
        "evaluation_guidelines": guidance
    }

async def add_criterion(evaluation_plan: EvaluationPlan): # Made async
    """
    Adds a new criterion to the evaluation plan using AI for weight assignment
    and re-normalizes all weights.
    """
    new_criterion_details = get_new_criterion_details_from_user()
    if not new_criterion_details:
        return
    
    # Await the async LLM function directly
    # Need to pass the list of Criterion objects to the LLM function
    relative_weight = await get_llm_weight_for_new_criterion(new_criterion_details, evaluation_plan.criteria)

    if relative_weight is None:
        print("⚠️ Could not get weight from AI. Aborting add.")
        return

    # Create a new Criterion object with the user details and the relative weight
    new_criterion = Criterion(
        id=str(uuid.uuid4()),
        category=new_criterion_details['category'],
        criteria=new_criterion_details['criteria'],
        weightage=int(relative_weight), # Use the relative weight for now
        evaluation_guidelines=new_criterion_details['evaluation_guidelines']
    )
    
    # Append the new Criterion object to the list
    evaluation_plan.criteria.append(new_criterion)
    
    # Re-normalize the entire list of criteria in-place (will normalize to 100 by default)
    normalize_weights(evaluation_plan.criteria)
    print("✅ Criterion added and all weights have been re-normalized.")


async def modify_criterion(evaluation_plan: EvaluationPlan): # Made async
    """Allows the user to modify an existing criterion and re-normalizes if weight changes."""
    if not evaluation_plan.criteria:
        print("No criteria to modify.")
        return

    display_criteria(evaluation_plan)
    
    idx = -1
    while True:
        try:
            idx_str = input("Enter the number of the criterion to modify: ").strip()
            idx = int(idx_str) - 1
            if 0 <= idx < len(evaluation_plan.criteria):
                break
            else:
                print("Invalid number.")
        except ValueError:
            print("Invalid input.")

    criterion_to_modify = evaluation_plan.criteria[idx]
    weight_changed = False

    print(f"\n--- Modifying Criterion {idx+1} ---")
    
    new_category = input(f"Category (Enter to keep '{criterion_to_modify.category}'): ").strip()
    if new_category: criterion_to_modify.category = new_category
    
    new_criteria_text = input(f"Criteria (Enter to keep '{criterion_to_modify.criteria}'): ").strip()
    if new_criteria_text: criterion_to_modify.criteria = new_criteria_text

    # Ask for a RELATIVE weight (1-5) to change its importance
    while True:
        new_weight_str = input(f"New RELATIVE weight (1-5, or Enter to keep current importance): ").strip()
        if not new_weight_str:
            break
        try:
            new_relative_weight = int(new_weight_str)
            if 1 <= new_relative_weight <= 5:
                # Update with the new relative weight; normalization will adjust absolute value
                criterion_to_modify.weightage = new_relative_weight
                weight_changed = True
                break
            else:
                print("Weight must be between 1 and 5.")
        except ValueError:
            print("Invalid input.")

    new_guidance = input(f"Guidance (Enter to keep '{criterion_to_modify.evaluation_guidelines}'): ").strip()
    if new_guidance: criterion_to_modify.evaluation_guidelines = new_guidance

    # If the relative weight was changed, re-normalize the entire list (will normalize to 100)
    if weight_changed:
        normalize_weights(evaluation_plan.criteria)
        print("✅ Criterion modified and all weights have been re-normalized.")
    else:
        print("✅ Criterion modified.")


def delete_criterion(evaluation_plan: EvaluationPlan):
    """Allows the user to delete a criterion and re-normalizes the remaining weights."""
    if not evaluation_plan.criteria:
        print("No criteria to delete.")
        return

    display_criteria(evaluation_plan)
    idx = -1
    while True:
        try:
            idx_str = input("Enter the number of the criterion to delete: ").strip()
            idx = int(idx_str) - 1
            if 0 <= idx < len(evaluation_plan.criteria):
                break
            else:
                print("Invalid number.")
        except ValueError:
            print("Invalid input.")

    deleted_criterion = evaluation_plan.criteria.pop(idx)
    print(f"✅ Criterion '{deleted_criterion.criteria}' deleted.")

    # Re-normalize the remaining criteria if any are left (will normalize to 100)
    if evaluation_plan.criteria:
        normalize_weights(evaluation_plan.criteria)
        print("✅ Remaining weights have been re-normalized.")

def save_evaluation_plan(evaluation_plan: EvaluationPlan):
    """Saves the current evaluation plan to a JSON file."""
    if not evaluation_plan or not evaluation_plan.criteria:
        print("⚠️ No criteria to save.")
        return

    output_filename = input("Enter filename to save evaluation plan (e.g., evaluation_plan.json): ").strip()
    if not output_filename:
        output_filename = "evaluation_plan.json" # Default filename

    try:
        # Use the Pydantic model's built-in JSON serialization for clean output
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(evaluation_plan.model_dump_json(indent=2))
        print(f"✅ Evaluation plan saved successfully to '{output_filename}'.")
    except Exception as e:
        print(f"❌ Error saving file: {e}")