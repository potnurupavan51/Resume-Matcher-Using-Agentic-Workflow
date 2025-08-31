import streamlit as st
import requests
import pandas as pd
import os
import json
import uuid
from docx import Document # Required for extract_text_from_file if used locally
from typing import List, Optional, Dict, Any

# --- Custom Module Imports ---
try:
    from src.models import Criterion, JobMatchResult
    from src.utils import normalize_weights
    from src.extract_jd_text import extract_text_from_file  # Updated import
except ImportError as e:
    st.error(f"Failed to import necessary modules: {e}. Please ensure your 'src' directory and its contents are correctly set up.")
    st.stop()

# --- CONFIGURATION & INITIALIZATION ---
API_BASE_URL = "http://127.0.0.1:8000"
RESUME_DIR = "RR"
st.set_page_config(page_title="Resume Matcher", layout="wide", page_icon="📄")

# --- HELPER FUNCTIONS ---

# def get_token_usage_stats():
#     """Fetch token usage statistics from the API."""
#     try:
#         response = requests.get(f"{API_BASE_URL}/usage/tokens", timeout=10)
#         if response.status_code == 200:
#             return response.json()
#         return None
#     except requests.exceptions.RequestException:
#         return None

# def display_token_usage():
#     """Display token usage in the sidebar."""
#     token_stats = get_token_usage_stats()
#     if token_stats:
#         st.sidebar.subheader("💰 Token Usage")
        
#         session_stats = token_stats.get("session_stats", {})
#         total_cost = session_stats.get("total_cost", 0)
#         total_tokens = session_stats.get("total_tokens", 0)
#         total_operations = token_stats.get("total_operations", 0)
        
#         # Display main metrics
#         col1, col2 = st.sidebar.columns(2)
#         with col1:
#             st.metric("Total Cost", f"${total_cost:.4f}")
#         with col2:
#             st.metric("Total Tokens", f"{total_tokens:,}")
        
#         st.sidebar.metric("Operations", total_operations)
        
#         # Show cost breakdown if available
#         cost_breakdown = session_stats.get("cost_breakdown", {})
#         if cost_breakdown:
#             st.sidebar.write("**Cost Breakdown:**")
#             for model, cost in cost_breakdown.items():
#                 if cost > 0:
#                     st.sidebar.write(f"• {model}: ${cost:.4f}")
        
#         # Operations breakdown
#         operations = session_stats.get("operations", {})
#         if operations:
#             st.sidebar.write("**Operations:**")
#             for op_type, count in operations.items():
#                 if count > 0:
#                     st.sidebar.write(f"• {op_type}: {count}")

def get_api_key():
    """Retrieves the OpenAI API Key from environment variables or session state."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    return st.session_state.get("openai_api_key")

def extract_text_from_file(file_path: str) -> Optional[str]:
    """Extracts text from a file on disk. Supports both DOCX and PDF files. (Used for chatbot context preparation)."""
    if not os.path.exists(file_path):
        st.warning(f"File not found for text extraction: {file_path}")
        return None
    try:
        # Use the updated function from extract_jd_text module
        from src.extract_jd_text import extract_text_from_file as extract_func
        return extract_func(file_path)
    except Exception as e:
        st.error(f"Error extracting text from {os.path.basename(file_path)}: {e}")
        return None

def display_evaluation_plan(plan_data: dict, title: str):
    """Displays the evaluation plan in a structured DataFrame."""
    st.subheader(title)
    if plan_data is None:
        st.warning("No evaluation plan to display.")
        return
    st.markdown(f"**Job Title:** *{plan_data.get('job_title', 'N/A')}*")
    st.markdown(f"**AI Summary:** {plan_data.get('overall_summary', 'N/A')}")

    criteria_list = plan_data.get('criteria', [])
    if not criteria_list:
        st.warning("No criteria in this plan.")
        return

    display_data = []
    for criterion in criteria_list:
        display_data.append({
            'Criteria': criterion.get('criteria', 'N/A'),
            'Category': criterion.get('category', 'N/A'),
            'Normalized Weight': criterion.get('weightage', 0)
        })

    display_df = pd.DataFrame(display_data)
    display_df.columns = ['Criteria', 'Category', 'Normalized Weight']

    max_weight_value = 100
    if not display_df.empty and 'Normalized Weight' in display_df.columns:
        current_max = display_df['Normalized Weight'].max()
        if pd.notna(current_max):
            max_weight_value = max(100, int(current_max))

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Normalized Weight": st.column_config.ProgressColumn(
                "Importance (Normalized to 100)",
                format="%d",
                min_value=0,
                max_value=max_weight_value,
            )
        }
    )

def display_results(results: List[JobMatchResult]):
    """Formats and displays the final candidate matches."""
    if not results:
        st.warning("The workflow completed, but no matching candidates were found.")
        return

    st.header(f"🏆 Top {len(results)} Candidate Matches")
    st.markdown("---")
    sorted_results = sorted(results, key=lambda x: x.overall_score if isinstance(x, JobMatchResult) and hasattr(x, 'overall_score') else 0, reverse=True)

    for i, candidate in enumerate(sorted_results):
        if not isinstance(candidate, JobMatchResult):
            st.error(f"Skipping display for invalid candidate data at index {i}.")
            continue

        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"#{i+1}: {candidate.candidate_name or 'N/A'}")
                if (resume_filename := candidate.resume_filename):
                    resume_path = os.path.join(RESUME_DIR, resume_filename)
                    if os.path.exists(resume_path):
                        try:
                            with open(resume_path, "rb") as file:
                                st.download_button(f"📥 Download Resume (`{resume_filename}`)", file, resume_filename, key=f"dl_{i}", use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not open or read resume file '{resume_filename}': {e}")
                    else:
                        st.warning(f"Resume file `{resume_filename}` not found in the '{RESUME_DIR}' folder.")
                else:
                    st.caption("📄 Resume File: N/A")
                st.info(f"**Overall Reasoning:** {candidate.overall_reasoning or 'N/A'}")
            with col2:
                st.metric("Overall Match Score", f"{candidate.overall_score*100:.2f}%")

            with st.expander("📊 Show Detailed Score Breakdown"):
                scores_data = candidate.criterion_scores if hasattr(candidate, 'criterion_scores') and isinstance(candidate.criterion_scores, list) else []

                if scores_data:
                    processed_scores = []
                    for score_item in scores_data:
                        item_dict = score_item.model_dump() if hasattr(score_item, 'model_dump') else score_item
                        if isinstance(item_dict, dict):
                            processed_scores.append({
                                'Criteria': item_dict.get('criteria', 'N/A'),
                                'Category': item_dict.get('category', 'N/A'),
                                'Weightage': item_dict.get('weightage', 0),
                                'Score (1-10)': item_dict.get('score', 0),
                                'Reasoning': item_dict.get('reason', 'No reasoning provided.')
                            })
                #     if processed_scores:
                #         df = pd.DataFrame(processed_scores)
                #         st.dataframe(df, use_container_width=True, hide_index=True)
                #     else:
                #         st.write("No valid detailed scores available to display.")
                # else:
                #     st.write("No detailed scores available.")

                    if processed_scores:
                        df = pd.DataFrame(processed_scores)
                        # Enhanced column configuration for better display
                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True,
                            height=400,  # Set fixed height for better scrolling
                            column_config={
                                "Criteria": st.column_config.TextColumn("Criteria", width="large"),
                                "Category": st.column_config.TextColumn("Category", width="large"), 
                                "Weightage": st.column_config.NumberColumn("Weightage", format="%d"),
                                "Score (1-10)": st.column_config.NumberColumn("Score (1-10)"),
                                "Reasoning": st.column_config.TextColumn(
                                    "Reasoning",
                                    width="large"  # Changed from "wrap" to "large" for full content visibility
                                )
                            }
                        )
                    else:
                        st.write("No valid detailed scores available to display.")
                else:
                    st.write("No detailed scores available.")

# --- NEW: API-based Chat Function ---
def get_llm_response_api(query: str, chat_history: List[Dict[str, str]], job_id: str) -> str:
    """
    Handles the LLM chat interaction via the API endpoint.
    """
    try:
        # Prepare chat history in the expected format
        formatted_history = []
        for msg in chat_history:
            if msg["role"] in ["user", "assistant"]:
                formatted_history.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Prepare the request payload
        chat_request = {
            "job_id": job_id,
            "query": query,
            "chat_history": formatted_history
        }
        
        # Make the API call
        response = requests.post(
            f"{API_BASE_URL}/jobs/{job_id}/chat",
            json=chat_request,
            timeout=60  # Chat might take longer than regular API calls
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "No response received from AI.")
        else:
            error_data = response.json() if response.headers.get('content-type') == 'application/json' else {"detail": response.text}
            return f"Chat API Error: {error_data.get('detail', 'Unknown error')}"
            
    except requests.exceptions.RequestException as e:
        return f"Connection error while chatting: {e}"
    except Exception as e:
        return f"Error processing chat request: {e}"

# --- STATE MANAGEMENT ---
def initialize_state():
    """Initializes all necessary session state variables."""
    defaults = {
        'job_status': 'awaiting_upload',
        'job_id': None,
        'evaluation_plan': None,
        'top_candidates': None,
        'additional_candidates': [], # New: for the "Want More Resumes?" feature
        'api_payload': None,
        'modified_plan_preview': None,
        'jd_file_name': None,
        'jd_file_content': None,
        # Chatbot specific states
        'chat_messages': [],
        "openai_api_key": None,
        "api_key_input_visible": False,
        # --- Resume Upload State ---
        'uploaded_resumes_info': [], # Stores dicts like {"filename": "...", "status": "uploaded"}
        'resume_upload_status': 'idle', # 'idle', 'uploading', 'success', 'error'
        # --- Resume Management State ---
        'show_resume_management': False,
        'resume_search_query': '',
        'selected_resumes_for_deletion': [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def start_over():
    """Clears the session to start a new job."""
    api_key = st.session_state.get("openai_api_key")
    st.session_state.clear()
    if api_key: # Preserve API key if it was manually entered
        st.session_state["openai_api_key"] = api_key
    st.rerun()

initialize_state()

# --- SIDEBAR FOR API KEY INPUT ---
with st.sidebar:
    st.header("Configuration")
    current_api_key = get_api_key()
    if not current_api_key:
        st.session_state.api_key_input_visible = True

    if st.session_state.api_key_input_visible:
        st.info("Please enter your OpenAI API Key to enable the chatbot.")
        manual_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            key="manual_api_key_input",
            help="Get your API key from https://platform.openai.com/api-keys"
        )
        if st.button("Save API Key"):
            if manual_api_key:
                st.session_state["openai_api_key"] = manual_api_key
                st.success("API Key saved!")
                st.session_state.api_key_input_visible = False
                st.rerun()
            else:
                st.warning("Please enter an API Key.")
    else:
        if current_api_key:
            st.success("OpenAI API Key is set.")
        if st.button("Change API Key"):
            st.session_state.api_key_input_visible = True
            st.rerun()
    
    # Display token usage statistics
    # display_token_usage()
    
    # Resume Management Section
    st.markdown("---")
    st.subheader("📁 Resume Management")
    
    if st.button("📋 Manage Resumes", use_container_width=True):
        st.session_state.show_resume_management = True
        st.rerun()

# --- MAIN APP UI & LOGIC ---
st.title("📄 Interactive Resume Matcher")

# The API key is now handled on the server side, but we still show it for transparency
if not get_api_key():
    st.info("💡 **Note:** OpenAI API Key should be configured on the server for chat functionality. You can still use the basic matching features.")

# --- Resume Management Interface ---
if st.session_state.get('show_resume_management', False):
    st.header("📁 Resume Database Management")
    
    # Tabs for different resume management functions
    tab1, tab2, tab3 = st.tabs(["📋 View All Resumes", "🔍 Search Resumes", "🗑️ Delete Resumes"])
    
    with tab1:
        st.subheader("All Indexed Resumes")
        try:
            response = requests.get(f"{API_BASE_URL}/resumes/list", timeout=30)
            if response.status_code == 200:
                all_resumes = response.json()
                if all_resumes:
                    # Create a DataFrame for better display
                    resume_df = pd.DataFrame(all_resumes)
                    
                    # Show key columns
                    display_columns = ['filename', 'id']
                    if 'name' in resume_df.columns:
                        display_columns.insert(1, 'name')
                    
                    st.dataframe(
                        resume_df[display_columns] if all(col in resume_df.columns for col in display_columns) else resume_df,
                        use_container_width=True,
                        column_config={
                            "filename": "Resume File",
                            "name": "Candidate Name",
                            "id": "Resume ID"
                        }
                    )
                    st.info(f"Total resumes in database: {len(all_resumes)}")
                else:
                    st.info("No resumes found in the database.")
            else:
                st.error(f"Failed to fetch resumes: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")
    
    with tab2:
        st.subheader("Search Resumes")
        search_query = st.text_input("Search by filename or candidate name:", key="resume_search_input")
        
        if st.button("🔍 Search", use_container_width=True) and search_query:
            try:
                search_payload = {"query": search_query}
                response = requests.post(f"{API_BASE_URL}/resumes/search", json=search_payload, timeout=30)
                
                if response.status_code == 200:
                    search_results = response.json()
                    st.success(search_results.get("message", "Search completed"))
                    
                    results = search_results.get("results", [])
                    if results:
                        for i, result in enumerate(results):
                            with st.container(border=True):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**{result.get('filename', 'N/A')}**")
                                    if result.get('name'):
                                        st.write(f"Candidate: {result['name']}")
                                    st.caption(f"ID: {result.get('id', 'N/A')}")
                                with col2:
                                    if result.get('filename'):
                                        resume_path = os.path.join(RESUME_DIR, result['filename'])
                                        if os.path.exists(resume_path):
                                            try:
                                                with open(resume_path, "rb") as file:
                                                    st.download_button(
                                                        "📥 Download",
                                                        file,
                                                        result['filename'],
                                                        key=f"search_dl_{i}",
                                                        use_container_width=True
                                                    )
                                            except Exception as e:
                                                st.error(f"Cannot read file: {e}")
                    else:
                        st.info("No resumes matched your search query.")
                else:
                    st.error(f"Search failed: {response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")
    
    with tab3:
        st.subheader("Delete Resumes")
        st.warning("⚠️ **Warning:** Deletion is permanent and will remove both the database entry and the physical file.")
        
        # Fetch all resumes for selection
        try:
            response = requests.get(f"{API_BASE_URL}/resumes/list", timeout=30)
            if response.status_code == 200:
                all_resumes = response.json()
                if all_resumes:
                    # Create multiselect for resume deletion
                    resume_options = {f"{resume.get('filename', 'N/A')} (ID: {resume.get('id', 'N/A')})": resume.get('id') 
                                    for resume in all_resumes if resume.get('id')}
                    
                    selected_resume_labels = st.multiselect(
                        "Select resumes to delete:",
                        list(resume_options.keys()),
                        key="resumes_to_delete"
                    )
                    
                    if selected_resume_labels:
                        selected_ids = [resume_options[label] for label in selected_resume_labels]
                        st.info(f"Selected {len(selected_ids)} resume(s) for deletion.")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            confirm_delete = st.checkbox("I understand this action is permanent", key="confirm_delete_checkbox")
                        with col2:
                            if st.button("🗑️ Delete Selected", type="primary", disabled=not confirm_delete, use_container_width=True):
                                try:
                                    delete_payload = {"resume_ids": selected_ids}
                                    response = requests.post(f"{API_BASE_URL}/resumes/delete", json=delete_payload, timeout=60)
                                    
                                    if response.status_code == 200:
                                        result = response.json()
                                        st.success(result.get("message", "Resumes deleted successfully"))
                                        # Clear the selection
                                        st.session_state.resumes_to_delete = []
                                        st.rerun()
                                    else:
                                        st.error(f"Deletion failed: {response.status_code}")
                                except requests.exceptions.RequestException as e:
                                    st.error(f"Connection error: {e}")
                else:
                    st.info("No resumes available for deletion.")
            else:
                st.error(f"Failed to fetch resumes: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")
    
    # Close resume management
    st.markdown("---")
    if st.button("⬅️ Back to Main Interface", use_container_width=True):
        st.session_state.show_resume_management = False
        st.rerun()
    
    st.stop()  # Stop here if in resume management mode

# --- Section 1: JD Analysis and Resume Upload ---
if st.session_state.job_status == 'awaiting_upload':
    
    # --- Job Description Analysis Section (Moved Up) ---
    st.subheader("Analyze Job Description")
    
    # Fetch cached JDs from API
    cached_jds = []
    try:
        resp = requests.get(f"{API_BASE_URL}/jds/list", timeout=1000)
        if resp.status_code == 200:
            cached_jds = resp.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch cached JDs: {e}. Is the backend running?")

    col1_jd, col2_jd = st.columns(2)
    with col1_jd:
        uploaded_file = st.file_uploader("Choose a Job Description file", type=["docx", "pdf"])
    with col2_jd:
        selected_jd_cache_key: Optional[str] = None
        if cached_jds:
            jd_options = {jd['display_name']: jd['cache_key'] for jd in cached_jds}
            selected_display = st.selectbox("Or select a cached JD", ["-- Select --"] + list(jd_options.keys()))
            if selected_display != "-- Select --":
                selected_jd_cache_key = jd_options[selected_display]


    st.subheader("⚙️ Configuration")
    col1_config, col2_config = st.columns(2)

    with col1_config:
        # Get current top_n value from API
        current_top_n = 5  # Default fallback
        try:
            config_resp = requests.get(f"{API_BASE_URL}/config/top_n", timeout=5)
            if config_resp.status_code == 200:
                current_top_n = config_resp.json().get("top_n", 5)
        except:
            pass  # Use default if API call fails

        top_n_input = st.number_input(
            "Number of top candidates to select",
            min_value=1,
            max_value=50,
            value=current_top_n, # This is the value from config or previous session
            help="Specify how many top candidates you want in the final results"
        )
        if top_n_input >= 10:
            st.warning(" **Processing Time Alert:** Selecting 10 or more resumes will significantly increase processing time and api cost and may take several minutes to complete. Consider using a smaller number for faster results.")

    with col2_config:
        st.info(f"📊 **Current Settings:**\n\n• Top candidates: {top_n_input}\n• RAG retrieval: {top_n_input * 4}")

    if uploaded_file and st.button("🚀 Start Analysis", type="primary", use_container_width=True):
        if uploaded_file and selected_jd_cache_key: # Prevent using both
             st.warning("Please either upload a JD file or select a cached JD, not both.")
        else:
            with st.spinner("Step 1/3: Analyzing Job Description..."):
                try:
                    files = {
                        "jd_file": (uploaded_file.name, uploaded_file.getvalue()),
                        # Add top_n to the files dictionary, FastAPI will parse it as Form
                        "top_n": (None, str(top_n_input)) # Send as string, FastAPI converts to int
                    }
                    response = requests.post(f"{API_BASE_URL}/jobs/start", files=files, timeout=1000)
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.job_id = data.get("job_id")
                        st.session_state.evaluation_plan = data.get("evaluation_plan")
                        st.session_state.job_status = 'plan_received'
                        st.session_state.jd_file_name = uploaded_file.name
                        st.session_state.jd_file_content = uploaded_file.getvalue()
                        st.rerun()
                    else:
                        st.error(f"API Error (start): {response.status_code} - {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection Error: {e}")

    elif selected_jd_cache_key and st.button("🚀 Start Analysis with Cached JD", type="primary", use_container_width=True, key="start_cached"):
        with st.spinner("Step 1/3: Loading Cached JD and Starting Analysis..."):
            try:
                # Use 'data' for form fields, 'files' for file uploads
                payload = {
                    "selected_jd": selected_jd_cache_key,
                    "top_n": str(top_n_input) # Send as string
                }
                response = requests.post(f"{API_BASE_URL}/jobs/start", data=payload, timeout=1000)

                if response.status_code == 200:
                    data = response.json()
                    st.session_state.job_id = data.get("job_id")
                    st.session_state.evaluation_plan = data.get("evaluation_plan")
                    st.session_state.job_status = 'plan_received'

                    # Store a placeholder for JD info (backend handles the actual file)
                    display_name = next((jd['display_name'] for jd in cached_jds if jd['cache_key'] == selected_jd_cache_key), selected_jd_cache_key)
                    st.session_state.jd_file_name = f"{display_name}.docx"
                    st.session_state.jd_file_content = b"" # No actual file content stored locally

                    st.rerun()
                else:
                    st.error(f"API Error (start from cache): {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Connection Error: {e}")

    # --- Resume Upload Section (Moved Down) ---
    st.markdown("---") # Separator
    st.subheader("Manage Candidate Resumes")
    st.info("📁 Upload resume files (.docx or .pdf) here. They will be **automatically processed and indexed in FAISS** for immediate availability in matching workflows.")

    # File uploader widget for selecting resume files
    uploaded_resume_files = st.file_uploader(
        "Choose resume files",
        type=["docx", "pdf"], # Common resume file types
        accept_multiple_files=True,
        key="resume_file_uploader",
        label_visibility="collapsed" # Hide label as subheader is descriptive
    )

    # Button to trigger the upload to the API
    if uploaded_resume_files and st.button("Upload Selected Resumes", key="upload_resumes_btn"):
        st.session_state.resume_upload_status = 'uploading'
        st.rerun() # Rerun to show spinner and process uploads

    # Display upload status and progress
    if st.session_state.resume_upload_status == 'uploading':
        with st.spinner("Uploading and indexing resumes..."):
            successful_uploads = []
            failed_uploads = []
            indexed_uploads = []
            # Define the API endpoint for uploading resumes (ensure this exists in api.py)
            API_UPLOAD_RESUME_ENDPOINT = f"{API_BASE_URL}/resumes/upload" 

            for resume_file in uploaded_resume_files:
                try:
                    # Prepare the file payload for the API request
                    # The API endpoint must expect 'multipart/form-data' with a field like 'resume_file'
                    files = {"resume_file": (resume_file.name, resume_file.getvalue())}
                    
                    # Make the POST request to your new API endpoint
                    response = requests.post(API_UPLOAD_RESUME_ENDPOINT, files=files, timeout=1000)
                    
                    if response.status_code == 200:
                        # Assuming the API returns JSON like {"filename": "resume_name.docx", "status": "uploaded"}
                        upload_info = response.json()
                        successful_uploads.append(upload_info.get('filename', resume_file.name))
                        
                        # Check if the resume was also indexed
                        message = upload_info.get('message', '')
                        if 'indexed in FAISS' in message or 'Already indexed' in message:
                            indexed_uploads.append(upload_info.get('filename', resume_file.name))
                        
                        # Store minimal info, the actual file is in RESUMES_DIR managed by the API
                        st.session_state.uploaded_resumes_info.append({
                            "filename": upload_info.get('filename', resume_file.name), 
                            "status": "uploaded",
                            "indexed": 'indexed in FAISS' in message or 'Already indexed' in message
                        })
                    else:
                        # Handle API errors
                        error_msg = f"Failed to upload {resume_file.name}: {response.status_code} - {response.text}"
                        failed_uploads.append((resume_file.name, error_msg))
                        st.error(error_msg)
                        
                except requests.exceptions.RequestException as e:
                    # Handle connection errors
                    error_msg = f"Connection error uploading {resume_file.name}: {e}"
                    failed_uploads.append((resume_file.name, error_msg))
                    st.error(error_msg)
            
            # Update status after attempting all uploads
            if successful_uploads:
                st.session_state.resume_upload_status = 'success'
                if indexed_uploads:
                    st.success(f"Successfully uploaded {len(successful_uploads)} resumes. {len(indexed_uploads)} were automatically indexed in FAISS.")
                else:
                    st.success(f"Successfully uploaded {len(successful_uploads)} resumes.")
            if failed_uploads:
                st.session_state.resume_upload_status = 'error'
                st.error(f"Failed to upload {len(failed_uploads)} resumes.")
            else: # Handle case where no files selected but button clicked, or all succeeded
                 st.session_state.resume_upload_status = 'idle' if not failed_uploads else 'error'

            st.rerun() # Rerun to update the UI based on status and potentially clear the uploader state

    # Display final status message
    if st.session_state.resume_upload_status == 'success':
        indexed_count = sum(1 for info in st.session_state.uploaded_resumes_info if info.get('indexed', False))
        total_count = len(st.session_state.uploaded_resumes_info)
        
        if indexed_count > 0:
            st.info(f"✅ Resumes are ready! {indexed_count}/{total_count} resumes were automatically indexed in FAISS. You can now upload a Job Description.")
        else:
            st.info(f"Resumes are uploaded. You can now upload a Job Description.")
        
        # Optionally list uploaded resumes with indexing status
        if st.session_state.uploaded_resumes_info:
            st.write("Available resumes:")
            for info in st.session_state.uploaded_resumes_info:
                filename = info.get('filename', 'Unknown')
                indexed_status = "🔍 Indexed" if info.get('indexed', False) else "📄 Uploaded"
                st.write(f"- {filename} {indexed_status}")
    elif st.session_state.resume_upload_status == 'error':
        st.error("Some resumes failed to upload. Please check messages above and retry.")
    # If status is 'uploading' and rerun happens, spinner handles it. If it becomes 'idle' or 'success', this block is skipped.


# --- Section 2: Plan Review & Action ---
elif st.session_state.job_status == 'plan_received':
    st.success(f"**Analysis Complete!** Job ID: `{st.session_state.job_id}`")
    display_evaluation_plan(st.session_state.evaluation_plan, "Initial Evaluation Plan")
    st.subheader("What would you like to do next?")
    col1, col2, col3 = st.columns(3)
    if col1.button("👍 Approve & Find Matches", type="primary", use_container_width=True):
        st.session_state.job_status = 'awaiting_final_approval'
        st.session_state.api_payload = {"action": "approve", "evaluation_plan": st.session_state.evaluation_plan}
        st.rerun()
    if col2.button("✏️ Modify Plan", use_container_width=True):
        st.session_state.job_status = 'modifying_plan'
        st.rerun()
    if col3.button("❌ Reject Plan", use_container_width=True):
        st.session_state.job_status = 'awaiting_final_approval'
        st.session_state.api_payload = {"action": "reject"}
        st.rerun()
# --- Section 3: Plan Modification ---
elif st.session_state.job_status == 'modifying_plan':
    st.header("Modify Evaluation Criteria")
    st.info("You can edit criteria, mark them for removal, or add new ones. Changes are applied when you click a button below.")

    # Initialize editable_criteria from the plan if it doesn't exist
    if 'editable_criteria' not in st.session_state:
        plan = st.session_state.evaluation_plan or {}
        criteria_list = plan.get('criteria', [])
        # Ensure each criterion is a dictionary with a unique 'id' and 'remove' flag
        st.session_state.editable_criteria = [
            {
                'id': c.get('id', str(uuid.uuid4())),
                'criteria': c.get('criteria', ''),
                'category': c.get('category', ''),
                'weightage': c.get('weightage', 0),
                'evaluation_guidelines': c.get('evaluation_guidelines', ''),
                'remove': False # Add a remove flag
            }
            for c in criteria_list
        ]

    # --- Start of the Form ---
    with st.form("modify_plan_form"):
        # Display existing criteria for editing
        for idx, c in enumerate(st.session_state.editable_criteria):
            with st.container(border=True):
                cols = st.columns([4, 3, 1, 1])
                # We still need keys here to differentiate between the criteria rows
                c['criteria'] = cols[0].text_input("Criteria", value=c.get('criteria', ''), key=f"crit_{idx}")
                c['category'] = cols[1].text_input("Category", value=c.get('category', ''), key=f"cat_{idx}")
                c['weightage'] = cols[2].number_input("Weight", min_value=0, max_value=100, value=int(c.get('weightage', 0)), key=f"weight_{idx}")
                c['remove'] = cols[3].checkbox("Remove", key=f"rem_{idx}")

        st.markdown("---")
        st.subheader("Add New Criterion")
        
        # Create input widgets WITHOUT keys. Their values will be retrieved on submit.
        # These variables will hold the current values from the input widgets when submitted.
        new_criteria_input = st.text_input("New Criteria", value="")
        new_category_input = st.text_input("New Category", value="")
        new_weight_input = st.number_input("New Weight", min_value=1, max_value=100, value=50)

        # --- Form Submit Buttons ---
        submitted_add = st.form_submit_button("➕ Add Criterion")
        submitted_finalize = st.form_submit_button("🔬 Preview & Finalize Changes", type="primary")

        # --- Form Submission Logic ---
        if submitted_add:
            # *** FIX IS HERE: Use the correct variable names ***
            # Use the values from the widgets directly (they are available in these variables)
            if new_criteria_input and new_category_input and new_weight_input > 0:
                st.session_state.editable_criteria.append({
                    'id': str(uuid.uuid4()),
                    'criteria': new_criteria_input, # Use the input widget variables
                    'category': new_category_input,
                    'weightage': new_weight_input,
                    'evaluation_guidelines': "Evaluate based on resume content.",
                    'remove': False
                })
                # Rerun to clear input fields and update the list display
                # Streamlit automatically clears text_input/number_input widgets if they don't have a 'key' 
                # and the form is resubmitted and rerendered.
                st.rerun() 
            else:
                st.warning("Please fill all fields for the new criterion.")

        if submitted_finalize:
            # Filter out criteria marked for removal or with zero weight
            final_criteria_data = [
                c for c in st.session_state.editable_criteria if not c.get('remove', False) and c.get('weightage', 0) > 0
            ]

            if not final_criteria_data:
                st.warning("You must have at least one criterion with weight > 0.")
            else:
                try:
                    pydantic_criteria = [Criterion(**data) for data in final_criteria_data]
                    normalize_weights(pydantic_criteria)

                    plan = st.session_state.evaluation_plan or {}
                    st.session_state.modified_plan_preview = {
                        "job_title": plan.get('job_title', 'N/A'),
                        "overall_summary": plan.get('overall_summary', 'N/A'),
                        "criteria": [c.model_dump() for c in pydantic_criteria]
                    }

                    # Clean up temporary state and transition
                    if 'editable_criteria' in st.session_state: del st.session_state['editable_criteria']
                    st.session_state.job_status = 'previewing_changes'
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing criteria: {e}")



# --- Section 4: Preview Changes ---
elif st.session_state.job_status == 'previewing_changes':
    st.success("Your changes have been processed and weights normalized.")
    display_evaluation_plan(st.session_state.modified_plan_preview, "Final Plan to be Submitted")
    col1, col2 = st.columns(2)
    if col1.button("✅ Confirm & Find Matches", type="primary", use_container_width=True):
        st.session_state.job_status = 'awaiting_final_approval'
        st.session_state.api_payload = {"action": "modify", "evaluation_plan": st.session_state.modified_plan_preview}
        st.rerun()
    if col2.button("⬅️ Go Back & Edit", use_container_width=True):
        st.session_state.job_status = 'modifying_plan'
        st.rerun()

# --- API Action Handler (Centralized) ---
elif st.session_state.job_status == 'awaiting_final_approval':
    action_text = "Finding top candidates..." if st.session_state.api_payload.get('action') != 'reject' else "Cancelling job..."
    with st.spinner(action_text):
        try:
            response = requests.post(
                f"{API_BASE_URL}/jobs/{st.session_state.job_id}/action",
                json=st.session_state.api_payload,
                timeout=1000
            )
            if response.status_code == 200:
                if st.session_state.api_payload.get('action') in ['approve', 'modify']:
                    # Convert response JSON (list of dicts) to JobMatchResult objects
                    st.session_state.top_candidates = [JobMatchResult(**item) for item in response.json()]
                    st.session_state.job_status = 'job_completed'
                else: # Reject action
                    st.session_state.job_status = 'job_rejected'
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                st.session_state.job_status = 'plan_received' # Revert to plan review
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {e}")
            st.session_state.job_status = 'plan_received' # Revert to plan review
    st.session_state.api_payload = None # Clear payload after use
    st.rerun()

# --- Final State Displays ---
elif st.session_state.job_status == 'job_completed':
    st.success("**Workflow Complete!**")

    # Display final token usage summary
    # token_stats = get_token_usage_stats()
    # if token_stats:
    #     st.info(f"💰 **Session Cost:** ${token_stats.get('session_stats', {}).get('total_cost', 0):.4f} | "
    #            f"🔢 **Total Tokens:** {token_stats.get('session_stats', {}).get('total_tokens', 0):,} | "
    #            f"⚡ **Operations:** {token_stats.get('total_operations', 0)}")

    jd_name = st.session_state.get('jd_file_name')
    # cached_jd_path=st.session_state.get('jd_file_name', 'job_description.docx') # Default name if not set

    jd_content = st.session_state.get('jd_file_content')
    evaluation_plan_data = st.session_state.get('evaluation_plan')

    if jd_name and evaluation_plan_data: # Check if we have a name and plan for details
        st.subheader("Original Job Description") # <-- Header for the JD details

        # Safely get job_title and overall_summary from evaluation_plan
        job_title = "N/A"
        overall_summary = "N/A"

        if isinstance(evaluation_plan_data, dict):
            job_title = evaluation_plan_data.get('job_title', 'N/A')
            overall_summary = evaluation_plan_data.get('overall_summary', 'N/A')
        elif hasattr(evaluation_plan_data, 'model_dump'): # If it's a Pydantic model (less likely in state, but safe)
            plan_dict = evaluation_plan_data.model_dump()
            job_title = plan_dict.get('job_title', 'N/A')
            overall_summary = plan_dict.get('overall_summary', 'N/A')

        st.markdown(f"**Job Title:** *{job_title}*") # <-- Displays the JD Title
        st.markdown(f"**AI Summary:** {overall_summary}") # <-- Displays the JD AI Summary


    # Display matching results
    display_results(st.session_state.top_candidates or [])
    
    # Add "Want More Resumes?" functionality
    st.markdown("---")
    st.subheader("🔍 Want More Candidates?")
    
    # Calculate the maximum additional resumes we can request
    job_id = st.session_state.get('job_id')
    if job_id:
        try:
            # Get configuration to determine max available
            config_response = requests.get(f"{API_BASE_URL}/config/top_n", timeout=10)
            if config_response.status_code == 200:
                config_data = config_response.json()
                current_top_n = config_data.get("top_n", 5)
                max_additional = (current_top_n * 4) - current_top_n  # RAG retrieval - already shown
                
                if max_additional > 0:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        additional_count = st.number_input(
                            f"Number of additional candidates (max {max_additional})",
                            min_value=1,
                            max_value=max_additional,
                            value=min(5, max_additional),
                            key="additional_count_input"
                        )
                    
                    with col2:
                        if st.button("Get More Candidates", type="primary", use_container_width=True):
                            with st.spinner(f"Fetching {additional_count} more candidates..."):
                                try:
                                    # Make API call to get more resumes
                                    payload = {"additional_count": str(additional_count)}
                                    response = requests.post(
                                        f"{API_BASE_URL}/jobs/{job_id}/more-resumes",
                                        data=payload,
                                        timeout=30
                                    )
                                    
                                    if response.status_code == 200:
                                        additional_candidates = [JobMatchResult(**item) for item in response.json()]
                                        
                                        # Store additional candidates in session state
                                        if 'additional_candidates' not in st.session_state:
                                            st.session_state.additional_candidates = []
                                        st.session_state.additional_candidates.extend(additional_candidates)
                                        
                                        st.success(f"✅ Successfully loaded {len(additional_candidates)} more candidates!")
                                        st.rerun()
                                        
                                    else:
                                        error_data = response.json() if response.headers.get('content-type') == 'application/json' else {"detail": response.text}
                                        st.error(f"Failed to fetch more candidates: {error_data.get('detail', 'Unknown error')}")
                                        
                                except requests.exceptions.RequestException as e:
                                    st.error(f"Connection error: {e}")
                                except Exception as e:
                                    st.error(f"Error processing request: {e}")
                else:
                    st.info("No additional candidates available within the RAG retrieval limit.")
            else:
                st.warning("Could not determine available candidate count.")
        except Exception as e:
            st.error(f"Error checking configuration: {e}")
    
    # Display additional candidates if any
    if 'additional_candidates' in st.session_state and st.session_state.additional_candidates:
        st.markdown("---")
        st.subheader(f"🆕 Additional Candidates ({len(st.session_state.additional_candidates)})")
        
        for i, candidate in enumerate(st.session_state.additional_candidates):
            if not isinstance(candidate, JobMatchResult):
                st.error(f"Skipping display for invalid candidate data at index {i}.")
                continue

            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                
                # Continue numbering from where top candidates ended
                base_number = len(st.session_state.top_candidates or [])
                
                with col1:
                    st.subheader(f"#{base_number + i + 1}: {candidate.candidate_name or 'N/A'}")
                    if (resume_filename := candidate.resume_filename):
                        resume_path = os.path.join(RESUME_DIR, resume_filename)
                        if os.path.exists(resume_path):
                            try:
                                with open(resume_path, "rb") as file:
                                    st.download_button(f"📥 Download Resume (`{resume_filename}`)", file, resume_filename, key=f"dl_additional_{i}", use_container_width=True)
                            except Exception as e:
                                st.error(f"Could not open or read resume file '{resume_filename}': {e}")
                        else:
                            st.warning(f"Resume file `{resume_filename}` not found in the '{RESUME_DIR}' folder.")
                    else:
                        st.caption("📄 Resume File: N/A")
                    st.info(f"**Overall Reasoning:** {candidate.overall_reasoning or 'N/A'}")
                with col2:
                    st.metric("Overall Match Score", f"{candidate.overall_score:.2f}%")

                with st.expander("📊 Show Detailed Score Breakdown"):
                    scores_data = candidate.criterion_scores if hasattr(candidate, 'criterion_scores') and isinstance(candidate.criterion_scores, list) else []

                    if scores_data:
                        processed_scores = []
                        for score_item in scores_data:
                            item_dict = score_item.model_dump() if hasattr(score_item, 'model_dump') else score_item
                            if isinstance(item_dict, dict):
                                processed_scores.append({
                                    'Criteria': item_dict.get('criteria', 'N/A'),
                                    'Category': item_dict.get('category', 'N/A'),
                                    'Weightage': item_dict.get('weightage', 0),
                                    'Score (1-10)': item_dict.get('score', 0),
                                    'Reasoning': item_dict.get('reason', 'No reasoning provided.')
                                })

                        if processed_scores:
                            df = pd.DataFrame(processed_scores)
                            st.dataframe(
                                df,
                                use_container_width=True,
                                hide_index=True,
                                height=400,  # Set fixed height for better scrolling
                                column_config={
                                    "Criteria": st.column_config.TextColumn("Criteria", width="large"),
                                    "Category": st.column_config.TextColumn("Category", width="large"),
                                    "Weightage": st.column_config.NumberColumn("Weightage", format="%d"),
                                    "Score (1-10)": st.column_config.NumberColumn("Score (1-10)"),
                                    "Reasoning": st.column_config.TextColumn(
                                        "Reasoning",
                                        width="large"  # Full content visibility
                                    )
                                }
                            )
                        else:
                            st.write("No valid detailed scores available to display.")
                    else:
                        st.write("No detailed scores available.")
    
    # Chatbot interface
    st.markdown("---")
    st.header("🤖 Chat with AI about Top Resumes")
    st.info("💡 The AI has access to all candidate resumes and can answer questions about their qualifications, experience, and how they compare to each other.")

    # Display previous chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("Ask about the top resumes..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            if not st.session_state.get('job_id'):
                ai_response = "I don't have a job ID to get context from. Please complete a job first."
                st.warning("No job ID available for chat context.")
            else:
                try:
                    # Use the new API-based chat function
                    ai_response = get_llm_response_api(
                        prompt,
                        st.session_state.chat_messages[:-1],  # Exclude the current message
                        st.session_state.job_id
                    )
                except Exception as e:
                    ai_response = f"An error occurred while getting the AI response: {e}"

            st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})
            with st.chat_message("assistant"):
                st.markdown(ai_response)
    
    st.markdown("---")
    if st.button("🔄 Start New Job", use_container_width=True):
        start_over()

# --- State after job rejection ---
elif st.session_state.job_status == 'job_rejected':
    st.warning(f"Job `{st.session_state.job_id}` has been successfully cancelled.")
    if st.button("🔄 Start New Job", use_container_width=True):
        start_over()













