
top_n=5 #this is a final selection value

top_k_retrieved_count = top_n*4  #This is a rag retrival value


score_display_threshold=0.60 #This is a final selection value thressold after coming from the llm

# Cache management settings
JD_CACHE_MAX_SIZE = 2 #used fifo 

CACHE_DIR = {
    "MAIN_DIR": "cache",
    "JD_CACHE_DIR": "cache/jd_cache",
    "JD_EVAL_PLAN_DIR": "cache/jd_cache/evaluation_plans",
    "JD_EMBEDDING_DIR": "cache/jd_cache/embeddings",
    "JD_PARSED_DIR": "cache/jd_cache/parsed_jds",
    "CV_CACHE_DIR": "cache/cv_cache",
    "RESUME_EMBEDDING_DIR": "cache/resume_cache/embeddings",
    "RESUME_PARSED_DIR": "cache/resume_cache/parsed_resumes",
    "RESUME_EVAL_PLAN_DIR": "cache/resume_cache/evaluation_plans"
}





















