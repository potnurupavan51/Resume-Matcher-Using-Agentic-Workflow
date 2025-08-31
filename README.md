# 📄 Interactive Resume Matcher

## Overview

The **Interactive Resume Matcher** is an advanced AI-powered system that intelligently matches resumes to job descriptions using vector similarity search, LLM-based evaluation, and semantic analysis. The system provides both a web interface and API for comprehensive resume screening and candidate evaluation.

## 🚀 Key Features

### Core Functionality
- **AI-Powered Job Description Analysis**: Automatically extracts evaluation criteria from job descriptions
- **Intelligent Resume Processing**: Parses resumes and extracts structured data (skills, experience, education)
- **FAISS Vector Search**: High-performance semantic similarity search for candidate matching
- **LLM-Based Evaluation**: GPT-4 powered candidate scoring with detailed reasoning
- **Interactive Web Interface**: User-friendly Streamlit application for complete workflow management
- **RESTful API**: FastAPI backend for programmatic access and integration

### Advanced Features
- **Immediate FAISS Indexing**: New resumes are automatically processed and indexed upon upload
- **Caching System**: Intelligent caching of job descriptions and evaluation plans
- **Resume Management**: Complete CRUD operations for resume database
- **Real-time Chat**: AI chatbot for discussing candidate results and comparisons
- **Dynamic Criteria Management**: Modify evaluation criteria with real-time weight normalization
- **Comprehensive Scoring**: Detailed criterion-by-criterion evaluation with reasoning

## 🏗️ Architecture

### System Components

```
├── Frontend (Streamlit)          # User Interface
├── Backend (FastAPI)             # REST API Server
├── LLM Processing (OpenAI)       # AI Analysis Engine
├── Vector Database (FAISS)       # Semantic Search
├── Workflow Engine (LangGraph)   # State Management
└── Caching Layer                 # Performance Optimization
```

### Data Flow
1. **Job Description Upload** → LLM Analysis → Evaluation Plan Generation
2. **Resume Upload** → Text Extraction → LLM Parsing → FAISS Indexing
3. **Matching Process** → Vector Search → LLM Evaluation → Ranking & Results
4. **Interactive Features** → Chat, Resume Management, Plan Modification

## 📁 Project Structure

```
📦 interactive-resume-matcher/
├── 📄 api.py                     # FastAPI backend server
├── 📄 app.py                     # Streamlit frontend application
├── 📄 main.py                    # CLI interface and workflow runner
├── 📄 config.py                  # Configuration settings
├── 📄 requirements.txt           # Python dependencies
├── 📄 .env                       # Environment variables (create this)
├── 📄 .gitignore                 # Git ignore rules
│
├── 📂 src/                       # Core source code
│   ├── 📄 __init__.py
│   ├── 📄 models.py              # Pydantic data models
│   ├── 📄 services.py            # Core business logic agents
│   ├── 📄 graph.py               # LangGraph workflow definition  
│   ├── 📄 db_manager.py          # FAISS vector database manager
│   ├── 📄 llm_handlers.py        # OpenAI LLM integration
│   ├── 📄 extract_jd_text.py     # Document text extraction
│   ├── 📄 resume_processor.py    # Resume processing utilities
│   ├── 📄 user_interface.py      # CLI user interface helpers
│   ├── 📄 utils.py               # Utility functions
│   └── 📄 config_manager.py      # Configuration management
│
├── 📂 cache/                     # Caching directories
│   ├── 📂 jd_cache/              # Job description cache
│   │   ├── 📂 embeddings/        # JD embeddings
│   │   ├── 📂 evaluation_plans/  # Generated evaluation plans
│   │   └── 📂 parsed_jds/        # Parsed JD data
│   └── 📂 cv_cache/              # Resume cache (if used)
│
├── 📂 faiss_index/               # FAISS vector database
│   ├── 📄 index.faiss            # Vector index file
│   ├── 📄 id_map.json            # ID mapping
│   └── 📄 docstore.json          # Document metadata
│
├── 📂 RR/                        # Resume storage directory

├── 📄 jd.py                      # Standalone JD analysis script
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- OpenAI API Key
- Git

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd interactive-resume-matcher
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Create Required Directories**
   ```bash
   mkdir -p RR temp_uploads faiss_index cache/jd_cache/{embeddings,evaluation_plans,parsed_jds}
   ```

## 🚀 Usage

### Method 1: Web Interface (Recommended)

1. **Start the Backend API**
   ```bash
   uvicorn api:app --reload --port 8000
   ```

2. **Launch the Frontend**
   ```bash
   streamlit run app.py
   ```

3. **Access the Application**
   - Open your browser to `http://localhost:8501`
   - Upload job descriptions and resumes
   - Review and modify evaluation plans
   - Get intelligent candidate matches

### Method 2: API Integration

**Start the API Server:**
```bash
uvicorn api:app --reload --port 8000
```

**API Documentation:** `http://localhost:8000/docs`

**Key Endpoints:**
- `POST /jobs/start` - Start new job with JD
- `POST /resumes/upload` - Upload and auto-index resumes
- `GET /resumes/list` - List all indexed resumes
- `POST /jobs/{job_id}/action` - Execute job actions
- `POST /jobs/{job_id}/chat` - Chat about results

### Method 3: Command Line Interface

```bash
python main.py
```

## 🔧 Configuration

### Core Settings (`config.py`)
```python
top_n = 5                          # Number of final candidates
top_k_retrieved_count = top_n * 4  # RAG retrieval count
score_display_threshold = 0.60     # Minimum score threshold
```

### Cache Directories
- **JD Cache**: Stores processed job descriptions
- **Resume Cache**: Stores processed resume data
- **FAISS Index**: Vector database for semantic search

## 🤖 AI Components

### LLM Integration
- **Model**: GPT-4o-mini for cost-effective processing
- **Embedding Model**: text-embedding-3-small (768 dimensions)
- **Use Cases**: 
  - Job description analysis
  - Resume parsing and structuring
  - Candidate evaluation and scoring
  - Interactive chat functionality

### Vector Database
- **Engine**: FAISS (Facebook AI Similarity Search)
- **Index Type**: IndexFlatIP (cosine similarity)
- **Features**:
  - Real-time indexing of new resumes
  - Semantic similarity search
  - Efficient vector operations
  - Automatic consistency validation

## 📊 Workflow Process

### 1. Job Description Analysis
```
Upload JD → Text Extraction → LLM Analysis → Criteria Generation → Plan Review
```

### 2. Resume Processing
```
Upload Resume → Text Extraction → LLM Parsing → Vector Embedding → FAISS Indexing
```

### 3. Candidate Matching
```
JD Embedding → FAISS Search → Candidate Retrieval → LLM Evaluation → Scoring & Ranking
```

### 4. Results & Interaction
```
Candidate Display → Detailed Scoring → Chat Interface → Additional Candidates
```

## 🔍 Features Deep Dive

### Resume Management
- **Immediate Indexing**: Resumes are processed and indexed upon upload
- **CRUD Operations**: Create, read, update, delete resume records
- **Search Functionality**: Find resumes by name or filename
- **Batch Operations**: Handle multiple resume uploads efficiently

### Evaluation System
- **Dynamic Criteria**: Modify evaluation criteria in real-time
- **Weight Normalization**: Automatic weight balancing for fair scoring
- **Detailed Scoring**: 1-10 scale with reasoning for each criterion
- **Threshold Filtering**: Configurable minimum score requirements

### Chat Interface
- **Context-Aware**: AI has access to all candidate data
- **Comparative Analysis**: Compare candidates across different criteria
- **Detailed Explanations**: Get insights into scoring decisions
- **Interactive Q&A**: Ask specific questions about candidates

## 🔧 Troubleshooting

### Common Issues

1. **FAISS Index Inconsistency**
   ```bash
   # Use the repair endpoint
   curl -X POST http://localhost:8000/resumes/repair-index
   ```

2. **Missing Dependencies**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **OpenAI API Errors**
   - Check API key validity
   - Verify sufficient credits
   - Check rate limits

4. **File Processing Errors**
   - Ensure files are valid .docx or .pdf format
   - Check file permissions
   - Verify file is not corrupted

### Log Analysis
- Backend logs: Check FastAPI console output
- Frontend logs: Check Streamlit browser console
- System logs: Check Python logging output

## 🔒 Security Considerations

- **API Keys**: Store in environment variables, never commit to code
- **File Uploads**: Validated file types and sanitized filenames
- **Data Privacy**: Resume content is processed locally
- **Access Control**: Consider implementing authentication for production use

## 🚀 Performance Optimization

### Caching Strategy
- **JD Caching**: Reuse processed job descriptions
- **Resume Caching**: Avoid reprocessing existing resumes
- **Embedding Caching**: Store computed embeddings for reuse

### Vector Database Optimization
- **Index Type**: Using IndexFlatIP for accurate cosine similarity
- **Batch Processing**: Efficient batch operations for multiple resumes
- **Memory Management**: Optimized for production workloads

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Document functions and classes
- Add appropriate logging

## 📋 Dependencies

### Core Dependencies
- **FastAPI**: Web framework for API
- **Streamlit**: Frontend application framework
- **LangChain**: LLM integration and workflows  
- **FAISS**: Vector similarity search
- **OpenAI**: LLM and embedding models
- **Pydantic**: Data validation and modeling

### Processing Dependencies
- **python-docx**: Word document processing
- **pdfminer.six**: PDF text extraction
- **numpy**: Numerical computations
- **pandas**: Data manipulation (frontend)

See `requirements.txt` for complete dependency list with versions.

## 📈 Future Enhancements

### Planned Features
- **Multi-language Support**: Process resumes in different languages
- **Advanced Filtering**: Industry-specific filters and preferences
- **Batch Job Processing**: Process multiple JDs simultaneously
- **Analytics Dashboard**: Detailed matching analytics and insights
- **Integration APIs**: Connect with ATS and HR systems

### Scalability Improvements
- **Database Integration**: PostgreSQL/MongoDB for metadata
- **Distributed Processing**: Celery for background tasks
- **Container Deployment**: Docker and Kubernetes support
- **Cloud Integration**: AWS/Azure deployment options

## 📞 Support

For issues, questions, or contributions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed information
4. Provide logs and error messages


---

**This project is made inside GyanSys**

*Last Updated: August 2025*
