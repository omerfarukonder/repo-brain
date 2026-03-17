# RepoBrain PRD

AI System for Repository Understanding, Architecture Mapping, and Change
Impact Analysis

Author: Omer Onder Version: 1.1 Status: MVP Design Target: Local-first
AI system with optional OpenAI API integration

------------------------------------------------------------------------

# 1. Overview

RepoBrain is a local-first AI system that analyzes software repositories
and builds a structured understanding of the codebase.

The system parses repository structure, extracts dependencies,
constructs architecture graphs, and generates natural language
explanations about how the system works.

RepoBrain also supports change impact analysis and effort estimation to
assist developers and technical planners.

The system is designed to run **fully locally**, but supports **optional
API integration with OpenAI** for higher-quality reasoning.

------------------------------------------------------------------------

# 2. Goals

Primary goals:

1.  Automatically analyze software repositories
2.  Extract architecture and dependency structures
3.  Generate explanations of code modules
4.  Answer natural language questions about the repository
5.  Estimate the impact of proposed changes
6.  Estimate implementation effort

Secondary goals:

-   Provide architecture diagrams
-   Detect code risk hotspots
-   Support multiple programming languages
-   Operate on private repositories without uploading code externally

------------------------------------------------------------------------

# 3. Non Goals

The following are out of scope for MVP:

-   Code generation
-   Code editing
-   CI/CD integration
-   Security vulnerability scanning
-   Production deployment tooling

------------------------------------------------------------------------

# 4. Target Users

Primary users:

-   software engineers
-   technical leads
-   engineering managers
-   AI researchers exploring code intelligence

Secondary users:

-   product managers
-   system architects

------------------------------------------------------------------------

# 5. Core Capabilities

## Repository Understanding

-   detect languages
-   detect frameworks
-   extract file structure
-   identify entry points

## Structural Analysis

-   extract modules
-   extract classes
-   extract functions
-   build dependency graph

## Architecture Detection

-   identify architecture layers
-   classify modules by role

## Code Intelligence

-   module summaries
-   system explanation
-   execution flow analysis

## Impact Analysis

-   determine modules affected by change request
-   analyze dependency propagation

## Effort Estimation

-   estimate engineering complexity
-   estimate effort range
-   detect risks

## Interactive Query

-   answer questions about repository behavior

------------------------------------------------------------------------

# 6. High-Level System Architecture

Repository ↓ Repo Scanner ↓ Code Parser ↓ Structure Extractor ↓
Dependency Graph Builder ↓ Architecture Analyzer ↓ Knowledge Index ↓ LLM
Reasoning Layer ↓ CLI / API Interface

------------------------------------------------------------------------

# 7. Technology Stack

Primary language: Python 3.11+

Core libraries:

tree-sitter (code parsing) networkx (dependency graph)
sentence-transformers (embeddings) chromadb or faiss (vector store)

LLM support:

Local: - Ollama - llama3 - mistral - qwen

API: - OpenAI API - GPT-4o

Visualization:

-   Graphviz
-   Mermaid diagrams

Optional UI:

-   Streamlit

------------------------------------------------------------------------

# 8. System Modules

## Repo Scanner

Scans repository and detects languages, frameworks, entry points and
structure.

Output example:

{ "languages": \["python","javascript"\], "framework": "fastapi",
"entry_points": \["main.py"\], "file_count": 328 }

------------------------------------------------------------------------

## Code Parser

Parses files using AST and extracts:

-   functions
-   classes
-   imports

Uses tree-sitter for multi-language parsing.

------------------------------------------------------------------------

## Dependency Graph Builder

Constructs graph relationships between modules.

Example:

auth_routes → auth_service auth_service → user_repository
user_repository → database

Uses NetworkX.

------------------------------------------------------------------------

## Architecture Analyzer

Detects architectural layers using heuristics.

Example output:

{ "pattern": "Layered Architecture", "layers":
\["API","Service","Data","Database"\] }

------------------------------------------------------------------------

## Module Summarizer

Uses LLM to explain module responsibilities.

Example:

Module: auth_service.py

Description: Handles authentication operations including login, token
validation and password hashing.

------------------------------------------------------------------------

## Execution Flow Mapper

Maps flow of requests.

Example:

HTTP Request → auth_routes.login() → auth_service.authenticate() →
user_repository.get_user() → database

------------------------------------------------------------------------

## Change Impact Analyzer

Input example:

"Add Google SSO login"

Output:

{ "affected_modules": \[ "auth_routes.py", "auth_service.py",
"user_model.py" \] }

------------------------------------------------------------------------

## Effort Estimation Engine

Score factors:

-   number of files affected
-   number of architecture layers touched
-   module centrality
-   dependency count
-   test coverage

Score formula:

effort_score = files_score + layer_score + dependency_score + risk_score

Output example:

Estimated Complexity: High Engineering Effort: 4--7 days Testing Effort:
1--2 days Confidence: 70%

------------------------------------------------------------------------

# 9. Hallucination Prevention & Answer Validity

RepoBrain must implement strict safeguards to prevent hallucinations.

Rules:

1.  The system must **only answer questions grounded in the analyzed
    repository data**.
2.  If a question cannot be answered from repository context, the system
    must respond:

"Unable to determine from repository analysis."

3.  If the user question is unrelated to the repository, the system must
    respond:

"This question is not related to the analyzed repository."

4.  The LLM must never fabricate modules, files, or behaviors that are
    not detected in the parsed repository.

5.  Retrieval-Augmented Generation (RAG) must always provide the LLM
    with real context chunks before generating an answer.

6.  If confidence is low, output:

"Low confidence answer --- repository evidence is limited."

------------------------------------------------------------------------

# 10. Multilingual Code & Comment Understanding

Repositories may contain comments or documentation in languages other
than English.

RepoBrain must support multilingual understanding.

Requirements:

1.  Detect language of comments using language detection libraries.
2.  Translate non-English comments into English before LLM reasoning.
3.  Preserve original comment text for reference.
4.  Support at minimum:

-   Turkish
-   Spanish
-   German
-   French
-   Chinese

Example:

Original comment: "Bu fonksiyon kullanıcı doğrulaması yapar."

Translated context: "This function performs user authentication."

The translated version should be used during analysis.

------------------------------------------------------------------------

# 11. CLI Interface

Example commands:

repobrain analyze ./repo

repobrain architecture ./repo

repobrain ask "Where is authentication implemented?"

repobrain impact "Add Google SSO login"

------------------------------------------------------------------------

# 12. Output Artifacts

/analysis

repository_summary.json parsed_code.json dependency_graph.graphml
architecture_report.json module_summaries.json impact_report.json
effort_estimation.json

Visualization:

architecture_diagram.png dependency_graph.png

------------------------------------------------------------------------

# 13. Local LLM Configuration

config.yaml example:

llm_provider: local model: llama3

openai_api_key: "" openai_model: gpt-4o

If provider = openai, the system uses OpenAI API.

------------------------------------------------------------------------

# 14. Performance Targets

Target repository size:

≤ 100k LOC

Expected analysis time:

30--120 seconds depending on repository size.

------------------------------------------------------------------------

# 15. Security & Privacy

RepoBrain is designed to run locally.

Source code must not leave the machine unless the user explicitly
enables API mode.

------------------------------------------------------------------------

# 16. MVP Feature Set

Required MVP:

-   repo scanning
-   AST parsing
-   dependency graph generation
-   architecture detection
-   module summaries
-   CLI interface
-   local LLM support
-   optional OpenAI API

Impact analysis and effort estimation can be expanded in phase 2.

------------------------------------------------------------------------

# 17. Success Criteria

RepoBrain is successful if it:

-   analyzes repositories automatically
-   generates architecture explanations
-   answers repository questions accurately
-   detects impact of changes
-   operates locally on private repositories

------------------------------------------------------------------------

# 18. Example End-to-End Run

repobrain analyze ./example_repo

Example output:

Languages detected: Python Framework detected: FastAPI Files analyzed:
214

Architecture: Layered

Main modules: auth_service user_service order_service

Dependency graph built Module summaries generated

Repository health score: 74/100

------------------------------------------------------------------------

# 19. Repository Structure

repobrain/

src/ scanner/ parser/ graph/ architecture/ summarizer/ impact/ effort/
llm/

cli/ config/ analysis/ tests/ README.md

------------------------------------------------------------------------

# End of PRD
