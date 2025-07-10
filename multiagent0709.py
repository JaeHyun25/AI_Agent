# LLM Orchestrator Agent v2.1 - Multi-Step & Self-Correcting
# 07-01 09:34 db 쿼리 했을 때 결과가 너무 많으면 요약하는 함수 (동적 라우팅 설계 추가)
# 07-02 13:05 memory_retiever가 질문-답만 기억했는데, 쿼리, 데이터 결과의 과정도 기억할 수 있게 수정
# '''07-03 final 코드가 잘 동작한다. 하지만 가끔 conversation_history에 집중해서 말하지않았는데 질문과 상관없이 이전 대화에 대한 것을 찾을때가 있다.
#그래서 그 부분에 대해 수정을 시작한게 0704버전이다.''' 더해서 토큰 측정하는 코드도 첨부했다.
# 07-04 11:08 플래너가 기존 대화를 생각해서 좋은 계획 세우게 하기 추가
# 07-04 13:00 기억의 전달 실패에 대한 수정(answer_synthesizer) -> 실행 컨텍스트 누적 기능(run_ai_agent)
# 07-04 17:21 현재 많이 수정했고 이 버전은 sql 프롬프트만 수정해봄. 현재 0704_2도 성능 좋음...
# report, capacity agent 추가 중(07-06 13:00)
# 0707은 temperature 설정했는데, 안한 기본 버전인 0706이 나은거같아 이어서 0708로...이번엔 핫스팟 관련 프롬프트 추가
# 0708_2는 anomaly 함수와 툴에 관해수정.
import os
import io
import re
import sys
import json
import warnings
import logging
from datetime import datetime, timedelta
from contextlib import redirect_stdout
from typing import Optional, List, Dict

# RAG를 위한 라이브러리
import google.generativeai as genai
import chromadb

# Third-party libraries
import matplotlib.pyplot as plt
import pandas as pd
import pymysql
import numpy as np

# 2025-06-25 추가 pandas의 모든 float 출력 형식을 소수점 한 자리로 설정
pd.options.display.float_format = '{:.1f}'.format
# --- 0. 로깅 설정 (기존과 동일) ---
def setup_file_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"agent_log_{datetime.now().strftime('%Y-%m-%d')}.log")
    file_logger = logging.getLogger('FileLogger')
    file_logger.setLevel(logging.INFO)
    if not file_logger.handlers:    # 핸들러는 로그 메시지를 특정 대상(예: 파일, 콘솔, 네트워크 등)에 전달하는 역할
        file_handler = logging.FileHandler(log_filename, encoding='utf-8') # FileHandler를 생성해 로그를 파일에 기록
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)    # 위 포맷들을 핸들러에 설정, 위 포맷대로 로그 메시지가 출력
        file_logger.addHandler(file_handler)    # 생성한 핸들러를 file_logger에 추가, 이 핸들러를 통해 로그 메시지가 '파일'에 기록
    return file_logger

# --- 1. 환경 설정 및 전역 변수 (그래프 한글 설정 및 DB 접속) ---
warnings.filterwarnings("ignore")
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    print("폰트 설정 경고: 'Malgun Gothic' 폰트를 찾을 수 없습니다.")

DB_HOST = "192.168.0.242"
DB_PORT = 3306
DB_USER = "asi_agent"
DB_PASSWORD = "agent@asi"

# Gemini API 키 
genai.configure(api_key="AIzaSyDGQWJ6sQWfc8JToxCw9ioXFegBIXdHQLE")
gemini_model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

# --- RAG 관련 설정 및 함수 ---
try:
    # 1. SQL 생성을 위한 RAG DB - DB에 나와있는 value 
    sql_rag_client = chromadb.PersistentClient(path="chroma_db")    # ChromaDB(벡터DB)의 클라이언트를 생성
    sql_rag_collection = sql_rag_client.get_collection(name="device_object_names") # 해당 DB에서 특정 컬렉션(테이블) 가져옴 - RAG_DB_GIST.py에서 생성한 컬렉션 이름 확인 필요
    print("✅ SQL RAG DB 로딩 성공.")
    
    # 2. 논문 지식 탐색을 위한 RAG DB 07.01 15:42 추가
    paper_rag_client = chromadb.PersistentClient(path="chroma_db_paper")
    paper_rag_collection = paper_rag_client.get_collection(name="paper_rag") # pdf_RAG.py에서 생성한 컬렉션 이름
    print("✅ Paper RAG DB 로딩 성공.")

    # [NEW] 3. 질문-SQL 족보를 위한 RAG DB - 07.03 15:51 추가
    qa_rag_client = chromadb.PersistentClient(path="chroma_db")
    qa_rag_collection = qa_rag_client.get_collection(name="gist_qa_v1")
    print("✅ Q&A RAG DB 로딩 성공.")

except Exception as e:
    print(f"❌ Vector DB 로딩 실패: {e}")
    sys.exit()

# 기존 retrieve_context 함수는 이름을 명확하게 변경 - 07.01 15:44 수정
def retrieve_sql_context(query: str, n_results: int = 10) -> str:
    """[SQL 생성용] 사용자 질문과 가장 관련 높은 장비/센서 이름을 Vector DB에서 검색"""
    if sql_rag_collection.count() == 0:
        return "참조할 SQL 지식 베이스가 비어있습니다."
    
    query_embedding = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="RETRIEVAL_QUERY"
    )['embedding']
    
    results = sql_rag_collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    retrieved_docs = results['documents'][0]
    context_str = "\n- ".join(retrieved_docs)
    return "- " + context_str if context_str else "관련된 장비/센서 이름을 찾지 못했습니다."

# [NEW] 논문 RAG 검색을 위한 새로운 함수 추가 - 07.01 15:44 수정
# [수정] 텍스트와 함께 출처(소스) 파일명도 반환하도록 변경
def retrieve_paper_context(query: str, n_results: int = 5) -> tuple[str, list]:
    """[지식 탐색용] Vector DB에서 컨텍스트와 해당 출처 파일명 리스트를 반환"""
    if paper_rag_collection.count() == 0:
        return "참조할 논문 지식 베이스가 비어있습니다.", []

    # 1. [추가] Google 모델로 직접 쿼리 텍스트를 임베딩합니다. (768차원 벡터 생성)
    query_embedding = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="RETRIEVAL_QUERY"
    )['embedding']
    
    # 2. [수정] 텍스트 대신 생성된 벡터(임베딩)를 전달합니다.
    results = paper_rag_collection.query(
        query_embeddings=[query_embedding], # query_texts -> query_embeddings
        n_results=n_results,
        include=['documents', 'metadatas']  # 검색 결과에 문서내용, 메타데이터를 함께 반환
    )
    
    retrieved_chunks = results['documents'][0]
    source_files = list(set(meta['filename'] for meta in results['metadatas'][0] if 'filename' in meta))
    context_str = "\n\n---\n\n".join(retrieved_chunks)
    
    return context_str, source_files

# [NEW] 질문-SQL 족보를 검색하는 새로운 함수 추가 07.03 15:53
def retrieve_qa_examples(query: str, n_results: int = 3) -> str:
    """[Q&A용] 사용자 질문과 가장 유사한 질문-SQL 쌍을 Vector DB에서 검색"""
    if qa_rag_collection.count() == 0:
        return "참고할 Q&A 예시가 없습니다."

    query_embedding = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="RETRIEVAL_QUERY"
    )['embedding']
    
    results = qa_rag_collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    retrieved_docs = results['documents'][0]
    # Q: ...\nSQL: ...\n--- 식으로 format!
    qa_examples = []
    for doc in retrieved_docs:
        # md/jsonl 구조라면 파싱해서 Q/SQL 분리, 예시 포맷팅
        # 예: {"question": "...", "sql_query": "..."}
        try:
            import json
            item = json.loads(doc)
            qa_examples.append(f"Q: {item['question']}\nSQL: {item['sql_query']}\n---")
        except Exception:
            qa_examples.append(doc) # fallback
    return "\n".join(qa_examples) if qa_examples else "관련된 Q&A 예시를 찾지 못했습니다."
    # # 각 예시를 구분선으로 나누어 보기 좋게 합침
    # return "\n\n---\n\n".join(retrieved_docs) if retrieved_docs else "관련된 Q&A 예시를 찾지 못했습니다."

# --- <<수정>> 프롬프트 템플릿: Planner, SQL, Python 템플릿 최신화 ---
planner_prompt_template = """### Instruction:
You are an expert planner AI. Your job is to understand the user's request and create a step-by-step plan to fulfill it.

# [CRITICAL RULE] When the user asks for a calculated or aggregated value (e.g., average, sum, count, min, max),
# the `db_querier` tool is responsible for performing that calculation directly within the SQL query.
# Your plan's description for the `db_querier` step MUST reflect this calculation task.
# Do NOT create a separate synthesizer/summarizer step for simple calculations that the database can handle.

# [CRITICAL RULE for RAG-based Planning]
# If the RAG context provides a complete, single SQL query that solves the entire user request, your plan MUST consist of a single 'db_querier' step. Do NOT break it down into multiple steps.

# [CRITICAL CONTEXTUAL PLANNING RULE]
# When the user's question contains pronouns (e.g., '그것', '그게', that, it) or contextual references ('방금', '아까', '이전', 'just now'), you MUST examine the `Conversation History` to resolve the context.
# 1. If the answer is DIRECTLY stated in the previous agent response, your plan MUST use the `memory_retriever` tool.
# 2. If the history only provides CONTEXT for a new database query (e.g., identifying "that PDU" is "PDU 16"), your plan MUST create a more SPECIFIC `db_querier` step with a description that includes the resolved context.

# [CRITICAL RULE for Final Step]
# - For requests that ask for a "report", "document", or "summary file" (보고서, 문서, 요약 파일), your plan MUST end with the `report_generator` tool.
# - For all other questions that require a synthesized answer from data, the plan should end with `answer_synthesizer`.
# - A single plan MUST NOT contain both `report_generator` and `answer_synthesizer`. Choose only one as the final step.

# [CRITICAL RULE for Comparisons]
# For questions that require comparing two or more items (e.g., "compare A and B"),
# you MUST create a single `db_querier` step that fetches all the necessary information for the comparison at once.
# Do NOT create separate `db_querier` steps for each item.

# Bad Plan:
# - Step 1: db_querier (Get data for A)
# - Step 2: db_querier (Get data for B)
# - Step 3: synthesizer (Compare A and B)

# Good Plan:
# - Step 1: db_querier (Get data for both A and B in a single query)
# - Step 2: synthesizer (Analyze the combined data and compare)

You have the following tools available:
- `db_querier`: Used to query a database to retrieve data.
- `visualizer`: Used to create a plot or chart from data.
- `anomaly_detector`: Used to analyze data to find anomalies (e.g., abnormally high values).
- `general_qa`: Used for conceptual or explanatory questions about technologies, principles, or definitions (e.g., "What is PUE?", "Explain how a bus duct works."). This tool will search a knowledge base of technical documents to answer the question.
- `answer_synthesizer`: Uses retrieved data to formulate a final, natural language answer to the user's original question.
- `memory_retriever`: When the user asks a follow-up question (using pronouns like 'that', 'it', '그것', '이전','아까','방금','그게') that can be answered from the recent conversation history, use this tool to extract the answer directly from memory without accessing the database.
- data_summarizer: Summarizes a large table of data into key statistics before final analysis.
- `report_generator`: Used to combine data, tables, and plots from previous steps into a single, structured report in Markdown format. Use this when the user asks for a "report", "summary document", or "briefing".


Based on the user's request, create a plan as a JSON array of steps. Each step must have a "tool" and a "description".

### Examples

**Request:** "GIST PUE 알려줘"
**Plan:**
```json
[
    {{
        "tool": "db_querier",
        "description": "Query the database to get the latest PUE value."
    }}
]

**Request (with simulated RAG context):**
- *User's Request:* "6월 한달간 온도 경보가 주 10회 이상 발생한 항온항습기를 찾아서, 해당 기간 중 최고 온도를 알려줘."
- *Simulated RAG Context:* (A complete, single SQL query for this question has been found in the RAG examples)

**Plan (GOOD - follows the RAG example):**
```json
[
    {{
        "tool": "db_querier",
        "description": "RAG 예시에서 찾은 단일 쿼리를 사용하여, 6월 한 달간 온도 경보가 주 10회 이상 발생한 항온항습기의 최고 온도와 시간을 한 번에 조회합니다."
    }}
]

**Request:** "이번 주 랙별 전력 사용량이 얼마야? 그래프로도 보여줘"
**Plan:**
```json
[
    {{
        "tool": "db_querier",
        "description": "이번 주 랙별 전력 사용량 데이터를 데이터베이스에서 조회합니다."
    }},
    {{
        "tool": "visualizer",
        "description": "1단계에서 얻은 데이터를 사용하여 랙별 전력 사용량을 막대 그래프로 생성합니다."
    }}
]

**Request:** "PDU Rack 중 전력에 이상이 있는 것을 알려줘"
**Plan:**
```json
[
    {{
        "tool": "db_querier",
        "description": "분석을 위해 최근 3일간의 모든 PDU 랙별 전력 데이터를 조회합니다."
    }},
    {{
        "tool": "anomaly_detector",
        "description": "1단계에서 얻은 전체 전력 데이터에서 통계적으로 이상치를 보이는 PDU 랙을 탐지합니다."
    }}
]   

**Request:** "온습도계 17번 온도가 25도를 넘었어?"
**Plan:**
```json
[
    {{
        "tool": "db_querier",
        "description": "온습도계 17번의 현재 온도를 데이터베이스에서 조회합니다."
    }},
    {{
        "tool": "answer_synthesizer",
        "description": "1단계에서 조회한 온도 값과 사용자의 질문('25도를 넘었어?')을 비교하여 최종적으로 '예/아니오' 답변을 생성합니다."
    }}
]

**Request:** (After being told that Chamber 5 had the lowest temperature) "그게 몇 번 챔버야?"
**Plan:**
```json
[
    {{
        "tool": "memory_retriever",
        "description": "Extract the chamber number from the previous agent's response in the conversation history."
    }}
]

**Request:** "최근 6시간동안 PDU의 전력이 얼마나 변동됐어?"
**Plan:**
```json
[
    {{
        "tool": "db_querier",
        "description": "최근 6시간 동안의 모든 PDU 전력 데이터를 데이터베이스에서 조회합니다."
    }},
    {{
        "tool": "data_summarizer",
        "description": "1단계에서 얻은 방대한 전력 데이터를 최소/최대/평균 값 등 핵심 통계로 요약합니다."
    }},
    {{
        "tool": "answer_synthesizer",
        "description": "2단계에서 요약된 통계 데이터를 바탕으로 전력 변동에 대한 최종 답변을 생성합니다."
    }}
]

**Request (with conversation history):**
- *Conversation History:*  #는 번호
  - user: 현재 PDU #번에 측정되는 지표들을 알려줘
  - agent: PDU #번에 측정되는 지표는 다음과 같습니다: PDU_#_Rack_%-output_current: 11.9 ...
- *Current User Request:* "방금 PDU의 전류는 몇이야?"

**Plan (GOOD - uses history to make a specific plan):**
```json
[
    {{
        "tool": "db_querier",
        "description": "대화 기록을 참고하여 사용자가 질문하는 '방금 PDU'가 'PDU #'임을 파악하고, PDU #의 현재 전류 값을 데이터베이스에서 조회합니다."
    }}
]
**Plan (BAD - ignores history and makes a generic plan):**
```json
[
    {{
        "tool": "db_querier",
        "description": "가장 최근의 PDU 전류 값을 데이터베이스에서 조회합니다."
    }}
]

**Request:** "지난 24시간 동안의 PDU별 평균 전력 사용량으로 주간 보고서를 만들어주고, 상위 5개 PDU를 막대그래프로 보여줘."
**Plan:**
```json
[
    {{
        "tool": "db_querier",
        "description": "지난 24시간 동안 모든 PDU의 전력 데이터를 데이터베이스에서 조회합니다."
    }},
    {{
        "tool": "visualizer",
        "description": "조회된 데이터를 바탕으로 PDU별 평균 전력 사용량 상위 5개를 막대 그래프로 생성합니다."
    }},
    {{
        "tool": "report_generator",
        "description": "1, 2단계의 데이터와 그래프를 종합하여 주간 보고서를 작성합니다."
    }}
]

**Request:** "어제 핫스팟이 발생했어?"
**Plan:**
```json
[
    {{
        "tool": "db_querier",
        "description": "어제 AB랙에서 발생한 핫스팟(평균 온도 35°C 초과) 데이터를 데이터베이스에서 조회합니다."
    }},
    {{
        "tool": "data_summarizer",
        "description": "1단계에서 조회된 핫스팟 데이터를 분석하여 발생 건수, 최고 온도, 발생 시간 등 핵심 정보를 요약합니다."
    }},
    {{
        "tool": "answer_synthesizer",
        "description": "2단계에서 요약된 핫스팟 분석 결과를 바탕으로, 어제 핫스팟 발생 여부와 상세 내역에 대한 최종 답변을 생성합니다."
    }}
]

**Request:** "데이터센터에 특이사항이나 문제점 있어?"
**Plan:**
```json
[
    {{
        "tool": "db_querier",
        "description": "어제부터 현재까지 발생한 핫스팟(AB랙 35°C 초과)이 있었는지 조회합니다."
    }},
    {{
        "tool": "db_querier",
        "description": "같은 기간 동안의 주요 지표(온도, 습도, PUE)의 시계열 데이터를 조회하여 통계적 이상치를 탐지할 준비를 합니다."
    }},
    {{
        "tool": "anomaly_detector",
        "description": "2단계에서 조회된 주요 지표 데이터에서 비정상적인 패턴이나 이상치를 탐지합니다."
    }},
    {{
        "tool": "answer_synthesizer",
        "description": "1, 3단계의 결과를 종합하여, 핫스팟 발생 여부와 통계적 이상치 분석을 포함한 데이터센터의 전체적인 특이사항에 대한 최종 답변을 생성합니다."
    }}
]




**User Request:** "{user_question}"
**Plan:**
```json
"""

# SQL 프롬프트 쿼리
sql_gen_prompt_template = """### Instruction: Your Task and Decision-Making Process

You are an expert MySQL engineer. Your goal is to generate one single, perfect, and efficient query to answer the user's question. Follow this step-by-step process meticulously:
날짜에 관한 질문 중에 연도가 없으면 올해(2025년) 연도로 검색하세요.

**Step 1: Check for a "Cheat Sheet" Match in RAG Examples.**
- First, look at the "Retrieved Context from Q&A Examples". Is the `User's Current Question` almost **identical** to one of the examples?
- If YES, your highest priority is to copy and adapt the corresponding "모범 SQL 쿼리". This is your most critical rule. This step takes precedence over all others.

**Step 2: If No Perfect Match, Synthesize a New Query.**
- If there is no identical example, you must build a new query from scratch. To do this, you MUST use the following resources in this order of importance:
  1.  **The User's Current Question**: This is the primary goal you must fulfill.
  2.  **Retrieved Q&A Examples**: Use these as **inspiration** for the query structure and logic (e.g., how to compare items, how to find min/max, how to use window functions).
  3.  **Retrieved Knowledge Base (Terms)**: Use these to find the exact `description` or `objectName` for the devices mentioned in the user's question.
  4.  **General Querying Rules (Best Practices)**: You must follow these rules for efficiency and correctness.
  5.  **MySQL Compatibility Rules**: Finally, ensure your query is syntactically correct according to these rules.

**Step 3: A Note on Conversation History.**
- The `Conversation History` is mainly for understanding pronouns or very direct follow-ups. For generating new SQL, the RAG examples and the current question are far more important.

---
### General Querying Rules (For reference in Step 2)
1.  **Specific vs. General**: Use `o.description` for specific devices, `o.objectName` for general metrics like 'PUE'.
2.  **Data Validation**: ALWAYS use `v.value BETWEEN 0 AND 100` for temperature and humidity. Always use `v.value >= 0` for power summations.
3.  **Completeness**: For min/max/avg requests, select the device name (`description`) along with the value.
4.  **Averages for General Queries**: If the user asks for a trend of a device type without a specific number, calculate the overall average by removing the device name from `SELECT` and `GROUP BY`.
5.  **Efficiency**: Use `GROUP BY` for trends over long periods. Do not use `UNION ALL` for trend-vs-average plots; let Python handle the final average calculation.
6.  **Clarity**: Use parentheses `()` when mixing `AND` and `OR` to define the order of operations.

---

### MySQL Compatibility Rules:
1. Database uses `sql_mode=only_full_group_by`, so ALL columns in SELECT must be in GROUP BY or used in aggregate functions.
2. DO NOT use WITH ROLLUP syntax.
3. Avoid complex COALESCE expressions with non-aggregated columns.
4. If you need overall averages and detailed data together, use UNION or separate queries.
5. For date grouping, use `GROUP BY DATE(timestamp)` and include `DATE(timestamp)` in the SELECT list.

### Conversation History:
{conversation_history}

### Retrieved Context from Knowledge Base (Terms):
# DB의 실제 용어들입니다. 
{retrieved_context}

### Retrieved Context from Q&A Examples:
# 현재 질문과 유사한 과거의 질문과 모범 SQL 쿼리입니다. 질문을 먼저 여기서 검색하고 패턴을 참고하여 쿼리를 작성하세요.
{retrieved_qa_examples}

### Korean-to-Database Term Mapping:
- '항온항습기': `Constant_Temp_and_Humi_Chamber_`
  - '항온항습기 온도': `Constant_Temp_and_Humi_Chamber_#-current_temperature` (# = 번호)
  - '항온항습기 습도': `Constant_Temp_and_Humi_Chamber_#-current_humidity` (# = 번호)
  - '항온항습기 운전 상태': `Constant_Temp_and_Humi_Chamber_#-set_running_status` (# = 번호)
- '온습도계 #번 온도': 'Thermo_Hygrometer_#-temperature_ch2'  # ch2(환기 온도)를 기본값으로 설정(# = 번호)
- '온습도계 #번 습도': 'Thermo_Hygrometer_#-humidity_ch2'   # 습도도 ch2를 기본값으로 설정(# = 번호)
- '분전반': `Distribution_Board_`
- '버스덕트': `Bus_Duct_`
- '전력', '파워': `power` or `output_power`
- '온도': `temperature` or `current_temperature`
- '습도': `humidity` or `current_humidity`
- 'PDU': `PDU_`
- 'PUE': `description = 'Post_processing_data-PUE'`
- '핫스팟': `AB_hot_average_temperture`

# 분전반 명칭 매핑 (Distribution Board Naming Map):
# 1. 분전반 1번 (LP1 패널)
- 사용자 용어: "분전반 1", "1번 LP", "1번 LP1 패널"
- DB 패턴: `Distribution_Board_1_LP_1_Panel-[측정항목]`
- 예시: "분전반 1번 LP1 패널의 주파수" -> `description = 'Distribution_Board_1_LP_1_Panel-frequency'`

# 2. 분전반 2번 (UPS 패널)
- 사용자 용어: "분전반 2", "2번 UPS", "UPS 패널"
- DB 패턴: `Distribution_Board_2_UPS_Panel-[측정항목]`
- 예시: "분전반 2번 UPS 패널의 전력" -> `description = 'Distribution_Board_2_UPS_Panel-power'`

# 3. 분전반 3번 (LP-AC1 패널)
- 사용자 용어: "분전반 3", "3번 AC1", "3번 LP-AC1 패널"
- DB 패턴: `Distribution_Board_3_LP_AC1_Panel-[측정항목]`
- 예시: "분전반 3번 LP-AC1 패널의 R상 라인 전압" -> `description = 'Distribution_Board_3_LP_AC1_Panel-line_voltage_r'`

# 4. 분전반 4번 (LP-AC2 패널)
- 사용자 용어: "분전반 4", "4번 AC2", "4번 LP-AC2 패널"
- DB 패턴: `Distribution_Board_4_LP_AC2_Panel-[측정항목]`
- 예시: "분전반 4번 LP-AC2 패널의 역률" -> `description = 'Distribution_Board_4_LP_AC2_Panel-power_factor'`

# 5. 분전반 5번 (메인 패널)
- 사용자 용어: "분전반 5", "5번 메인"
- DB 패턴: `Distribution_Board_5_Main-[측정항목]`
- 예시: "분전반 5번 메인 S상 전류" -> `description = 'Distribution_Board_5_Main-current_s'`

# 측정항목 일반 규칙
- '전류': 별도 상(r, s, t) 언급 없으면 `-current_r`
- '라인 전압': 별도 상 언급 없으면 `-line_voltage_r`
- '상 전압': 별도 상 언급 없으면 `-phase_voltage_r`
- '전력': `-power`
- '역률': `-power_factor`
- '주파수': `-frequency`

### 항온항습기 상태 매핑 (Thermo-Hygrostat Status Mapping):
# 사용자가 항온항습기의 특정 작동 상태에 대해 질문할 경우, 아래 패턴을 사용하여 'description'을 구성합니다.
# 참고: 아래 항목들의 value가 1이면 '작동 중(ON)', 0이면 '정지(OFF)'를 의미합니다.

# 1. 냉방 (Cooling)
- 사용자 용어: "냉방", "냉방 중", "냉방 모드"
- DB 패턴: `Constant_Temp_and_Humi_Chamber_[장비번호]-running_status_coldroom`

# 2. 난방 (Heating)
- 사용자 용어: "난방", "난방 중", "난방 모드"
- DB 패턴: `Constant_Temp_and_Humi_Chamber_[장비번호]-running_status_warmroom`

# 3. 제습 (Dehumidifying)
- 사용자 용어: "제습", "제습 중", "제습 작동"
- DB 패턴: `Constant_Temp_and_Humi_Chamber_[장비번호]-running_decrease_humidity`

# 4. 가습 (Humidifying)
- 사용자 용어: "가습", "가습 중", "가습 기능"
- DB 패턴: `Constant_Temp_and_Humi_Chamber_[장비번호]-running_increase_humidity`

### 시멘틱 그룹 정의 (Semantic Group Definitions):
# 사용자가 추상적인 그룹(예: 전체 전력, IT 부하)에 대해 질문할 경우, 아래 정의를 사용하여 쿼리를 구성합니다.

# 1. '데이터센터 총 전력 소비량 (Total Power Consumption)'
#    - '총 전력량'은 아래 패턴에 해당하는 모든 장비의 '최신' 전력 값을 합산한 것입니다.
#    - 단위는 와트(W)이므로, 킬로와트(kW)로 변환하려면 1000으로 나누어야 합니다.
#    - 포함되어야 할 항목들 (LIKE 사용):
#      - 'PDU_%-output_power'
#      - 'Distribution_Board_%-power'
#      - 'Bus_Duct_%-power'
#      - 'Chamber_Power_Meter_%-active_power' -- 'active_power'가 실제 유효 전력입니다.
#      - 'Post_processing_data-Server_power'
#    - 제외되어야 할 항목들:
#      - 'power_factor'(역률), 'reactive_power'(무효전력) 등은 실제 소비 전력이 아니므로 합산에서 제외합니다.

### Additive Querying Rules:
1. **Choose the correct column (`objectName` vs. `description`)**:
   - For a **specific device and metric** (e.g., "PDU 1's power"), query the `o.description` column for the highest accuracy (e.g., `o.description LIKE 'PDU_1_%-output_power'`).
   - For a **general metric name** that acts as a standalone point (e.g., "PUE"), query the `o.objectName` column (e.g., `o.objectName = 'PUE'`).
2. **Use Precise `LIKE` Patterns**:
   - For 'PDU 1': use `LIKE 'PDU_1_%'`. The underscore `_` is a wildcard for a single character.
   - For '항온항습기 2번 온도': use `LIKE 'Constant_Temp_and_Humi_Chamber_2-current_temperature'`
   - For '항온항습기 2번 습도': use `LIKE 'Constant_Temp_and_Humi_Chamber_2-current_humidity'`
3. **Correctly `JOIN` tables** (`fms_object_value` as v, `fms_object_list` as o, `fms_device_list` as d) when you need information across them.
4. **Distinguish 'current' vs. 'overall' requests**: For "current" or "latest" data, use `ORDER BY v.timestamp DESC LIMIT 1`. For overall trends or averages without a time constraint, do not limit the time range.
5.  **Apply Data Validation Filters**: To prevent outliers, add reasonable range conditions to the `WHERE` clause.
    - For temperature (`온도`) queries, ALWAYS add `AND v.value BETWEEN 0 AND 100`.
    - For humidity (`습도`) queries, ALWAYS add `AND v.value BETWEEN 0 AND 100`.
6.  **Assume the current year** if a date is mentioned without a year. The current date is **{current_date}**.
7.  **Single Query**: Generate only one single, executable MySQL query.
8.  **Provide Complete Information**: When the user asks for a superlative (min, max, avg), you MUST select both the value itself AND the `description` or `deviceName` of the record it belongs to. Do not select only the value.
9. **Overall Average for General Queries**: If the user asks for a trend of a device type (e.g., 'PDU', '온습도계') *without* specifying a number or a specific name, generate a query that calculates the **overall average** for all devices of that type. Do this by removing the specific device name column (like `o.description`) from the `SELECT` and `GROUP BY` clauses.
10.  **Aggregate for Trends**: For trend analysis over long periods (e.g., 'a week', 'a month'), DO NOT fetch raw data. Instead, calculate hourly or daily averages directly in the SQL query using `AVG()` and `GROUP BY`. This is much more efficient.
12. **Avoid Complex Unions for Trend + Average**: For a "trend vs. average" plot, DO NOT use `UNION ALL` to combine trend data and the overall average in one query. It is much more stable and efficient to:
    1. Generate a simple query for the trend data only (e.g., `GROUP BY DATE(timestamp)`).
    2. Let the subsequent Python code calculate the overall average from that trend data (e.g., `df['value'].mean()`). This is the preferred method.

---
### Query Examples

#### Example 1: Specific device and metric -> Use `description`
Question: "GIST의 PDU 1번 장비의 전력을 알려줘"
MySQL Query:
USE gist_agent_test;
SELECT v.timestamp, v.value FROM fms_object_value v JOIN fms_object_list o ON v.object_ID = o.id WHERE o.description LIKE 'PDU_1_%-output_power' ORDER BY v.timestamp DESC LIMIT 1;

#### Example 2: General standalone metric -> Use `objectName`
Question: "최근 1시간 동안의 PUE 트렌드를 보여줘"
MySQL Query:
USE gist_agent_test;
SELECT v.timestamp, v.value FROM fms_object_value v JOIN fms_object_list o ON v.object_ID = o.id WHERE o.objectName = 'PUE' AND v.timestamp >= NOW() - INTERVAL 1 HOUR ORDER BY v.timestamp ASC;

#### Example 3: Korean term for specific device -> Use `description` with correct pattern
Question: "온습도계 2번의 현재 온도는 얼마야?"
MySQL Query:
USE gist_agent_test;
SELECT o.description, v.value, v.timestamp FROM fms_object_value v JOIN fms_object_list o ON v.object_ID = o.id WHERE o.description LIKE 'Thermo_Hygrometer_2-temperature_ch2' ORDER BY v.timestamp DESC LIMIT 1;

#### Example 4: Daily averages with overall average
Question: "지난 7일간 PUE 트렌드를 1일 간격으로 보여주고, 평균과 비교해줘"
MySQL Query:
USE gist_agent_test;
-- 일별 PUE 평균
SELECT DATE(v.timestamp) AS trend_date, AVG(v.value) AS daily_pue FROM fms_object_value v JOIN fms_object_list o ON v.object_ID = o.id WHERE o.objectName = 'PUE' AND v.timestamp >= DATE_SUB(CURDATE(), INTERVAL 7 DAY) GROUP BY DATE(v.timestamp) ORDER BY trend_date ASC;

#### Example 5: 항온항습기 온도/습도 조회 (correct pattern)
Question: "어제부터 오늘까지 항온 항습기에서 가장 높은 온도를 기록한게 몇 번 챔버야?"
MySQL Query:
USE gist_agent_test;
SELECT o.description AS chamber_name, v.value AS highest_temperature_celsius, v.timestamp FROM fms_object_value AS v JOIN fms_object_list AS o ON v.object_ID = o.id WHERE o.description LIKE 'Constant_Temp_and_Humi_Chamber_%-current_temperature' AND v.timestamp >= CURDATE() - INTERVAL 1 DAY AND v.timestamp <= NOW() AND v.value BETWEEN 0 AND 100 ORDER BY v.value DESC LIMIT 1;

#### Example 6: 항온항습기 습도 조회 (correct pattern)
Question: "2번 항온항습기의 습도가 얼마야?"
MySQL Query:
USE gist_agent_test;
SELECT o.description, v.value AS humidity_percent, v.timestamp FROM fms_object_value AS v JOIN fms_object_list AS o ON v.object_ID = o.id WHERE o.description = 'Constant_Temp_and_Humi_Chamber_2-current_humidity' ORDER BY v.timestamp DESC LIMIT 1;

#### Example 7: Finding Min/Max values with device info (BEST PRACTICE)
Question: "어제 항온항습기의 최저 온도는 몇 도 였어?"
MySQL Query:
USE gist_agent_test;
SELECT o.description, v.value FROM fms_object_value AS v JOIN fms_object_list AS o ON v.object_ID = o.id WHERE o.description LIKE 'Constant_Temp_and_Humi_Chamber_%-current_temperature' AND DATE(v.timestamp) = CURDATE() - INTERVAL 1 DAY AND v.value BETWEEN 0 AND 100 ORDER BY v.value ASC LIMIT 1;

#### Example 8:  경보 관련 -- v.value = 1인 '경보가 활성 상태'임을 의미
Question: "지금 경보 뜬 항온항습기 있어?"
MySQL Query:
USE gist_agent_test;
SELECT o.deviceName, o.objectName, v.timestamp FROM fms_object_value v JOIN fms_object_list o ON v.object_ID = o.id WHERE o.deviceName LIKE 'Constant_Temp_and_Humi_Chamber_%' AND o.objectName LIKE 'warn_%' AND v.value = 1 AND v.timestamp >= NOW() - INTERVAL 2 HOUR ORDER BY v.timestamp DESC;

#### Example 9 : 냉방, 난방 관련 ('난방' 이면 'Constant_Temp_and_Humi_Chamber_#-running_status_warmroom', #는 번호)
Question : "항온항습기 4번 지금 냉방 중이야?"
MySQL Query:
USE gist_agent_test;
SELECT o.description, v.value, v.timestamp FROM fms_object_value v JOIN fms_object_list o ON v.object_ID = o.id WHERE o.description = 'Constant_Temp_and_Humi_Chamber_4-running_status_coldroom' ORDER BY v.timestamp DESC LIMIT 1;

#### Example 10 : '이번 주' 라고 물었을 때 기간 설정
Question : "이번 주 항온항습기 9의 평균 습도는 어느정도야?"
MySQL Query:
USE gist_agent_test;
SELECT o.description, AVG(v.value) AS average_humidity_percent FROM fms_object_value v JOIN fms_object_list o ON v.object_ID = o.id WHERE o.description = 'Constant_Temp_and_Humi_Chamber_9-current_humidity' AND WEEK(v.timestamp) = WEEK(CURDATE()) AND YEAR(v.timestamp) = YEAR(CURDATE()) AND v.value BETWEEN 0 AND 100;

#### Example 11 : 
Question : "여러 분전반 중 현재 전류가 가장 낮은 패널은 어디이고, 전류 값은 얼마야? "
MySQL Query:
USE gist_agent_test; 
SELECT o.deviceName, v.value FROM fms_object_value v JOIN fms_object_list o ON v.object_ID = o.id WHERE o.objectName = 'current_r' AND o.deviceName LIKE 'Distribution_Board_%' ORDER BY v.value ASC LIMIT 1;

#### Example 12: 두 개를 비교 할 때
Question : "분전반 2 UPS 패널이랑 분전반 4 AC2 패널 중 어디 전류가 더 높아?"
MySQL Query:
USE gist_agent_test; 
SELECT (SELECT v.value FROM fms_object_value v JOIN fms_object_list o ON v.object_ID = o.id WHERE o.description = 'Distribution_Board_2_UPS_Panel-current_r' ORDER BY v.timestamp DESC LIMIT 1) AS db2_current, (SELECT v.value FROM fms_object_value v JOIN fms_object_list o ON v.object_ID = o.id WHERE o.description = 'Distribution_Board_4_LP_AC2_Panel-current_r' ORDER BY v.timestamp DESC LIMIT 1) AS db4_current;

#### Example 13: 두 장비의 최신 값 비교 (효율적인 방법)
Question: "항온항습기 4번과 5번의 현재 온도를 비교해줘."
MySQL Query:
# This query efficiently finds the latest timestamp for each specified device first, then joins to get the values.
USE gist_agent_test;
SELECT o.description, v.value FROM fms_object_value AS v JOIN (SELECT object_ID, MAX(timestamp) AS max_timestamp FROM fms_object_value WHERE object_ID IN (SELECT id FROM fms_object_list WHERE description IN ('Constant_Temp_and_Humi_Chamber_4-current_temperature', 'Constant_Temp_and_Humi_Chamber_5-current_temperature')) GROUP BY object_ID) AS latest_data ON v.object_ID = latest_data.object_ID AND v.timestamp = latest_data.max_timestamp JOIN fms_object_list AS o ON v.object_ID = o.id;

#### Example 14: 두 장비의 최신 값 비교 (효율적인 방법)
Question: "PDU 4랑 5의 현재 전류 값 비교해줘."
MySQL Query:
USE gist_agent_test; 
SELECT (SELECT v.value FROM fms_object_value v JOIN fms_object_list o ON v.object_ID = o.id WHERE o.description = 'PDU_4_Rack_A1-output_current' AND v.timestamp >= NOW() - INTERVAL 2 HOUR ORDER BY v.timestamp DESC LIMIT 1) AS pdu4_current, 
(SELECT v.value FROM fms_object_value v JOIN fms_object_list o ON v.object_ID = o.id WHERE o.description = 'PDU_5_Rack_A2-output_current' AND v.timestamp >= NOW() - INTERVAL 2 HOUR ORDER BY v.timestamp DESC LIMIT 1) AS pdu5_current;

#### Example 15: 전류 값 묻기
Question : "어제 PDU 3에서 사용된 최대 전류는 몇 A였어?"
MySQL Query:
USE gist_agent_test;
SELECT MAX(v.value) AS max_current 
FROM fms_object_value v JOIN fms_object_list o ON v.object_ID = o.id 
WHERE o.description = 'PDU_3_RACK_A1-output_current' AND DATE(v.timestamp) = CURDATE() - INTERVAL 1 DAY;

#### Example 16: 온도 묻기
Question : "현재 데이터센터 평균 온도는 몇 도야?"
MySQL Query:
USE gist_agent_test;
SELECT AVG(v.value) avg_temp
FROM fms_object_value v JOIN fms_object_list o ON v.object_ID = o.id
WHERE o.description LIKE 'Thermo_Hygrometer_%-temperature_ch2' AND v.timestamp >= NOW() - INTERVAL 2 HOUR AND v.value BETWEEN 0 AND 100;

#### Example 17: Specific vs. Overall Average Trend
-- Scenario A: User asks for a SPECIFIC device
Question: "지난 주 10번 온습도계의 습도 추세는 어땠어?"
MySQL Query:
USE gist_agent_test;
SELECT
  DATE_FORMAT(v.timestamp, '%Y-%m-%d %H:00:00') AS hour_timestamp,
  AVG(v.value) AS avg_humidity
FROM fms_object_value AS v
JOIN fms_object_list AS o ON v.object_ID = o.id
WHERE o.description = 'Thermo_Hygrometer_10-humidity_ch2' 
  AND v.timestamp >= CURDATE() - INTERVAL 7 DAY AND v.value BETWEEN 0 AND 100
GROUP BY hour_timestamp
ORDER BY hour_timestamp ASC;

-- Scenario B: User asks for the GENERAL device type
Question: "지난 주 온습도계의 전반적인 습도 추세는 어땠어?"
MySQL Query:
USE gist_agent_test;
SELECT
  DATE_FORMAT(v.timestamp, '%Y-%m-%d %H:00:00') AS hour_timestamp,
  AVG(v.value) AS overall_avg_humidity 
FROM fms_object_value AS v
JOIN fms_object_list AS o ON v.object_ID = o.id
WHERE o.description LIKE 'Thermo_Hygrometer_%-humidity_ch2'
  AND v.timestamp >= CURDATE() - INTERVAL 7 DAY AND v.value BETWEEN 0 AND 100
GROUP BY hour_timestamp 
ORDER BY hour_timestamp ASC;

#### Example 18: Comparing Multiple Devices and Metrics (Best Practice)
Question: "항온항습기 1과 2의 오늘 평균 온도와 습도를 알려줘. 어느 기기가 더 시원하게 유지되고 있어?"
MySQL Query:
USE gist_agent_test;
SELECT
    AVG(CASE WHEN o.description = 'Constant_Temp_and_Humi_Chamber_1-current_temperature' THEN v.value END) AS c1_temp_avg,
    AVG(CASE WHEN o.description = 'Constant_Temp_and_Humi_Chamber_1-current_humidity' THEN v.value END) AS c1_hum_avg,
    AVG(CASE WHEN o.description = 'Constant_Temp_and_Humi_Chamber_2-current_temperature' THEN v.value END) AS c2_temp_avg,
    AVG(CASE WHEN o.description = 'Constant_Temp_and_Humi_Chamber_2-current_humidity' THEN v.value END) AS c2_hum_avg
FROM fms_object_value AS v JOIN fms_object_list AS o ON v.object_ID = o.id 
WHERE o.description IN ('Constant_Temp_and_Humi_Chamber_1-current_temperature',
        'Constant_Temp_and_Humi_Chamber_1-current_humidity',
        'Constant_Temp_and_Humi_Chamber_2-current_temperature',
        'Constant_Temp_and_Humi_Chamber_2-current_humidity')
    AND DATE(v.timestamp) = CURDATE()
    AND v.value BETWEEN 0 AND 100;

#### Example 20: Checking Multiple States of a Single Device
Question: "항온항습기 2의 냉방 상태와 난방 상태가 지금 둘 다 켜져 있는지 알려줄래?" -- 참고: 상태값에서 1은 '켜짐(ON)', 0은 '꺼짐(OFF)'을 의미합니다.

MySQL Query:
USE gist_agent_test;
SELECT
    (SELECT v.value FROM fms_object_value v JOIN fms_object_list o ON v.object_ID = o.id WHERE o.description = 'Constant_Temp_and_Humi_Chamber_2-running_status_coldroom' ORDER BY v.timestamp DESC LIMIT 1) AS coldroom_status,
    (SELECT v.value FROM fms_object_value v JOIN fms_object_list o ON v.object_ID = o.id WHERE o.description = 'Constant_Temp_and_Humi_Chamber_2-running_status_warmroom' ORDER BY v.timestamp DESC LIMIT 1) AS warmroom_status;
    

#### Example 21:
Question: "지금 데이터센터의 주요 환경 지표와 전력 지표들을 간단히 알려줘" -- 참고:(예: 평균 온도, 습도, PUE)
MySQL Query:
USE gist_agent_test;
SELECT (SELECT AVG(v.value) FROM fms_object_value v JOIN fms_object_list o ON v.object_ID = o.id WHERE o.objectName = 'current_temperature') AS avg_temp_all, (SELECT AVG(v.value) FROM fms_object_value v JOIN fms_object_list o ON v.object_ID = o.id WHERE o.objectName = 'current_humidity') AS avg_hum_all, (SELECT v.value FROM fms_object_value v JOIN fms_object_list o ON v.object_ID = o.id WHERE o.objectName = 'PUE' ORDER BY v.timestamp DESC LIMIT 1) AS current_PUE;

#### Example 22: Comparing Multiple Devices - "()" 괄호 사용!
Question : " 모든 항온항습기의 지난 3일간 평균 온도와 습도를 알려줘." 
MySQL Query:
USE gist_agent_test;
SELECT
  AVG(CASE WHEN o.description LIKE 'Constant_Temp_and_Humi_Chamber_%-current_temperature' THEN v.value END) AS average_temperature_celsius,
  AVG(CASE WHEN o.description LIKE 'Constant_Temp_and_Humi_Chamber_%-current_humidity' THEN v.value END) AS average_humidity_percent
FROM fms_object_value AS v
JOIN fms_object_list AS o
  ON v.object_ID = o.id
WHERE
  (o.description LIKE 'Constant_Temp_and_Humi_Chamber_%-current_temperature' OR o.description LIKE 'Constant_Temp_and_Humi_Chamber_%-current_humidity')
  AND v.timestamp >= CURDATE() - INTERVAL 3 DAY
  AND v.value BETWEEN 0 AND 100;

#### Example 23: Querying a Specific Distribution Board Panel
Question: "분전반 4번 LP-AC2 패널의 현재 역률은 얼마야?"
MySQL Query:
USE gist_agent_test;
SELECT
  v.value
FROM fms_object_value AS v
JOIN fms_object_list AS o
  ON v.object_ID = o.id
WHERE
  o.description = 'Distribution_Board_4_LP_AC2_Panel-power_factor'
ORDER BY
  v.timestamp DESC
LIMIT 1;

#### Example 24 : Calculating Total Power Consumption using a Semantic Group (Advanced)
Question : "현재 데이터센터 전체 전력 소비량은 몇 kW 정도야?"
MySQL Query:
USE gist_agent_test;
SELECT SUM(latest_power_values.value / 1000) total_current_power_kW FROM (SELECT v.value, ROW_NUMBER() OVER (PARTITION BY o.id ORDER BY v.timestamp DESC) rn 
FROM fms_object_value v JOIN fms_object_list o ON v.object_ID = o.id 
WHERE (o.description LIKE 'PDU_%-output_power' or o.description like 'Distribution_Board_%-power' or o.description like 'Bus_Duct_%-power' or o.description like 'Chamber_Power_Meter_%-active_power' or o.description = 'Post_processing_data-Server_power') 
AND v.value IS NOT NULL AND v.value >= 0 AND v.timestamp >= now() - interval 1 DAY) as latest_power_values 
WHERE latest_power_values.rn = 1;

#### Example 25 : Hot spot question
Question : "어제(최근) 핫스팟 발생했어?"
MySQL Query:
USE gist_agent_test;
SELECT v.timestamp, v.vlaue 
FROM fms_object_value v JOIN fms_object_list o ON v.object_ID = o.id
WHERE o.objectName = 'AB_hot_average_temperature' AND v.value > 35 AND DATE(v.timestamp) = CURDATE() - INTERVAL 1 DAY 
ORDER BY v.timestamp DESC;

Question: {user_question}
### MySQL Query:
"""

# Python  코드 생성용 프롬프트
python_gen_prompt_template = """### Instruction:
You are a Python expert specializing in pandas, matplotlib, and file operations.
Based on the user's request, generate a single, executable block of Python code to perform the task.

### CRITICAL RULE:
- A pandas DataFrame named `df` has ALREADY been loaded with all necessary data.
- You MUST use this `df` variable.
- **DO NOT, under any circumstances, connect to a database, read a file, or create your own dummy data.** Your entire script must assume `df` exists and work with it. Any code that tries to load data will be rejected.

### Context for the plot:
- The user wants to visualize the data based on this request: "{description}"
- The available columns in the `df` are: {df_columns}

### Your Task:
Write a Python script that takes the existing `df` and creates a plot according to the user's request and the styling guidelines.


### Design & Style Guidelines:

# [MANDATORY] You MUST include the following code block at the beginning of your script to set up a Korean font.
# You MUST then use the `font_prop` variable for all text elements to ensure Korean characters are displayed correctly.
""
# --- 한글 폰트 설정 (그래프 코드에 반드시 포함) ---
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

try:
    # Windows 환경에서 '맑은 고딕' 폰트 경로를 직접 지정
    font_path = 'c:/Windows/Fonts/malgun.ttf'
    font_prop = fm.FontProperties(fname=font_path, size=12)
    plt.rc('font', family='Malgun Gothic')
except FileNotFoundError:
    # '맑은 고딕'이 없는 경우, 나눔고딕으로 대체 시도 (주로 Linux/Mac)
    try:
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
        font_prop = fm.FontProperties(fname=font_path, size=12)
        plt.rc('font', family='NanumGothic')
    except FileNotFoundError:
        # 어떤 폰트도 찾을 수 없는 경우, 경고 메시지 출력
        print("한글 폰트를 찾을 수 없습니다. 기본 폰트로 표시됩니다.")
        font_prop = fm.FontProperties(size=12) # 오류 방지를 위한 기본 폰트 속성

plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
# --- 폰트 설정 끝 ---
""

# [CRITICAL] Apply the `font_prop` to ALL text elements like this:
# - plt.title("그래프 제목", fontproperties=font_prop)
# - plt.xlabel("X축 레이블", fontproperties=font_prop)
# - plt.ylabel("Y축 레이블", fontproperties=font_prop)
# - plt.legend(prop=font_prop)  <-- Note: legend uses 'prop'
# - plt.xticks(fontproperties=font_prop)
# - plt.yticks(fontproperties=font_prop)

# [READABILITY] To prevent X-axis labels from overlapping:
# 1. Rotate the labels: `plt.xticks(rotation=45, ha='right')`
# 2. Use automatic date locators for time-series data to show fewer ticks if needed.
#    (e.g., from matplotlib.dates import DayLocator)
# 3. Always call `plt.tight_layout()` at the end to adjust spacing.

# ### Rules:
# 1. Assume the necessary libraries like pandas (as pd), matplotlib.pyplot (as plt), and datetime are imported.
# 2. If the data is already provided as 'df' DataFrame, use it directly. 
# 3. **For database access, ALWAYS use pymysql with these parameters**:
#    ```
#    conn = pymysql.connect(
#        host='192.168.0.242',
#        user='asi_agent',
#        password='agent@asi',
#        database='gist_agent_test',
#        charset='utf8mb4'
#    )
#    ```
# 4. **For PUE data, use this SQL query**:
#    ```sql
#    SELECT v.timestamp, v.value as PUE 
#    FROM fms_object_value v 
#    JOIN fms_object_list o ON v.object_ID = o.id 
#    WHERE o.objectName = 'PUE' 
#    AND v.timestamp >= DATE_SUB(NOW(), INTERVAL <N> DAY)
#    ORDER BY v.timestamp
#    ```
#    Replace <N> with the number of days needed.
# 5. **For PDU power data, use this query pattern**:
#    SELECT v.timestamp, v.value as power, o.description
#    FROM fms_object_value v 
#    JOIN fms_object_list o ON v.object_ID = o.id 
#    WHERE o.description LIKE 'PDU_%_Rack_%-output_power'
#    AND v.timestamp >= DATE_SUB(NOW(), INTERVAL <N> DAY)
#    ORDER BY v.timestamp, o.description
# 6. **For processing PDU power data by rack, use this pattern**:
#    # Extract rack information
#    df['rack'] = df['description'].str.extract(r'(Rack_[A-Za-z0-9]+)')[0]
   
#    # Create pivot table
#    pivot_df = df.pivot_table(
#        index='timestamp', 
#        columns='rack',
#        values='power',
#        aggfunc='mean'
#    )
   
#    # Resample to desired time interval
#    df_resampled = pivot_df.resample('2H').mean()
# 7. Always save plots with a descriptive filename:
#     timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"plot_name_{{timestamp_str}}.png"
#     plt.savefig(filename)
#     print(f"그래프를 '{{filename}}' 파일로 저장했습니다.")
# 8. Always display the plot using plt.show()
# 9. Ensure the code is a single, executable block and uses utf-8 encoding for file I/O.
# 10. NEVER use simulated or random data when real data can be obtained from the database.

### Conversation History:
{conversation_history}

Request: {user_question}
### Python Code:
"""

# --- 2. 헬퍼 함수 및 도구(Tool) 정의 ---
# 07-04 토큰 측정 함수 추가
# [NEW] 세션 동안의 총 토큰 사용량을 기록하기 위한 전역 변수

token_tracker = {
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_tokens": 0
}

def call_gemini_with_token_tracking(prompt: str, purpose: str, file_logger: logging.Logger) -> str:
    """
    Gemini API를 호출하고, 토큰 사용량을 계산하여 로깅한 후, 텍스트 결과를 반환하는 래퍼 함수
    """
    try:
        response = gemini_model.generate_content(prompt)

        # 토큰 사용량 추출 및 기록
        prompt_tokens = response.usage_metadata.prompt_token_count  # 입력에 사용되는 토큰 수
        completion_tokens = response.usage_metadata.candidates_token_count  # Gemini가 생성한 답변(출력)에 사용된 토큰 수
        total_tokens = response.usage_metadata.total_token_count    # 위 입력 + 출력 토큰 합계 

        log_message = (
            f"🪙 Token Usage ({purpose}): "
            f"Input={prompt_tokens}, Output={completion_tokens}, Total={total_tokens}"
        )
        file_logger.info(log_message)

        # 전역 트래커에 토큰 사용량 누적
        token_tracker["total_prompt_tokens"] += prompt_tokens
        token_tracker["total_completion_tokens"] += completion_tokens
        token_tracker["total_tokens"] += total_tokens

        return response.text
    
    except Exception as e:
        error_message = f"❌ Gemini API 호출 중 오류 ({purpose}): {e}"
        file_logger.error(error_message)
        # API 호출 실패 시, 빈 텍스트나 에러 메시지를 반환할 수 있습니다.
        return f"API 호출에 실패했습니다: {e}"


def extract_code(llm_output: str) -> str:
    """LLM의 다양한 출력 형식에 대응하여 코드 또는 JSON 블럭을 안정적으로 추출합니다."""
    match = re.search(r"```(?:python|sql|json)?\s*(.*?)```", llm_output, re.DOTALL) # LLM 출력에서 python, sql, json 매칭, DOTALL은 줄바꿈 포함하여 전체 텍스트 매칭
    if match:
        return match.group(1).strip()

    json_match = re.search(r'\[.*\]|\{.*\}', llm_output, re.DOTALL) # JSON 형식의 데이터 ([] or {}) 추출
    if json_match:
        return json_match.group(0).strip()
        
    keywords = ["### Python Code:", "### MySQL Query:"]
    for keyword in keywords:
        if keyword in llm_output:
            return llm_output.split(keyword, 1)[1].strip()

    return llm_output.strip()

def execute_sql(sql_query: str) -> pd.DataFrame:
    """SQL을 실행하고 결과를 DataFrame으로 반환합니다."""
    db_to_use = "gist_agent_test"
    
    # mysql 키워드 제거
    if sql_query.lower().startswith("mysql"):
        sql_query = sql_query[5:].strip()   # mysql SELECT * FROM table ~ 에서 mysql 제거 후 앞,뒤 공백 제거
    
    use_match = re.search(r"USE\s+`?(\w+)`?;", sql_query, re.IGNORECASE)    # re.IGNOREDCASE는 대소문자를 구분하지 않음
    if use_match:
        db_to_use = use_match.group(1)  # group(1)은 USE 뒤의 gist_agent_test를 반환
        sql_query = re.sub(r"USE\s+`?(\w+)`?;", "", sql_query, flags=re.IGNORECASE).strip() # sub로 sql쿼리에서 USE 구문 제거
    
    # WITH ROLLUP 구문 제거 (WITH ROLLUP - MySQL에서 그룹화된 데이터의 요약을 계산하는데, 더 복잡해질 까봐 제거)
    sql_query = re.sub(r"GROUP BY\s+(.+?)\s+WITH\s+ROLLUP", r"GROUP BY \1", sql_query, flags=re.IGNORECASE)
    
    # 비호환 COALESCE 패턴 수정 - LLM이 COALESCE 를 생성할 수 있어서 미리 제거. 일부 MySQL에서 지원안해서.
    if "COALESCE(DATE(v.timestamp)" in sql_query:
        sql_query = sql_query.replace(
            "COALESCE(DATE(v.timestamp), 'Overall Average') AS trend_date",
            "DATE(v.timestamp) AS trend_date"
        )
    
    print(f"실행할 SQL 쿼리: {sql_query}")
    print(f"접속할 DB: {db_to_use}")
    
    try:
        conn = pymysql.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD,
            database=db_to_use, charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor
        )
        
        # 커서 방식으로 실행 
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql_query)
                results = cursor.fetchall()
            
            # 결과가 있으면 DataFrame으로 변환
            if results:
                df = pd.DataFrame(results)
                print(f"쿼리 결과: {len(df)} 행 반환됨")
                conn.close()
                return df
            else:
                print("쿼리 결과가 없습니다.")
                conn.close()
                return pd.DataFrame()
                
        except Exception as e1:
            print(f"커서로 쿼리 실행 실패: {e1}")
            
            # 오류 발생 시 pandas 방식 시도 
            try:
                safe_sql_query = sql_query
                df = pd.read_sql_query(safe_sql_query, conn)    # read_sql_query 메서드를 사용하여 SQL 쿼리 사용 가능.
                print(f"pandas로 쿼리 결과: {len(df)} 행 반환됨")
                conn.close()
                return df
            except Exception as e2:
                print(f"pandas로 쿼리 실행 실패: {e2}")
                conn.close()
                return pd.DataFrame()
    except Exception as e:
        print(f"❌ SQL 실행 중 오류: {e}")
        return pd.DataFrame()

def execute_python(python_code: str, exec_globals: dict = {}) -> str:
    """Python 코드를 실행하고, 동적 변수(DataFrame 등)를 주입받습니다."""
    output_buffer = io.StringIO()   # Python 코드 실행 중 생성된 출력(print 같은거)을 저장할 버퍼를 생성.
    # 자주 사용되는 라이브러리를 기본 전역 변수(default_globals)로 설정
    try:
        default_globals = {
            "pd": pd, "plt": plt, "os": os, "np": np, 
            "datetime": datetime, "timedelta": timedelta, "io": io, "re": re
        }
        combined_globals = {**default_globals, **exec_globals}
        with redirect_stdout(output_buffer):
            exec(python_code, combined_globals)
        
        output = output_buffer.getvalue()   # output_buffer에 저장된 실행 결과를 문자열로 가져옴.

        if 'plt.savefig' in python_code:
            match = re.search(r"plt\.savefig\(['\"](.*?)['\"]\)", python_code)
            if match and os.path.exists(match.group(1)):
                output += f"\n✅ '{match.group(1)}' 파일이 생성되었습니다."
        return output if output else "코드가 실행되었으나, 별도의 출력은 없습니다."
    except Exception as e:
        return f"❌ Python 코드 실행 중 오류: {e}\n{output_buffer.getvalue()}"

# 수정된 범용적인 이상치 탐지 함수
def detect_anomalies(df: pd.DataFrame, metric_col: str, group_by_col: str) -> pd.DataFrame:
    """
    주어진 데이터프레임에서 특정 측정값(metric)의 이상치를 탐지하는 '범용' 도구 함수
    :param df: 분석할 원본 데이터프레임
    :param metric_col: 이상치를 탐지할 측정값 컬럼 이름 (예: 'power', 'average_temperature_celsius')
    :param group_by_col: 그룹화할 기준 컬럼 이름 (예: 'deviceName', 'measurement_date')
    """
    # 함수가 올바른 컬럼을 받았는지 확인
    if metric_col not in df.columns or group_by_col not in df.columns:
        return pd.DataFrame() 

    # 'power', 'deviceName' 대신 인자로 받은 컬럼 이름을 사용
    stats = df.groupby(group_by_col)[metric_col].agg(['mean', 'std']).reset_index()
    df_with_stats = pd.merge(df, stats, on=group_by_col)
    
    # NaN 값을 처리하기 위해 fillna(0) 추가 (표준편차가 0인 경우 NaN이 될 수 있음)
    df_with_stats['std'] = df_with_stats['std'].fillna(0)

    # 이상치 계산
    df_with_stats['is_anomaly'] = df_with_stats[metric_col] > (df_with_stats['mean'] + 2 * df_with_stats['std'])
    
    anomalies = df.loc[df_with_stats[df_with_stats['is_anomaly']].index]
    
    # 결과가 비어있지 않을 때만 요약 로직 실행
    if not anomalies.empty:
        anomaly_summary = anomalies.loc[anomalies.groupby(group_by_col)[metric_col].idxmax()]
        # 반환하는 컬럼도 동적으로 선택
        return anomaly_summary[[group_by_col, metric_col, 'timestamp']]
    else:
        return pd.DataFrame()

# --- 멀티스텝을 위한 모듈화된 함수들 --- 0704 토큰 출력때문에 수정
def get_plan(user_question: str, file_logger: logging.Logger) -> List[Dict]:    # file_logger를 인자로 받도록 수정
    """Planner를 호출하여 행동 계획을 JSON으로 수립합니다."""
    prompt = planner_prompt_template.format(user_question=user_question)
    # 새 함수를 호출하고, 목적을 'Planner'로 명시
    response_text = call_gemini_with_token_tracking(prompt, "Planner", file_logger)
    plan_str = extract_code(response_text)  # plan_str은 JSON 형식의 문자열 (예:[{ "tool": "db_querier", "description":"이번주~조회"}])
    return json.loads(plan_str)     # json.loads 는 JSON 문자열을 Python 객체로 변환하는 함수
    # loads 는 'load string'의 약자로, 문자열 형태의 JSON 데이터를 Python 객체로 '로드' 한다는 의미.

# 0704 기록 관련 기억때문에 수정
def generate_sql_code(description: str, conversation_history: List[Dict], file_logger: logging.Logger) -> str:  # file_logger 인자 유지
    """SQL 코드 생성을 전담하는 함수(두 종류의 RAG 모두 활용)"""
    # [수정] SQL 생성 시에는 과거 대화 기록을 참조하지 않아 LLM의 혼란을 방지합니다. (RAG와 대화기록이 중요합니다.)
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
    
    # [수정] 두 종류의 컨텍스트를 모두 검색
    retrieved_terms = retrieve_sql_context(description) # deviceName RAG
    retrieved_qa_examples = retrieve_qa_examples(description)   # Q&A Set RAG 새로 추가

    current_date_str = datetime.now().strftime("%Y-%m-%d")
    
    # [수정] 프롬프트에 두 컨텍스트를 모두 전달
    prompt = sql_gen_prompt_template.format(
        conversation_history="", # <-- 이 부분을 빈 문자열로 변경
        retrieved_context=retrieved_terms,
        retrieved_qa_examples=retrieved_qa_examples, # 새로 추가
        user_question=description,
        current_date=current_date_str
    )

    # 중간 확인을 위해 로그 출력
    print("[RAG Q&A 예시]\n", retrieved_qa_examples)
    print("[최종 프롬프트 일부]\n", prompt[:1000])

    response_text = call_gemini_with_token_tracking(prompt, "SQL_Generator", file_logger)    # 수정된 함수 호출
    return extract_code(response_text)

def generate_viz_code(description: str, df: pd.DataFrame, conversation_history: list, file_logger: logging.Logger) -> str:    
    """시각화 코드 생성을 전담하는 함수"""  
    # 대화 기록이 없으면 빈 문자열 사용 - description은 시각화에 대한 설명 "~ 그래프로 그려줘"
    history_str = ""
    if conversation_history:
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
    
    prompt = python_gen_prompt_template.format(
        df_columns=str(df.columns.to_list()),
        description=description,
        user_question=description,  # python_gen_prompt_template에서 템플릿 유연성을 위해 description을 2개로 사용
        conversation_history=history_str
    )
    # 07-07 수정 모두 call~token 변수로
    response_text = call_gemini_with_token_tracking(prompt, "Visualizer_Code_Generator", file_logger)
    return extract_code(response_text)

# 0701 
def summarize_table(df: pd.DataFrame) -> str:
    """모든 수치형 컬럼(센서값, 전력, 온도 등)에 대해 주요 변화/변동폭 요약"""
    if df.empty:
        return "변동 요약을 할 데이터가 없습니다."
    # 수치형 컬럼 자동 탐색
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not num_cols:
        return "수치형(숫자형) 컬럼이 없습니다. 변동 요약 불가."
    # 데이터 전체 행에 대해 ID/이름/장치/설비/센서명 컬럼 찾기
    id_cols = [c for c in df.columns if any(k in c.lower() for k in ["pdu", "id", "sensor", "chamber", "name", "description"])]
    lines = []
    for col in num_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        col_mean = df[col].mean()
        col_std = df[col].std()
        col_range = col_max - col_min
        lines.append(f"- '{col}'의 요약: min={col_min:.2f}, max={col_max:.2f}, mean={col_mean:.2f}, std={col_std:.2f}, 변동폭={col_range:.2f}")
        # ID 별 각 장치/센서/그룹별로 상위 5개 변동폭 요약
        for id_col in id_cols:
            group_stats = df.groupby(id_col)[col].agg(['min', 'max', 'mean', 'std'])
            group_stats['range'] = group_stats['max'] - group_stats['min']
            top_var = group_stats.sort_values('range', ascending=False).head(5)
            lines.append(f"  {id_col}별 '{col}' 변동폭 TOP5:")
            for idx, row in top_var.iterrows():
                lines.append(f"    • {idx}: min={row['min']:.1f}, max={row['max']:.1f}, mean={row['mean']:.1f}, std={row['std']:.1f}, 변동폭={row['range']:.1f}")
    return "\n".join(lines)

def synthesize_answer_from_summary(user_question: str, conversation_history: list, summary: str, file_logger: logging.Logger) -> str:
    """
    요약(summary) 텍스트만 가지고 LLM이 자연어 답변 생성.
    """
    history_str = "\n".join([f"## {msg['role']}:\n{msg['content']}" for msg in conversation_history])
    prompt = f"""### Instruction:
너는 데이터 분석 AI야. 아래 summary만 참고해서 사용자의 질문에 대해 명확하고 간결하게 답해. 추측이나 근거 없는 확장 설명은 금지. summary의 숫자/항목만 인용.

### Context:
- 사용자 질문: "{user_question}"
- 대화 기록:
{history_str}
- 데이터 요약(summary):
{summary}

### 최종 답변(한국어):
"""
    # 일관성을 위해 래퍼 함수 사용
    return call_gemini_with_token_tracking(prompt, "Synthesize_from_Summary", file_logger)
# 07-09 17:16 수정
# sql_query 인자 추가
def synthesize_answer_from_dataframe(user_question: str, conversation_history: list, df: pd.DataFrame, sql_query: str, file_logger: logging.Logger) -> str:
    """
    DataFrame과 이를 생성한 SQL 쿼리까지 함께 LLM에 제공하여, 더 높은 품질의 자연어 답변을 생성합니다.
    """
    if df.empty:
        # 이전에 1단계에서 "정보를 찾지 못했습니다"와 같은 결과가 나왔을 때, 
        # 최종 답변 생성 단계에서 이 메시지를 활용할 수 있습니다.
        return "관련 정보를 찾지 못했습니다."

    if len(df) > 20:
        table_str = df.head(20).to_markdown(index=False) + "\n(이하 생략...)"
    else:
        table_str = df.to_markdown(index=False)

    history_str = "\n".join([f"## {msg['role']}:\n{msg['content']}" for msg in conversation_history])
    
    prompt = f"""### Instruction:
너는 전문 데이터 분석가 AI야. 사용자의 질문과, 그 질문에 답하기 위해 실행된 SQL 쿼리, 그리고 그 결과 데이터를 보고 최종 답변을 생성해.

### Context:
- **사용자 원본 질문**: "{user_question}"

- **실행된 SQL 쿼리**:
```sql
{sql_query}
- **쿼리 실행 결과 데이터**:
{table_str}

### 최종 답변(한국어):
위의 모든 맥락을 종합하여, 사용자의 질문에 대한 자연스러운 최종 답변을 생성해줘. 쿼리에 있는 조건들을 참고하여 왜 이 데이터가 나왔는지 설명에 포함할 수 있어. 

"""
    # 일관성을 위해 래퍼 함수 사용
    return call_gemini_with_token_tracking(prompt, "Synthesize_from_DataFrame", file_logger)
# 07-06 report_generator agent

def generate_report(user_question: str, execution_context: dict, conversation_history: list, plan: list) -> str:
    """
    이전 단계들에서 수집된 데이터(테이블, 그래프 등)를 종합하여
    구조화된 마크다운(Markdown) 보고서를 생성합니다.
    """
    print("📝 보고서 생성 중...")
    
    # 실행 컨텍스트에서 보고서에 포함할 모든 결과물을 수집
    report_elements = []
    for key, value in execution_context.items():
        if key.startswith('output_of_step_'):
            step_num = key.split('_')[-1]
            step_plan = plan[int(step_num) - 1] # 해당 단계의 계획 정보 가져오기
            header = f"### [분석 {step_num}: {step_plan['tool']} - {step_plan['description']}]"
            
            element_body = ""
            if isinstance(value, pd.DataFrame):
                element_body = value.to_markdown(index=False) if not value.empty else "결과 데이터 없음."
            elif "파일이 생성되었습니다" in str(value):
                # 시각화 결과에서 파일 경로 추출
                match = re.search(r"'([^']+\.png)'", str(value))
                if match:
                    filepath = match.group(1)
                    element_body = f"생성된 그래프: {filepath}"
            else:
                element_body = str(value)
            
            report_elements.append(f"{header}\n{element_body}")

    combined_context = "\n\n---\n\n".join(report_elements)
    history_str = "\n".join([f"## {msg['role']}:\n{msg['content']}" for msg in conversation_history])

    # 보고서 생성을 위한 프롬프트
    report_prompt = f"""### Instruction:
You are a professional data center operations analyst. Your task is to generate a formal report in Korean based on the provided data and visualizations. The report should be well-structured, easy to understand, and directly address the user's original request.

### Report Generation Context:
- User's Original Request: "{user_question}"
- Conversation History: {history_str}
- Collected Data and Analysis Results:
{combined_context}

### Report Structure:
1.  **제목 (Title)**: 사용자의 요청을 기반으로 보고서의 제목을 명확하게 작성하세요 (예: 주간 데이터센터 전력 사용량 보고서).
2.  **개요 (Summary)**: 분석된 핵심 내용을 1~2 문장으로 요약하세요.
3.  **세부 분석 내용 (Detailed Analysis)**: 각 분석 단계의 결과를 항목별로 나누어 설명하세요. 데이터 테이블과 그래프 결과를 명확히 참조하여 서술하세요.
4.  **결론 및 제언 (Conclusion & Recommendation)**: 전체 분석 결과를 바탕으로 최종 결론을 내리고, 필요한 경우 운영상 제언을 덧붙이세요.

### Final Report (in Korean Markdown format):
"""
    # Gemini를 호출하여 최종 보고서 생성
    final_report = call_gemini_with_token_tracking(report_prompt, "Report_Generator", file_logger) # logger 전달
    return final_report


# --- 3. 메인 AI 에이전트 함수 (Orchestrator) ---
def run_ai_agent(user_question: str, file_logger: logging.Logger, conversation_history: list):
    """(멀티스텝 아키텍처) 계획에 따라 여러 도구를 순차적으로 실행합니다."""
    print("\n🤔 생각 중...")
    file_logger.info(f"🤖 사용자 질문: {user_question}")
    
    final_output = ""
    intermediate_outputs = []   # 대화 이력 결과를 저장할 리스트(순서있어서) - 최종 결과 반환 전 확인 및 디버깅용
    execution_context = {}      # DB 조회한 결과 (DataFrame)를 저장, 시각화, 이상치 탐지 단계에서 재사용

    try:
        # [수정]
        plan = get_plan(user_question, file_logger)
        file_logger.info(f"🧠 수립된 계획: {plan}")
        print(f" 수립된 계획 :")
        for i, step in enumerate(plan): # 플래너가 생성한 계획을 순회하며 실행(2개 이상)
            print(f" {i+1}. {step['tool']}: {step['description']}")
        # 06-27 16:30 수정 - 06-30 16:53 "result": [] 삭제

        ## --- 1. 정보 수집 단계(for 루프) --- ##
        # 이 루프의 목표는 계획에 따라 각 도구를 실행하고, 그 결과를 execution_context에 저장하는 것.
        for i, step in enumerate(plan):
            tool = step['tool']
            description = step['description']
            
            #[*수정] answer_synthesizer는 모든 정보 수집이 끝난 후 마지막에 따로 처리하므로, 루프에서는 건너뜀.
            if tool == 'answer_synthesizer':
                continue

            print(f"--- {i+1}/{len(plan)}단계 실행: {tool} ({description}) ---")
            file_logger.info(f"▶ {i+1}단계 실행: {tool} - {description}")

            # 각 단계의 결과물을 담을 일시 변수. 이 변수에는 DataFrame 또는 str이 담길 수 있습니다.
            step_result_data = None

            # 각 도구는 자신의 역할(정보 수집)에만 집중합니다.
            if tool == 'db_querier':
                sql_code = generate_sql_code(description, conversation_history, file_logger)
                file_logger.info(f"💻 생성된 SQL:\n---\n{sql_code}\n---")
                
                # SQL 실행과 오류 처리
                try:    # result_df를 step_result_data로 고침. 
                    step_result_data = execute_sql(sql_code)
                    execution_context[f'dataframe_step_{i+1}'] = step_result_data   # 각 단계별 결과를 개별적으로 저장하여 추적 가능하게 함.
                    # 0630 16:16 수정(GPT)
                    execution_context['dataframe'] = step_result_data  # 이게 핵심 - 최신 데이터만 저장(항상 덮어쓰기)

                    # 결과 형식 확인 및 처리
                    if isinstance(step_result_data, pd.DataFrame):     # DataFrame 타입인지 검사
                        if step_result_data.empty:
                            final_output = "데이터베이스에서 관련 정보를 찾지 못했습니다."
                        else:
                            if len(step_result_data) == 1 and len(step_result_data.columns) == 1:      # SQL 쿼리 결과가 단일 값인 경우 - row, column 수가 1개인지 확인    
                                # final_output = f"데이터 조회를 완료했습니다: {result_df.iloc[0, 0]}"
                                the_actual_value = step_result_data.iloc[0, 0]
                                if isinstance(the_actual_value, (int, float)):
                                    final_output = f"데이터 조회를 완료했습니다: {the_actual_value:.1f}"
                                else:
                                    final_output = f"데이터 조회를 완료했습니다: {the_actual_value}"
                            else:
                                final_output = f"데이터 조회를 완료했습니다. {len(step_result_data)}개의 행을 찾았습니다.\n(미리보기)\n{step_result_data.head().to_string()}"
                    
                    elif isinstance(step_result_data, str):    # SQL 코드가 str 타입인지 검사
                        # 문자열로 반환된 경우 (기존 에이전트와 호환성 유지)
                        final_output = step_result_data
                        # 문자열 결과를 DataFrame으로 변환 시도
                        try:
                            if "데이터가 없습니다" not in step_result_data:    # '데이터가 없습니다' 메시지를 처리하지 않도록 위함
                                import ast
                                data = ast.literal_eval(step_result_data)  # 문자열을 Python 객체로 변환
                                execution_context[f'dataframe_step_{i+1}'] = pd.DataFrame(data) # 실행 컨텍스트에 저장
                        except:
                            pass
                    else:
                        final_output = f"데이터 조회를 완료했습니다. 결과 유형: {type(step_result_data)}"
                except Exception as e:
                    final_output = f"SQL 실행 중 오류가 발생했습니다: {str(e)}"
                    file_logger.error(final_output, exc_info=True)
                
                #  동적 분기 로직 추가 :plan 마지막 단계일 때만
                is_last_step = (i == len(plan) - 1) # 단순 db 데이터 탐색 일 때 쓰기 위한 로직
                plan_only_db_querier = (len(plan)==1 and tool =="db_querier")

                if is_last_step and plan_only_db_querier:
                    if isinstance(step_result_data, pd.DataFrame) and not step_result_data.empty:
                        # 조건부 요약 및 답변 생성
                        if len(step_result_data) > 100: # 결과 행 수 기준, 상황에 맞게 임계값 조정
                            print("--- 동적 추가: data_summarizer + answer_synthesizer ---")
                            # 요약 실행(100행 초과의 대용량 데이터)
                            summary = summarize_table(step_result_data)
                            # 요약된 내용 기반 답변(API 호출)
                            final_output = synthesize_answer_from_summary(user_question, conversation_history, summary, file_logger)
                        else:
                            print("--- 동적 추가: answer_synthesizer ---")
                            final_output = synthesize_answer_from_dataframe(user_question, conversation_history, step_result_data, sql_code, file_logger)
                        break   # 동적 분기에서 최종 답변 만들었으면 for문 탈출

            elif tool == 'visualizer':
                # 25-06-30 16:46 추가
                df_to_visualize = execution_context.get('dataframe', pd.DataFrame())

                if df_to_visualize.empty:
                    final_output = "그래프를 그릴 데이터가 없습니다."
                    break
                # 07-07 수정       
                python_code = generate_viz_code(description, df_to_visualize, conversation_history, file_logger)
                file_logger.info(f"💻 생성된 Python Code:\n---\n{python_code}\n---")
                step_result_data = execute_python(python_code, exec_globals={'df': df_to_visualize})
            
            elif tool == 'anomaly_detector':
                # 25-06-30 16:46 추가 07-08 16:40 변경
                source_df = execution_context.get('dataframe', pd.DataFrame())
                if source_df.empty:
                    # 0704 15:53 수정. 분석할 데이터가 없으면, 빈 데이터프레임을 결과로 설정
                    step_result_data = pd.DataFrame()
                else:
                    # 07-08 더 다양한 분석 로직
                    metric_col = None
                    group_by_col = None

                    # 1. 그룹화할 컬럼(group_by_col) 찾기 (예: deviceName, description, measurement_date 등)
                    possible_group_cols = ['deviceName', 'description', 'measurement_date', 'name', 'id']
                    for col in possible_group_cols:
                        if col in source_df.columns:
                            group_by_col = col
                            break
                    
                    # 2. 분석할 측정값 컬럼(metric_col) 찾기 (숫자형이고, 그룹 컬럼이나 타임스탬프가 아닌 컬럼)
                    for col in source_df.columns:
                        if pd.api.types.is_numeric_dtype(source_df[col]) and col not in possible_group_cols and 'timestamp' not in col:
                            metric_col = col
                            break
                    
                    # 3. 분석 대상 컬럼을 찾았으면, 범용 함수 호출
                    if metric_col and group_by_col:
                        print(f"🔬 이상치 분석 실행: 그룹({group_by_col}), 측정값({metric_col})")
                        anomalies_df = detect_anomalies(source_df, metric_col, group_by_col)
                        step_result_data = anomalies_df
                    else:
                        # 분석 대상을 찾지 못한 경우
                        print("⚠️ 이상치 분석을 위한 적절한 컬럼을 찾지 못했습니다.")
                        step_result_data = pd.DataFrame()
                    # --- 스마트 분석 로직 끝 ---
                
                # 다음 단계를 위해 결과를 컨텍스트에 저장
                execution_context['dataframe'] = step_result_data
                    
                
            
            # 데이터 요약 추가 0701-10:06 수치 데이터들에 대한 텍스트 결과 요약
            elif tool == 'data_summarizer':
                # [수정]  요약 텍스트 결과만 반환
                source_df = execution_context.get('dataframe', pd.DataFrame())
                if source_df.empty:
                    step_result_data = '요약할 데이터가 없습니다.'
                else:
                    summary_text = summarize_table(source_df)
                    #요약 결과를 step_result_data에 할당
                    step_result_data = summary_text
 
            elif tool == 'memory_retriever':
                # [설명] memory_retriever는 그 자체로 답변을 생성하는 '완결된' 도구입니다.
                # 따라서 이 도구가 만든 결과가 사실상 최종 답변이 됩니다.
                
                if not conversation_history or len(conversation_history) < 2:
                    step_result_data = "참조할 이전 대화 기록이 없습니다."
                else:
                    # 가장 최근의 사용자 질문과 에이전트 답변을 가져옵니다.
                    last_user_question = conversation_history[-2]['content']
                    last_agent_turn = conversation_history[-1]

                    # 직전 대화에 상세 실행 내역(steps)이 있는지 확인
                    if last_agent_turn['role'] == 'agent' and 'steps' in last_agent_turn and last_agent_turn['steps']:
                        steps_summary = []
                        for step_info in last_agent_turn['steps']:
                            steps_summary.append(
                                f"--- 단계 {step_info['step']}: {step_info['tool']} ---\n"
                                f"요청: {step_info['description']}\n"
                                f"결과:\n{step_info['result']}"
                            )
                        detailed_context = "\n\n".join(steps_summary)

                        # 상세 내역 기반 프롬프트
                        extraction_prompt = f"""### Instruction:
You are an expert at information extraction. Answer the "User's Follow-up Question" based on the context from the "Previous Conversation". The context includes the detailed step-by-step execution log of how the previous answer was generated.

### Previous Conversation Context:
- Previous User Question: "{last_user_question}"
- Detailed Execution Log from Previous Task:
{detailed_context}

### User's Follow-up Question:
{user_question}

### Extracted Information (Answer directly and concisely in Korean):
"""
                        step_result_data = call_gemini_with_token_tracking(extraction_prompt, "Memory_Retriever_Detailed", file_logger)
                    else:
                        # 상세 실행 내역이 없는 경우, 간단한 답변만 참고
                        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
                        extraction_prompt = f"""### Instruction:
You are an expert at information extraction. From the 'Full Conversation History', find the specific piece of information needed to answer the 'User's Follow-up Question'.

### Full Conversation History:
{history_str}

### User's Follow-up Question:
{user_question}

### Extracted Information (Answer concisely):
"""
                        step_result_data = call_gemini_with_token_tracking(extraction_prompt, "Memory_Retriever_Simple", file_logger)

                # [수정] 이 단계의 결과를 step_result_data에 할당합니다.
                # 이 값은 루프 마지막에서 final_output에 할당되어 최종 답변으로 사용됩니다.
                final_output = step_result_data

# elif tool == 'general_qa' 블록을 아래 코드로 교체합니다.

            elif tool == 'general_qa':
                print("--- 'general_qa' 도구 실행: 논문 RAG DB 검색 시도 ---")

                # 1. 논문 RAG DB에서 관련 컨텍스트를 먼저 검색합니다.
                context, sources = retrieve_paper_context(user_question)
                
                # 이 단계의 결과물을 담을 변수
                step_result_data = ""

                # 2. RAG 검색 성공 여부에 따라 분기 처리
                if "찾지 못했습니다" not in context and context.strip():
                    # [경로 A] RAG 검색 성공: 검색된 내용을 바탕으로 답변 생성
                    print(f"✅ RAG 검색 성공. 참고문서: {sources}")
                    file_logger.info(f"📄 참고한 문서: {sources}")
                    
                    # RAG 기반 답변 생성 프롬프트 - 엄격한 규칙에서 논문에 없는 부분은 일반적인 LLM 지식으로 답변하도록 수정!
                    synthesis_prompt = f"""### Instruction:
You are a helpful AI assistant. Your primary goal is to answer the "User's Question" in Korean.

1.  First, you MUST thoroughly review the "Retrieved Knowledge" provided below. This is your most important reference.
2.  If the "Retrieved Knowledge" contains a direct and complete answer to the "User's Question", base your answer primarily on that information.
3.  If the "Retrieved Knowledge" is relevant but insufficient to fully answer the question, **use it as a starting point or key reference, and supplement it with your general knowledge** to provide a comprehensive and helpful answer.
4.  When you use information directly from the "Retrieved Knowledge", it is good practice to mention that the information is from the provided documents (e.g., "논문에 따르면...").

### Retrieved Knowledge:
{context}

### User's Question:
{user_question}

### Final Answer (Korean):
"""
                    step_result_data = call_gemini_with_token_tracking(synthesis_prompt, "General_QA_with_RAG", file_logger)

                else:
                    # [경로 B] RAG 검색 실패: LLM의 일반 지식으로 답변 (Fallback)
                    print("...RAG 검색 실패. LLM 일반 지식으로 답변을 시도합니다.")
                    file_logger.info("📄 논문 DB에 관련 정보가 없어, LLM 일반 지식으로 답변 시도.")

                    # LLM의 일반 지식을 활용하는 간단한 프롬프트
                    fallback_prompt = f"""Please answer the following question in Korean based on your general knowledge.
Question: "{user_question}"
Answer (Korean):"""
                    step_result_data = call_gemini_with_token_tracking(fallback_prompt, "General_QA_Fallback", file_logger)

            elif tool == 'report_generator':
                # 이 도구는 항상 마지막에 실행되어야 함
                # 이전 단계들의 결과가 담긴 execution_context를 사용.
                report_content = generate_report(user_question, execution_context, conversation_history, plan)
                step_result_data = report_content

            # --- 중간 결과 저장 --- 
            # [핵심 수정] 각 단계의 결과(step_result_data)를 다음 최종 종합 단계를 위해 execution_context에 차곡차곡 저장합니다.
            execution_context[f'output_of_step_{i+1}'] = step_result_data
            
            # 사용자와 로그에 보여줄 중간 요약본 생성
            if isinstance(step_result_data, pd.DataFrame):
                if step_result_data.empty:
                    final_output = "데이터베이스에서 관련 정보를 찾지 못했습니다."
                else:
                    final_output = f"데이터 조회를 완료했습니다. {len(step_result_data)}개의 행을 찾았습니다.\n(미리보기)\n{step_result_data.head().to_string()}"
            else: # DataFrame이 아닌 경우 (텍스트 등)
                final_output = str(step_result_data)

            step_log_entry = {
                "step": i + 1, "tool": tool, "description": description,
                "result": final_output[:500] + ("..." if len(final_output) > 500 else "")
            }
            intermediate_outputs.append(step_log_entry)

            if len(plan) > 1 and tool != plan[-1]['tool']: # 마지막 단계가 아닐 경우에만 중간 결과 출력
                print(f"[중간 결과 {i+1}] {final_output[:100]}...")
            
            file_logger.info(f"📊 {i+1}단계 결과:\n---\n{final_output}\n---")

        # 추가 07-04 13:03
        # [NEW] for 루프가 끝난 후, 계획에 answer_synthesizer가 있었는지 확인하고 실행
        final_synthesis_step = next((step for step in plan if step['tool'] == 'answer_synthesizer'), None)

        if final_synthesis_step:
            print(f"--- 최종 단계 실행: answer_synthesizer ({final_synthesis_step['description']}) ---")

            # execution_context에 저장된 모든 중간 결과들을 하나로 합칩니다.
            combined_context = []
            # 루프는 synthesizer를 제외한 단계까지만 돕니다.
            for i in range(len(plan) - 1):
                step_key = f"output_of_step_{i+1}"
                step_plan = plan[i]
                
                if step_key in execution_context:
                    step_output = execution_context[step_key]
                    header = f"### [결과 from Step {i+1}: {step_plan['tool']} - {step_plan['description']}]"
                    
                    if isinstance(step_output, pd.DataFrame):
                        if step_output.empty:
                            body = "결과 없음."
                        else:
                            body = step_output.to_markdown(index=False)
                    else:
                        body = str(step_output)
                        
                    combined_context.append(f"{header}\n{body}")
            
            retrieved_data = "\n\n---\n\n".join(combined_context)
            
            # 사용자님이 완성한 강력한 최종 답변 생성 프롬프트를 여기서 사용합니다.
            history_str = "\n".join([f"## {msg['role']}:\n{msg['content']}" for msg in conversation_history])
            synthesis_prompt = f"""### Persona and Goal
You are a professional AI data analyst for a data center. Your goal is to provide a final, insightful, and user-friendly answer in Korean based on the provided information.

### Contextual Information Provided
1.  **User's Current Question**: The user's most recent query.
2.  **Data From Previous Step**: A collection of results from all preceding steps. This can be empty.
3.  **Operational Policy**: The official operational standards for the data center.
4.  **Conversation History**: The log of the conversation so far.

### Step-by-Step Instructions
1.  Thoroughly analyze the `User's Current Question` to understand their primary intent.
2.  Examine the `Data From Previous Step`. This contains all the information gathered so far.
3.  **[CRITICAL] How to Handle Empty Data**:
    - If the `Data From Previous Step` is empty or contains "결과 없음", DO NOT just say "데이터가 없습니다."
    - Instead, look at the `User's Current Question`. If the question was asking *if something exists* (e.g., "any alarms?"), an empty result definitively means "no, none exist." State this clearly.
    - Example: If the question was "Are there any active alarms?" and the data is empty, your answer MUST be "현재 활성화된 알람이 없습니다."
4.  **How to Handle Available Data**:
    - If data exists, synthesize all pieces of information into a coherent answer.
    - First, state the key facts from the data (e.g., "The current temperature is 25°C.").
    - Next, compare these facts with the `Operational Policy`.
    - Make a clear judgment based on the policy (e.g., "This is within the normal range.").
5.  Formulate a final, concise, and insightful answer in Korean.

---
### Provided Information

#### [User's Current Question]
{user_question}

#### [Data From Previous Step]
{retrieved_data}
#### [Operational Policy]
- **온도 (Temperature)**: 18°C ~ 27°C (정상), 27°C 초과 (주의), 32°C 초과 (경고)
- **습도 (Humidity)**: 40% ~ 60% (정상)
- **PUE**: 1.2 이하 (매우 우수), 1.3~1.6 이하 (우수), 1.6~2.0 이하 (보통), 2.0 이상 (에너지 효율 낮음)
- **알람(Alarm)**: 값이 1이면 즉시 확인 필요

#### [Conversation History]
{history_str}
---

### Final Answer (in Korean):
"""
            final_output = call_gemini_with_token_tracking(synthesis_prompt, "Final_Answer_Synthesis", file_logger)

    except Exception as e:
        final_output = f"💥 에이전트 실행 중 오류가 발생했습니다: {e}"
        file_logger.error(final_output, exc_info=True)

    # --- 대화 기록 저장 및 최종 출력 ---
    conversation_history.append({"role": "user", "content": user_question})
    conversation_history.append({"role": "agent", "content": final_output, "steps": intermediate_outputs})
    
    MAX_HISTORY_TURNS = 5
    if len(conversation_history) > MAX_HISTORY_TURNS * 2:
        conversation_history = conversation_history[-(MAX_HISTORY_TURNS * 2):]

    print(f"\n[🤖에이전트 답변]\n{final_output}\n")
    file_logger.info("="*40 + "\n")

    # 0710 eval 코드 때문에 추가
    return final_output

# --- 4. 스크립트 실행 지점 ---
if __name__ == "__main__":
    file_logger = setup_file_logger()
    conversation_history = []
    print("🤖 LLM 에이전트가 준비되었습니다. 질문을 입력해주세요. (종료하려면 'exit' 또는 'quit' 입력)")
    
    while True:
        try:
            user_input = input("🧑당신: ")
            if user_input.lower() in ["exit", "quit"]:

                # [NEW] 종료 시 총 토큰 사용량 출력
                print("\n" + "="*40)
                print("📊 세션 총 토큰 사용량 요약")
                print(f" - 총 입력 토큰: {token_tracker['total_prompt_tokens']}")
                print(f" - 총 출력 토큰: {token_tracker['total_completion_tokens']}")
                print(f" - 합계 토큰: {token_tracker['total_tokens']}")
                print("="*40)
                print("👋 에이전트를 종료합니다.")
                break
            if not user_input.strip():  # 빈 입력은 무시하고 다음 입력을 받음(실수로 엔터만 쳤을 때 처리하기 위함)
                continue
            run_ai_agent(user_input, file_logger, conversation_history)
        except KeyboardInterrupt:
            print("\n👋 사용자에 의해 에이전트가 중지되었습니다.")
            break
        except Exception as e:
            error_msg = f"💥 메인 루프에서 예상치 못한 오류 발생: {e}"
            print(error_msg)
            file_logger.error(error_msg, exc_info=True)