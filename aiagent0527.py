# 다 통합하는 오케스트레이터 코드

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.language_models.llms import LLM
from typing import Any, List, Mapping, Optional, Dict
import warnings
import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import sys
from contextlib import redirect_stdout
import re

# --- 0. 환경 설정 및 경고 무시 ---
warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'NanumGothic' # <-- 예시: 나눔고딕 폰트 설정 (시스템에 설치 필요)
plt.rcParams['axes.unicode_minus'] = False # <-- 마이너스 폰트 깨짐 방지

# --- 1. CustomGemmaLLM 클래스 정의 (이전 코드 재사용) ---
class CustomGemmaLLM(LLM):
    tokenizer: Any = None
    model: Any = None
    device: str = "cuda:0"
    model_path: str = None

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        print("1. 토크나이저 로딩 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("2. 모델 로딩 중...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16
        ).to(self.device).eval()
        print(f"   - 모델을 {self.model.device} 장치로 로드했습니다.")

    @property
    def _llm_type(self) -> str:
        return "custom_gemma"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024, # 코드 생성을 위해 토큰 수 늘림
                do_sample=True,
                temperature=0.2, # 코드 생성 시에는 약간 낮춤
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            )
        input_length = inputs['input_ids'].shape[1]
        output_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        return output_text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_path": self.model_path}

# --- 2. LLM 및 DB 정보 설정 ---
model_path = "/home/ubuntu/models/gemma-3-12b-it/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80"
llm = CustomGemmaLLM(model_path=model_path)

db_host = "127.0.0.1"
db_user = "root"
db_password = "rootpass"
db_name = "sakila"

# --- 3. 프롬프트 템플릿 정의 ---
planner_prompt_template = """### Instruction:
You are an intelligent AI assistant. Your job is to understand the user's request and determine which capability is needed to fulfill it.
You have the following capabilities:
1.  **db_querier**: Sakila 데이터베이스 정보(배우, 영화, 고객 등)를 물어볼 때 사용합니다.
2.  **excel_analyzer**: Excel(.xlsx) 또는 CSV 파일의 데이터를 읽거나 분석/처리할 때 사용합니다.
3.  **visualizer**: 데이터를 기반으로 그래프나 차트(막대, 선 등)를 그려달라고 요청할 때 사용합니다.
4.  **file_reader**: 로컬 텍스트 파일의 내용을 읽어달라고 요청할 때 사용합니다.
5.  **file_writer**: 정보를 로컬 텍스트 파일에 저장해달라고 요청할 때 사용합니다.
6.  **general_qa**: 위의 어느 것에도 해당하지 않는 일반적인 질문에 답할 때 사용합니다.
Based on the user's question below, output *only* the name of the single most relevant capability from the list above.

Question: {user_question}
### Capability:"""

sql_gen_prompt_template = """### Instruction:
You are a MySQL expert. You know the Sakila database schema.
Key tables:
- actor(actor_id, first_name, last_name)
- film(film_id, title, description, release_year)
- film_actor(actor_id, film_id)
Based on the user's question, generate a single, executable MySQL query.
Make sure to JOIN tables correctly if needed.
Use actor.first_name and actor.last_name for filtering.
Only SELECT the requested columns. Add LIMIT if asked.

Question: {user_question}
### MySQL Query:"""

python_gen_prompt_template = """### Instruction:
You are a Python expert specializing in pandas, matplotlib, and file operations.
Based on the user's request, generate only the Python code required to perform the task.
Assume pandas is imported as pd and matplotlib.pyplot as plt.
Use standard Python file I/O (with utf-8 encoding).
Ensure any plots are saved to a file (default: 'output.png').
Ensure the code is a single, executable block.

Request: {user_question}
### Python Code:"""

# --- 4. 헬퍼 함수 정의 ---
# LLM의 전체 출력 문자열에서 실제 실행 가능한 코드 부분만 추출하는 함수
def extract_code(llm_output: str) -> str:
    """LLM 출력에서 코드 블럭(```) 또는 특정 키워드 이후를 추출합니다."""
    match = re.search(r"```(?:python|sql)?\s*(.*?)```", llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()    # 실제 코드 내용 부분 가져와서 공백 제거 하고 반환
    
    keywords = ["### Python Code:", "### MySQL Query:"]    # 우리가 프롬프트에서 사용한 키워드를 기준으로 코드를 추
    for keyword in keywords:
        if keyword in llm_output:
            return llm_output.split(keyword)[1].strip()    # 키워드를 기준으로 문자열을 나누고, 키워드 이후의 부분을 가져와 앞뒤 공백 제거하고 반환
            
    # 위 패턴이 없으면, 그냥 전체 출력을 반환 (LLM이 코드만 출력했다고 가정)
    return llm_output.strip()
    
def execute_sql(sql_query: str) -> str:
    """SQL 쿼리를 실제 데이터베이스에서 실행하고 결과를 문자열로 반환합니다."""
    try:
        conn = pymysql.connect(
            host=db_host, user=db_user, password=db_password, database=db_name,
            charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor             # 결과를 딕셔너리 형태로 받기 위한 커서 설정, 문자 인코딩을 utf8mb4로 설정하여 한글 등 다국어 지원
        )
        with conn.cursor() as cursor:
            cursor.execute(sql_query.replace(';', ''))    # pymysql은 ; 없어도 잘 실행되서 세미콜론 제거하고 sql 쿼리 실행
            results = cursor.fetchall()    # 결과 딕셔너리 리스트 형태 실행 결과 가져옴
        conn.close()
        return str(results) if results else "데이터가 없습니다."
    except Exception as e:
        return f"❌ SQL 실행 중 오류: {e}"

def execute_python(python_code: str) -> str:
    """Python 코드를 실행하고 표준 출력을 문자열로 반환합니다."""
    f = io.StringIO()    # 출력을 io.StringIO 객체로 보내기위해 생
    try:
        with redirect_stdout(f):
            exec(python_code, {"pd": pd, "plt": plt, "os": os})
        output = f.getvalue()
        # 파일 생성 여부 확인 (visualizer 경우)
        if 'plt.savefig' in python_code:
             # 파일 이름 추출 시도 (간단한 방식)
             match = re.search(r"plt\.savefig\(['\"](.*?)['\"]\)", python_code)
             if match and os.path.exists(match.group(1)):
                 output += f"\n✅ '{match.group(1)}' 파일이 생성되었습니다."
             else:
                 output += "\n⚠️ 그래프 파일 생성 여부를 확인하지 못했습니다."
        return output if output else "코드가 실행되었으나, 출력은 없습니다."
    except Exception as e:
        return f"❌ Python 코드 실행 중 오류: {e}\n{f.getvalue()}"

# 사용자 질문을 받아, 어떤 기능을 사용해야 할 지 결정하는 '플래너' 역할을 하는 함수
def get_capability(user_question: str, llm: LLM) -> str:
    """플래너 LLM을 호출하여 기능을 결정합니다."""
    prompt = planner_prompt_template.format(user_question=user_question)    # 미리 정의된 planner_prompt_template에 사용자 질문을 넣어 전체 프롬프트를 완성.
    response = llm.invoke(prompt)    # 완성된 프롬프트를 llm(CustomGemmaLLM 인스턴스)에 전달하여 응답(기능 이름)을 받는다.
    return extract_code(response) # 코드 추출 함수 재활용

# 결정된 기능과 사용자 질문을 받아, 해당 기능을 수행할 코드(SQL or Python)를 생성하는 함
def generate_code(capability: str, user_question: str, llm: LLM) -> Optional[str]:
    """기능에 맞는 프롬프트를 사용하여 코드를 생성합니다."""
    if capability == 'db_querier':
        prompt = sql_gen_prompt_template.format(user_question=user_question)
    elif capability in ['excel_analyzer', 'visualizer', 'file_reader', 'file_writer']:
        prompt = python_gen_prompt_template.format(user_question=user_question)
    else:
        return None
        
    generated_output = llm.invoke(prompt)    # 선택된 프롬프트 템플릿에 사용자 질문을 넣어 전체 프롬프트를 완성하고, llm에 전달하여 코드 생성 요청
    return extract_code(generated_output)    # LLM이 생성한 전체 출력에서 실제 코드 부분만 추출하여 반환

# --- 5. 메인 AI 에이전트 함수 ---
def run_ai_agent(user_question: str):
    print(f"\n========================================")
    print(f"🤖 사용자 질문: {user_question}")
    print(f"========================================")

    # 1. 기능 결정 (플래너)
    capability = get_capability(user_question, llm)
    print(f"🧠 에이전트 판단: '{capability}' 기능이 필요합니다.")

    # 2. 코드 생성 (코드 생성기)
    if capability in ['db_querier', 'excel_analyzer', 'visualizer', 'file_reader', 'file_writer']:
        code_to_run = generate_code(capability, user_question, llm)
        if not code_to_run:
            print("❌ 코드를 생성하지 못했습니다.")
            return

        print(f"\n💻 생성된 코드:\n---\n{code_to_run}\n---")

        # 3. 코드 실행 (실행기)
        if capability == 'db_querier':
            result = execute_sql(code_to_run)
        else:
            result = execute_python(code_to_run)
            
        print(f"\n📊 실행 결과:\n---\n{result}\n---")

    elif capability == 'general_qa':
        print("\n🗣️ 일반 질문입니다. LLM에게 직접 물어봅니다...")
        # 간단한 QA를 위해 LLM 직접 호출 (프롬프트 개선 필요)
        qa_prompt = f"Please answer the following question: {user_question}"
        answer = llm.invoke(qa_prompt)
        print(f"\n📊 LLM 답변:\n---\n{answer}\n---")
    else:
        print("🤷‍♂️ 이 질문은 어떻게 처리해야 할지 잘 모르겠습니다.")

    print(f"========================================")
