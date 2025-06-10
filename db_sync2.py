import pymysql
# MySQL 데이터베이스와의 연결 및 상호 작용을 위한 PyMySQL 라이브러리를 가져옵니다.
import pandas as pd
# 강력한 데이터 조작 도구인 pandas 라이브러리를 가져옵니다. 여기서는 주로 DB에서 가져온 행(딕셔너리 리스트)을 DataFrame으로 변환하여 컬럼 삭제 등 처리를 용이하게 합니다.
from datetime import datetime
# datetime 모듈을 가져오지만, 현재 제공된 코드 스니펫에서는 사용되지 않습니다. 타임스탬프 조작이나 로깅을 위해 의도되었을 수 있습니다.
from tqdm import tqdm
# 진행률 표시줄을 위한 tqdm을 가져옵니다. 이는 많은 행을 upsert하는 것과 같이 오래 걸리는 작업의 진행 상황을 시각화하는 데 도움을 줍니다.

# --- 설정 ---
# 설정 섹션임을 나타내는 주석입니다.
SOURCE_DBS = {
# 모든 소스 데이터베이스의 설정을 담는 딕셔너리를 정의합니다.
    "gist": {
    # 'gist' 소스 데이터베이스의 설정입니다.
        "host": "210.125.69.5", "port": 65012, "user": "gist_collector", "password": "asigwangjugist", "db": "GWANGJU_GIST",
        # GIST 데이터베이스의 연결 정보(호스트, 포트, 사용자, 비밀번호, 데이터베이스 이름)입니다.
        "tables": ["fms_device_list", "fms_object_list", "fms_object_value"]
        # GIST 데이터베이스에서 동기화할 테이블 목록입니다.
    },
    "centralcity": {
    # 'centralcity' 소스 데이터베이스의 설정입니다.
        "host": "218.50.4.180", "port": 3306, "user": "centralcity_collector", "password": "asicentralcity", "db": "main_centralcity",
        # Centralcity 데이터베이스의 연결 정보입니다.
        "tables": [
        # Centralcity 데이터베이스에서 동기화할 테이블 목록입니다.
            "department_bacnet_device_list", "department_bacnet_object_list", "department_bacnet_object_value",
            "terminal_bacnet_device_list", "terminal_bacnet_object_list", "terminal_bacnet_object_value"
        ]
    }
}

TARGET_DB = {
# 대상 데이터베이스의 설정을 담는 딕셔너리를 정의합니다.
    "host": "localhost", "port": 3306, "user": "asi_agent", "password": "agent@asi",
    # 대상 MySQL 서버의 연결 정보(localhost, 기본 MySQL 포트, 사용자, 비밀번호)입니다.
    "gist": "gist_agent_test",
    # GIST 데이터를 위한 대상 데이터베이스 이름입니다.
    "centralcity": "centralcity_agent_test"
    # Centralcity 데이터를 위한 대상 데이터베이스 이름입니다.
}

CHUNK_SIZE = 100000  # 대용량 테이블 처리용 chunk 크기
# 대용량 테이블에서 데이터를 가져오고 처리할 때 메모리 부족 문제를 방지하고 트랜잭션 크기를 관리하기 위해 사용되는 청크 크기 상수입니다.

# --- DB 연결 함수 ---
# 데이터베이스 연결 함수 시작을 나타내는 주석입니다.
def connect_mysql(conf):
# MySQL 데이터베이스에 연결을 설정하는 함수를 정의합니다. 설정 딕셔너리를 입력으로 받습니다.
    return pymysql.connect(
    # PyMySQL 연결 객체를 반환합니다.
        host=conf["host"], port=conf["port"], user=conf["user"],
        # 'conf' 딕셔너리에서 호스트, 포트, 사용자 값을 사용합니다.
        password=conf["password"], database=conf["db"], charset="utf8mb4",
        # 'conf'에서 비밀번호와 데이터베이스 이름을 사용하고, 전체 유니코드 지원을 위해 문자 집합을 'utf8mb4'로 설정합니다.
        cursorclass=pymysql.cursors.DictCursor
        # DictCursor를 지정하여 쿼리 결과가 튜플 대신 딕셔너리로 반환되도록 합니다. 이는 컬럼 이름으로 더 쉽게 접근할 수 있게 합니다.
    )

# --- UPSERT 실행 함수 ---
# UPSERT 실행 함수 시작을 나타내는 주석입니다.
def _upsert_rows(rows, target_conf, table, db_label):
# UPSERT 작업을 수행하는 비공개 헬퍼 함수(선행 밑줄로 표시)를 정의합니다.
# 행 리스트, 대상 DB 설정, 테이블 이름, 데이터베이스 레이블을 입력으로 받습니다.
    if not rows:
    # 'rows' 리스트가 비어 있는지 확인합니다. 비어 있다면 upsert할 것이 없습니다.
        return
        # 행이 제공되지 않으면 함수를 종료합니다.

    df = pd.DataFrame(rows)
    # 딕셔너리 리스트(행)를 pandas DataFrame으로 변환합니다. 이는 균일한 데이터 처리에 유용합니다.
    if "description" in df.columns:
    # DataFrame에 'description' 컬럼이 있는지 확인합니다.
        df.drop(columns=["description"], inplace=True)
        # 'description'이 존재하면 해당 컬럼을 삭제합니다. 이는 대상 테이블에 이 컬럼이 없거나, 동기화에 필요하지 않거나 원하지 않는 경우일 수 있습니다. 'inplace=True'는 DataFrame을 직접 수정합니다.

    with connect_mysql({**target_conf, "db": target_conf[db_label]}) as tgt_conn:
    # 대상 데이터베이스에 연결을 설정합니다. `db_label`을 사용하여 `target_conf`에서 데이터베이스 이름을 동적으로 선택합니다.
        with tgt_conn.cursor() as tgt_cur:
        # SQL 쿼리를 실행할 수 있는 커서 객체를 생성합니다.
            # 생성된 컬럼 조회
            # 주석: 생성된 컬럼을 쿼리합니다.
            tgt_cur.execute(f"SHOW COLUMNS FROM `{table}`")
            # 대상 테이블의 모든 컬럼에 대한 정보를 얻기 위해 SQL 쿼리를 실행합니다.
            columns_info = tgt_cur.fetchall()
            # 실행된 쿼리에서 모든 결과를 가져옵니다.
            generated_cols = {col["Field"] for col in columns_info if 'Generated' in col["Extra"].upper()}
            # 컬럼 정보에서 생성된 컬럼의 이름을 추출합니다. 생성된 컬럼은 데이터베이스에 의해 파생되므로 일반적으로 INSERT/UPDATE 문에 포함되어서는 안 됩니다.

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Upserting {db_label}.{table}"):
            # `iterrows()`를 사용하여 DataFrame의 각 행을 반복하고 `tqdm`을 사용하여 진행률 표시줄을 표시합니다.
                row_dict = {k: (None if pd.isna(v) else v) for k, v in row.items() if k not in generated_cols}
                # 현재 행에 대한 딕셔너리를 생성하며, NaN 값을 None(SQL NULL)으로 변환하고 생성된 컬럼은 제외합니다.

                cols = ', '.join(f'`{k}`' for k in row_dict.keys())
                # SQL을 위해 백틱으로 올바르게 인용된 컬럼 이름의 콤마로 구분된 문자열을 생성합니다.
                placeholders = ', '.join(['%s'] * len(row_dict))
                # 파라미터화된 쿼리를 위한 '%s' 플레이스홀더의 콤마로 구분된 문자열을 생성합니다. 이는 컬럼 수와 일치합니다.
                update_clause = ', '.join(f"{k}=VALUES({k})" for k in row_dict.keys())
                # UPSERT 문의 ON DUPLICATE KEY UPDATE 절을 생성합니다. 각 컬럼에 대해, INSERT 부분에서 제공된 새 값으로 컬럼을 설정합니다.

                sql = (
                # 전체 SQL UPSERT (INSERT ... ON DUPLICATE KEY UPDATE) 문을 구성합니다.
                    f"INSERT INTO `{table}` ({cols}) VALUES ({placeholders}) "
                    f"ON DUPLICATE KEY UPDATE {update_clause}"
                )
                tgt_cur.execute(sql, tuple(row_dict.values()))
                # SQL UPSERT 문을 행의 값(튜플 형식)과 함께 실행합니다. 이는 SQL 인젝션을 방지하고 다양한 데이터 유형을 올바르게 처리합니다.
            tgt_conn.commit()
            # 모든 변경 사항을 대상 데이터베이스에 저장하기 위해 트랜잭션을 커밋합니다.

# --- 테이블 복사 함수 ---
# 테이블 복사 함수 시작을 나타내는 주석입니다.
def sync_table(source_conf, target_conf, table, db_label):
# 단일 테이블을 동기화하는 메인 함수를 정의합니다. 소스 및 대상 설정, 테이블 이름, 데이터베이스 레이블을 입력으로 받습니다.
    is_value_table = table.endswith("_value")
    # 테이블 이름이 "_value"로 끝나는지 확인합니다. 이는 특별한 처리(예: 타임스탬프 필터링)가 필요한 테이블을 식별하기 위한 휴리스틱입니다.

    with connect_mysql(source_conf) as src_conn, src_conn.cursor() as src_cur:
    # 소스 데이터베이스에 연결하고 커서를 생성합니다.
        if is_value_table:
        # value 테이블인 경우, 최근 기록(3일 이내)만 카운트합니다.
            src_cur.execute(f"SELECT COUNT(*) as cnt FROM `{table}` WHERE `timestamp` >= NOW() - INTERVAL 3 DAY")
            # 테이블의 행 수를 세는 쿼리를 실행합니다. 특히 "value" 테이블의 경우 지난 3일간의 데이터만 필터링합니다. 이는 잠재적으로 매우 큰 시계열 테이블에서 가져오는 데이터를 크게 줄여줍니다.
        else:
        # 다른 테이블의 경우 모든 기록을 카운트합니다.
            src_cur.execute(f"SELECT COUNT(*) as cnt FROM `{table}`")
            # 테이블의 모든 행 수를 세는 쿼리를 실행합니다.
        row_count = src_cur.fetchone()["cnt"]
        # 행 수를 가져옵니다.

    print(f"[{db_label}] {table}: ", end="")
    # 현재 처리 중인 테이블을 표시하기 위해 데이터베이스 레이블과 테이블 이름을 출력합니다.
    if row_count > CHUNK_SIZE:
    # 행 수가 정의된 CHUNK_SIZE를 초과하는지 확인합니다.
        print(f"{row_count:,}건 복사 (chunk {CHUNK_SIZE}씩)...")
        # 대용량 테이블인 경우, 청크 단위 복사를 나타내는 메시지를 출력합니다.
        for offset in range(0, row_count, CHUNK_SIZE):
        # 오프셋을 사용하여 청크 단위로 데이터를 반복 처리합니다.
            with connect_mysql(source_conf) as src_conn, src_conn.cursor() as src_cur:
            # 각 청크 가져오기마다 소스 연결 및 커서를 다시 설정합니다. 이는 장기 실행 트랜잭션이나 연결 시간 초과를 방지하기 위함일 수 있습니다.
                if is_value_table:
                # value 테이블인 경우, 지난 3일간의 데이터를 청크 단위로 가져옵니다.
                    src_cur.execute(
                        f"SELECT * FROM `{table}` WHERE `timestamp` >= NOW() - INTERVAL 3 DAY "
                        f"LIMIT {CHUNK_SIZE} OFFSET {offset}"
                    )
                    # value 테이블에서 데이터 청크를 선택하며, 다시 3일 타임스탬프 필터를 적용합니다.
                else:
                # 다른 테이블의 경우, 모든 데이터를 청크 단위로 가져옵니다.
                    src_cur.execute(f"SELECT * FROM `{table}` LIMIT {CHUNK_SIZE} OFFSET {offset}")
                    # 테이블에서 모든 데이터의 청크를 선택합니다.
                rows = src_cur.fetchall()
                # 현재 청크의 모든 행을 가져옵니다.
                _upsert_rows(rows, target_conf, table, db_label)
                # 가져온 행을 대상 데이터베이스에 upsert하기 위해 헬퍼 함수를 호출합니다.
    else:
    # 행 수가 CHUNK_SIZE보다 작거나 같으면 한 번에 모두 복사합니다.
        with connect_mysql(source_conf) as src_conn, src_conn.cursor() as src_cur:
        # 소스 데이터베이스에 연결하고 커서를 생성합니다.
            if is_value_table:
            # value 테이블인 경우, 모든 최근 데이터를 가져옵니다.
                src_cur.execute(f"SELECT * FROM `{table}` WHERE `timestamp` >= NOW() - INTERVAL 3 DAY")
                # 청크 없이 value 테이블에서 모든 최근 데이터를 가져옵니다.
            else:
            # 다른 테이블의 경우, 모든 데이터를 가져옵니다.
                src_cur.execute(f"SELECT * FROM `{table}`")
                # 청크 없이 테이블에서 모든 데이터를 가져옵니다.
            rows = src_cur.fetchall()
            # 모든 행을 가져옵니다.
            print(f"{len(rows)}건 전체 복사 진행 중...")
            # 전체 복사가 진행 중임을 나타내는 메시지를 출력합니다.
            _upsert_rows(rows, target_conf, table, db_label)
            # 가져온 모든 행을 대상 데이터베이스에 upsert하기 위해 헬퍼 함수를 호출합니다.

# --- 실행 ---
# 실행 블록을 나타내는 주석입니다.
for db_label, conf in SOURCE_DBS.items():
# SOURCE_DBS에 정의된 각 소스 데이터베이스 설정을 반복합니다.
    print(f"=== {db_label.upper()} DB 복사 시작 ===")
    # 현재 데이터베이스의 동기화 시작을 나타내는 헤더를 출력합니다.
    for table in conf["tables"]:
    # 현재 소스 데이터베이스에 지정된 각 테이블을 반복합니다.
        sync_table(conf, TARGET_DB, table, db_label)
        # 현재 테이블을 동기화하기 위해 `sync_table` 함수를 호출합니다.
    print(f"=== {db_label.upper()} DB 복사 완료 ===\n")
    # 현재 데이터베이스의 동기화 완료를 나타내는 푸터를 출력합니다.

---

### 전반적인 리뷰 및 개선 가능성:

이 코드는 잘 구조화되어 있으며, 대용량 테이블 및 UPSERT 로직과 같은 일반적인 데이터베이스 동기화 문제를 잘 처리합니다. `pymysql.cursors.DictCursor` 사용은 가독성에 매우 좋습니다. `tqdm` 진행률 표시줄은 사용자 경험을 향상시키는 좋은 요소입니다.

다음은 고려할 만한 몇 가지 사항입니다.

* **오류 처리**: 현재 코드는 데이터베이스 연결 오류, 쿼리 실행 오류 또는 pandas 작업에 대한 명시적인 `try-except` 블록을 포함하지 않습니다. 프로덕션 환경에서는 견고한 오류 처리가 중요합니다. 예를 들어, 소스 데이터베이스가 다운되면 스크립트가 중단될 것입니다.
* **로깅**: 단순히 `print` 문을 사용하는 대신 파이썬의 `logging` 모듈을 사용하는 것을 고려해보세요. 이는 보다 유연한 출력(콘솔, 파일 등), 다양한 로그 수준(INFO, WARNING, ERROR) 및 더 나은 추적성을 제공합니다.
* **청크 단위 트랜잭션 관리**: 각 청크의 upsert 후에 `tgt_conn.commit()`이 호출되지만, *모든* 청크 가져오기마다 `src_conn` 및 `src_cur`를 다시 설정하는 것은 약간 비효율적입니다. 일반적으로 `sync_table` 호출 기간 동안 소스 연결을 열어두고 필요한 경우에만 커서를 다시 가져오는 것이 좋습니다. 그러나 매우 오래 실행되는 프로세스의 경우 다시 연결하는 것이 소스 측의 연결 시간 초과에 대한 보호 장치가 될 수 있습니다.
* **대상 테이블의 인덱스/기본 키**: `ON DUPLICATE KEY UPDATE` 절은 대상 테이블에 적절한 `PRIMARY KEY` 또는 `UNIQUE` 인덱스가 있는지에 크게 의존합니다. 이들이 없으면 UPSERT는 모든 행에 대해 `INSERT`로 변환되어 잠재적으로 중복 항목이 발생하거나 암시적인 고유 제약 조건이 있는 경우 `Duplicate entry` 오류가 발생할 수 있습니다. 대상 테이블이 이미 올바르게 설정되어 있다고 가정합니다.
* **Description 컬럼 삭제**: `description` 컬럼을 삭제하는 것은 특정 비즈니스 규칙처럼 보입니다. 처리된다는 점은 좋지만, 이 컬럼을 삭제하는 이유(예: "대상 스키마에 컬럼이 없음", "데이터가 필요 없음")를 문서화하는 것이 도움이 될 것입니다.
* **`datetime` 임포트**: `datetime` 모듈은 임포트되었지만 사용되지 않습니다. 깔끔한 임포트를 위해 제거할 수 있습니다.
* **설정 보안**: 비밀번호가 스크립트에 하드코딩되어 있습니다. 프로덕션 시스템의 경우, 민감한 자격 증명을 안전하게 저장하기 위해 환경 변수, 구성 파일(예: JSON, YAML) 또는 비밀 관리 시스템을 사용하는 것을 강력히 권장합니다.
* **Value 테이블의 타임스탬프 필터링**: `NOW() - INTERVAL 3 DAY` 필터는 `_value` 테이블에 대한 좋은 최적화입니다. 그러나 이 3일 창이 항상 충분한지 고려해보세요. 오래된 데이터를 다시 채워야 할 필요가 있다면 어떻게 할까요? 이 기간을 구성 가능하게 하거나 전체 이력 동기화를 위한 옵션을 추가하는 것을 고려할 수 있습니다.
* **생성된 컬럼 처리**: 생성된 컬럼을 동적으로 감지하는 것은 좋은 방어적 프로그래밍 관행입니다. 이는 이러한 컬럼이 있는 테이블에 upsert할 때 발생하는 일반적인 오류를 방지합니다.

전반적으로, 이 코드는 데이터베이스 동기화 스크립트를 위한 견고한 기반입니다. 더 강력한 오류 처리와 민감한 설정의 외부화를 추가하면 프로덕션 준비가 될 것입니다.

혹시 더 자세히 알아보고 싶으신 부분이 있으신가요?
