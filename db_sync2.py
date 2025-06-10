import pymysql
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# --- 설정 ---
SOURCE_DBS = {
    "gist": {
        "host": "210.125.69.5", "port": 65012, "user": "gist_collector", "password": "asigwangjugist", "db": "GWANGJU_GIST",
        "tables": ["fms_device_list", "fms_object_list", "fms_object_value"]
    },
    "centralcity": {
        "host": "218.50.4.180", "port": 3306, "user": "centralcity_collector", "password": "asicentralcity", "db": "main_centralcity",
        "tables": [
            "department_bacnet_device_list", "department_bacnet_object_list", "department_bacnet_object_value",
            "terminal_bacnet_device_list", "terminal_bacnet_object_list", "terminal_bacnet_object_value"
        ]
    }
}

TARGET_DB = {
    "host": "localhost", "port": 3306, "user": "asi_agent", "password": "agent@asi",
    "gist": "gist_agent_test",
    "centralcity": "centralcity_agent_test"
}

CHUNK_SIZE = 100000  # 대용량 테이블 처리용 chunk 크기

# --- DB 연결 함수 ---
def connect_mysql(conf):
    return pymysql.connect(
        host=conf["host"], port=conf["port"], user=conf["user"],
        password=conf["password"], database=conf["db"], charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )

# --- UPSERT 실행 함수 ---
def _upsert_rows(rows, target_conf, table, db_label):
    if not rows:
        return

    df = pd.DataFrame(rows)
    if "description" in df.columns:
        df.drop(columns=["description"], inplace=True)

    with connect_mysql({**target_conf, "db": target_conf[db_label]}) as tgt_conn:
        with tgt_conn.cursor() as tgt_cur:
            # 생성된 컬럼 조회
            tgt_cur.execute(f"SHOW COLUMNS FROM `{table}`")
            columns_info = tgt_cur.fetchall()
            generated_cols = {col["Field"] for col in columns_info if 'Generated' in col["Extra"].upper()}

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Upserting {db_label}.{table}"):
                row_dict = {k: (None if pd.isna(v) else v) for k, v in row.items() if k not in generated_cols}

                cols = ', '.join(f'`{k}`' for k in row_dict.keys())
                placeholders = ', '.join(['%s'] * len(row_dict))
                update_clause = ', '.join(f"{k}=VALUES({k})" for k in row_dict.keys())

                sql = (
                    f"INSERT INTO `{table}` ({cols}) VALUES ({placeholders}) "
                    f"ON DUPLICATE KEY UPDATE {update_clause}"
                )
                tgt_cur.execute(sql, tuple(row_dict.values()))
            tgt_conn.commit()

# --- 테이블 복사 함수 ---
def sync_table(source_conf, target_conf, table, db_label):
    is_value_table = table.endswith("_value")

    with connect_mysql(source_conf) as src_conn, src_conn.cursor() as src_cur:
        if is_value_table:
            src_cur.execute(f"SELECT COUNT(*) as cnt FROM `{table}` WHERE `timestamp` >= NOW() - INTERVAL 3 DAY")
        else:
            src_cur.execute(f"SELECT COUNT(*) as cnt FROM `{table}`")
        row_count = src_cur.fetchone()["cnt"]

    print(f"[{db_label}] {table}: ", end="")
    if row_count > CHUNK_SIZE:
        print(f"{row_count:,}건 복사 (chunk {CHUNK_SIZE}씩)...")
        for offset in range(0, row_count, CHUNK_SIZE):
            with connect_mysql(source_conf) as src_conn, src_conn.cursor() as src_cur:
                if is_value_table:
                    src_cur.execute(
                        f"SELECT * FROM `{table}` WHERE `timestamp` >= NOW() - INTERVAL 3 DAY "
                        f"LIMIT {CHUNK_SIZE} OFFSET {offset}"
                    )
                else:
                    src_cur.execute(f"SELECT * FROM `{table}` LIMIT {CHUNK_SIZE} OFFSET {offset}")
                rows = src_cur.fetchall()
                _upsert_rows(rows, target_conf, table, db_label)
    else:
        with connect_mysql(source_conf) as src_conn, src_conn.cursor() as src_cur:
            if is_value_table:
                src_cur.execute(f"SELECT * FROM `{table}` WHERE `timestamp` >= NOW() - INTERVAL 3 DAY")
            else:
                src_cur.execute(f"SELECT * FROM `{table}`")
            rows = src_cur.fetchall()
            print(f"{len(rows)}건 전체 복사 진행 중...")
            _upsert_rows(rows, target_conf, table, db_label)

# --- 실행 ---
for db_label, conf in SOURCE_DBS.items():
    print(f"=== {db_label.upper()} DB 복사 시작 ===")
    for table in conf["tables"]:
        sync_table(conf, TARGET_DB, table, db_label)
    print(f"=== {db_label.upper()} DB 복사 완료 ===\n")
