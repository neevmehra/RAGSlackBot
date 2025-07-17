import os, sys, array, json, re, time, sqlite3, PyPDF2
import oracledb, oci
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from telemetry import tracer
from opentelemetry import trace


# ================== ORACLE DB SETUP ==================
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

oracledb.init_oracle_client(lib_dir=os.path.join(BASE_DIR, "instantclient_23_8"))
os.environ["TNS_ADMIN"] = os.path.join(BASE_DIR, "wallet")

un = os.getenv("DB_USER")
pw = os.getenv("DB_PASS")
cs = os.getenv("DB_DSN")
topK = 3

# ================== EMBEDDING MODEL ==================
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

# ================== OCI GENERATIVE AI SETUP ==================
compartment_id = "ocid1.compartment.oc1..aaaaaaaaawkpra4vxusalnxjz3aztkizm7jnxis5docvbj2cssqau3a4xlaq"
model_id = "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyapnibwg42qjhwaxrlqfpreueirtwghiwvv2whsnwmnlva"
endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

CONFIG_PROFILE = "DEFAULT"
config_path = os.path.join(BASE_DIR, ".oci", "config")
config = oci.config.from_file(config_path, CONFIG_PROFILE)

generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
    config=config,
    service_endpoint=endpoint,
    retry_strategy=oci.retry.NoneRetryStrategy(),
    timeout=(10, 240)
)

# ================== FILE INGESTION + CHUNKING ==================
def embed_and_store(file_path, table_name, schema):
    with tracer.start_as_current_span("embed_and_store") as span:
        span.set_attribute("file_path", file_path)
        span.set_attribute("table_name", table_name)
        span.set_attribute("file_type", "json" if file_path.endswith('.json') else "other")

        encoder = SentenceTransformer('all-MiniLM-L12-v2')
        docs = []

        # =================== File Parsing ====================
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)

                    # === Flatten and normalize ticket structure ===
                    if isinstance(data, list):
                        if all(isinstance(x, list) for x in data):
                            tickets = [item for sublist in data for item in sublist]
                        else:
                            tickets = data
                    elif isinstance(data, dict):
                        if "tickets" in data and isinstance(data["tickets"], list):
                            tickets = data["tickets"]
                        else:
                            tickets = [data]
                    else:
                        raise ValueError("Unsupported JSON format")

                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON format")


            for ticket in tickets:
                doc_text = "\n".join([
                    f"SR: {ticket.get('sr_number', 'N/A')}",
                    f"Ticket ID: {ticket.get('ticket_id', 'N/A')}",
                    f"Status: {ticket.get('status', 'N/A')}",
                    f"Priority: {ticket.get('priority', 'N/A')}",
                    f"Product: {ticket.get('product', 'N/A')}",
                    f"Summary: {ticket.get('summary', 'N/A')}",
                    f"Description: {ticket.get('description', 'N/A')}",
                    f"Resolution: {ticket.get('resolution_description', 'N/A')}",
                    f"Root Cause: {ticket.get('root_cause', 'N/A')}",
                    f"Customer: {ticket.get('customer_account', 'N/A')}"
                    f"Ticket URL: {ticket.get('ticket_url', 'N/A')}"
                ])
                docs.append({"text": doc_text})

        elif file_path.endswith('.pdf'):
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"

                paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
                for idx, paragraph in enumerate(paragraphs):
                    if len(paragraph) > 50:
                        docs.append({
                            "text": f"PDF Content (Page {idx + 1}): {paragraph}",
                            "source": f"PDF_{os.path.basename(file_path)}"
                        })

        else:
            raise ValueError("Unsupported file type. Only .json and .pdf are supported.")

        # =================== Embedding ====================
        data = [{"id": idx, "vector_source": doc["text"], "payload": doc} for idx, doc in enumerate(docs)]
        texts = [row['vector_source'] for row in data]
        embeddings = encoder.encode(texts, batch_size=32, show_progress_bar=True)

        for row, embedding in zip(data, embeddings):
            row['vector'] = array.array("f", embedding)

        # =================== Oracle Insert ====================
        try:
            with oracledb.connect(user=un, password=pw, dsn=cs) as connection:
                with connection.cursor() as cursor:
                    qualified_table = f"{schema}.{table_name}"

                    cursor.execute("""
                        SELECT COUNT(*) FROM all_tables
                        WHERE table_name = :table_name AND owner = :owner
                    """, {'table_name': table_name.upper(), 'owner': schema.upper()})
                    exists = cursor.fetchone()[0] > 0

                    if not exists:
                        cursor.execute(f"""
                            CREATE TABLE {qualified_table} (
                                id NUMBER PRIMARY KEY,
                                payload CLOB,
                                vector VECTOR
                            )
                        """)

                    prepared_data = [(row['id'], json.dumps(row['payload']), row['vector']) for row in data]

                    cursor.executemany(
                        f"INSERT INTO {qualified_table} (id, payload, vector) VALUES (:1, :2, :3)",
                        prepared_data
                    )

                connection.commit()
                span.set_attribute("records_inserted", len(prepared_data))
                span.set_status(trace.Status(trace.StatusCode.OK))

        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
 
# ================== VECTOR SEARCH + GENERATE ==================
def vector_search(user_query, schema):
    with tracer.start_as_current_span("vector_search") as span:
        span.set_attribute("query", user_query)
        span.set_attribute("topK", topK)

        tables_searched = 0

        try:
            print("[DEBUG] vector_search() started.")
            embedding = list(embedding_model.encode(user_query))
            vec = array.array("f", embedding)
            retrieved_docs = []

            with oracledb.connect(user=un, password=pw, dsn=cs) as connection:
                with connection.cursor() as cursor:
                    # Fetch tables once
                    cursor.execute("""
                        SELECT table_name FROM all_tables
                        WHERE owner = :owner AND table_name NOT LIKE 'BIN$%'
                    """, {'owner': schema.upper()})
                    tables = [row[0] for row in cursor.fetchall()]

                    for table_name in tables:
                        tables_searched += 1
                        print(f"[DEBUG] Searching table: {table_name}")
                        try:
                            print(f"[DEBUG] Attempting vector search in table: {table_name}")
                            sql_retrieval = f'''
                                SELECT payload, VECTOR_DISTANCE(vector, :vector, COSINE) as score 
                                FROM {schema}.{table_name}
                                ORDER BY score 
                                FETCH APPROX FIRST {topK} ROWS ONLY
                            '''
                            rows = list(cursor.execute(sql_retrieval, vector=vec))

                            if not rows:
                                print(f"[DEBUG] No rows returned from {table_name}")

                            for (info, score) in rows:
                                info_str = info.read() if isinstance(info, oracledb.LOB) else info
                                print(f"[DEBUG] Score from table {table_name}: {score:.4f}")
                                retrieved_docs.append((score, json.loads(info_str)["text"]))

                        except Exception as e:
                            print(f"[WARNING] Skipping table {table_name}: {e}")
                            continue

            retrieved_docs.sort(key=lambda x: x[0])
            final_docs = [text for _, text in retrieved_docs[:topK]]

            span.set_attribute("documents_retrieved", len(final_docs))
            span.set_attribute("tables_searched", tables_searched)
            span.set_status(trace.Status(trace.StatusCode.OK))

            return final_docs

        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise

def clean_llm_response(text: str) -> str:
    # Remove repetitive intros
    text = re.sub(r"(?i)^based on the provided context.*?:\s*", "", text)

    # Remove ticket boilerplate like "Here are some possible causes" or "Steps to troubleshoot:"
    text = re.sub(r"(?i)(steps to troubleshoot:|possible issues and solutions:)\s*", "", text)

    # Trim long whitespace
    text = re.sub(r'\n{2,}', '\n\n', text.strip())

    # Optionally, cut off hallucinated headers
    text = re.sub(r"(?i)^summary:\s*", "", text)

    return text.strip()

def generate_answer(user_query, retrieved_docs):

    with tracer.start_as_current_span("generate_answer") as span: 
        span.set_attribute("query", user_query)
        #relevant documents in search 
        span.set_attribute("context_docs_count", len(retrieved_docs))
        #which model is being used 
        span.set_attribute("model_id", model_id)

    context = "\n==========\n".join(retrieved_docs)
   
    try: 
        context = "\n==========\n".join(retrieved_docs)

        span.set_attribute("context_length", len(context))
        user_prompt = (
            f"Context:\n{context}\n\nUser query:\n{user_query}"
        )
    
        span.set_attribute("prompt_length", len(user_prompt))

        chat_request = oci.generative_ai_inference.models.CohereChatRequest(
            message=user_prompt,
            max_tokens=600,
            temperature=0.5,
            top_p=0.75,
            top_k=0
        )

        chat_detail = oci.generative_ai_inference.models.ChatDetails(
            compartment_id=compartment_id,
            serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(model_id=model_id),
            chat_request=chat_request
        )

        chat_response = generative_ai_inference_client.chat(chat_detail)
        # Apply post-processing filter
        raw_response = chat_response.data.chat_response.text

        span.set_attribute("raw_response_length", len(raw_response))

        
        return clean_llm_response(raw_response)
    
    except Exception as e: 
        span.record_exception(e)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
        raise 

# ================== SCHEMA MANIPULATION ==================
def create_schema_if_not_exists(schema_name):
    admin_user = os.getenv("DB_USER")
    admin_pass = os.getenv("DB_PASS")
    dsn = os.getenv("DB_DSN")

    schema_name = schema_name.upper()
    password = "TempStrongPass123"  # Randomize this or store securely

    with oracledb.connect(user=admin_user, password=admin_pass, dsn=dsn) as conn:
        with conn.cursor() as cur:
            # Check if schema (user) already exists
            cur.execute("SELECT COUNT(*) FROM dba_users WHERE username = :name", {"name": schema_name})
            exists = cur.fetchone()[0] > 0

            if not exists:
                cur.execute(f"CREATE USER {schema_name} IDENTIFIED BY {password}")
                cur.execute(f"GRANT CONNECT, RESOURCE TO {schema_name}")
                cur.execute(f"ALTER USER {schema_name} DEFAULT TABLESPACE users TEMPORARY TABLESPACE temp")
                cur.execute(f"ALTER USER {schema_name} QUOTA UNLIMITED ON data")
                conn.commit()

# ================== ENTRY POINT ==================
if __name__ == "__main__":
    mode = input("Type 'embed' to load/encode data or 'ask' to query: ").strip().lower()

    if mode == 'embed':
        path = input("Enter path to your FAQ file: ").strip()
        table_name = input("Enter the table name to store this data: ").strip()
        schema = input("Enter schema name (e.g., TeamA): ").strip()
        embed_and_store(path, table_name, schema)
        print("Embedding + DB insert complete.")

    elif mode == 'ask':
        while True:
            user_input = input("\nAsk your question (or type quit): ").strip()
            if user_input.lower() in ["quit", "exit"]:
                break
            schema = input("Enter schema name (e.g., TeamA): ").strip()
            docs = vector_search(user_input, schema)
            response = generate_answer(user_input, docs)
            print("\nGenerated Answer:\n", response)
