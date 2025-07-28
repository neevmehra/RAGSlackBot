import os, sys, array, json, re, time, sqlite3, subprocess
import oracledb, oci
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from telemetry import tracer
from opentelemetry import trace
from PyPDF2 import PdfReader
import pandas as pd


# ================== ORACLE DB SETUP ==================
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

oracledb.init_oracle_client(lib_dir=os.path.join(BASE_DIR, "instantclient_23_8"))
os.environ["TNS_ADMIN"] = os.path.join(BASE_DIR, "wallet")

un = os.getenv("DB_USER")
pw = os.getenv("DB_PASS")
cs = os.getenv("DB_DSN")
initial_topK = 20
final_topK = 5

# ================== EMBEDDING MODEL ==================
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

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

def compress_pdf(input_path, output_path): #compression using qpdf
    
        subprocess.run(["qpdf", "--linearize", input_path, output_path], check=True)
        return output_path

def parse_pdf(file_path): #pdf parsing using PyPDF2
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + "\n"   
    return text

def parse_csv(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    df.columns = df.columns.str.strip()
    records = df.to_dict('records')
    parsed_records = []
    for idx, record in enumerate(records):
        sr_number = record.get('SR Number', 'N/A')
        global_party_name = record.get('Global Parent Party Name', 'N/A')
        country = record.get('Country Name', 'N/A')
        title = record.get('Title', 'N/A')
        error_message = record.get('Error Message', 'N/A')
        severity = record.get('Severity', 'N/A')
        status = record.get('Status', 'N/A')
        product_line = record.get('Product Line', 'N/A')
        product_category = record.get('Product Category(Legacy)', 'N/A')
        category = record.get('Category', 'N/A')
        product_version = record.get('Product Version', 'N/A')
        platform = record.get('Platform', 'N/A')
        root_cause = record.get('Root Cause', 'N/A')
        resolution_range = record.get('Resolution Range', 'N/A')
        creation_date = record.get('Creation Date', 'N/A')
        created_month = record.get('Create Month', 'N/A')
        closed_month = record.get("Close Month", "N/A")
        date_closed = record.get('Date Closed', 'N/A')
        resolution_date = record.get('Resolution Date', 'N/A')
        sr_type = record.get('SR Type', 'N/A')
        source = record.get('Source', 'N/A')
        level_of_service = record.get('Level of Service', 'N/A')
        functional_description = record.get('Functional Product Description', 'N/A')

        parsed_record = {
            "sr_number": sr_number,
            "customer": global_party_name,
            "product_line": product_line,
            "root_cause": root_cause,
            "severity": severity,
            "status": status,
            "platform": platform,
            "resolution_range": resolution_range,
            "creation_date": creation_date,
            "created_date": creation_date, 
            "created_month": created_month, 
            "closed_month": closed_month,
            "date_closed": date_closed,
            "source_file": os.path.basename(file_path),
            "title": title,
            "record_index": idx,
            "chunk_id": f"csv_sr_{sr_number}_record_{idx}",
            "source": f"CSV_{os.path.basename(file_path)}"
        }

        parsed_records.append(parsed_record)
    
    return parsed_records

chunk_size = 300  # Define a chunk size in characters for text splitting
def chunk_text_by_tokens(text, max_tokens=300):
    return [text[i:i+chunk_size]for i in range(0, len(text), chunk_size)]


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
        
        elif file_path.endswith('.csv'):
            try:
                csv_records = parse_csv(file_path)
                if not csv_records:
                    raise ValueError("No records found in CSV file")

                for record in csv_records:
                    doc_text = "\n".join([
                        f"Service Request: {record['sr_number']}",
                        f"Customer: {record['customer']}",
                        f"Issue Title: {record['title']}",
                        f"Status: {record['status']}",
                        f"Severity: {record['severity']}",
                        f"Product: {record['product_line']}",
                        f"Root Cause: {record['root_cause']}",
                        f"Create Date: {record['creation_date']}",
                        f"Date Closed: {record['date_closed']}",
                        f"Create Month: {record['created_month']}",
                        f"Closed Month: {record['closed_month']}",
                        f"Resolution Range: {record['resolution_range']}",
                       
                    ])
                    docs.append({"text": doc_text})
            except Exception as e:
                print(f"MINIMAL TEST ERROR: {str(e)}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, f"CSV parsing error: {str(e)}"))
                raise ValueError(f"Error parsing CSV file: {str(e)}")

        elif file_path.endswith('.pdf'):

            compress_path = os.path.join("/tmp", f"compressed._{os.path.basename(file_path)}")
            compress_pdf(file_path, compress_path) # compress the pdf file first
            
            text = parse_pdf(compress_path)

            paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 50]

            for i, paragraph in enumerate(paragraphs):
                chunks = chunk_text_by_tokens(paragraph)
                for j, chunk in enumerate(chunks):
                    docs.append({
                        "text": chunk,
                        "chunk_id": f"pdf_para{i+1}_chunk{j+1}",
                        "source": f"PDF_{os.path.basename(file_path)}"
            }) ###end of the pdf extraction

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
        span.set_attribute("topK", initial_topK)

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
                                FETCH APPROX FIRST {initial_topK} ROWS ONLY
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
            raw_texts = [text for _, text in retrieved_docs[:initial_topK]]
            final_docs = rerank_passages(user_query, raw_texts, top_n=final_topK)

            span.set_attribute("documents_retrieved", len(final_docs))
            span.set_attribute("tables_searched", tables_searched)
            span.set_status(trace.Status(trace.StatusCode.OK))

            return final_docs

        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise

# ================== RERANKER MODEL ==================
def rerank_passages(query, passages, top_n=5):
    pairs = [(query, passage) for passage in passages]
    scores = reranker.predict(pairs, batch_size=8)
    ranked = sorted(zip(passages, scores), key = lambda x: x[1], reverse=True)
    return [p for p, _ in ranked[:top_n]]

# ================== LLM OUTPUT ==================
def clean_llm_response_slack(text: str) -> str:
    # Remove "#"
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)

    # Convert "**" to "*" for bolding
    text = re.sub(r"\*\*(.*?)\*\*", r"*\1*", text)

    # Remove extra spaces
    return text.strip()

def clean_llm_response_web(text: str) -> str:
    # Remove markdown headers
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)

    # Convert **bold** and *bold* to <strong>...</strong>
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.*?)\*", r"<strong>\1</strong>", text)

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
        "You are an expert assistant trained to help users based on internal support documents.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{user_query}\n\n"
        "Answer clearly and concisely with step-by-step resolution processes and include a citation section at the end citing your sources."

        "If the user explicitly asks for a root cause analysis for a specific company, respond using the following structure:\n"
        "1. **Event Summary**: Briefly describe what happened.\n"
        "2. **Root Cause Details**: Explain the underlying issue, how it was identified, and why it occurred.\n"
        "3. **Corrective Actions**: Identify any recurring patterns or issues for that company and suggest specific steps to fix or prevent them.\n"
        "4. **Event Timeline**: Provide key timestamps and events in chronological order.\n\n"
        "For all other questions, respond clearly and concisely with step-by-step resolution processes when appropriate, "
        "and include a citation section at the end citing your sources."
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

        
        return raw_response
    
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
