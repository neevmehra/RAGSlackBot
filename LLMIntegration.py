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
topK = 5

# ================== EMBEDDING MODEL ==================
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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

    # Wrap the entire function in a span
    with tracer.start_as_current_span("embed_and_store") as span:
        # Add attributes to track what we're processing

        # tracks which attribute is being processed 
        span.set_attribute("file_path", file_path)
        # tracks what database table the data goes into 
        span.set_attribute("table_name", table_name)
        # tracks what kind of file is being processed 
        span.set_attribute("file_type", "json" if file_path.endswith('.json') else "other")

    encoder = SentenceTransformer('all-MiniLM-L12-v2')
    docs = []

    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                # Handle single objects and arrays
                data = json.load(f)
                tickets = [data] if isinstance(data, dict) else data

                if not tickets:
                    return {"error": "No tickets found in the JSON file."}
                
            except json.JSONDecodeError:
                return {"error": "Invalid JSON format"}
        
        for ticket in tickets:
            # Build comprehensive chunk with all relevant fields
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
            ])
            docs.append({"text": doc_text})

    elif file_path.endswith('.pdf'):
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"

                    paragraphs = [long.strip() for long in text.split('\n')
                    if long.strip()]
                    for idx, paragraph in enumerate(paragraphs):
                        if len(paragraph) > 50:
                            docs.append({
                                "text": f"PDF Content (Page {idx + 1}): {paragraph}",
                                "source": f"PDF_{os.path.basename (file_path)}" })  ###end of the pdf extraction
                            
    else:
        return {"error": "Unsupported file type. Only .json and .pdf are supported."}

    
    # Existing embedding and DB storage logic below
    data = [{"id": idx, "vector_source": doc["text"], "payload": doc} for idx, doc in enumerate(docs)]
    texts = [row['vector_source'] for row in data]
    embeddings = encoder.encode(texts, batch_size=32, show_progress_bar=True)
    
    for row, embedding in zip(data, embeddings):
        row['vector'] = array.array("f", embedding)


        try:   
            encoder = SentenceTransformer('all-MiniLM-L12-v2')
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        # Handle single objects and arrays
                        data = json.load(f)
                        tickets = [data] if isinstance(data, dict) else data
                    except json.JSONDecodeError:
                        return {"error": "Invalid JSON format"}

                docs = []
                for ticket in tickets:
                    # Build comprehensive chunk with all relevant fields
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
                    ])
                    docs.append({"text": doc_text})

            # Existing embedding and DB storage logic below
            data = [{"id": idx, "vector_source": doc["text"], "payload": doc} for idx, doc in enumerate(docs)]
            texts = [row['vector_source'] for row in data]
            embeddings = encoder.encode(texts, batch_size=32, show_progress_bar=True)

            for row, embedding in zip(data, embeddings):
                row['vector'] = array.array("f", embedding)

            with oracledb.connect(user=un, password=pw, dsn=cs) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(f"""
                        SELECT COUNT(*) FROM all_tables
                        WHERE table_name = :table_name AND owner = :owner
                    """, {'table_name': table_name.upper(), 'owner': schema.upper()})
                    exists = cursor.fetchone()[0] > 0

                    if not exists:
                        cursor.execute(f"""
                            CREATE TABLE {schema}.{table_name} (
                                id NUMBER PRIMARY KEY,
                                payload CLOB CHECK (payload IS JSON),
                                vector VECTOR
                            )
                        """)

                    prepared_data = [(row['id'], json.dumps(row['payload']), row['vector']) for row in data]
                    cursor.executemany(
                        f"INSERT INTO {schema}.{table_name} (id, payload, vector) VALUES (:1, :2, :3)",
                        prepared_data)

            if not exists:
                print("Creating table with:", schema, table_name)
                cursor.execute(f"""
                    CREATE TABLE {schema}.{table_name} (
                        id NUMBER PRIMARY KEY,
                        payload CLOB,
                        vector VECTOR
                    )
                """) 
                
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
        #Add attributes to track what we're processing
        #keeping track of question asked from the user 
        span.set_attribute("query", user_query)
        #how many results are requested from the table 
        span.set_attribute("topK", topK)
    

        try: 
            embedding = list(embedding_model.encode(user_query))
            vec = array.array("f", embedding)
            retrieved_docs = []

            with oracledb.connect(user=un, password=pw, dsn=cs) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("""
                        SELECT table_name FROM all_tables
                        WHERE owner = :owner AND table_name NOT LIKE 'BIN$%'
                    """, {'owner': schema.upper()})
                    tables = [row[0] for row in cursor.fetchall()]

                for table_name in tables:
                    try:
                        sql_retrieval = f'''
                            SELECT payload, VECTOR_DISTANCE(vector, :vector, EUCLIDEAN) as score 
                            FROM {schema}.{table_name}
                            ORDER BY score 
                            FETCH APPROX FIRST {topK} ROWS ONLY
                        '''
                        for (info, score,) in cursor.execute(sql_retrieval, vector=vec):
                            info_str = info.read() if isinstance(info, oracledb.LOB) else info
                            retrieved_docs.append((score, json.loads(info_str)["text"]))
                    except Exception:
                        continue  # skip tables that don't match schema

            retrieved_docs.sort(key=lambda x: x[0])
            final_docs = [text for _, text in retrieved_docs[:topK]]

            span.set_attribute("documents_retrieved", len(final_docs))
            span.set_status(trace.Status(trace.StatusCode.OK))

            return final_docs 
        
        except Exception as e: 
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e))) 
            raise 
        
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
            f"Context:\n{context}\n\n"
            f"User Question: {user_query}\n\n"
            f"You are helping a support engineer troubleshoot a customer issue using historical ticket data. "
            f"Never suggest escalating to or contacting support — the engineer reading this IS support. "
            f"Answer the question ONLY with technical solutions, past ticket examples, root causes, and known resolutions. "
            f"If no resolution is available, say 'No known resolution found in ticket history. Consider further investigation.'\n"
            f"Answer their question by:\n"
            f"- Citing specific ticket IDs (e.g., TECH-8926) when referencing similar cases\n"
            f"- Explaining what worked in past resolutions\n"
            f"- Providing actionable troubleshooting steps based on successful tickets\n"
            f"- Including resolution timeframes and root causes from the ticket data\n"
            f"- Add a separate citations section for the tickets you used\n"
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
