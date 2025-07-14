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
                                vector VECTOR(384)
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
            f"You are helping a support engineer troubleshoot a customer issue using historical ticket data. "
            f"Never suggest escalating to or contacting support — the engineer reading this IS support. "
            f"Answer with technical solutions, past ticket examples, root causes, and known resolutions.\n\n"
            
            f"Examples of expected responses:\n\n"
            
            f"Q: How would I resolve an issue pertaining to database misconfiguration?\n"
            f"A: Here's the step-by-step resolution for database misconfiguration issues:\n"
            f"1. **Check connection settings** - Verify hostname, port, database name, and credentials in your config file for typos or incorrect values.\n"
            f"2. **Test server connectivity** - Use `telnet <hostname> <port>` to confirm network access and ensure the database service is running.\n"
            f"3. **Validate authentication** - Confirm user credentials have proper permissions and the account isn't locked or expired.\n"
            f"4. **Update and restart** - Correct any misconfigurations in your database config file and restart the application service.\n"
            f"5. **Verify resolution** - Test the database connection and run a simple query to ensure full functionality.\n"
            f"**Source:** Ticket #DB-2847 - Database Connection Failure Resolution\n"
            f"**Link:** https://your-ticketing-system.com/ticket/DB-2847\n\n"
            
            f"Q: The API is returning 500 errors intermittently.\n"
            f"A: Here's the step-by-step resolution for intermittent API 500 errors:\n"
            f"1. **Check server logs** - Review application and web server error logs to identify the root cause of the 500 errors.\n"
            f"2. **Monitor resource usage** - Verify CPU, memory, and database connections aren't hitting limits during error periods.\n"
            f"3. **Test database connectivity** - Ensure database connections are stable and not timing out during high load.\n"
            f"4. **Review recent deployments** - Check if any recent code changes correlate with the error pattern.\n"
            f"5. **Implement error handling** - Add retry logic and improve error logging to handle transient issues gracefully.\n"
            f"**Source:** Ticket #API-5672 - Intermittent 500 Error Investigation\n"
            f"**Link:** https://your-ticketing-system.com/ticket/API-5672\n\n"
            
            f"Q: Users can't log in to the application.\n"
            f"A: Here's the step-by-step resolution for login issues:\n"
            f"1. **Verify authentication service** - Check if the authentication server is running and responding to requests.\n"
            f"2. **Test database connections** - Ensure the user database is accessible and user table queries are working.\n"
            f"3. **Check session management** - Verify session storage (Redis/database) is functioning and not full.\n"
            f"4. **Review security settings** - Confirm firewall rules, rate limiting, and account lockout policies aren't blocking users.\n"
            f"5. **Clear cache and test** - Clear application cache, restart services, and test login with known good credentials.\n"
            f"**Source:** Ticket #AUTH-9134 - User Authentication Failure Resolution\n"
            f"**Link:** https://your-ticketing-system.com/ticket/AUTH-9134\n\n"
            
            f"Context from ticket history:\n{context}\n\n"
            f"User Question: {user_query}\n\n"
            f"Based on the examples above and the ticket history context, provide a structured response with:\n"
            f"- Step-by-step technical resolution\n"
            f"- Source ticket information with links\n"
            f"- No suggestions to contact support or escalate\n"
            f"If no resolution is available, say 'No known resolution found in ticket history. Consider further investigation.'\n"
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
