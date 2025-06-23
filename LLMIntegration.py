import os, sys, array, json, re, time
import oracledb, oci
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

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
def embed_and_store(file_path, table_name):
    encoder = SentenceTransformer('all-MiniLM-L12-v2')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        tickets = json.load(f)  # Load JSON array of tickets

    # Extract and format relevant fields from each ticket
    docs = []
    for ticket in tickets:
        sr_num = ticket.get('sr_number', 'N/A')
        ticket_id = ticket.get('ticket_id', 'N/A')
        problem = ticket.get('description', ticket.get('summary', 'N/A'))
        solution = ticket.get('resolution_description', 'N/A')
        root_cause = ticket.get('root_cause', 'N/A')
        doc_text = (
            f"SR: {sr_num} | Ticket: {ticket_id}\n"
            f"Problem: {problem}\n"
            f"Solution: {solution}\n"
            f"Root Cause: {root_cause}"
        )
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
            """, {'table_name': table_name.upper(), 'owner': un.upper()})
            exists = cursor.fetchone()[0] > 0

            if not exists:
                cursor.execute(f"""
                    CREATE TABLE {table_name} (
                        id NUMBER PRIMARY KEY,
                        payload CLOB CHECK (payload IS JSON),
                        vector VECTOR
                    )
                """)

            prepared_data = [(row['id'], json.dumps(row['payload']), row['vector']) for row in data]
            cursor.executemany(
                f"INSERT INTO {table_name} (id, payload, vector) VALUES (:1, :2, :3)",
                prepared_data
            )
            connection.commit()

# ================== VECTOR SEARCH + GENERATE ==================
def vector_search(user_query):
    embedding = list(embedding_model.encode(user_query))
    vec = array.array("f", embedding)
    retrieved_docs = []

    with oracledb.connect(user=un, password=pw, dsn=cs) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT table_name FROM all_tables
                WHERE owner = :owner AND table_name NOT LIKE 'BIN$%'
            """, {'owner': un.upper()})
            tables = [row[0] for row in cursor.fetchall()]

            for table_name in tables:
                try:
                    sql_retrieval = f'''
                        SELECT payload, VECTOR_DISTANCE(vector, :vector, EUCLIDEAN) as score 
                        FROM {table_name}
                        ORDER BY score 
                        FETCH APPROX FIRST {topK} ROWS ONLY
                    '''
                    for (info, score,) in cursor.execute(sql_retrieval, vector=vec):
                        info_str = info.read() if isinstance(info, oracledb.LOB) else info
                        retrieved_docs.append((score, json.loads(info_str)["text"]))
                except Exception:
                    continue  # skip tables that don't match schema

    retrieved_docs.sort(key=lambda x: x[0])
    return [text for _, text in retrieved_docs[:topK]]

def generate_answer(user_query, retrieved_docs):
    context = "\n==========\n".join(retrieved_docs)
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"User Question: {user_query}\n\n"
        "Answer (cite the context when relevant with a specific support ticket name/number):"
    )

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
    return chat_response.data.chat_response.text

# ================== ENTRY POINT ==================
if __name__ == "__main__":
    mode = input("Type 'embed' to load/encode data or 'ask' to query: ").strip().lower()

    if mode == 'embed':
        path = input("Enter path to your FAQ file: ").strip()
        table_name = input("Enter the table name to store this data: ").strip()
        embed_and_store(path, table_name)
        print("Embedding + DB insert complete.")

    elif mode == 'ask':
        while True:
            user_input = input("\nAsk your question (or type quit): ").strip()
            if user_input.lower() in ["quit", "exit"]:
                break
            docs = vector_search(user_input)
            response = generate_answer(user_input, docs)
            print("\nGenerated Answer:\n", response)
