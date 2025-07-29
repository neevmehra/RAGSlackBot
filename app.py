# IMPORTS
import threading, requests, os, re, redis, json, sqlite3
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from LLMIntegration import vector_search, generate_answer, embed_and_store, create_schema_if_not_exists, clean_llm_response_slack, get_all_schemas, clean_llm_response_web
from telemetry import setup_telemetry 
from opentelemetry import trace
from dotenv import load_dotenv
import time 
from requests_oauthlib import OAuth2Session
from werkzeug.middleware.proxy_fix import ProxyFix
from base64 import urlsafe_b64decode

# Connect to local Redis instance (data cache)
redis_client = redis.Redis(host='localhost', port=6379, db=0)

load_dotenv()
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1)
CORS(app)
app.secret_key = os.getenv("APP_SECRET_KEY")
tracer = setup_telemetry(app)
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
file_cache = {}

USERS = {
    "admin": "1234"
}

@app.route("/login")
def login():
    oauth = get_oauth_session()
    authorization_url, state = oauth.authorization_url(os.getenv("OAUTH_AUTH_URL"))

    # Store state in session to validate later
    session["oauth_state"] = state
    return redirect(authorization_url)

@app.route("/login/callback")
def callback():
    oauth = get_oauth_session(state=session.get("oauth_state"))
    token = oauth.fetch_token(
        os.getenv("OAUTH_TOKEN_URL"),
        client_secret=os.getenv("OAUTH_CLIENT_SECRET"),
        authorization_response=request.url,
    )

    session["oauth_token"] = token

    # Get user info from ID token (JWT) or make request to userinfo endpoint if available
    id_token = token.get("id_token")
    if id_token:

        payload_part = id_token.split('.')[1]
        padded = payload_part + '=' * (4 - len(payload_part) % 4)
        user_info = json.loads(urlsafe_b64decode(padded))
        session["user"] = user_info.get("email", "Unknown")
    else:
        session["user"] = "Unknown"

    return redirect("/")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

@app.route("/")
def index():
    if "user" not in session:
        return redirect("/login")
    return render_template("upload.html")


@app.route("/slack/events", methods=["POST"])
def slack_events():

    with tracer.start_as_current_span("slack_events_handler"):
        #Initialize data to None
        data = None
        
        # Handle JSON requests first
        if request.is_json:
            data = request.get_json()
        
        # Process Events API payloads
        if data:
            # URL verification challenge
            if "challenge" in data:
                return jsonify({"challenge": data["challenge"]})
            
            # Handle file_shared event
            if data.get("event", {}).get("type") == "file_shared":
                event = data["event"]
                file_id = event["file_id"]
                user_id = event["user_id"]
                channel_id = event.get("channel_id")
                if channel_id:
                    file_cache[(user_id, channel_id)] = {
                        "file_id": file_id,
                        "timestamp": event["event_ts"]
                    }
                return jsonify({}), 200
            
            # Handle message event with file_share subtype
            if (data.get("event", {}).get("type") == "message" and 
                data.get("event", {}).get("subtype") == "file_share"):
                event = data["event"]
                files = event.get("files", [])
                if files:
                    file_id = files[0]["id"]
                    user_id = event.get("user")
                    channel_id = event.get("channel")
                    if user_id and channel_id:
                        file_cache[(user_id, channel_id)] = {
                            "file_id": file_id,
                            "timestamp": event["ts"]
                        }
                return jsonify({}), 200

    # Handle Slash Commands (form-encoded)
    user_input = request.form.get("text", "").strip()
    command = request.form.get("command", "").strip()
    response_url = request.form.get("response_url")
    user_id = request.form.get("user_id")
    channel_id = request.form.get("channel_id")

    if command == "/oracleembed":
        schema = get_schema_for_user(user_id)
        if not schema:
            return jsonify({ "text": "You are not assigned to any team. Use /setteam." })

        table_match = re.search(r'table_name=(\w+)', user_input)

        if not table_match:
            return jsonify({
                "response_type": "ephemeral",
                "text": "Missing table_name parameter. Usage: /oracleembed table_name=your_table"
            })
        
        table_name = f"{schema}.{table_match.group(1)}"
        file_info = file_cache.get((user_id, channel_id))
        
        if not file_info:
            return jsonify({
                "response_type": "ephemeral",
                "text": "No recent file found. Please upload a file first."
            })
        
        threading.Thread(
            target=process_slack_embedding,
            args=(user_id, file_info["file_id"], table_name, response_url)
        ).start()
        
        return jsonify({
            "response_type": "ephemeral",
            "text": "Processing your file embedding..."
        })
    
    elif command == "/resetcontext":
        memory_key = f"context:{user_id}:{channel_id}"
        redis_client.delete(memory_key)
        return jsonify({"text": "🔁 Context memory has been cleared."})

    elif command == "/oraclebot":
        schema = get_schema_for_user(user_id)
        thread_ts = request.form.get("thread_ts") or request.form.get("message_ts")
        print("Form Data:", request.form.to_dict())

        if not schema:
            return jsonify({ "text": "You are not assigned to any team. Use /setteam." })

        if not user_input or not response_url:
            return jsonify({
                "response_type": "ephemeral",
                "text": "Please enter a question."
            })
        
        def process_query_and_respond(thread_ts):
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span("oraclebot_response_latency") as span:
                start_time = time.perf_counter()
                try:
                    # Fetch previous memory if any
                    memory_key = f"context:{user_id}:{channel_id}"
                    prior_memory = redis_client.get(memory_key)

                    if prior_memory:
                        memory_text = json.loads(prior_memory)
                    else:
                        memory_text = []

                    memory_text = memory_text[-7:]  # Trim after assigning

                    # Combine vector search with short-term memory
                    docs = vector_search(user_input, schema)
                    all_context = memory_text + docs
                    response = generate_answer(user_input, all_context)
                    response = clean_llm_response_slack(response)

                    # Update the memory with this latest turn
                    new_entry = f"User: {user_input}\nBot: {response}"
                    memory_text.append(new_entry)

                    # Save back to Redis with TTL = 15 mins (900 seconds)
                    redis_client.setex(memory_key, 900, json.dumps(memory_text))

                except Exception as e:
                    response = f"Error: {str(e)}"

                requests.post(response_url, json={
                    "response_type": "in_channel",
                    "text": response,
                    "thread_ts": thread_ts
                })

                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000  # convert to milliseconds
                span.set_attribute("oraclebot.latency_ms", latency_ms)
                print(f"[Telemetry] OracleBot Response Latency: {latency_ms:.2f} ms")
        
        threading.Thread(target=process_query_and_respond, args=(thread_ts,)).start()
        return jsonify({
            "response_type": "ephemeral",
            "text": "Working on it...⏳"
        })
    
    return jsonify({"error": "Unsupported command"}), 400

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    question = data.get("question")
    schema = data.get("schema")

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if not schema:
        return jsonify({"error": "No schema provided"}), 400

    try:
        docs = vector_search(question, schema)  # Query from the correct schema
        response = generate_answer(question, docs)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"}), 500

@app.route("/embed", methods=["POST"])
def embed_file():
    schema = request.form.get("schema")
    new_team = request.form.get("new_team")
    table_name = request.form.get("table_name")
    file = request.files.get("file")

    # Handle creation of new team schema
    if new_team and new_team.strip():
        schema = new_team.strip()
        create_schema_if_not_exists(schema)
        user_id = session.get("user_id")
        if user_id:
            update_user_team(user_id, schema)

    if not file or not table_name or not schema:
        return jsonify({"error": "Missing file, table_name, or schema"}), 400

    with tracer.start_as_current_span("embed_file_handler"):
        try:
            temp_path = f"/tmp/{file.filename}"
            file.save(temp_path)
            embed_and_store(temp_path, table_name, schema)
            os.remove(temp_path)
            return jsonify({
                "status": "success",
                "message": f"✅ File embedded into {table_name}."
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"❌ Embedding failed: {str(e)}"
            }), 500

@app.route("/schemas")
def get_schemas():
    schemas = get_all_schemas()
    return jsonify({"schemas": schemas})

@app.route("/slack/commands", methods=["POST"])
def slack_commands():
    command = request.form.get("command")
    user_id = request.form.get("user_id")
    text = request.form.get("text", "").strip()

    if command == "/setteam":
        team = text.strip().lower()
        create_schema_if_not_exists(team)
        update_user_team(user_id, team)
        return jsonify({"text": f"✅ Your team has been set to {team} and schema created (if needed)."})
    
    if command == "/unsetteam":
        conn = sqlite3.connect("user_team.db")
        cur = conn.cursor()
        cur.execute("DELETE FROM user_team WHERE slack_user_id = ?", (user_id,))
        conn.commit()
        conn.close()
        return jsonify({"text": "✅ Your team assignment has been removed."})

def get_oauth_session(state=None, token=None):
    return OAuth2Session(
        client_id=os.getenv("OAUTH_CLIENT_ID"),
        redirect_uri=os.getenv("OAUTH_REDIRECT_URI"),
        scope=["openid", "email", "profile"],
        state=state,
        token=token
    )

def update_user_team(user_id, team):
    conn = sqlite3.connect("user_team.db")
    cur = conn.cursor()
    cur.execute("""
                CREATE TABLE IF NOT EXISTS user_team (
                slack_user_id TEXT PRIMARY KEY,
                team_schema TEXT
                )
                """)
    cur.execute("""
        INSERT INTO user_team (slack_user_id, team_schema)
        VALUES (?, ?)
        ON CONFLICT(slack_user_id) DO UPDATE SET team_schema=excluded.team_schema
    """, (user_id, team))
    conn.commit()
    conn.close()

def get_schema_for_user(user_id):
    conn = sqlite3.connect("user_team.db")
    cur = conn.cursor()
    cur.execute("SELECT team_schema FROM user_team WHERE slack_user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

def download_slack_file(file_id):
    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
    file_info = requests.get(
        f"https://slack.com/api/files.info?file={file_id}",
        headers=headers
    ).json()
    if not file_info.get("ok"):
        raise Exception(f"File info error: {file_info.get('error')}")
    download_url = file_info["file"]["url_private_download"]
    file_response = requests.get(download_url, headers=headers)
    if file_response.status_code != 200:
        raise Exception(f"Download failed: HTTP {file_response.status_code}")
    return file_response.content, file_info["file"]["name"]

def process_slack_embedding(user_id, file_id, table_name, response_url):
    
    schema = get_schema_for_user(user_id)

    try:
        file_content, filename = download_slack_file(file_id)
        temp_path = f"/tmp/{filename}"
        with open(temp_path, "wb") as f:
            f.write(file_content)
        embed_and_store(temp_path, table_name, schema)
        os.remove(temp_path)
        requests.post(response_url, json={
            "response_type": "in_channel",
            "text": f"✅ File {filename} embedded into table {table_name}"
        })
    except Exception as e:
        requests.post(response_url, json={
            "response_type": "ephemeral",
            "text": f"❌ Embedding failed: {str(e)}"
        })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)