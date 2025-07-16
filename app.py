# IMPORTS
import threading, requests, os, re, redis, json, sqlite3
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from LLMIntegration import vector_search, generate_answer, embed_and_store, create_schema_if_not_exists
from telemetry import setup_telemetry 
from opentelemetry import trace
import time 
# Connect to local Redis instance (data cache)
redis_client = redis.Redis(host='localhost', port=6379, db=0)

load_dotenv()
app = Flask(__name__)
CORS(app)
tracer = setup_telemetry(app)
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
print("SLACK_BOT_TOKEN =", SLACK_BOT_TOKEN)

file_cache = {}

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
                "text": "Missing table_name parameter. Usage: `/oracleembed table_name=your_table`"
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
        user_input = request.form.get("text", "").strip()

        if not schema:
            return jsonify({"text": "You are not assigned to any team. Use /setteam."})

        if not user_input:
            return jsonify({
                "response_type": "ephemeral",
                "text": "Please enter a question."
            })

        # Step 1: Post the user's question publicly with buttons
        question_post = {
            "channel": channel_id,
            "text": f":question: <@{user_id}> asked:\n>*{user_input}*",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":question: *<@{user_id}> asked:*\n>{user_input}"
                    }
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Ask a question"},
                            "action_id": "ask_new_question"
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Set team"},
                            "action_id": "set_team"
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Unset team"},
                            "action_id": "unset_team"
                        }
                    ]
                }
            ]
        }

        question_response = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={
                "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
                "Content-Type": "application/json"
            },
            json=question_post
        ).json()

        if not question_response.get("ok"):
            return jsonify({
                "response_type": "ephemeral",
                "text": f"Failed to post question: {question_response.get('error')}"
            })

        thread_ts = question_response.get("ts")

        # Step 2: Process answer in a thread
        def process_query_and_respond():
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span("oraclebot_response_latency") as span:
                start_time = time.perf_counter()
                try:
                    memory_key = f"context:{user_id}:{channel_id}"
                    prior_memory = redis_client.get(memory_key)

                    memory_text = json.loads(prior_memory) if prior_memory else []
                    memory_text = memory_text[-7:]  # keep last 7 turns

                    docs = vector_search(user_input, schema)
                    all_context = memory_text + docs
                    response = generate_answer(user_input, all_context)

                    memory_text.append(f"User: {user_input}\nBot: {response}")
                    redis_client.setex(memory_key, 900, json.dumps(memory_text))

                except Exception as e:
                    response = f"Error: {str(e)}"

                requests.post(
                    "https://slack.com/api/chat.postMessage",
                    headers={
                        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "channel": channel_id,
                        "thread_ts": thread_ts,
                        "text": response
                    }
                )

                latency_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("oraclebot.latency_ms", latency_ms)

        threading.Thread(target=process_query_and_respond).start()

        return jsonify({
            "response_type": "ephemeral",
            "text": "Working on it...⏳"
        })

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
        docs = vector_search(question, schema)
        response = generate_answer(question, docs)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})

@app.route("/embed", methods=["POST"])
def embed_file():
    with tracer.start_as_current_span("embed_file_handler"):
        schema = request.form.get("schema")  # Use schema directly now (from HTML)
        table_name = request.form.get("table_name")
        file = request.files.get("file")

        if not file or not table_name or not schema:
            return jsonify({"error": "Missing file, table_name, or schema"}), 400


        #full_table_name = f"{schema}.{table_name}"
        #table_name = table_match.group(1)

        try:
            temp_path = f"/tmp/{file.filename}"
            file.save(temp_path)
            embed_and_store(temp_path, table_name, schema)
            os.remove(temp_path)
            return jsonify({
                "status": "success",
                "message": f"✅ File embedded into `{table_name}`."
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"❌ Embedding failed: {str(e)}"
            }), 500
    #full_table_name = f"{schema}.{table_name}"

    try:
        temp_path = f"/tmp/{file.filename}"
        file.save(temp_path)
        embed_and_store(temp_path, table_name, schema)
        os.remove(temp_path)
        return jsonify({
            "status": "success",
            "message": f"✅ File embedded into `{table_name}`."
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"❌ Embedding failed: {str(e)}"
        }), 500

    
@app.route("/slack/commands", methods=["POST"])
def slack_commands():
    command = request.form.get("command")
    user_id = request.form.get("user_id")
    text = request.form.get("text", "").strip()

    if command == "/setteam":
        team = text.strip().lower()
        create_schema_if_not_exists(team)
        update_user_team(user_id, team)
        return jsonify({"text": f"✅ Your team has been set to `{team}` and schema created (if needed)."})
    
    if command == "/unsetteam":
        conn = sqlite3.connect("user_team.db")
        cur = conn.cursor()
        cur.execute("DELETE FROM user_team WHERE slack_user_id = ?", (user_id,))
        conn.commit()
        conn.close()
        return jsonify({"text": "✅ Your team assignment has been removed."})

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
            "text": f"✅ File `{filename}` embedded into table `{table_name}`"
        })
    except Exception as e:
        requests.post(response_url, json={
            "response_type": "ephemeral",
            "text": f"❌ Embedding failed: {str(e)}"
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
