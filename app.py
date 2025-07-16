import os, re, json, threading, requests, sqlite3, redis
from flask import Flask, request, jsonify
from flask_cors import CORS
from LLMIntegration import vector_search, generate_answer, embed_and_store, create_schema_if_not_exists

app = Flask(__name__)
CORS(app)

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_schema_for_user(user_id):
    conn = sqlite3.connect("user_team.db")
    cur = conn.cursor()
    cur.execute("SELECT team_schema FROM user_team WHERE slack_user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.get_json()
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    event = data.get("event", {})
    if event.get("type") == "message" and not event.get("bot_id"):
        channel_id = event.get("channel")
        thread_ts = event.get("thread_ts", event.get("ts"))
        send_support_button(channel_id, thread_ts)
        return '', 200
    return '', 200

def send_support_button(channel_id, thread_ts):
    payload = {
        "channel": channel_id,
        "thread_ts": thread_ts,
        "text": "How can we help?",
        "blocks": [
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Technical Support"},
                        "action_id": "technical_support_click"
                    }
                ]
            }
        ]
    }
    requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}", "Content-Type": "application/json"},
        json=payload
    )

@app.route("/slack/interactive", methods=["POST"])
def slack_interactive():
    payload = json.loads(request.form["payload"])
    # Handle button click
    if payload["type"] == "block_actions":
        action = payload["actions"][0]
        if action["action_id"] == "technical_support_click":
            trigger_id = payload["trigger_id"]
            channel_id = payload["channel"]["id"]
            message_ts = payload["message"]["ts"]
            open_support_modal(trigger_id, channel_id, message_ts)
            return '', 200
    elif payload["type"] == "view_submission":
        user_id = payload["user"]["id"]
        question = payload["view"]["state"]["values"]["question_block"]["question_input"]["value"]
        metadata = json.loads(payload["view"]["private_metadata"])
        channel_id = metadata.get("channel_id")
        thread_ts = metadata.get("thread_ts")
        # Starts working in background and acknowledges modal
        threading.Thread(target=process_tech_support, args=(user_id, question, channel_id, thread_ts)).start()
        return jsonify({"response_action": "clear"}), 200
    return '', 200

def open_support_modal(trigger_id, channel_id, thread_ts):
    modal = {
        "type": "modal",
        "callback_id": "tech_support_submit",
        "private_metadata": json.dumps({"channel_id": channel_id, "thread_ts": thread_ts}),
        "title": {"type": "plain_text", "text": "Technical Support"},
        "submit": {"type": "plain_text", "text": "Submit"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "blocks": [
            {
                "type": "input",
                "block_id": "question_block",
                "label": {"type": "plain_text", "text": "What's your question?"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "question_input",
                    "multiline": True
                }
            }
        ]
    }
    payload = {"trigger_id": trigger_id, "view": modal}
    requests.post(
        "https://slack.com/api/views.open",
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}", "Content-Type": "application/json"},
        json=payload
    )

def process_tech_support(user_id, question, channel_id, thread_ts):
    # Notify user
    requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}", "Content-Type": "application/json"},
        json={
            "channel": channel_id,
            "thread_ts": thread_ts,
            "text": f"Got your question, <@{user_id}>! Let me think…"
        }
    )
    try:
        schema = get_schema_for_user(user_id)
        if not schema:
            answer = "You are not assigned to any team. Use /setteam."
        else:
            docs = vector_search(question, schema)
            answer = generate_answer(question, docs)
    except Exception as e:
        answer = f"Error processing your question: {str(e)}"
    # Reply with answer
    message = (
        f"*Tech Support Request from* <@{user_id}>\n"
        f"*Question:*\n>{question}\n\n"
        f"*Answer:*\n{answer}"
    )
    requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}", "Content-Type": "application/json"},
        json={"channel": channel_id, "thread_ts": thread_ts, "text": message}
    )

### Slash command `/oraclebot` and `/oracleembed` retained as in your current app
@app.route("/slack/commands", methods=["POST"])
def slack_commands():
    command = request.form.get("command")
    user_id = request.form.get("user_id")
    text = request.form.get("text", "").strip()
    channel_id = request.form.get("channel_id")
    response_url = request.form.get("response_url")
    thread_ts = request.form.get("thread_ts") or request.form.get("message_ts")
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
    if command == "/oracleembed":
        schema = get_schema_for_user(user_id)
        if not schema:
            return jsonify({ "text": "You are not assigned to any team. Use /setteam." })
        table_match = re.search(r'table_name=(\w+)', text)
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
    if command == "/oraclebot":
        schema = get_schema_for_user(user_id)
        if not schema:
            return jsonify({ "text": "You are not assigned to any team. Use /setteam." })
        if not text or not response_url:
            return jsonify({
                "response_type": "ephemeral",
                "text": "Please enter a question."
            })
        def process_query_and_respond(thread_ts):
            try:
                memory_key = f"context:{user_id}:{channel_id}"
                prior_memory = redis_client.get(memory_key)
                memory_text = json.loads(prior_memory) if prior_memory else []
                memory_text = memory_text[-7:]
                docs = vector_search(text, schema)
                all_context = memory_text + docs
                response = generate_answer(text, all_context)
                new_entry = f"User: {text}\nBot: {response}"
                memory_text.append(new_entry)
                redis_client.setex(memory_key, 900, json.dumps(memory_text))
            except Exception as e:
                response = f"Error: {str(e)}"
            requests.post(response_url, json={
                "response_type": "in_channel",
                "text": response,
                "thread_ts": thread_ts
            })
        threading.Thread(target=process_query_and_respond, args=(thread_ts,)).start()
        return jsonify({
            "response_type": "ephemeral",
            "text": "Working on it...⏳"
        })
    return jsonify({"error": "Unsupported command"}), 400

def update_user_team(user_id, team):
    conn = sqlite3.connect("user_team.db")
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_team (
        slack_user_id TEXT PRIMARY KEY,
        team_schema TEXT
    )""")
    cur.execute("""
    INSERT INTO user_team (slack_user_id, team_schema)
    VALUES (?, ?)
    ON CONFLICT(slack_user_id) DO UPDATE SET team_schema=excluded.team_schema
    """, (user_id, team))
    conn.commit()
    conn.close()

file_cache = {}

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
    app.run(host="0.0.0.0", port=5000)