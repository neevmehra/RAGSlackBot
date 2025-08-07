import json
import os
import re
import sqlite3
import threading
from base64 import urlsafe_b64decode
import time
import oci
import requests
import redis
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from opentelemetry import trace
from dotenv import load_dotenv
from requests_oauthlib import OAuth2Session
from werkzeug.middleware.proxy_fix import ProxyFix

from LLMIntegration import (
    clean_llm_response_slack,
    clean_llm_response_web,
    create_schema_if_not_exists,
    embed_and_store,
    generate_answer,
    get_all_schemas,
    vector_search
)
from telemetry import setup_telemetry, push_custom_metric

# Connect to local Redis instance (data cache)
redis_client = redis.Redis(host="localhost", port=6379, db=0)

load_dotenv()
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1)
CORS(app)
app.secret_key = os.getenv("APP_SECRET_KEY")
tracer = setup_telemetry(app)
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
file_cache = {}
SLACK_BOT_USER_ID = os.getenv("SLACK_BOT_USER_ID")
config = oci.config.from_file("/home/opc/oracle-bot/.oci/config")
# Remove debug print for config
# print(config["user"], config["region"])

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
    # Get user info from ID token (JWT) if available
    id_token = token.get("id_token")
    if id_token:
        payload_part = id_token.split(".")[1]
        padded = payload_part + "=" * (4 - len(payload_part) % 4)
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
    data = request.get_json()
    # Handle Slack URL verification challenge
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})
    event = data.get("event", {})
    if event.get("type") != "message" or event.get("subtype"):
        return jsonify({}), 200  # Ignore bot messages, joins, edits, etc.
    text = event.get("text", "")
    user_id = event.get("user")
    channel_id = event.get("channel")
    thread_ts = event.get("thread_ts") or event.get("ts")
    # Determine if bot was explicitly mentioned
    bot_mentioned = f"<@{SLACK_BOT_USER_ID}>" in text
    # Check if this is a follow-up in an active thread
    thread_active = redis_client.get(f"active_thread:{channel_id}:{thread_ts}")
    if not bot_mentioned and not thread_active:
        return jsonify({}), 200  # Ignore unrelated messages
    schema = get_schema_for_user(user_id)
    if not schema:
        return jsonify({}), 200  # User has no team/schema assigned
    # Mark thread as active for follow-ups (15 min TTL)
    redis_client.setex(f"active_thread:{channel_id}:{thread_ts}", 900, "active")

    def process_and_reply():
        memory_key = f"context:{user_id}:{channel_id}"
        prior_memory = redis_client.get(memory_key)
        memory_text = json.loads(prior_memory) if prior_memory else []
        memory_text = memory_text[-7:]  # Keep recent history
        try:
            docs = vector_search(text, schema)
            all_context = memory_text + docs
            response = generate_answer(text, all_context)
            response = clean_llm_response_slack(response)
            # Update memory
            new_entry = f"User: {text}\nBot: {response}"
            memory_text.append(new_entry)
            redis_client.setex(memory_key, 900, json.dumps(memory_text))
        except Exception as e:
            response = f"⚠️ Error generating response: {str(e)}"
        headers = {
            "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "channel": channel_id,
            "thread_ts": thread_ts,
            "text": response
        }
        requests.post("https://slack.com/api/chat.postMessage", headers=headers, data=json.dumps(payload))

    threading.Thread(target=process_and_reply).start()
    return jsonify({}), 200

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
    if command == "/resetcontext":
        channel_id = request.form.get("channel_id")
        memory_key = f"context:{user_id}:{channel_id}"
        redis_client.delete(memory_key)
        return jsonify({"text": "🔁 Context memory has been cleared."})


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