import threading
import requests
import os
import re
from flask import Flask, request, jsonify
from LLMIntegration import vector_search, generate_answer, embed_and_store
import sqlite3

app = Flask(__name__)

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
file_cache = {}

@app.route("/slack/events", methods=["POST"])
def slack_events():
    # Initialize data to None
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
    
    elif command == "/oraclebot":
        schema = get_schema_for_user(user_id)

        if not schema:
            return jsonify({ "text": "You are not assigned to any team. Use /setteam." })

        if not user_input or not response_url:
            return jsonify({
                "response_type": "ephemeral",
                "text": "Please enter a question."
            })
        
        def process_query_and_respond():
            try:
                docs = vector_search(user_input, schema)
                response = generate_answer(user_input, docs)
            except Exception as e:
                response = f"Error: {str(e)}"
            requests.post(response_url, json={
                "response_type": "in_channel",
                "text": response
            })
        
        threading.Thread(target=process_query_and_respond).start()
        return jsonify({
            "response_type": "ephemeral",
            "text": "Working on it...⏳"
        })
    
    return jsonify({"error": "Unsupported command"}), 400

@app.route("/embed", methods=["POST"])
def embed_file():
    schema = request.form.get("schema")  # Use schema directly now (from HTML)
    table_name = request.form.get("table_name")
    file = request.files.get("file")

    if not file or not table_name or not schema:
        return jsonify({"error": "Missing file, table_name, or schema"}), 400

    full_table_name = f"{schema}.{table_name}"

    try:
        temp_path = f"/tmp/{file.filename}"
        file.save(temp_path)
        embed_and_store(temp_path, full_table_name, schema)
        os.remove(temp_path)
        return jsonify({
            "status": "success",
            "message": f"✅ File embedded into `{full_table_name}`."
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
        team = text.lower()
        update_user_team(user_id, team)
        return jsonify({"text": f"Your team has been set to `{team}`."})
    
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

# Rest of the code remains unchanged
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
