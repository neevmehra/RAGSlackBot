import threading
import requests
import os
import re
from flask import Flask, request, jsonify
from LLMIntegration import vector_search, generate_answer, embed_and_store

app = Flask(__name__)

# Slack bot token from environment variables
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

# Global cache for file metadata (in production, use persistent storage)
file_cache = {}

@app.route("/slack/events", methods=["POST"])
def slack_events():
    # Handle Events API verification
    if request.json and "challenge" in request.json:
        return jsonify({"challenge": request.json["challenge"]})
    
    # Process file_shared events
    if request.json and request.json.get("event", {}).get("type") == "file_shared":
        event = request.json["event"]
        file_id = event["file_id"]
        user_id = event["user_id"]
        channel_id = event["channel_id"]
        
        # Store file metadata
        file_cache[(user_id, channel_id)] = {
            "file_id": file_id,
            "timestamp": event["event_ts"]
        }
        return jsonify({}), 200
    
    # Handle slash commands
    user_input = request.form.get("text", "").strip()
    command = request.form.get("command", "").strip()
    response_url = request.form.get("response_url")
    user_id = request.form.get("user_id")
    channel_id = request.form.get("channel_id")

    if command == "/oracleembed":
        # Extract table name from command text
        table_match = re.search(r'table_name=(\w+)', user_input)
        if not table_match:
            return jsonify({"response_type": "ephemeral", "text": "Missing table_name parameter. Usage: `/oracleembed table_name=your_table`"})
        
        table_name = table_match.group(1)
        
        # Retrieve cached file info
        file_info = file_cache.get((user_id, channel_id))
        if not file_info:
            return jsonify({"response_type": "ephemeral", "text": "No recent file found. Please upload a file first."})
        
        # Start processing in background thread
        threading.Thread(
            target=process_slack_embedding,
            args=(file_info["file_id"], table_name, response_url)
        ).start()
        
        return jsonify({"response_type": "ephemeral", "text": "Processing your file embedding..."})
    
    elif command == "/oraclebot":
        # Original LLM chat flow
        if not user_input or not response_url:
            return jsonify({"response_type": "ephemeral", "text": "Please enter a question."})
        
        def process_query_and_respond():
            try:
                docs = vector_search(user_input)
                response = generate_answer(user_input, docs)
            except Exception as e:
                response = f"Error: {str(e)}"
            requests.post(response_url, json={
                "response_type": "in_channel",
                "text": response
            })
        
        threading.Thread(target=process_query_and_respond).start()
        return jsonify({"response_type": "ephemeral", "text": "Working on it...⏳"})
    
    else:
        return jsonify({"error": "Unsupported command"}), 400

def download_slack_file(file_id):
    """Download file from Slack using file ID"""
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

def process_slack_embedding(file_id, table_name, response_url):
    """Process file embedding in background"""
    try:
        file_content, filename = download_slack_file(file_id)
        
        # Save to temp file
        temp_path = f"/tmp/{filename}"
        with open(temp_path, "wb") as f:
            f.write(file_content)
        
        # Process embedding
        embed_and_store(temp_path, table_name)
        os.remove(temp_path)
        
        # Send success response
        requests.post(response_url, json={
            "response_type": "in_channel",
            "text": f"✅ File `{filename}` embedded into table `{table_name}`"
        })
    except Exception as e:
        requests.post(response_url, json={
            "response_type": "ephemeral",
            "text": f"❌ Embedding failed: {str(e)}"
        })

# Existing embed endpoint for non-Slack uploads
@app.route("/embed", methods=["POST"])
def embed_file():
    file = request.files.get("file")
    table_name = request.form.get("table_name")
    if not file or not table_name:
        return jsonify({"error": "Missing file or table_name"}), 400
    try:
        temp_path = f"/tmp/{file.filename}"
        file.save(temp_path)
        embed_and_store(temp_path, table_name)
        os.remove(temp_path)
        return jsonify({"status": "success", "message": f"File embedded into table {table_name}."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
