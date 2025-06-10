import threading
import requests
from flask import Flask, request, jsonify
from LLMIntegration import vector_search, generate_answer, embed_and_store
import os

app = Flask(__name__)

@app.route("/slack/events", methods=["POST"])
def slack_query():
    user_input = request.form.get("text", "").strip()
    response_url = request.form.get("response_url")

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
