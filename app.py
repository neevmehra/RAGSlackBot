from flask import Flask, request, jsonify
from LLMIntegration import vector_search, generate_answer, embed_and_store
import os

app = Flask(__name__)

@app.route("/slack/events", methods=["POST"])
def slack_query():
    user_input = request.form.get("text", "").strip()
    if not user_input:
        return jsonify({"response_type": "ephemeral", "text": "Please enter a question."})

    try:
        docs = vector_search(user_input)
        response = generate_answer(user_input, docs)
        return jsonify({"response_type": "in_channel", "text": response})
    except Exception as e:
        return jsonify({"response_type": "ephemeral", "text": f"Error: {str(e)}"})


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
