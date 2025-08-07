# Customer Support Service AI Agent

## Overview

The Customer Support Service AI Agent is an AI-powered assistant designed to enhance the efficiency of Oracle support engineers. It acts as a first line of defense, helping engineers triage service tickets faster by retrieving relevant historical issues, documents, and RCA data in real-time.

## Problem Statement

Customer support workflows at Oracle involve manually reproducing issues and combing through various documents and past tickets, often taking 30+ minutes per ticket. This leads to:

- Delays in resolution time
- Reduced customer satisfaction
- Lower engineering productivity

## Proposed Solution

An AI-powered agent that:
- Answers customer queries in real-time
- Surfaces similar past issues and RCAs
- Recommends solutions based on historical support data
- Streamlines triage and improves SLA compliance (15-minute RCA & 2-hour solution targets)

## Setup Instructions 
1. Oracle Setup
Download and extract Oracle Instant Client into instantclient_23_8[https://www.oracle.com/database/technologies/instant-client/downloads.html]
Place Wallet files in /wallet directory: cwallet.sso, sqlnet.ora, etc.
2. Environment File .env
    DB_USER=ADMIN
    DB_PASS=Oracle123456
    DB_DSN=eckoel9tcpqzr2nl_high
    SLACK_BOT_TOKEN=...
    APP_SECRET_KEY=...
    OAUTH_CLIENT_ID=...
    OAUTH_CLIENT_SECRET=...
    OAUTH_AUTH_URL=...
    OAUTH_TOKEN_URL=...
    OAUTH_REDIRECT_URI=...
3. Install & Run
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python app.py

## User Experience

Support engineers can interact with the agent through:

- **Slack Bot**: Ask questions, get AI-generated answers, and link to past tickets in a threaded, organized Slack experience.
- **Web Interface**: Upload tickets (JSON), guides (PDF), and RCA logs (CSV). Chat with the AI directly via [customersupportagent.com](https://customersupportagent.com).

## Features

- Vector search with cosine similarity to retrieve semantically relevant results
- Context-aware RAG (Retrieval-Augmented Generation) pipeline
- Secure integration with Oracle systems
- Slack threading + natural language interface
- Re-ranking via Cross-Encoder for high-accuracy results
- Escalation workflows to SMEs if AI can’t resolve an issue

### Technologies Used

| Component | Purpose |
|----------|---------|
| **Python** | Backend logic and orchestration |
| **Cohere Embed v4.0** | Embedding model |
| **23AI Vector Search** | Fast semantic similarity search |
| **Cross-Encoder (MiniLM)** | Re-ranking layer |
| **Cohere Command-a-03-2025** | LLM for response generation |
| **Oracle Autonomous DB** | Metadata and ticket storage |
| **OCI Logging/Monitoring** | Logs and KPIs |
| **Redis** | Context memory (15 min TTL) |
| **Slack + Flask** | Interface and routing |
| **HTML UI + PDF/CSV ingestion** | File support in web portal |

## System Flow

1. **Data Ingestion & Processing**
   - Ticket data, RCAs, and documents are chunked and embedded
   - Stored in OCI Autonomous DB & vector storage

2. **Query Handling**
   - User query is embedded
   - Vector search retrieves top-K results
   - Cross-Encoder re-ranks top results

3. **Response Generation**
   - RAG combines query + context
   - LLM generates accurate, source-cited responses

4. **Delivery**
   - Output is sent to Slack or Web UI

## Security & Privacy

- Data privacy enforced via access control & encryption
- Role-based access for sensitive support data
- Sensitive fields tokenized in embedding workflows
- Confidence scoring to reduce over-reliance on AI

## Testing

Used `unittest` and `pytest` with mocks for:

- Database calls
- LLM generation
- Vector search
- Slack integration

Logic tested in isolation to ensure clean, testable code.

## File Upload & Embedding (Web)

- Upload files through `/embed` endpoint
- JSON: Support tickets
- PDF: Troubleshooting guides
- CSV: RCA logs

Files are embedded and stored with associated schema/team context.

## Slack Commands

| Command | Description |
|---------|-------------|
| `@oraclebot` | Ask a support question |
| `/oracleembed table_name=name` | Embed uploaded file into specified table |
| `/resetcontext` | Clear chat memory |
| `/setteam team_name` | Set your Slack user to a team/schema |
| `/unsetteam` | Remove team assignment |


