import requests
from requests.auth import HTTPBasicAuth
import json
import os

# Config 
JIRA_DOMAIN = "https://jira-sd.mc1.oracleiaas.com/"
API_ENDPOINT = f"{JIRA_DOMAIN}/rest/api/3/search"
EMAIL = "misha.faruki@oracle.com"
API_TOKEN = os.getenv("JIRA_API_TOKEN")

# Define JQL to get tickets from the support project
jql_query = 'project = SUPPORT AND status != Done ORDER BY created DESC'

# Parameters for the API request
params = {
    "jql": jql_query,
    "maxResults": 50,
    "fields": "key,summary,status,assignee,created"
}

# Make request
response = requests.get(
    API_ENDPOINT,
    headers={"Accept": "application/json"},
    params=params,
    auth=HTTPBasicAuth(EMAIL, API_TOKEN)
)


# Check and parse JSON
if response.status_code == 200:
    data = response.json()
    print(json.dumps(data, indent=2)) # pretty-print

else:
    print(f"Error: {response.status_code}")
    print(response.text)