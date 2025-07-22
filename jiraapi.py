import os
import requests
from requests.auth import HTTPBasicAuth
import logging
from jira import JIRA, exceptions

# Enable debug logging for troubleshooting
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration (use environment variables for security)
JIRA_DOMAIN = "https://jira-sd.mc1.oracleiaas.com/"
EMAIL = "misha.faruki@oracle.com"

API_TOKEN = os.getenv("JIRA_API_TOKEN")

def test_basic_auth_connection():
    """Test connection using raw requests library"""
    try:
        response = requests.get(
            f"{JIRA_DOMAIN}/rest/api/3/serverInfo",
            auth=HTTPBasicAuth(EMAIL, API_TOKEN),
            timeout=10
        )
        
        if response.status_code == 200:
            print("Basic Auth Success! Server version:", response.json().get('version'))
            return True
        else:
            print(f"Auth Failed (HTTP {response.status_code}): {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Network error: {str(e)}")
        return False

def test_jira_library_connection():
    """Test connection using official Jira library"""
    try:
        jira = JIRA(
            server=JIRA_DOMAIN,
            basic_auth=(EMAIL, "hello!"),
            timeout=10
        )
        print("Jira Library Success! User:", jira.myself()['displayName'])
        return True
        
    except exceptions.JIRAError as e:
        print(f"Jira Library Error ({e.status_code}): {e.text}")
        return False
    except Exception as e:
        print(f"General error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Basic Auth connection...")
    basic_result = test_basic_auth_connection()
    
    print("\nTesting Jira Library connection...")
    jira_lib_result = test_jira_library_connection()
    
    if not (basic_result or jira_lib_result):
        print("\nTroubleshooting Tips:")
        print("- Verify VPN connection is active")
        print("- Check API token validity at https://id.atlassian.com/manage-profile/security/api-tokens")
        print("- Ensure DNS resolves 'jira-sd.mc1.oracleiaas.com' (try: nslookup jira-sd.mc1.oracleiaas.com)")
        print("- If behind proxy, configure HTTP_PROXY/HTTPS_PROXY environment variables")
