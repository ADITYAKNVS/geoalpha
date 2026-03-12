import os
from fyers_apiv3 import fyersModel

# -------------------------------------------------------------------
# Fyers Token Generation Script
# -------------------------------------------------------------------
# Fyers requires you to generate a daily Access Token using your 
# App ID and Secret Key. Run this script once every morning to 
# generate the token and automatically save it to your .env file!
# -------------------------------------------------------------------

def get_env_var(filepath, key):
    try:
        with open(filepath, "r") as f:
            for line in f:
                if line.startswith(f"{key}="):
                    return line.strip().split("=", 1)[1]
    except FileNotFoundError:
        pass
    return ""

def set_env_var(filepath, key, value):
    lines = []
    found = False
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        pass

    with open(filepath, "w") as f:
        for line in lines:
            if line.startswith(f"{key}="):
                f.write(f"{key}={value}\n")
                found = True
            else:
                f.write(line)
        if not found:
            f.write(f"{key}={value}\n")

# Settings
env_file = ".env"
client_id = get_env_var(env_file, "FYERS_CLIENT_ID")
secret_key = get_env_var(env_file, "FYERS_SECRET_KEY")
redirect_uri = get_env_var(env_file, "FYERS_REDIRECT_URI")

print("=====================================================")
print("          Fyers Daily Token Generator")
print("=====================================================")

# Step 1: Prompt for keys if not in .env or if user wants to reset
if client_id:
    print(f"Found existing App ID: {client_id}")
    reset = input("Do you want to enter new credentials? (y/N): ").strip().lower()
    if reset == 'y':
        client_id = ""
        secret_key = ""
        redirect_uri = ""

if not client_id:
    print("\n[NOTE] Make sure your App ID usually ends in '-100' for Fyers API v3")
    client_id = input("Enter your Fyers App ID (Client ID): ").strip()
    set_env_var(env_file, "FYERS_CLIENT_ID", client_id)

if not secret_key:
    secret_key = input("Enter your Fyers Secret Key: ").strip()
    set_env_var(env_file, "FYERS_SECRET_KEY", secret_key)
    
if not redirect_uri:
    print("\n[NOTE] The Redirect URI MUST EXACTLY MATCH what you put in the Fyers API Dashboard!")
    print("If you are unsure, go to https://myapi.fyers.in/dashboard and check your App settings.")
    print("Common default: https://trade.fyers.in/api-login/redirect-uri/index.html")
    uri_input = input("Enter your Redirect URI (press enter for default): ").strip()
    redirect_uri = uri_input if uri_input else "https://trade.fyers.in/api-login/redirect-uri/index.html"
    set_env_var(env_file, "FYERS_REDIRECT_URI", redirect_uri)

# Step 2: Generate Auth Code URL
response_type = "code" 
grant_type = "authorization_code"

try:
    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type=response_type,
        grant_type=grant_type
    )

    auth_link = session.generate_authcode()

    print("\n--- STEP 1 ---")
    print("Open the following link in your browser and log in to Fyers:")
    print(f"\n{auth_link}\n")

    print("--- STEP 2 ---")
    print("After logging in, you will be redirected.")
    print("Look at the URL in your browser's address bar. It will look something like:")
    print(f"{redirect_uri}?auth_code=xxxxxxxxxxxxxxxxxxxxx&state=None")

    print("\nCopy ONLY the actual auth_code parameter from the URL (the part after auth_code= and before &state=)")
    auth_code = input("\nPaste the auth_code here: ").strip()

    # Step 3: Set auth code and generate access token
    session.set_token(auth_code)
    response = session.generate_token()
    
    if "access_token" in response:
        access_token = response["access_token"]
        set_env_var(env_file, "FYERS_ACCESS_TOKEN", access_token)
        print("\n✅ Success! The daily FYERS_ACCESS_TOKEN has been generated and saved to your .env file.")
        print("You can now run your main application using `streamlit run app.py`")
    else:
        print(f"\n❌ Error generating token. Fyers response: {response}")
        print("This usually means the auth_code was invalid, expired, or pasted incorrectly.")
except Exception as e:
    print(f"\n❌ Error during token generation: {e}")
    print("Check if your client_id and redirect_uri are exactly matching your dashboard.")
