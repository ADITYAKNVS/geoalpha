import os
from dotenv import load_dotenv
load_dotenv()
from fyers_apiv3 import fyersModel

FYERS_CLIENT_ID = os.environ.get("FYERS_CLIENT_ID", "")
FYERS_ACCESS_TOKEN = os.environ.get("FYERS_ACCESS_TOKEN", "")

fyers = fyersModel.FyersModel(client_id=FYERS_CLIENT_ID, is_async=False, token=FYERS_ACCESS_TOKEN, log_path="")

data = {
    "symbols": "NSE:NIFTY50-INDEX,MCX:CRUDEOIL26MARFUT"
}
response = fyers.quotes(data=data)
print(response)
