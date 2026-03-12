import re

with open("app.py", "r") as f:
    text = f.read()

# Add import
if 'fetch_historical_data' not in text:
    text = text.replace('STOCK_PICK_REASONS,\n)', 'STOCK_PICK_REASONS,\n    fetch_historical_data,\n)')

# Replace yf.Ticker(X).history(period=Y) with fetch_historical_data(X, period=Y)
text = re.sub(r'yf\.Ticker\(([^)]+)\)\.history\(period=([^)]+)\)', r'fetch_historical_data(\1, period=\2)', text)

# Add float() explicitly around iloc inside rounding logic to fix pyre errors too in get_delta etc if possible.
# Wait, let's just replace the yf.Ticker first because the Pyre errors are mostly on list indexing or yf usages.
# Actually, the python script will replace ALL occurrences of yf.Ticker(...).history(...) in app.py

with open("app.py", "w") as f:
    f.write(text)
print("Replaced successfully.")
