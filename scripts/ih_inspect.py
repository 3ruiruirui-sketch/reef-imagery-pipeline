import requests
import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
url = "https://www.hidrografico.pt/previsao-mares.php"

def main():
    print(f"Fetching {url}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print(f"Content-Type: {response.headers.get('Content-Type')}")
    
    html = response.text
    output_path = _PROJECT_ROOT / "ih_response.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved raw HTML to {output_path}")
    
    print("\n--- Lines containing 'Faro' ---")
    lines = html.splitlines()
    for line in lines:
        if 'Faro' in line or 'faro' in line.lower():
            print(line.strip())
            
    print("\n--- HTML Tags Analysis ---")
    select_count = len(re.findall(r'<select[^>]*>', html, re.IGNORECASE))
    option_count = len(re.findall(r'<option[^>]*>', html, re.IGNORECASE))
    table_count = len(re.findall(r'<table[^>]*>', html, re.IGNORECASE))
    
    print(f"<select> tags count: {select_count}")
    if select_count > 0:
        first_select = re.search(r'<select[^>]*>', html, re.IGNORECASE)
        print(f"First <select>: {first_select.group(0)}")
        
    print(f"<option> tags count: {option_count}")
    if option_count > 0:
        first_option = re.search(r'<option[^>]*>.*?</option>', html, re.IGNORECASE)
        if first_option:
            print(f"First <option>: {first_option.group(0)}")
        
    print(f"<table> tags count: {table_count}")
    if table_count > 0:
        first_table = re.search(r'<table[^>]*>', html, re.IGNORECASE)
        print(f"First <table>: {first_table.group(0)}")

if __name__ == "__main__":
    main()
