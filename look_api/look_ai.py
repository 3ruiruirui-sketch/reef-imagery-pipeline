import requests

def check_headers(url, headers=None):
    """
    Sends a GET request to the specified URL and prints the response headers.
    
    Args:
        url (str): The target URL.
        headers (dict, optional): Custom headers to include in the request. Defaults to None.
        
    Returns:
        requests.Response: The response object if successful; None otherwise.
    """
    try:
        response = requests.get(url, headers=headers)
        print(f"\nStatus Code: {response.status_code}")
        print("Response Headers:")
        for header, value in response.headers.items():
            print(f"{header}: {value}")
        return response
    except Exception as e:
        print(f"Request failed: {e}\n")
        return None

# Replace 'your@academic.institution.pt' with your actual academic email
headers = {"X-User-Email": "your@academic.institution.pt"}
url = "https://dgt-be.a.incd.pt:8081/collections/MDT-50cm/items"

# Run the function and capture the response
response = check_headers(url, headers)

