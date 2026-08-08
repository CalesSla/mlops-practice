import requests

r = requests.get("https://example.com")
reveal_type(r)
print(r.status_code)
