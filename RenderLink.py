import requests

RENDER_BASE = "https://audioapi-g2ru.onrender.com"
AUTH_TOKEN = "Potato"

def SetGradioURL(url: str) -> str:
    if not url.startswith("http"):
        print("Invalid URL. Must start with http.")
        return "error:invalid-url"
    if not AUTH_TOKEN:
        print("Missing AUTH_TOKEN.")
        return "error:no-token"

    try:
        print(f"\n→ Sending POST /set to {RENDER_BASE} ...")
        r = requests.post(
            f"{RENDER_BASE}/set",
            json={"url": url},
            headers={"X-Auth-Token": AUTH_TOKEN},
            timeout=10
        )
        print(f"Status: {r.status_code}")
        print(f"Response: {r.text}")

        if r.status_code == 200:
            print(f"Successfully set URL to: {url}")
            return "success"
        else:
            print("Error:", r.text)
            return f"error:{r.status_code}"

    except Exception as e:
        print("Exception while sending:", e)
        return "error:exception"

def GetGradioURL():
    try:
        print(f"\n→ Sending GET /current to {RENDER_BASE} ...")
        r = requests.get(f"{RENDER_BASE}/current", timeout=10)
        print(f"Status: {r.status_code}")

        if r.status_code == 200:
            data = r.json()
            url = data.get("url", "")
            if url:
                print(f"Current stored URL: {url}")
                return ["success", url]
            else:
                print("No URL currently stored.")
                return ["error:empty", None]
        else:
            print("Error:", r.text)
            return [f"error:{r.status_code}", None]

    except Exception as e:
        print("Exception while fetching:", e)
        return ["error:exception", None]
