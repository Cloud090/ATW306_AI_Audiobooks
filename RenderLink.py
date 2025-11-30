# This module provides functions for communicating with the Render “static endpoint” used to store and retrieve the current Gradio API URL.

import requests

# Render URL address, and authentication token
RENDER_BASE = "https://audioapi-g2ru.onrender.com"
AUTH_TOKEN = "Potato"

# Function to set the Gradio URL in the static endpoint, once the Gradio API has been initialised.
def SetGradioURL(url: str) -> str:
    # Checks that the URL starts with "http", and that an authentication token exists.
    if not url.startswith("http"):
        print("Invalid URL. Must start with http.")
        return "error:invalid-url"
    if not AUTH_TOKEN:
        print("Missing AUTH_TOKEN.")
        return "error:no-token"

    try:
        print(f"\n→ Sending POST /set to {RENDER_BASE} ...")

        # Sends a POST request to the "/set" option in Render, with the new URL and the authentication token
        r = requests.post(
            f"{RENDER_BASE}/set",
            json={"url": url},
            headers={"X-Auth-Token": AUTH_TOKEN},
            timeout=10
        )
        print(f"Status: {r.status_code}")
        print(f"Response: {r.text}")

        # Returns "success" if a response is received, or the error code if it fails
        if r.status_code == 200:
            print(f"Successfully set URL to: {url}")
            return "success"
        else:
            print("Error:", r.text)
            return f"error:{r.status_code}"

    # Returns "error:exception" if the POST request causes an error (e.g. timeout, network problems)
    except Exception as e:
        print("Exception while sending:", e)
        return "error:exception"

# Function to get the Gradio URL from the static endpoint,
def GetGradioURL():
    try:
        print(f"\n→ Sending GET /current to {RENDER_BASE} ...")

        # Sends a GET request to the "/current" option in Render
        r = requests.get(f"{RENDER_BASE}/current", timeout=10)
        print(f"Status: {r.status_code}")

        # Returns "success" and the current URL if a URL is received, or "error:empty" if no URL is set.
        if r.status_code == 200:
            data = r.json()
            url = data.get("url", "")
            if url:
                print(f"Current stored URL: {url}")
                return ["success", url]
            else:
                print("No URL currently stored.")
                return ["error:empty", None]

        # Returns error code if the request fails
        else:
            print("Error:", r.text)
            return [f"error:{r.status_code}", None]

    # Returns "error:exception" if the GET request causes an error (e.g. timeout, network problems)
    except Exception as e:
        print("Exception while fetching:", e)
        return ["error:exception", None]
