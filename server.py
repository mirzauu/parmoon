# ─────────────────────────────────────────────────────────────
#  parmanoo.com  |  Traffic Incident Feed Server
#  Serves: /feed/mumbai  /feed/delhi  /feed/lucknow
#  Stack:  Python 3.10+  |  Flask  |  Anthropic API
# ─────────────────────────────────────────────────────────────

import os
import json
import time
import re
import anthropic
import requests
from datetime import datetime, timezone
from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS

# Load .env file if it exists
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key.strip()] = value.strip()

app = Flask(__name__)
CORS(app)  # Allows any screen/browser to call this feed


# ─── CONFIGURATION ───────────────────────────────────────────
# Keys are loaded from environment variables.
# For local development, they are set in .env file.

print(f"[INIT] RAPIDAPI_KEY ends with: ...{RAPIDAPI_KEY[-4:]}", flush=True)
print(f"[INIT] ANTHROPIC_KEY ends with: ...{ANTHROPIC_KEY[-4:]}", flush=True)

POLICE_ACCOUNTS = {
    "mumbai":  "MTPHereToHelp",   # @MTPHereToHelp  — Mumbai Traffic Police
    "delhi":   "dtptraffic",      # @dtptraffic     — Delhi Traffic Police
    "lucknow": "lkopolice",       # @lkopolice      — Lucknow Police (traffic updates)
}

TWEETS_TO_FETCH   = 20   # Fetch last 20 tweets per account, keep top 3 incidents
CACHE_TTL_SECONDS = 900  # Cache results for 15 minutes (reduces API calls)


# ─── IN-MEMORY CACHE ─────────────────────────────────────────
# Stores the last fetched result per city so we don't
# hit the Twitter API on every screen refresh.

_cache = {}   # Structure: { "mumbai": { "data": [...], "fetched_at": timestamp } }


# ─── STEP 1: FETCH TWEETS FROM X/TWITTER ─────────────────────

def fetch_tweets(username: str) -> list[str]:
    """
    Calls the RapidAPI 'twitter-v24' endpoint to get recent
    tweets from a public account.
    """
    url = "https://twitter-v24.p.rapidapi.com/user/tweets"
    headers = {
        "x-rapidapi-key":  RAPIDAPI_KEY,
        "x-rapidapi-host": "twitter-v24.p.rapidapi.com",
    }
    print(f"\n[TWITTER API] Fetching tweets for user: @{username}", flush=True)
    print(f"[TWITTER API] URL: {url}", flush=True)
    
    params = {"username": username, "limit": TWEETS_TO_FETCH}
    response = requests.get(url, headers=headers, params=params, timeout=12)
    print(f"[TWITTER API] Status Code: {response.status_code}", flush=True)
    response.raise_for_status()
    raw = response.json()
    print(f"[TWITTER API] Raw response received. Parsing text...", flush=True)

    # Reusable recursive search for 'full_text' or 'text' in the complex response
    texts = []
    def extract_text(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ["full_text", "text"] and isinstance(v, str):
                    texts.append(v)
                else:
                    extract_text(v)
        elif isinstance(obj, list):
            for item in obj:
                extract_text(item)

    extract_text(raw)
    
    # Return unique texts to avoid duplicates from the 'result' vs 'legacy' structures
    seen = set()
    cleaned = []
    for t in texts:
        if t not in seen:
            cleaned.append(t)
            seen.add(t)
            
    print(f"[TWITTER API] Found {len(cleaned)} unique tweets.", flush=True)
    return cleaned[:TWEETS_TO_FETCH]


# ─── STEP 2: EXTRACT INCIDENTS USING CLAUDE AI ───────────────

def extract_incidents(tweets: list[str], city: str) -> list[dict]:
    """
    Sends the raw tweets to Claude.
    Claude reads them and returns only genuine traffic incidents,
    each cleaned into a short structured format.
    Returns a list of up to 3 incident dicts.
    """
    if not tweets:
        return []

    tweet_block = "\n\n".join(
        f"Tweet {i+1}: {text}" for i, text in enumerate(tweets)
    )

    prompt = f"""
You are a traffic data processor for {city}, India.

Below are recent tweets from the {city} traffic police account. {city.upper()} IS IN INDIA.
Your job: extract only genuine traffic incidents (accidents, breakdowns, road blocks, waterlogging, diversions).

IGORE: awareness campaigns, recruitment, general advisories, congratulations, and non-traffic content.

For each real incident, return a JSON object with exactly these fields:
- "road": the road or area name (short, e.g. "Western Express Hwy")
- "issue": what the problem is (e.g. "Vehicle breakdown", "Minor accident", "Waterlogging")
- "detail": one clean sentence describing the situation (max 12 words)
- "severity": one of "high", "medium", or "low"

Return a JSON array of up to 3 incidents, ordered by severity (high first).
If there are no real incidents, return an empty array [].
Return ONLY valid JSON. No explanation, no markdown.

TWEETS:
{tweet_block}
"""
    print(f"\n[CLAUDE AI] Sending {len(tweets)} tweets to Claude for extraction...", flush=True)
    print(f"[CLAUDE AI] Prompting model: claude-3-5-sonnet-20241022", flush=True)
    
    try:
        # Move client init inside try to catch dependency/initialization errors
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        
        # Log a snippet of the prompt for debugging
        print(f"[CLAUDE AI] First 100 chars of instruction: {prompt[:100]}...", flush=True)
        
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        raw_text = message.content[0].text.strip()
        print(f"[CLAUDE AI] Raw Response from Claude:\n{raw_text}", flush=True)
        
        # Strip markdown code fences if Claude adds them
        raw_text = re.sub(r"^```(?:json)?|```$", "", raw_text, flags=re.MULTILINE).strip()
        incidents = json.loads(raw_text)
        print(f"[CLAUDE AI] Successfully extracted {len(incidents)} incidents.", flush=True)
    except Exception as e:
        print(f"AI Extraction Error: {e}", flush=True)
        # Fallback: very simple extraction for just the first incident if AI fails
        if tweets:
            first = tweets[0][:120] # Take a bit more for detail
            incidents = [{
                "road": "Current Traffic Update",
                "issue": "Note: AI Extraction Unavailable",
                "detail": first,
                "severity": "medium"
            }]
        else:
            return []

    # Add a timestamp to each incident so the screen can show "X min ago"
    now_iso = datetime.now(timezone.utc).isoformat()
    for inc in incidents:
        inc["reported_at"] = now_iso

    return incidents[:3]


# ─── STEP 3: CACHE WRAPPER ───────────────────────────────────

def get_incidents_cached(city: str, force: bool = False) -> list[dict]:
    """
    Returns cached incidents if fresh (< 5 min old).
    Otherwise fetches new data from Twitter + Claude.
    This prevents hammering the APIs on every screen request.
    """
    now = time.time()
    cached = _cache.get(city)

    if not force and cached and (now - cached["fetched_at"]) < CACHE_TTL_SECONDS:
        print(f"[CACHE] Serving {city} from memory. TTL: {int(CACHE_TTL_SECONDS - (now - cached['fetched_at']))}s remaining.", flush=True)
        return cached["data"]

    # Cache is stale or empty — fetch fresh data
    print(f"[CACHE] Cache {'expired' if cached else 'empty'} or refresh forced for {city}. Fetching fresh data...", flush=True)
    username = POLICE_ACCOUNTS[city]
    tweets   = fetch_tweets(username)
    incidents = extract_incidents(tweets, city)

    _cache[city] = {
        "data":       incidents,
        "fetched_at": now,
    }

    return incidents


# ─── STEP 4: API ENDPOINTS ───────────────────────────────────

@app.route("/feed/<city>", methods=["GET"])
def feed(city: str):
    """
    Detects who is calling:
    - A browser (human)   → returns a clean readable HTML page
    - A screen / machine  → returns JSON as before
    Both use the same cached data underneath.
    """
    if city not in POLICE_ACCOUNTS:
        return jsonify({"error": f"City '{city}' not supported. Use: mumbai, delhi, lucknow"}), 404

    try:
        force = request.args.get("refresh") == "1"
        incidents = get_incidents_cached(city, force=force)
    except requests.RequestException as e:
        # Fallback to stale cache if network error occurs
        cached = _cache.get(city)
        if cached:
            incidents = cached["data"]
            updated = datetime.fromtimestamp(cached["fetched_at"], tz=timezone.utc).strftime("%-I:%M %p UTC")
            is_stale = True
        else:
            if e.response is not None and e.response.status_code == 403:
                return jsonify({
                    "error": "Twitter API Subscription Required",
                    "detail": "Please check your subscription to 'Twitter V24' on RapidAPI.",
                    "url": "https://rapidapi.com/alexanderstapi-alexanderstapi-default/api/twitter-v24"
                }), 403
            return jsonify({"error": "Could not reach Twitter API", "detail": str(e)}), 503
    except Exception as e:
        return jsonify({"error": "Unexpected server error", "detail": str(e)}), 500

    if 'updated' not in locals():
        updated = datetime.now(timezone.utc).strftime("%-I:%M %p UTC")
    
    is_stale = locals().get('is_stale', False)

    # ── If a browser is asking, return a human-readable HTML page ──
    accepts = request.headers.get("Accept", "")
    if "text/html" in accepts:
        return render_template_string(HTML_TEMPLATE,
            city=city.title(),
            city_slug=city,
            incidents=incidents,
            updated=updated,
            is_stale=is_stale,
            cities=list(POLICE_ACCOUNTS.keys())
        )

    # ── Otherwise return JSON (for screens and machines) ──
    print(f"[ROUTE] Returning JSON response for {city}.", flush=True)
    return jsonify({
        "city":       city,
        "source":     "parmanoo.com",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "incidents":  incidents,
    })


# ─── HTML TEMPLATE ────────────────────────────────────────────
# Shown when a human opens the feed URL in a browser.
# Clean, minimal, readable on any device.

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{{ city }} Traffic · parmanoo.com</title>
  <link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700&family=DM+Sans:wght@400;500&family=DM+Mono:wght@400&display=swap" rel="stylesheet">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'DM Sans', sans-serif;
      background: #faf8f4;
      color: #18160f;
      min-height: 100vh;
      padding: 32px 20px;
    }
    .container { max-width: 640px; margin: 0 auto; }

    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 28px;
      padding-bottom: 20px;
      border-bottom: 1px solid #e4ddd2;
    }
    .brand {
      font-family: 'DM Mono', monospace;
      font-size: 11px;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: #d4500a;
      margin-bottom: 5px;
    }
    .updated {
      font-size: 11px;
      color: #9a948a;
      display: flex;
      align-items: center;
      gap: 5px;
    }
    .live-dot {
      width: 6px; height: 6px;
      border-radius: 50%;
      background: #2d7a4f;
      animation: blink 2s ease-in-out infinite;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

    .city-select {
      appearance: none;
      background: #fff;
      border: 1px solid #e4ddd2;
      border-radius: 8px;
      padding: 9px 32px 9px 14px;
      font-family: 'Syne', sans-serif;
      font-size: 13px;
      font-weight: 700;
      color: #18160f;
      cursor: pointer;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%239a948a' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 10px center;
      min-width: 120px;
    }
    .city-select:focus { outline: none; border-color: #d4500a; }

    .refresh-btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: #18160f;
      color: #fff;
      border: none;
      border-radius: 8px;
      padding: 9px 16px;
      font-family: 'Syne', sans-serif;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s ease;
      text-decoration: none;
    }
    .refresh-btn:hover { background: #d4500a; transform: translateY(-1px); }
    .refresh-btn:active { transform: translateY(0); }
    .refresh-btn svg { width: 14px; height: 14px; }
    
    .loading-overlay {
        display: none;
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(250, 248, 244, 0.8);
        backdrop-filter: blur(4px);
        z-index: 100;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        gap: 16px;
    }
    .spinner {
        width: 32px; height: 32px;
        border: 3px solid #e4ddd2;
        border-top-color: #d4500a;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    .section-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      margin-bottom: 8px;
    }
    .section-label {
      font-family: 'DM Mono', monospace;
      font-size: 9px;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: #b0a898;
      margin-bottom: 4px;
    }

    .incident {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 4px 24px;
      align-items: start;
      padding: 16px 0;
      border-bottom: 1px solid #e4ddd2;
    }
    .incident:last-child { border-bottom: none; }

    .incident-road {
      font-family: 'Syne', sans-serif;
      font-size: 13px;
      font-weight: 700;
      color: #18160f;
      margin-bottom: 4px;
    }
    .incident-detail {
      font-size: 13px;
      color: #5c5850;
      line-height: 1.5;
    }
    .incident-time {
      font-family: 'DM Mono', monospace;
      font-size: 11px;
      color: #b0a898;
      white-space: nowrap;
      padding-top: 2px;
    }

    .empty {
      padding: 48px 0;
      color: #9a948a;
      font-size: 13px;
    }

    .footer {
      margin-top: 24px;
      padding-top: 16px;
      border-top: 1px solid #e4ddd2;
      display: flex;
      justify-content: space-between;
      font-family: 'DM Mono', monospace;
      font-size: 10px;
      color: #c0b8b0;
    }
    .footer strong { color: #d4500a; }
  </style>
</head>
<body>
<div class="container">

  <div class="loading-overlay" id="loading">
    <div class="spinner"></div>
    <div class="section-label">Fetching Live Updates...</div>
  </div>

  <div class="header">
    <div>
      <div class="brand">parmanoo.com</div>
      <div class="updated">
        <div class="live-dot" {% if is_stale %}style="background: #d4a017; animation: none;"{% endif %}></div>
        Last update: {{ updated }} {% if is_stale %}(Offline Mode){% endif %}
      </div>
    </div>
    <select class="city-select" onchange="window.location='/feed/'+this.value">
      {% for c in cities %}
      <option value="{{ c }}" {% if c == city_slug %}selected{% endif %}>{{ c.title() }}</option>
      {% endfor %}
    </select>
  </div>

  <div class="section-header">
    <div class="section-label">
        {% if incidents %}
            {{ incidents|length }} incident{% if incidents|length != 1 %}s{% endif %} reported
        {% else %}
            Status: Clear
        {% endif %}
    </div>
    <a href="?refresh=1" class="refresh-btn" onclick="document.getElementById('loading').style.display='flex'">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/></svg>
      Refresh
    </a>
  </div>

  {% if incidents %}
    {% for inc in incidents %}
    <div class="incident">
      <div>
        <div class="incident-road">{{ inc.road }}</div>
        <div class="incident-detail">{{ inc.detail }}</div>
      </div>
      <div class="incident-time">{{ inc.reported_at }}</div>
    </div>
    {% endfor %}
  {% else %}
    <div class="empty">No incidents reported right now. Road conditions appear normal.</div>
  {% endif %}

  <div class="footer">
    <span>Data by <strong>parmanoo.com</strong></span>
    <strong>Data available in JSON</strong>
  </div>

</div>
</body>
</html>"""


@app.route("/health", methods=["GET"])
def health():
    """ Simple health check — confirms the server is running. """
    return jsonify({"status": "ok", "service": "parmanoo.com traffic feed"})


# ─── START SERVER ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
