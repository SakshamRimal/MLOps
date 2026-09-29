import json 
import math 
import httpx

# tool implementation (the actual python code for the tool) goes here

def calculator(expression: str) -> dict:
    """
    Safely evaluate a basic arithmetic expression 
    """
    allowed_name = {"sqrt":math.sqrt, "pow":math.pow, "abs":abs, "round":round}
    try:
        # Evaluate the expression using eval in a restricted environment
        result = eval(expression, {"__builtins__": None}, allowed_name)
        return {"result": result}
    
    except Exception as e:
        return {"error": f"Error evaluating expression: {str(e)}"}
    
def web_search(query: str) -> dict:
    """
    Placeholder web search tool.Replace with a real API (eg. SerpAPI, Bing Search API) for production use.
    Kept as a stub so the tool calling loop works wihtuout extra signups 
    """
    return {
        "results":[
            {"title":f"Stub result for '{query}'", "url":"https://example.com", "snippet":"This is a stub search result. Replace with a real search API."}
        ]
    }
    
def query_knowledge_base(query: str) -> dict:
    """Retrieves relevant chunks from the vector store for the given query."""
    from app.rag.retriever import retrieve

    chunks = retrieve(query, top_k=4)
    if not chunks:
        return {"chunks": [], "note": "No relevant information found in knowledge base."}

    return {
        "chunks": [
            {"document": c["metadata"]["source"], "chunk_id": c["chunk_id"], "text": c["text"]}
            for c in chunks
        ]
    }

from datetime import datetime, timezone


def get_current_time() -> dict:
    """Returns the current date and time in UTC."""
    now = datetime.now(timezone.utc)
    return {
        "iso": now.isoformat(),
        "utc_time": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


_WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def weather(city: str) -> dict:
    """
    Current weather for a city via the free Open-Meteo API (no API key required).
    Uses Open-Meteo geocoding to resolve the city name to coordinates first.
    """
    try:
        geo_resp = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=8.0,
        )
        geo_resp.raise_for_status()
        results = geo_resp.json().get("results") or []
        if not results:
            return {"error": f"No location found for '{city}'."}

        place = results[0]
        lat, lon = place["latitude"], place["longitude"]

        forecast_resp = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": (
                    "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "precipitation,weather_code,wind_speed_10m"
                ),
                "timezone": "auto",
            },
            timeout=8.0,
        )
        forecast_resp.raise_for_status()
        current = forecast_resp.json().get("current", {})
        code = current.get("weather_code")

        label = ", ".join(p for p in (place.get("name"), place.get("country")) if p)
        return {
            "location": label,
            "local_time": current.get("time"),
            "condition": _WMO_CODES.get(code, f"Unknown (code {code})"),
            "temperature_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "units": {"temperature": "celsius", "wind_speed": "km/h"},
        }
    except httpx.HTTPError as e:
        return {"error": f"Weather request failed: {e}"}


# tool schemas (what we tell the model is available, and how to call it)

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression. Allowed functions: sqrt, pow, abs, round. Example: 'sqrt(16) + pow(2,3)'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression to evaluate.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_base",
            "description": "Search the internal document knowledge base for information relevant to the user's query. Returns a list of document chunks that may contain the answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's query to search for in the knowledge base.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Placeholder web search. Returns dummy results only - do NOT use it for factual answers; prefer the weather tool for weather and query_knowledge_base for documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get real-time current weather conditions for a city or location (temperature, feels-like, humidity, precipitation, wind, condition).",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City or location name, e.g. 'Paris' or 'Kathmandu, Nepal'.",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current real-world UTC timestamp and date.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "calculator": calculator,
    "query_knowledge_base": query_knowledge_base,
    "web_search": web_search,
    "weather": weather,
    "get_current_time": get_current_time,
}

def execute_tool(name: str, arguments_json: str) -> str:
    """Execute a tool by name and return a JSON string result."""
    if name not in TOOL_FUNCTIONS:
        return json.dumps({"error": f"Tool '{name}' not found"})

    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON arguments: {str(e)}"})

    try:
        result = TOOL_FUNCTIONS[name](**args)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {str(e)}"})
