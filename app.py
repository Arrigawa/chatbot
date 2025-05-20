from flask import Flask, request, jsonify
import requests
from serpapi import GoogleSearch
import os
from dotenv import load_dotenv # Import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure CORS to allow requests from your frontend
# In a production environment, configure this more securely
from flask_cors import CORS
CORS(app)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"  # pull ollama model 

# Add debug logging
def log_error(message, error=None):
    print(f"ERROR: {message}")
    if error:
        print(f"Details: {str(error)}")

def retry_request(func, retries=3, delay=1):
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            if i == retries - 1:
                raise e
            print(f"Attempt {i+1} failed, retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2

def check_ollama_model():
    try:
        response = requests.get(f"http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            print(f"Available models: {models}")
            return True
        return False
    except Exception as e:
        print(f"Error checking Ollama: {e}")
        return False

# --- SerpApi Configuration ---
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
if not SERPAPI_KEY:
    print("Error: SERPAPI_KEY not found in environment variables or .env file.")
    
# ---------------------------

@app.route('/chat', methods=['POST'])
def chat():
    # Check if Ollama is running and model is available
    if not check_ollama_model():
        return jsonify({"error": "Ollama tidak aktif atau model tidak tersedia. Pastikan Ollama berjalan dengan perintah 'ollama run llama2'"}), 503

    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "Pesan tidak diberikan"}), 400

    # --- Perform Searches ---
    search_results_text = ""
    video_data = []
    
    if not SERPAPI_KEY:
        search_results_text = "\n\nFungsi pencarian tidak dikonfigurasi (SERPAPI_KEY tidak ditemukan)."
    else:
        try:
            # Regular web search
            search = GoogleSearch({"q": user_message, "api_key": SERPAPI_KEY})
            results = search.get_dict()
            
            # Enhanced YouTube search with better parameters
            if any(word in user_message.lower() for word in ["cara", "belajar", "jelaskan", "tutorial", "panduan", "kursus", "mengajar"]):
                youtube_search = GoogleSearch({
                    "engine": "youtube",
                    "q": f"tutorial terbaik {user_message}",
                    "api_key": SERPAPI_KEY,
                    "num": 3  # Get top 3 results
                })
                youtube_results = youtube_search.get_dict()
                
                if "video_results" in youtube_results and youtube_results["video_results"]:
                    # Get multiple videos and sort by views/rating
                    for video in youtube_results["video_results"][:3]:  # Take top 3 videos
                        video_info = {
                            "title": video.get("title", ""),
                            "link": video.get("link", ""),
                            "thumbnail": video.get("thumbnail", {}).get("static", ""),
                            "duration": video.get("duration", ""),
                            "views": video.get("views", ""),
                            "description": video.get("description", "")
                        }
                        video_data.append(video_info)

            # Extract relevant snippets from web search results
            snippets = []
            if "organic_results" in results:
                for result in results["organic_results"][:3]:
                    if "snippet" in result:
                        snippets.append(result["snippet"])
                    elif "snippet_highlighted_words" in result:
                        snippets.append(" ".join(result["snippet_highlighted_words"]))

            if snippets:
                search_results_text = "\n\nHasil Pencarian:\n" + "\n---\n".join(snippets)
            else:
                search_results_text = "\n\nTidak ditemukan hasil pencarian yang relevan."

        except Exception as e:
            print(f"Error during search: {e}")
            search_results_text = "\n\nTidak dapat melakukan pencarian."

    # --- Augment Prompt with Search Results ---
    payload = {
        "model": MODEL_NAME,
        "prompt": user_message,
        "system": "Anda adalah asisten AI berbahasa Indonesia. Berikan jawaban yang jelas dan informatif.",
        "stream": False
    }

    try:
        def make_request():
            return requests.post(OLLAMA_API_URL, json=payload, timeout=60)  # Increased timeout to 60 seconds
        
        response = retry_request(make_request)
        
        if response.status_code != 200:
            log_error(f"Ollama error response: {response.text}")
            return jsonify({"error": "Model AI tidak dapat memproses permintaan"}), 500

        try:
            ollama_response = response.json()
            assistant_message = ollama_response.get('response', '')
            
            if not assistant_message:
                return jsonify({"error": "Tidak ada jawaban dari AI"}), 500

            return jsonify({
                "reply": assistant_message,
                "videos": video_data
            })

        except ValueError as e:
            log_error("JSON parsing error", e)
            return jsonify({"error": "Format respons tidak valid"}), 500

    except requests.exceptions.Timeout:
        return jsonify({"error": "Waktu habis, silakan coba lagi"}), 504
    except Exception as e:
        log_error("Unexpected error", e)
        return jsonify({"error": "Terjadi kesalahan sistem"}), 500

if __name__ == '__main__':
    if not check_ollama_model():
        print("WARNING: Ollama tidak aktif atau model tidak tersedia")
    app.run(debug=True, port=5000)
