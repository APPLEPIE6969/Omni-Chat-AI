import os
import requests
import json
import base64
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- CONFIGURATION ---
GEMINI_KEY = os.environ.get("GEMINI")

# --- MODEL ROSTER ---
MODELS = {
    "GEMINI": "gemini-3-flash-preview",                   # Standard Brain
    "GEMMA": "gemma-3-27b-it",                    # Creative / Open Model
    "DIRECTOR": "gemini-3-flash-preview",                 # Deep Think Reviewer
    "NATIVE_AUDIO": "gemini-2.5-flash-native-audio-dialog", # Voice Mode
    "NEURAL_TTS": "gemini-2.5-flash-tts"          # Fallback TTS
}

# --- HELPER: TTS ---
def generate_neural_speech(text):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELS['NEURAL_TTS']}:generateContent?key={GEMINI_KEY}"
    payload = { "contents": [{ "parts": [{ "text": text }] }] }
    try:
        r = requests.post(url, json=payload)
        data = r.json()
        if "candidates" in data:
            for part in data["candidates"][0]["content"]["parts"]:
                if "inline_data" in part: return part["inline_data"]["data"]
    except: return None
    return None

# --- DEEP THINK LOGIC ---
def director_review(prompt, initial_response):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELS['DIRECTOR']}:generateContent?key={GEMINI_KEY}"
    
    review_prompt = (
        f"User Prompt: {prompt}\n"
        f"AI Draft Response: {initial_response}\n\n"
        "You are the Director. Review the draft for accuracy, tone, and safety. "
        "If it's good, return it exactly as is. If it needs improvement, rewrite it better."
    )
    
    payload = { "contents": [{ "parts": [{ "text": review_prompt }] }] }
    try:
        r = requests.post(url, json=payload)
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return initial_response # Fallback to original if review fails

# --- MAIN AI CALLER ---
def call_ai(mode, model_id=None, prompt=None, audio_data=None, image_data=None, deep_think=False):
    
    # 1. TEXT/IMAGE MODE (Silent)
    if mode == "text":
        target_model = MODELS.get(model_id, MODELS["GEMINI"])
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={GEMINI_KEY}"
        
        parts = [{ "text": f"User: {prompt}" }]
        
        if image_data:
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg", 
                    "data": image_data
                }
            })

        payload = { "contents": [{ "parts": parts }] }
        
        try:
            r = requests.post(url, json=payload)
            response_text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            
            # Apply Deep Think if requested
            if deep_think:
                response_text = director_review(prompt, response_text)
                
            return {"text": response_text, "audio": None}
        except Exception as e: 
            return {"text": f"Error: {str(e)}", "audio": None}

    # 2. VOICE MODE (Audible)
    if mode == "voice":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELS['NATIVE_AUDIO']}:generateContent?key={GEMINI_KEY}"
        payload = {
            "contents": [{
                "parts": [
                    { "text": "Listen and respond naturally." },
                    { "inline_data": { "mime_type": "audio/mp3", "data": audio_data } }
                ]
            }]
        }
        try:
            r = requests.post(url, json=payload)
            data = r.json()
            resp_text = "Audio Message."
            resp_audio = None
            
            if "candidates" in data:
                parts = data["candidates"][0]["content"]["parts"]
                for part in parts:
                    if "text" in part: resp_text = part["text"]
                    if "inline_data" in part: resp_audio = part["inline_data"]["data"]

            if not resp_audio: resp_audio = generate_neural_speech(resp_text)
            return {"text": resp_text, "audio": resp_audio}
        except Exception as e: return {"text": str(e), "audio": None}

# --- WEB SERVER ---

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Omni-Chat Ultimate</title>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
        <meta name="theme-color" content="#050508">
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            :root {
                --bg: #050508;
                --header-bg: rgba(20, 20, 30, 0.95);
                --glass-border: rgba(255, 255, 255, 0.08);
                --primary: #00f2ea;
                --secondary: #7000ff;
                --gemma: #ff0055;
                --director: #ffd700;
                --text: #ffffff;
                --user-bubble: linear-gradient(135deg, var(--primary) 0%, #00a8a2 100%);
                --ai-bubble: rgba(255, 255, 255, 0.05);
            }

            * { box-sizing: border-box; margin: 0; padding: 0; outline: none; -webkit-tap-highlight-color: transparent; }

            body {
                background: var(--bg); color: var(--text); font-family: 'Outfit', sans-serif;
                height: 100dvh; display: flex; flex-direction: column; overflow: hidden; position: relative;
            }

            /* --- ANIMATED BACKGROUND --- */
            .orb { position: absolute; border-radius: 50%; filter: blur(80px); opacity: 0.3; z-index: -1; animation: float 10s infinite alternate; }
            .orb-1 { width: 400px; height: 400px; background: var(--secondary); top: -10%; left: -10%; }
            .orb-2 { width: 300px; height: 300px; background: var(--primary); bottom: -10%; right: -10%; animation-delay: 2s; }
            @keyframes float { 0% { transform: translate(0,0); } 100% { transform: translate(30px, 30px); } }

            /* --- HEADER --- */
            .header {
                padding: 10px 15px; background: var(--header-bg); backdrop-filter: blur(20px);
                border-bottom: 1px solid var(--glass-border); z-index: 10;
                display: flex; flex-direction: column; gap: 10px;
            }
            .top-row { display: flex; align-items: center; justify-content: space-between; width: 100%; }
            
            .brand { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 18px; }
            .dot { width: 8px; height: 8px; background: var(--primary); border-radius: 50%; box-shadow: 0 0 10px var(--primary); }
            
            .controls-row { display: flex; justify-content: space-between; width: 100%; align-items: center; }

            /* Switcher */
            .switcher {
                background: rgba(0,0,0,0.3); border: 1px solid var(--glass-border);
                border-radius: 20px; padding: 2px; display: flex;
            }
            .model-btn {
                padding: 5px 10px; border-radius: 16px; font-size: 10px; font-weight: 600;
                color: #888; cursor: pointer; transition: 0.3s;
            }
            .model-btn.active-gemini { background: rgba(0, 242, 234, 0.2); color: var(--primary); }
            .model-btn.active-gemma { background: rgba(255, 0, 85, 0.2); color: var(--gemma); }

            /* Deep Think Toggle */
            .dt-toggle {
                display: flex; align-items: center; gap: 8px; font-size: 11px; color: #888; cursor: pointer;
            }
            .dt-box {
                width: 14px; height: 14px; border: 1px solid #555; border-radius: 3px; display: flex; align-items: center; justify-content: center;
            }
            .dt-toggle.active { color: var(--director); text-shadow: 0 0 10px rgba(255, 215, 0, 0.3); }
            .dt-toggle.active .dt-box { background: var(--director); border-color: var(--director); box-shadow: 0 0 8px var(--director); }

            /* --- CHAT --- */
            .chat-container {
                flex-grow: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; scroll-behavior: smooth;
            }
            .message {
                max-width: 85%; padding: 12px 16px; border-radius: 18px; font-size: 15px; line-height: 1.5;
                position: relative; word-wrap: break-word; animation: popIn 0.2s ease;
            }
            .user-msg { align-self: flex-end; background: var(--user-bubble); color: #000; font-weight: 500; border-bottom-right-radius: 4px; }
            .ai-msg { align-self: flex-start; background: var(--ai-bubble); color: #eee; border: 1px solid var(--glass-border); border-bottom-left-radius: 4px; }
            .img-preview { max-width: 100%; border-radius: 10px; margin-top: 5px; display: block; }
            @keyframes popIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

            /* --- INPUT --- */
            .input-area {
                padding: 15px; background: var(--header-bg); backdrop-filter: blur(20px);
                border-top: 1px solid var(--glass-border); display: flex; gap: 10px; align-items: flex-end;
                padding-bottom: max(15px, env(safe-area-inset-bottom));
            }
            .input-wrapper { flex-grow: 1; position: relative; }
            
            textarea {
                width: 100%; background: rgba(0,0,0,0.4); border: 1px solid var(--glass-border);
                padding: 12px 15px; border-radius: 20px; color: #fff; font-size: 16px; font-family: 'Outfit', sans-serif;
                resize: none; height: 48px; max-height: 120px; transition: 0.3s;
            }
            textarea:focus { border-color: var(--primary); }

            .icon-btn {
                width: 48px; height: 48px; border-radius: 50%; border: 1px solid var(--glass-border);
                background: rgba(255,255,255,0.05); color: #aaa; font-size: 18px;
                display: flex; align-items: center; justify-content: center; transition: 0.2s; cursor: pointer; flex-shrink: 0;
            }
            .icon-btn:hover { color: var(--primary); border-color: var(--primary); }
            #micBtn.recording { color: #ff0055; border-color: #ff0055; animation: breathe 1.5s infinite; }
            @keyframes breathe { 0% { box-shadow: 0 0 0 0 rgba(255, 0, 85, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(255, 0, 85, 0); } }

            /* Image Preview */
            #imageUploadPreview {
                position: absolute; bottom: 60px; left: 15px; width: 60px; height: 60px;
                border-radius: 10px; object-fit: cover; border: 2px solid var(--primary);
                display: none; background: #000; z-index: 20;
            }
            .remove-img {
                position: absolute; top: -8px; right: -8px; background: red; color: white;
                width: 18px; height: 18px; border-radius: 50%; font-size: 12px;
                display: flex; align-items: center; justify-content: center; cursor: pointer;
            }

        </style>
    </head>
    <body>

        <div class="orb orb-1"></div>
        <div class="orb orb-2"></div>

        <div class="header">
            <div class="top-row">
                <div class="brand"><div class="dot"></div> Omni-Chat</div>
                <div class="switcher">
                    <div class="model-btn active-gemini" id="btnGemini" onclick="setModel('GEMINI')">Gemini 3</div>
                    <div class="model-btn" id="btnGemma" onclick="setModel('GEMMA')">Gemma 27B</div>
                </div>
            </div>
            <div class="controls-row">
                <div class="dt-toggle" id="dtToggle" onclick="toggleDT()">
                    <div class="dt-box"><i class="fa-solid fa-check" style="font-size: 8px; color: black; display: none;" id="dtCheck"></i></div>
                    Deep Think (Director)
                </div>
            </div>
        </div>

        <div class="chat-container" id="chat">
            <div class="message ai-msg">Online. Shift+Enter for new line.</div>
        </div>

        <div class="input-area">
            <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="handleFile(this)">
            <div id="previewContainer" style="display:none;">
                <img id="imageUploadPreview">
                <div class="remove-img" onclick="clearImage()">×</div>
            </div>

            <button class="icon-btn" onclick="document.getElementById('fileInput').click()">
                <i class="fa-solid fa-paperclip"></i>
            </button>

            <div class="input-wrapper">
                <textarea id="prompt" placeholder="Message..." rows="1"></textarea>
            </div>

            <button class="icon-btn" id="micBtn" ontouchstart="startRec()" ontouchend="stopRec()" onmousedown="startRec()" onmouseup="stopRec()">
                <i class="fa-solid fa-microphone"></i>
            </button>
            <button class="icon-btn" style="background: var(--primary); color: #000; border: none;" onclick="sendText()">
                <i class="fa-solid fa-arrow-up"></i>
            </button>
        </div>

        <audio id="audioPlayer" style="display:none"></audio>

        <script>
            let currentModel = 'GEMINI';
            let deepThinkEnabled = false;
            let currentImageBase64 = null;

            function setModel(model) {
                currentModel = model;
                document.getElementById('btnGemini').className = model === 'GEMINI' ? 'model-btn active-gemini' : 'model-btn';
                document.getElementById('btnGemma').className = model === 'GEMMA' ? 'model-btn active-gemma' : 'model-btn';
            }

            function toggleDT() {
                deepThinkEnabled = !deepThinkEnabled;
                document.getElementById('dtToggle').className = deepThinkEnabled ? 'dt-toggle active' : 'dt-toggle';
                document.getElementById('dtCheck').style.display = deepThinkEnabled ? 'block' : 'none';
            }

            function addMsg(text, type, img=null) {
                let div = document.createElement("div");
                div.className = "message " + type;
                if (img) {
                    let i = document.createElement("img");
                    i.src = img;
                    i.className = "img-preview";
                    div.appendChild(i);
                }
                let t = document.createElement("div");
                t.innerText = text;
                div.appendChild(t);
                let container = document.getElementById("chat");
                container.appendChild(div);
                container.scrollTop = container.scrollHeight;
            }

            // Image Logic
            function handleFile(input) {
                if (input.files && input.files[0]) {
                    let reader = new FileReader();
                    reader.onload = function(e) {
                        currentImageBase64 = e.target.result.split(',')[1];
                        let preview = document.getElementById('imageUploadPreview');
                        preview.src = e.target.result;
                        document.getElementById('previewContainer').style.display = 'block';
                        preview.style.display = 'block';
                    }
                    reader.readAsDataURL(input.files[0]);
                }
            }
            function clearImage() {
                currentImageBase64 = null;
                document.getElementById('fileInput').value = "";
                document.getElementById('previewContainer').style.display = 'none';
            }

            // Textarea Logic (Shift+Enter)
            const promptInput = document.getElementById("prompt");
            promptInput.addEventListener("input", function() { this.style.height = "auto"; this.style.height = (this.scrollHeight) + "px"; });
            promptInput.addEventListener("keydown", function(e) {
                if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendText(); }
            });

            function sendText() {
                let txt = promptInput.value.trim();
                if(!txt && !currentImageBase64) return;

                let imgSrc = currentImageBase64 ? "data:image/jpeg;base64," + currentImageBase64 : null;
                addMsg(txt, "user-msg", imgSrc);
                
                promptInput.value = "";
                promptInput.style.height = "48px";
                
                let payload = { prompt: txt, model: currentModel, deep_think: deepThinkEnabled };
                if (currentImageBase64) { payload.image = currentImageBase64; clearImage(); }

                if (deepThinkEnabled) addMsg("Director is reviewing...", "ai-msg");

                fetch("/process_text", {
                    method: "POST", headers: {"Content-Type": "application/json"},
                    body: JSON.stringify(payload)
                }).then(r=>r.json()).then(d => {
                    addMsg(d.text, "ai-msg");
                });
            }

            // Voice Logic
            let recorder, chunks = [];
            async function startRec() {
                document.getElementById("micBtn").classList.add("recording");
                try {
                    let stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    recorder = new MediaRecorder(stream);
                    chunks = [];
                    recorder.ondataavailable = e => chunks.push(e.data);
                    recorder.start();
                } catch(e) { alert("Mic denied"); }
            }

            function stopRec() {
                document.getElementById("micBtn").classList.remove("recording");
                if(!recorder) return;
                recorder.stop();
                recorder.onstop = () => {
                    let blob = new Blob(chunks, { type: 'audio/webm' });
                    let reader = new FileReader();
                    reader.readAsDataURL(blob);
                    reader.onloadend = () => {
                        let b64 = reader.result.split(',')[1];
                        addMsg("🎤 Processing Audio...", "user-msg");
                        fetch("/process_voice", {
                            method: "POST", headers: {"Content-Type": "application/json"},
                            body: JSON.stringify({audio: b64})
                        }).then(r=>r.json()).then(d => {
                            addMsg(d.text, "ai-msg");
                            if(d.audio) {
                                let aud = document.getElementById("audioPlayer");
                                aud.src = "data:audio/mp3;base64," + d.audio;
                                aud.play();
                            }
                        });
                    };
                };
            }
        </script>
    </body>
    </html>
    '''

# --- BACKEND ---

@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.json
    res = call_ai("text", model_id=data.get('model'), prompt=data.get('prompt'), image_data=data.get('image'), deep_think=data.get('deep_think'))
    return jsonify({"text": res["text"]})

@app.route('/process_voice', methods=['POST'])
def process_voice():
    res = call_ai("voice", audio_data=request.json.get('audio'))
    return jsonify({"text": res["text"], "audio": res["audio"]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
