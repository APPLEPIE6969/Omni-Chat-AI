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
    "BRAIN": "gemini-3-flash-preview",                    # Text Logic
    "NATIVE_AUDIO": "gemini-2.5-flash-native-audio-dialog", # Voice Logic
    "NEURAL_TTS": "gemini-2.5-flash-tts"          # Fallback TTS
}

# --- HELPER: GEMINI TEXT-TO-SPEECH ---
def generate_neural_speech(text):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELS['NEURAL_TTS']}:generateContent?key={GEMINI_KEY}"
    payload = { "contents": [{ "parts": [{ "text": text }] }] }
    try:
        response = requests.post(url, json=payload)
        data = response.json()
        if "candidates" in data:
            for part in data["candidates"][0]["content"]["parts"]:
                if "inline_data" in part: return part["inline_data"]["data"]
    except: return None
    return None

# --- MAIN AI CALLER ---
def call_ai(mode, prompt=None, audio_data=None):
    # 1. TEXT MODE (Gemini 3.0 - Silent)
    if mode == "text":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELS['BRAIN']}:generateContent?key={GEMINI_KEY}"
        payload = { "contents": [{ "parts": [{ "text": f"You are a helpful assistant. User says: {prompt}" }] }] }
        try:
            r = requests.post(url, json=payload)
            return {"text": r.json()["candidates"][0]["content"]["parts"][0]["text"], "audio": None}
        except: return {"text": "Connection error.", "audio": None}

    # 2. VOICE MODE (Native Audio 2.5 - Speaks)
    if mode == "voice":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELS['NATIVE_AUDIO']}:generateContent?key={GEMINI_KEY}"
        payload = {
            "contents": [{
                "parts": [
                    { "text": "Listen to this audio. Respond naturally with Audio." },
                    { "inline_data": { "mime_type": "audio/mp3", "data": audio_data } }
                ]
            }]
        }
        try:
            r = requests.post(url, json=payload)
            data = r.json()
            resp_text = "Audio Message Received."
            resp_audio = None
            
            if "candidates" in data:
                parts = data["candidates"][0]["content"]["parts"]
                for part in parts:
                    if "text" in part: resp_text = part["text"]
                    if "inline_data" in part: resp_audio = part["inline_data"]["data"]

            if not resp_audio: resp_audio = generate_neural_speech(resp_text)
            return {"text": resp_text, "audio": resp_audio}
        except Exception as e: return {"text": f"Error: {str(e)}", "audio": None}

# --- WEB SERVER ---

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Omni-Chat</title>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
        <meta name="theme-color" content="#050508">
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            :root {
                --bg: #050508;
                --header-bg: rgba(20, 20, 30, 0.9);
                --glass-border: rgba(255, 255, 255, 0.08);
                --primary: #00f2ea;
                --primary-dark: #00a8a2;
                --secondary: #7000ff;
                --text: #ffffff;
                --text-dim: #8888aa;
                --user-bubble: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
                --ai-bubble: rgba(255, 255, 255, 0.05);
            }

            * { box-sizing: border-box; margin: 0; padding: 0; outline: none; -webkit-tap-highlight-color: transparent; }

            body {
                background: var(--bg);
                color: var(--text);
                font-family: 'Outfit', sans-serif;
                height: 100dvh; /* Full mobile height */
                display: flex;
                flex-direction: column;
                overflow: hidden;
                position: relative;
            }

            /* --- ANIMATED BACKGROUND --- */
            .orb { position: absolute; border-radius: 50%; filter: blur(80px); opacity: 0.3; z-index: -1; animation: float 10s infinite alternate ease-in-out; }
            .orb-1 { width: 400px; height: 400px; background: var(--secondary); top: -10%; left: -10%; }
            .orb-2 { width: 300px; height: 300px; background: var(--primary); bottom: -10%; right: -10%; animation-delay: 2s; }
            @keyframes float { 0% { transform: translate(0,0); } 100% { transform: translate(30px, 30px); } }

            /* --- HEADER --- */
            .header {
                padding: 15px 20px;
                background: var(--header-bg);
                backdrop-filter: blur(20px);
                border-bottom: 1px solid var(--glass-border);
                z-index: 10;
                display: flex; align-items: center; justify-content: space-between;
            }
            .brand { display: flex; align-items: center; gap: 10px; }
            h1 { font-size: 18px; font-weight: 700; margin: 0; background: linear-gradient(135deg, #fff, #aaa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            .dot { width: 8px; height: 8px; background: var(--primary); border-radius: 50%; box-shadow: 0 0 10px var(--primary); animation: pulse 2s infinite; }
            @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } }
            
            .badges { display: flex; gap: 5px; }
            .badge { font-size: 9px; background: rgba(0, 242, 234, 0.1); color: var(--primary); padding: 4px 8px; border-radius: 6px; border: 1px solid rgba(0, 242, 234, 0.2); font-weight: 600; text-transform: uppercase; }

            /* --- CHAT AREA --- */
            .chat-container {
                flex-grow: 1;
                padding: 20px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 15px;
                scroll-behavior: smooth;
            }
            
            /* Scrollbar styling */
            .chat-container::-webkit-scrollbar { width: 4px; }
            .chat-container::-webkit-scrollbar-thumb { background: #333; border-radius: 10px; }

            .message {
                max-width: 80%;
                padding: 14px 18px;
                border-radius: 18px;
                font-size: 15px;
                line-height: 1.5;
                position: relative;
                animation: popIn 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                word-wrap: break-word;
            }

            .user-msg {
                align-self: flex-end;
                background: var(--user-bubble);
                color: #000;
                font-weight: 500;
                border-bottom-right-radius: 4px;
                box-shadow: 0 5px 15px rgba(0, 242, 234, 0.15);
            }

            .ai-msg {
                align-self: flex-start;
                background: var(--ai-bubble);
                color: #eee;
                border: 1px solid var(--glass-border);
                border-bottom-left-radius: 4px;
            }

            .ai-msg.thinking { color: var(--text-dim); font-style: italic; font-size: 13px; padding: 10px 15px; }

            @keyframes popIn { from { opacity: 0; transform: translateY(20px) scale(0.9); } to { opacity: 1; transform: translateY(0) scale(1); } }

            /* --- INPUT AREA --- */
            .input-area {
                padding: 15px;
                background: var(--header-bg);
                backdrop-filter: blur(20px);
                border-top: 1px solid var(--glass-border);
                display: flex;
                gap: 10px;
                align-items: center;
                padding-bottom: max(15px, env(safe-area-inset-bottom));
            }

            input {
                flex-grow: 1;
                background: rgba(0,0,0,0.4);
                border: 1px solid var(--glass-border);
                padding: 14px 20px;
                border-radius: 25px;
                color: #fff;
                font-size: 16px; /* Prevents Zoom */
                font-family: 'Outfit', sans-serif;
                transition: 0.3s;
            }
            input:focus { border-color: var(--primary); box-shadow: 0 0 15px rgba(0, 242, 234, 0.1); }

            /* Buttons */
            .icon-btn {
                width: 50px; height: 50px;
                border-radius: 50%;
                border: 1px solid var(--glass-border);
                background: rgba(255,255,255,0.05);
                color: var(--text-dim);
                font-size: 20px;
                cursor: pointer;
                display: flex; align-items: center; justify-content: center;
                transition: 0.2s;
                flex-shrink: 0;
            }
            
            /* Mic State */
            #micBtn:hover { border-color: var(--primary); color: var(--primary); background: rgba(0, 242, 234, 0.05); }
            #micBtn.recording { 
                background: rgba(255, 0, 85, 0.15); 
                border-color: #ff0055; 
                color: #ff0055; 
                box-shadow: 0 0 15px rgba(255, 0, 85, 0.3);
                animation: breathe 1.5s infinite; 
            }
            @keyframes breathe { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }

            /* Send State */
            #sendBtn { background: var(--primary); color: #000; border: none; }
            #sendBtn:active { transform: scale(0.9); }

        </style>
    </head>
    <body>

        <div class="orb orb-1"></div>
        <div class="orb orb-2"></div>

        <div class="header">
            <div class="brand">
                <div class="dot"></div>
                <h1>Omni-Chat</h1>
            </div>
            <div class="badges">
                <span class="badge">Gemini 3.0</span>
                <span class="badge">Live Audio</span>
            </div>
        </div>

        <div class="chat-container" id="chat">
            <div class="message ai-msg">
                System Online. <br>
                Text me for silent logic.<br>
                Speak to me for conversation.
            </div>
        </div>

        <div class="input-area">
            <button class="icon-btn" id="micBtn" ontouchstart="startRec()" ontouchend="stopRec()" onmousedown="startRec()" onmouseup="stopRec()">
                <i class="fa-solid fa-microphone"></i>
            </button>
            <input type="text" id="prompt" placeholder="Message..." autocomplete="off">
            <button class="icon-btn" id="sendBtn" onclick="sendText()">
                <i class="fa-solid fa-arrow-up"></i>
            </button>
        </div>

        <audio id="audioPlayer" style="display:none"></audio>

        <script>
            function addMsg(text, type, isTemp=false) {
                let div = document.createElement("div");
                div.className = "message " + type;
                if(isTemp) div.id = "tempMsg";
                div.innerText = text;
                let container = document.getElementById("chat");
                container.appendChild(div);
                container.scrollTop = container.scrollHeight;
            }

            function removeTemp() {
                let temp = document.getElementById("tempMsg");
                if(temp) temp.remove();
            }

            // --- TEXT MODE ---
            function sendText() {
                let p = document.getElementById("prompt");
                let txt = p.value.trim();
                if(!txt) return;

                addMsg(txt, "user-msg");
                p.value = "";
                addMsg("Thinking...", "ai-msg thinking", true);

                fetch("/process_text", {
                    method: "POST", headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({prompt: txt})
                }).then(r=>r.json()).then(d => {
                    removeTemp();
                    addMsg(d.text, "ai-msg");
                });
            }

            // --- VOICE MODE ---
            let recorder, chunks = [];

            async function startRec() {
                let btn = document.getElementById("micBtn");
                btn.classList.add("recording");
                try {
                    let stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    recorder = new MediaRecorder(stream);
                    chunks = [];
                    recorder.ondataavailable = e => chunks.push(e.data);
                    recorder.start();
                } catch(e) { 
                    btn.classList.remove("recording");
                    alert("Microphone denied."); 
                }
            }

            function stopRec() {
                document.getElementById("micBtn").classList.remove("recording");
                if(!recorder) return;
                
                addMsg("Listening...", "user-msg", true); // Temp user msg
                recorder.stop();
                
                recorder.onstop = () => {
                    let blob = new Blob(chunks, { type: 'audio/webm' });
                    let reader = new FileReader();
                    reader.readAsDataURL(blob);
                    reader.onloadend = () => {
                        let b64 = reader.result.split(',')[1];
                        
                        // Switch temp user msg to 'Processing'
                        let temp = document.getElementById("tempMsg");
                        if(temp) temp.innerText = "Processing Audio...";

                        fetch("/process_voice", {
                            method: "POST", headers: {"Content-Type": "application/json"},
                            body: JSON.stringify({audio: b64})
                        }).then(r=>r.json()).then(d => {
                            removeTemp(); // Remove 'Processing'
                            addMsg("🎤 Audio Input", "user-msg"); // Static marker
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
    p = request.json.get('prompt')
    res = call_ai("text", prompt=p)
    return jsonify({"text": res["text"]}) # No Audio

@app.route('/process_voice', methods=['POST'])
def process_voice():
    b64 = request.json.get('audio')
    res = call_ai("voice", audio_data=b64)
    return jsonify({"text": res["text"], "audio": res["audio"]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
