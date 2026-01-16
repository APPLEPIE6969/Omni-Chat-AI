import os
import requests
import json
import threading
import time
import base64
from flask import Flask, request, jsonify, send_file
from gtts import gTTS
import io

app = Flask(__name__)

# --- CONFIGURATION ---
GEMINI_KEY = os.environ.get("GEMINI")

# --- THE OMNI-ASSISTANT ROSTER ---
MODELS = {
    # The "Brain" that manages conversation flow
    "DIRECTOR": "gemini-3-flash", 
    # The "Ear" - Can understand uploaded audio files natively
    "LISTENER": "gemini-2.5-flash", 
    # The "Voice" - Optimized for conversation generation
    "TALKER": "gemini-2.5-flash-native-audio-dialog",
    # The "Worker" - General tasks
    "WORKER": "gemini-2.5-flash",
    # The "Fast" router
    "SCOUT": "gemini-2.5-flash-lite"
}

# Global State
current_status = {
    "message": "System Online",
    "logs": [],
    "last_response": "",
    "is_processing": False
}

def log_event(text, highlight=False):
    prefix = "✨ " if highlight else ">> "
    print(f"{prefix}{text}")
    current_status["logs"].append(f"{prefix}{text}")
    current_status["message"] = text

# --- ROBUST AI CALLER ---
def call_ai(primary_role, prompt, system_instruction, image_data=None, audio_data=None):
    """
    Handles Text, Image, and Audio inputs with Fallbacks.
    """
    primary_model = MODELS.get(primary_role, MODELS["WORKER"])
    fallback_model = MODELS["WORKER"]

    def _send(model_name, p, s, img=None, aud=None):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_KEY}"
        
        # Base content part
        parts = [{"text": f"{s}\n\nUser Input: {p}"}]
        
        # If there is audio (Voice Chat), we append the audio blob
        if aud:
            parts.append({
                "inline_data": {
                    "mime_type": "audio/mp3", # Assumes MP3/WAV input
                    "data": aud
                }
            })
            
        payload = { "contents": [{ "parts": parts }] }
        return requests.post(url, json=payload)

    # 1. Try Primary
    try:
        response = _send(primary_model, prompt, system_instruction, image_data, audio_data)
        data = response.json()
        if "candidates" in data:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        
        err = data.get("error", {}).get("message", "Unknown Error")
        log_event(f"⚠️ {primary_role} Busy: {err}")
        log_event(f"🔄 Switching to Fallback...")

    except Exception as e:
        log_event(f"❌ Connection Fail: {e}")

    # 2. Try Fallback
    try:
        response = _send(fallback_model, prompt, system_instruction, image_data, audio_data)
        data = response.json()
        if "candidates" in data:
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except:
        log_event("💀 CRITICAL: All agents failed.")
    
    return "I am having trouble connecting to the neural network."

# --- WEB SERVER ENDPOINTS ---

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Omni-Assistant</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
        <!-- FontAwesome for Icons -->
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            :root {
                --bg: #050508;
                --card-bg: rgba(20, 20, 30, 0.7);
                --glass-border: rgba(255, 255, 255, 0.08);
                --primary: #00f2ea;
                --primary-glow: rgba(0, 242, 234, 0.4);
                --secondary: #7000ff;
                --text: #ffffff;
                --text-dim: #8888aa;
                --code-bg: #08080c;
            }

            * { box-sizing: border-box; margin: 0; padding: 0; outline: none; }

            body {
                background: var(--bg);
                color: var(--text);
                font-family: 'Outfit', sans-serif;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                overflow-x: hidden;
                position: relative;
            }

            /* --- ANIMATED BACKGROUND --- */
            .orb { position: absolute; border-radius: 50%; filter: blur(120px); opacity: 0.4; z-index: -1; animation: float 10s infinite alternate ease-in-out; }
            .orb-1 { width: 600px; height: 600px; background: var(--secondary); top: -20%; left: -10%; }
            .orb-2 { width: 500px; height: 500px; background: var(--primary); bottom: -10%; right: -10%; animation-delay: 2s; }
            @keyframes float { 0% { transform: translate(0,0); } 100% { transform: translate(40px, 40px); } }

            /* --- MAIN INTERFACE --- */
            .interface {
                width: 90%; max-width: 700px;
                background: var(--card-bg);
                backdrop-filter: blur(30px);
                -webkit-backdrop-filter: blur(30px);
                border: 1px solid var(--glass-border);
                border-radius: 24px;
                padding: 40px;
                box-shadow: 0 20px 50px rgba(0,0,0,0.5);
                transition: transform 0.3s;
                position: relative;
                z-index: 10;
            }

            h1 {
                font-size: 28px; font-weight: 700; margin-bottom: 5px; letter-spacing: -0.5px;
                background: linear-gradient(135deg, #fff 0%, #aaa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                display: flex; align-items: center; gap: 15px;
            }
            .status-indicator { width: 10px; height: 10px; background: var(--primary); border-radius: 50%; box-shadow: 0 0 15px var(--primary); animation: pulse 2s infinite; }
            @keyframes pulse { 0% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(0.8); } 100% { opacity: 1; transform: scale(1); } }
            
            p.subtitle { color: var(--text-dim); font-size: 14px; margin-bottom: 30px; font-weight: 300; display: flex; gap: 10px; align-items: center;}
            .model-badge { font-size: 10px; background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px; color: var(--primary); border: 1px solid rgba(0, 242, 234, 0.2); }

            /* --- INPUT AREA --- */
            .input-container { position: relative; margin-bottom: 20px; display: flex; gap: 10px; }
            
            input[type="text"] {
                flex-grow: 1;
                background: rgba(0,0,0,0.4);
                border: 1px solid var(--glass-border);
                padding: 18px;
                border-radius: 16px;
                color: #fff;
                font-size: 16px;
                font-family: 'Outfit', sans-serif;
                transition: all 0.3s ease;
            }
            input:focus { border-color: var(--primary); box-shadow: 0 0 20px var(--primary-glow); }

            /* --- MIC BUTTON --- */
            #micBtn {
                width: 60px; height: 60px;
                border-radius: 50%;
                border: 1px solid var(--glass-border);
                background: rgba(255,255,255,0.05);
                color: var(--text);
                font-size: 20px;
                cursor: pointer;
                display: flex; align-items: center; justify-content: center;
                transition: 0.3s;
            }
            #micBtn:hover { background: rgba(255,255,255,0.1); border-color: var(--primary); color: var(--primary); }
            #micBtn.recording { 
                background: rgba(255, 0, 85, 0.2); 
                border-color: #ff0055; 
                color: #ff0055; 
                animation: breathe 1.5s infinite; 
            }
            @keyframes breathe { 0% { box-shadow: 0 0 0 0 rgba(255, 0, 85, 0.4); } 70% { box-shadow: 0 0 0 15px rgba(255, 0, 85, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 0, 85, 0); } }

            /* --- ACTION BUTTON --- */
            #sendBtn {
                width: 100%; padding: 16px; border: none; border-radius: 16px;
                background: linear-gradient(135deg, var(--primary) 0%, #00c2bb 100%);
                color: #000; font-size: 16px; font-weight: 700; cursor: pointer;
                transition: 0.3s; margin-bottom: 20px;
                box-shadow: 0 10px 20px rgba(0, 242, 234, 0.15);
            }
            #sendBtn:hover { transform: translateY(-2px); box-shadow: 0 15px 30px rgba(0, 242, 234, 0.3); }

            /* --- CONSOLE & RESPONSE --- */
            .console-window {
                background: var(--code-bg);
                border-radius: 12px;
                padding: 15px;
                height: 150px;
                overflow-y: auto;
                font-family: 'JetBrains Mono', monospace;
                font-size: 12px;
                border: 1px solid var(--glass-border);
                margin-bottom: 20px;
            }
            .console-window::-webkit-scrollbar { width: 5px; }
            .console-window::-webkit-scrollbar-thumb { background: #333; border-radius: 10px; }
            .log-entry { margin-bottom: 6px; color: #aaa; display: flex; gap: 8px; }
            
            .response-area {
                background: rgba(255,255,255,0.03);
                border-radius: 12px;
                padding: 20px;
                min-height: 100px;
                font-size: 15px;
                line-height: 1.6;
                color: #e0e0e0;
                border: 1px solid var(--glass-border);
                display: none;
                animation: slideUp 0.5s ease;
            }
            @keyframes slideUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

        </style>
    </head>
    <body>

        <div class="orb orb-1"></div>
        <div class="orb orb-2"></div>

        <div class="interface">
            <h1><div class="status-indicator"></div> Omni-Assistant</h1>
            <p class="subtitle">
                Powered by Gemini Swarm
                <span class="model-badge">2.5-Flash-TTS</span>
                <span class="model-badge">Native-Audio</span>
            </p>

            <div class="input-container">
                <input type="text" id="prompt" placeholder="Type a message or click Mic to speak..." autocomplete="off">
                <button id="micBtn" onmousedown="startRecording()" onmouseup="stopRecording()" ontouchstart="startRecording()" ontouchend="stopRecording()">
                    <i class="fa-solid fa-microphone"></i>
                </button>
            </div>

            <button id="sendBtn" onclick="sendText()">Send Message</button>

            <div class="console-window" id="console">
                <div class="log-entry">>> System Online. Ready for Text or Voice input.</div>
            </div>

            <div class="response-area" id="responseBox"></div>
            
            <!-- Hidden Audio Player -->
            <audio id="audioPlayer" style="display:none"></audio>
        </div>

        <script>
            // --- LOGGING ---
            function log(txt) { 
                let c = document.getElementById("console");
                c.innerHTML += `<div class="log-entry">${txt}</div>`;
                c.scrollTop = c.scrollHeight;
            }

            // --- TEXT CHAT ---
            function sendText() {
                let p = document.getElementById("prompt");
                let btn = document.getElementById("sendBtn");
                if(!p.value) return;

                btn.innerText = "Thinking...";
                document.getElementById("responseBox").style.display = "none";
                
                fetch("/process_text", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({prompt: p.value})
                }).then(r => r.json()).then(handleResponse);
                
                p.value = "";
            }

            // --- VOICE CHAT LOGIC ---
            let mediaRecorder;
            let audioChunks = [];

            async function startRecording() {
                document.getElementById("micBtn").classList.add("recording");
                log("🎤 Listening...");
                
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.start();
            }

            async function stopRecording() {
                document.getElementById("micBtn").classList.remove("recording");
                if (!mediaRecorder) return;
                
                mediaRecorder.stop();
                mediaRecorder.onstop = async () => {
                    log("⬆️ Uploading Audio to Gemini 2.5...");
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' }); // Webm is standard for browsers
                    
                    // Convert Blob to Base64 to send via JSON (easier for Flask)
                    const reader = new FileReader();
                    reader.readAsDataURL(audioBlob);
                    reader.onloadend = () => {
                        const base64Audio = reader.result.split(',')[1];
                        
                        fetch("/process_voice", {
                            method: "POST",
                            headers: {"Content-Type": "application/json"},
                            body: JSON.stringify({audio: base64Audio})
                        }).then(r => r.json()).then(handleResponse);
                    };
                };
            }

            // --- RESPONSE HANDLER ---
            function handleResponse(data) {
                let btn = document.getElementById("sendBtn");
                btn.innerText = "Send Message";
                
                if(data.success) {
                    // Show Text
                    let box = document.getElementById("responseBox");
                    box.innerText = data.text;
                    box.style.display = "block";
                    log("✅ Response Received.");

                    // Play Audio (If available)
                    if(data.audio_url) {
                        log("🔊 Playing Neural Voice...");
                        let player = document.getElementById("audioPlayer");
                        player.src = data.audio_url + "?t=" + new Date().getTime(); // Prevent caching
                        player.play();
                    }
                } else {
                    log("❌ Error: " + data.error);
                }
            }

            // --- STATUS LOOP ---
            setInterval(() => {
                fetch("/status").then(r=>r.json()).then(d => {
                    if(d.logs.length > 0) {
                        let c = document.getElementById("console");
                        c.innerHTML = "";
                        d.logs.forEach(l => { c.innerHTML += `<div class="log-entry">${l}</div>`; });
                        c.scrollTop = c.scrollHeight;
                    }
                });
            }, 1000);
        </script>
    </body>
    </html>
    '''

# --- BACKEND LOGIC ---

@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.json
    prompt = data.get('prompt')
    current_status["logs"] = []
    
    log_event(f"📝 Text Input: {prompt}")
    
    # 1. Ask the "Talker" model (optimized for dialog)
    response_text = call_ai("TALKER", prompt, "You are a helpful, empathetic AI Assistant. Keep answers concise.")
    
    # 2. Generate Audio (Fallback TTS because specific Gemini Audio API is complex)
    log_event("🔊 Generating Voice...")
    try:
        tts = gTTS(text=response_text, lang='en')
        tts.save("static_response.mp3")
        audio_ready = True
    except:
        audio_ready = False

    return jsonify({
        "success": True, 
        "text": response_text, 
        "audio_url": "/get_audio" if audio_ready else None
    })

@app.route('/process_voice', methods=['POST'])
def process_voice():
    data = request.json
    audio_b64 = data.get('audio')
    current_status["logs"] = []
    
    log_event("🎧 Processing Voice Input...")
    
    # 1. Send Audio to "Listener" (Gemini 2.5 Flash handles audio natively)
    response_text = call_ai(
        "LISTENER", 
        "Listen to this audio and respond helpfully.", 
        "You are a Voice Assistant.", 
        audio_data=audio_b64
    )
    
    # 2. Generate Audio Output
    log_event("🔊 Synthesizing Reply...")
    try:
        tts = gTTS(text=response_text, lang='en')
        tts.save("static_response.mp3")
        audio_ready = True
    except:
        audio_ready = False

    return jsonify({
        "success": True, 
        "text": response_text, 
        "audio_url": "/get_audio" if audio_ready else None
    })

@app.route('/get_audio')
def get_audio():
    try:
        return send_file("static_response.mp3", mimetype="audio/mp3")
    except:
        return "No audio", 404

@app.route('/status')
def status():
    return jsonify(current_status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
