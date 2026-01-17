import os
import json
import base64
import asyncio
import requests
import markdown2
import io
from flask import Flask, request, jsonify
from flask_sock import Sock
from gtts import gTTS  # <--- Essential for the button to work
from google import genai
from google.genai import types

app = Flask(__name__)
sock = Sock(app)

# --- CONFIGURATION ---
GEMINI_KEY = os.environ.get("GEMINI")

# --- MODEL CHAINS ---
MODEL_CHAINS = {
    "GEMINI": [
        "gemini-3-flash-preview", 
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite"
    ],
    "GEMMA": [
        "gemma-3-27b-it",
        "gemma-3-12b-it",
        "gemma-3-4b-it",
        "gemma-3-2b-it"
    ],
    "DIRECTOR": [
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemma-3-27b-it",
        "gemma-3-12b-it",
        "gemma-3-4b-it",
        "gemma-3-2b-it"
    ],
    "NATIVE_AUDIO": ["gemini-2.5-flash-native-audio-dialog"]
}

# --- MARKDOWN PARSING ---
def parse_markdown(text):
    try:
        return markdown2.markdown(text, extras=["tables", "fenced-code-blocks", "strike", "break-on-newline"])
    except: return text

# --- HELPER: ROBUST REQUEST ---
def try_model_chain(chain_key, payload):
    models = MODEL_CHAINS.get(chain_key, MODEL_CHAINS["GEMINI"])
    last_error = "No models available"

    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}"
        try:
            r = requests.post(url, json=payload)
            if r.status_code != 200:
                print(f"⚠️ {model} Failed ({r.status_code}). Switching...")
                continue
            
            data = r.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            
            if "error" in data:
                print(f"⚠️ {model} API Error. Switching...")
                continue
                
        except Exception as e:
            last_error = str(e)
            continue

    return f"Error: All models failed. ({last_error})"

# --- REST AI CALLER ---
def call_ai_text(model_id, prompt, image_data=None, deep_think=False):
    chain_key = model_id if model_id in MODEL_CHAINS else "GEMINI"
    
    if deep_think:
        prompt = f"CRITICAL INSTRUCTION: Review your own answer for accuracy/tone before replying.\n\nUser: {prompt}"
        chain_key = "DIRECTOR" 

    parts = [{ "text": prompt }]
    if image_data:
        parts.append({ "inline_data": { "mime_type": "image/jpeg", "data": image_data } })
    
    payload = { "contents": [{ "parts": parts }] }
    
    return try_model_chain(chain_key, payload)

# --- HELPER: GTTS GENERATION (FIXED) ---
@app.route('/generate_tts', methods=['POST'])
def generate_tts():
    text = request.json.get('text')
    if not text: return jsonify({"error": "No text"}), 400

    try:
        # We use gTTS library instead of Gemini API for the button
        # This is much faster and more reliable for reading text bubbles
        tts = gTTS(text=text, lang='en')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode()
        return jsonify({"audio": b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- WEBSOCKET LIVE CALL (Google GenAI SDK) ---
@sock.route('/ws/live')
def live_socket(ws):
    client = genai.Client(api_key=GEMINI_KEY, http_options={'api_version': 'v1alpha'})
    
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"], 
        output_audio_transcription=types.AudioTranscriptionConfig()
    )
    
    async def session_loop():
        try:
            # We grab the first model from the NATIVE_AUDIO chain
            target_model = MODEL_CHAINS["NATIVE_AUDIO"][0]
            
            async with client.aio.live.connect(model=target_model, config=config) as session:
                print(f">> Connected to {target_model}")
                
                async def send_audio():
                    while True:
                        try:
                            data = await asyncio.to_thread(ws.receive)
                            if not data: break
                            msg = json.loads(data)
                            if "audio" in msg:
                                await session.send(input={"data": msg["audio"], "mime_type": "application/pcm"}, end_of_turn=False)
                            elif "commit" in msg:
                                await session.send(input={}, end_of_turn=True)
                        except: break

                async def receive_response():
                    while True:
                        async for response in session.receive():
                            payload = {}
                            if response.server_content and response.server_content.model_turn:
                                for part in response.server_content.model_turn.parts:
                                    if part.inline_data:
                                        payload["audio"] = base64.b64encode(part.inline_data.data).decode('utf-8')
                            if response.server_content and response.server_content.output_transcription:
                                payload["text"] = response.server_content.output_transcription.text
                            if payload:
                                await asyncio.to_thread(ws.send, json.dumps(payload))
                await asyncio.gather(send_audio(), receive_response())
        except Exception as e:
            print(f"WS Error: {e}")

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(session_loop())
    except: pass

# --- WEB SERVER ---

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Omni-Chat Live</title>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
        <meta name="theme-color" content="#050508">
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            :root { --bg: #050508; --header: rgba(20,20,30,0.95); --border: rgba(255,255,255,0.1); --primary: #00f2ea; --secondary: #7000ff; --text: #fff; }
            * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
            body { background: var(--bg); color: var(--text); font-family: 'Outfit', sans-serif; height: 100dvh; display: flex; flex-direction: column; margin: 0; overflow: hidden; }

            .orb { position: absolute; border-radius: 50%; filter: blur(80px); opacity: 0.3; z-index: -1; animation: float 10s infinite alternate; }
            .orb-1 { width: 400px; height: 400px; background: var(--secondary); top: -10%; left: -10%; }
            .orb-2 { width: 300px; height: 300px; background: var(--primary); bottom: -10%; right: -10%; animation-delay: 2s; }
            @keyframes float { 0% { transform: translate(0,0); } 100% { transform: translate(30px, 30px); } }

            .header { padding: 10px 15px; background: var(--header); border-bottom: 1px solid var(--border); z-index: 10; display: flex; flex-direction: column; gap: 10px; }
            .top { display: flex; justify-content: space-between; align-items: center; }
            .brand { font-weight: 700; font-size: 18px; display: flex; gap: 10px; align-items: center; }
            .dot { width: 8px; height: 8px; background: var(--primary); border-radius: 50%; box-shadow: 0 0 10px var(--primary); animation: pulse 2s infinite; }
            @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } }

            .switcher { background: rgba(0,0,0,0.3); border: 1px solid var(--border); border-radius: 20px; padding: 2px; display: flex; }
            .mod-btn { padding: 5px 10px; border-radius: 16px; font-size: 10px; font-weight: 600; color: #888; cursor: pointer; transition: 0.3s; }
            .mod-btn.active { background: rgba(0, 242, 234, 0.2); color: var(--primary); }
            
            .dt-toggle { font-size: 11px; color: #888; display: flex; align-items: center; gap: 5px; cursor: pointer; }
            .dt-box { width: 14px; height: 14px; border: 1px solid #555; border-radius: 3px; display: flex; align-items: center; justify-content: center; transition: 0.3s; }
            .dt-toggle.active { color: #ffd700; }
            .dt-toggle.active .dt-box { background: #ffd700; border-color: #ffd700; color: #000; box-shadow: 0 0 8px #ffd700; }

            .chat { flex-grow: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; }
            .msg { max-width: 85%; padding: 12px 16px; border-radius: 18px; font-size: 15px; line-height: 1.5; word-wrap: break-word; animation: pop 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275); position: relative; }
            .user { align-self: flex-end; background: linear-gradient(135deg, var(--primary), #00a8a2); color: #000; font-weight: 500; border-bottom-right-radius: 4px; }
            .ai { align-self: flex-start; background: rgba(255,255,255,0.05); border: 1px solid var(--border); border-bottom-left-radius: 4px; }
            .img-prev { max-width: 100%; border-radius: 10px; margin-top: 5px; display: block; }
            @keyframes pop { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

            /* LOADING INDICATOR */
            .loading { display: flex; align-items: center; gap: 8px; color: #aaa; font-style: italic; padding: 10px 16px; }
            .spinner { width: 12px; height: 12px; border: 2px solid var(--primary); border-top-color: transparent; border-radius: 50%; animation: spin 1s linear infinite; }
            @keyframes spin { to { transform: rotate(360deg); } }

            .ai p { margin: 5px 0; }
            .ai code { background: rgba(0,242,234,0.1); color: var(--primary); padding: 2px 4px; border-radius: 4px; font-family: monospace; }
            .ai pre { background: rgba(0,0,0,0.5); padding: 10px; border-radius: 8px; overflow-x: auto; margin: 10px 0; }

            /* TTS Button */
            .tts-btn {
                position: absolute; bottom: -25px; right: 0; background: rgba(255,255,255,0.1); 
                color: #aaa; border: none; border-radius: 50%; width: 24px; height: 24px;
                display: flex; align-items: center; justify-content: center; cursor: pointer; font-size: 10px; transition: 0.2s;
            }
            .tts-btn:hover { color: var(--primary); background: rgba(0,242,234,0.1); }

            .input-area { padding: 15px; background: var(--header); border-top: 1px solid var(--border); display: flex; gap: 10px; align-items: flex-end; padding-bottom: max(15px, env(safe-area-inset-bottom)); }
            .txt-box { flex-grow: 1; position: relative; }
            textarea { width: 100%; background: rgba(0,0,0,0.4); border: 1px solid var(--border); padding: 12px 15px; border-radius: 20px; color: #fff; font-size: 16px; resize: none; height: 48px; max-height: 120px; transition: 0.3s; font-family: inherit; }
            textarea:focus { border-color: var(--primary); box-shadow: 0 0 15px rgba(0,242,234,0.2); }
            
            .icon-btn { width: 48px; height: 48px; border-radius: 50%; border: 1px solid var(--border); background: rgba(255,255,255,0.05); color: #aaa; font-size: 18px; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: 0.2s; flex-shrink: 0; }
            .icon-btn:hover { color: var(--primary); border-color: var(--primary); }
            .send-btn { background: var(--primary); color: #000; border: none; }

            /* LIVE CALL MODAL */
            .call-modal { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(5,5,8,0.95); z-index: 100; display: none; flex-direction: column; align-items: center; justify-content: center; backdrop-filter: blur(10px); animation: fadeIn 0.3s ease; }
            .call-status { font-size: 24px; font-weight: 700; color: #fff; margin-bottom: 10px; }
            .call-subtitle { font-size: 14px; color: #aaa; margin-bottom: 30px; text-align: center; max-width: 80%; }
            .call-visualizer { display: flex; gap: 5px; height: 50px; align-items: center; margin-bottom: 40px; }
            .bar { width: 6px; background: var(--primary); border-radius: 3px; animation: wave 1s infinite ease-in-out; height: 10px; }
            @keyframes wave { 0%, 100% { height: 10px; opacity: 0.5; } 50% { height: 40px; opacity: 1; } }
            @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
            
            .call-controls { display: flex; gap: 20px; }
            .call-btn { width: 70px; height: 70px; border-radius: 50%; border: none; display: flex; align-items: center; justify-content: center; font-size: 24px; cursor: pointer; transition: 0.2s; }
            .mute-btn { background: #333; color: #fff; }
            .mute-btn.active { background: #fff; color: #000; }
            .end-btn { background: #ff0055; color: #fff; transform: scale(1.1); }

            #fileInput, #previewContainer { display: none; }
            #previewContainer { position: absolute; bottom: 60px; left: 15px; }
            #imageUploadPreview { width: 60px; height: 60px; border-radius: 10px; object-fit: cover; border: 2px solid var(--primary); }

        </style>
    </head>
    <body>

        <div class="orb orb-1"></div><div class="orb orb-2"></div>

        <div class="header">
            <div class="top">
                <div class="brand"><div class="dot"></div> Omni-Chat</div>
                <div class="switcher">
                    <div class="mod-btn active" id="btnGemini" onclick="setMod('GEMINI')">Gemini</div>
                    <div class="mod-btn" id="btnGemma" onclick="setMod('GEMMA')">Gemma</div>
                </div>
            </div>
            <div class="dt-toggle" id="dtToggle" onclick="toggleDT()">
                <div class="dt-box"><i class="fa-solid fa-check" style="display:none" id="dtCheck"></i></div> Director Review
            </div>
        </div>

        <div class="chat" id="chat">
            <div class="msg ai">Online. Click the mic for Live Call.</div>
        </div>

        <div class="input-area">
            <input type="file" id="fileInput" accept="image/*" onchange="handleFile(this)">
            <div id="previewContainer"><img id="imageUploadPreview"></div>

            <button class="icon-btn" onclick="document.getElementById('fileInput').click()"><i class="fa-solid fa-paperclip"></i></button>
            
            <div class="txt-box">
                <textarea id="prompt" placeholder="Message..." rows="1"></textarea>
            </div>

            <button class="icon-btn" onclick="startLiveCall()"><i class="fa-solid fa-microphone"></i></button>
            <button class="icon-btn send-btn" onclick="sendText()"><i class="fa-solid fa-arrow-up"></i></button>
        </div>

        <!-- LIVE CALL MODAL -->
        <div class="call-modal" id="callModal">
            <div class="call-status" id="callStatus">Connecting...</div>
            <div class="call-subtitle" id="callSub"></div>
            <div class="call-visualizer">
                <div class="bar" style="animation-delay:0s"></div><div class="bar" style="animation-delay:0.1s"></div>
                <div class="bar" style="animation-delay:0.2s"></div><div class="bar" style="animation-delay:0.3s"></div>
            </div>
            <div class="call-controls">
                <button class="call-btn mute-btn" id="muteBtn" onclick="toggleMute()"><i class="fa-solid fa-microphone-slash"></i></button>
                <button class="call-btn end-btn" onclick="endCall()"><i class="fa-solid fa-phone-slash"></i></button>
            </div>
        </div>

        <audio id="audioPlayer" style="display:none"></audio>

        <script>
            let currMod = 'GEMINI';
            let dtEnabled = false;
            let imgBase64 = null;
            let mediaRecorder = null;
            let ws = null;
            let audioContext = null;
            let audioQueue = [];
            let isPlaying = false;

            function setMod(m) {
                currMod = m;
                document.getElementById('btnGemini').className = m === 'GEMINI' ? 'mod-btn active' : 'mod-btn';
                document.getElementById('btnGemma').className = m === 'GEMMA' ? 'mod-btn active' : 'mod-btn';
            }

            function toggleDT() {
                dtEnabled = !dtEnabled;
                document.getElementById('dtToggle').className = dtEnabled ? 'dt-toggle active' : 'dt-toggle';
                document.getElementById('dtCheck').style.display = dtEnabled ? 'block' : 'none';
            }

            function addMsg(txt, type, isHtml=false, isLive=false) {
                let d = document.createElement("div");
                d.className = "msg " + type;
                
                let contentDiv = document.createElement("div");
                if(isHtml) contentDiv.innerHTML = txt; else contentDiv.innerText = txt;
                d.appendChild(contentDiv);

                if (type === "ai" && !isLive) {
                    let btn = document.createElement("button");
                    btn.className = "tts-btn";
                    btn.innerHTML = '<i class="fa-solid fa-volume-high"></i>';
                    btn.onclick = () => playTTS(contentDiv.innerText);
                    d.appendChild(btn);
                }

                let c = document.getElementById("chat");
                c.appendChild(d);
                c.scrollTop = c.scrollHeight;
            }

            // LOADING UI LOGIC
            function addLoading() {
                let d = document.createElement("div");
                d.className = "msg ai loading";
                d.id = "loadingMsg";
                d.innerHTML = 'Thinking <div class="spinner"></div>';
                document.getElementById("chat").appendChild(d);
                document.getElementById("chat").scrollTop = document.getElementById("chat").scrollHeight;
            }

            function removeLoading() {
                let el = document.getElementById("loadingMsg");
                if(el) el.remove();
            }

            function playTTS(text) {
                fetch("/generate_tts", {
                    method: "POST", headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({text: text})
                }).then(r=>r.json()).then(d => {
                    if(d.audio) {
                        let aud = document.getElementById("audioPlayer");
                        aud.src = "data:audio/mp3;base64," + d.audio;
                        aud.play();
                    }
                });
            }

            const txtIn = document.getElementById("prompt");
            txtIn.addEventListener("input", function() { this.style.height = "auto"; this.style.height = this.scrollHeight + "px"; });
            txtIn.addEventListener("keydown", function(e) { if(e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendText(); } });

            function sendText() {
                let t = txtIn.value.trim();
                if(!t && !imgBase64) return;
                
                addMsg(t, "user");
                txtIn.value = "";
                txtIn.style.height = "48px";
                
                let p = { prompt: t, model: currMod, deep_think: dtEnabled };
                if(imgBase64) { p.image = imgBase64; imgBase64 = null; document.getElementById('previewContainer').style.display='none'; }

                addLoading();

                fetch("/process_text", {
                    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(p)
                }).then(r=>r.json()).then(d => {
                    removeLoading();
                    addMsg(d.html || d.text, "ai", true);
                });
            }

            function handleFile(input) {
                if (input.files[0]) {
                    let r = new FileReader();
                    r.onload = e => {
                        imgBase64 = e.target.result.split(',')[1];
                        document.getElementById('imageUploadPreview').src = e.target.result;
                        document.getElementById('previewContainer').style.display = 'block';
                    };
                    r.readAsDataURL(input.files[0]);
                }
            }

            // --- LIVE CALL LOGIC ---
            async function startLiveCall() {
                document.getElementById('callModal').style.display = 'flex';
                document.getElementById('callStatus').innerText = "Connecting...";
                
                try {
                    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } });
                    
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    ws = new WebSocket(`${protocol}//${window.location.host}/ws/live`);
                    
                    ws.onopen = () => {
                        document.getElementById('callStatus').innerText = "Live";
                        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
                        mediaRecorder.ondataavailable = (e) => {
                            if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
                                const reader = new FileReader();
                                reader.onload = () => {
                                    const b64 = reader.result.split(',')[1];
                                    ws.send(JSON.stringify({ audio: b64 }));
                                };
                                reader.readAsDataURL(e.data);
                            }
                        };
                        mediaRecorder.start(100); 
                    };

                    ws.onmessage = async (event) => {
                        const data = JSON.parse(event.data);
                        if(data.audio) playPCM(data.audio);
                        if(data.text) document.getElementById('callSub').innerText = data.text;
                    };

                    ws.onclose = () => endCall();

                } catch(e) {
                    alert("Call Failed: " + e);
                    endCall();
                }
            }

            function playPCM(b64) {
                const binaryString = window.atob(b64);
                const len = binaryString.length;
                const bytes = new Uint8Array(len);
                for (let i = 0; i < len; i++) bytes[i] = binaryString.charCodeAt(i);
                
                const int16 = new Int16Array(bytes.buffer);
                const float32 = new Float32Array(int16.length);
                for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

                const buffer = audioContext.createBuffer(1, float32.length, 24000);
                buffer.getChannelData(0).set(float32);

                audioQueue.push(buffer);
                schedulePlayback();
            }

            function schedulePlayback() {
                if (isPlaying || audioQueue.length === 0) return;
                isPlaying = true;
                const buffer = audioQueue.shift();
                const source = audioContext.createBufferSource();
                source.buffer = buffer;
                source.connect(audioContext.destination);
                source.onended = () => { isPlaying = false; schedulePlayback(); };
                source.start();
            }

            function toggleMute() {
                if(mediaRecorder) {
                    if(mediaRecorder.state === "recording") { mediaRecorder.pause(); document.getElementById('muteBtn').classList.add('active'); }
                    else { mediaRecorder.resume(); document.getElementById('muteBtn').classList.remove('active'); }
                }
            }

            function endCall() {
                if(ws) ws.close();
                if(mediaRecorder) mediaRecorder.stop();
                if(audioContext) audioContext.close();
                document.getElementById('callModal').style.display = 'none';
            }

        </script>
    </body>
    </html>
    '''

# --- BACKEND REST ---

@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.json
    p = data.get('prompt')
    m = data.get('model')
    dt = data.get('deep_think')
    img = data.get('image')
    
    text_res = call_ai_text(m, p, img, dt)
    html = parse_markdown(text_res)
    return jsonify({"text": text_res, "html": html})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
