import os
import json
import base64
import asyncio
import requests
import markdown2
import io
import concurrent.futures
import numpy as np
from flask import Flask, request, jsonify
from flask_sock import Sock
from gtts import gTTS
from google import genai
from google.genai import types

app = Flask(__name__)
sock = Sock(app)

# --- CONFIGURATION ---
GEMINI_KEY = os.environ.get("GEMINI_KEY")
GROQ_API_KEY = os.environ.get("GROQ")
SKYREELS_API_KEY = os.environ.get("SKYREELS")

# --- SERVER-SIDE CONFIG ---
MODEL_CHAINS = {
    "NATIVE_AUDIO": ["gemini-2.0-flash-exp"], 
    "NEURAL_TTS": ["gemini-2.5-flash-tts"]
}

# --- SERVER HELPERS ---
def parse_markdown(text):
    try:
        return markdown2.markdown(text, extras=["tables", "fenced-code-blocks", "strike", "break-on-newline"])
    except: return text

# --- ENDPOINTS ---
@app.route('/generate_tts', methods=['POST'])
def generate_tts():
    text = request.json.get('text')
    if not text: return jsonify({"error": "No text"}), 400
    try:
        tts = gTTS(text=text, lang='en')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        b64 = base64.b64encode(fp.read()).decode()
        return jsonify({"audio": b64})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/ws/live')
def live_socket(ws):
    client = genai.Client(api_key=GEMINI_KEY, http_options={'api_version': 'v1alpha'})
    config = types.LiveConnectConfig(response_modalities=["AUDIO"], output_audio_transcription=types.AudioTranscriptionConfig())
    async def session_loop():
        try:
            async with client.aio.live.connect(model=MODEL_CHAINS["NATIVE_AUDIO"][0], config=config) as session:
                async def send_audio():
                    while True:
                        try:
                            data = await asyncio.to_thread(ws.receive)
                            if not data: break
                            msg = json.loads(data)
                            if "audio" in msg: await session.send(input={"data": msg["audio"], "mime_type": "application/pcm"}, end_of_turn=False)
                            elif "commit" in msg: await session.send(input={}, end_of_turn=True)
                        except: break
                async def receive_response():
                    while True:
                        async for response in session.receive():
                            payload = {}
                            if response.server_content and response.server_content.model_turn:
                                for part in response.server_content.model_turn.parts:
                                    if part.inline_data: payload["audio"] = base64.b64encode(part.inline_data.data).decode('utf-8')
                            if response.server_content and response.server_content.output_transcription:
                                payload["text"] = response.server_content.output_transcription.text
                            if payload: await asyncio.to_thread(ws.send, json.dumps(payload))
                await asyncio.gather(send_audio(), receive_response())
        except: pass
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(session_loop())
    except: pass

@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.json
    prompt = data.get('prompt', '')
    model = data.get('model', 'gemini-3-flash-preview')
    
    try:
        # Use GEMINI_KEY for all models (Gemini and Gemma)
        client = genai.Client(api_key=GEMINI_KEY, http_options={'api_version': 'v1alpha'})
        
        # Configure the model
        config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048
        )
        
        # Generate content
        response = client.generate_content(
            model=model,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config=config
        )
        
        # Extract text from response
        text = ""
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text'):
                    text += part.text
        
        return jsonify({"text": text or "No response generated"})
        
    except Exception as e:
        return jsonify({"text": f"Error processing request: {str(e)}"})

@app.route('/generate_video', methods=['POST'])
def generate_video():
    if not SKYREELS_API_KEY:
        return jsonify({"error": "SKYREELS_API_KEY not configured"}), 500
    
    data = request.json
    prompt = data.get('prompt')
    ref_images = data.get('ref_images', [])
    duration = data.get('duration', 5)
    aspect_ratio = data.get('aspect_ratio', '16:9')
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    try:
        # Submit video generation task
        submit_response = requests.post(
            'https://apis.skyreels.ai/api/v1/video/multiobject/submit',
            headers={'Content-Type': 'application/json'},
            json={
                'api_key': SKYREELS_API_KEY,
                'prompt': prompt,
                'ref_images': ref_images,
                'duration': duration,
                'aspect_ratio': aspect_ratio
            }
        )
        
        if submit_response.status_code != 200:
            return jsonify({"error": f"SKYREELS API submit error: {submit_response.status_code}"}), 500
        
        submit_result = submit_response.json()
        if submit_result.get('code') != 200:
            return jsonify({"error": f"SKYREELS API error: {submit_result.get('msg', 'Unknown error')}"}), 500
        
        task_id = submit_result.get('task_id')
        if not task_id:
            return jsonify({"error": "Failed to get task ID from SKYREELS API"}), 500
        
        # Poll for task completion
        max_attempts = 60  # 5 minutes with 5-second intervals
        for attempt in range(max_attempts):
            import time
            time.sleep(5)  # Wait 5 seconds between polls
            
            query_response = requests.get(
                f'https://apis.skyreels.ai/api/v1/video/multiobject/task/{task_id}'
            )
            
            if query_response.status_code != 200:
                continue
            
            query_result = query_response.json()
            status = query_result.get('status')
            
            if status == 'success':
                video_data = query_result.get('data', {})
                return jsonify({
                    "video_url": video_data.get('video_url'),
                    "duration": video_data.get('duration'),
                    "resolution": video_data.get('resolution'),
                    "cost_credits": video_data.get('cost_credits'),
                    "status": "success"
                })
            elif status == 'failed':
                return jsonify({"error": f"Video generation failed: {query_result.get('msg', 'Unknown error')}"}), 500
            elif status == 'unknown':
                return jsonify({"error": "Unknown task status from SKYREELS API"}), 500
            # Continue polling for 'submitted', 'pending', or 'running'
        
        return jsonify({"error": "Video generation timed out after 5 minutes"}), 500
            
    except Exception as e:
        return jsonify({"error": f"Video generation failed: {str(e)}"}), 500

# --- WEB SERVER ---
@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Omni-Chat</title>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
        <meta name="theme-color" content="#050508">
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <script src="https://js.puter.com/v2/"></script>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

        <style>
            :root { --bg: #050508; --header: rgba(20,20,30,0.95); --border: rgba(255,255,255,0.1); --primary: #00f2ea; --secondary: #7000ff; --text: #fff; }
            * { box-sizing: border-box; }
            body { background: var(--bg); color: var(--text); font-family: 'Outfit', sans-serif; height: 100dvh; display: flex; flex-direction: column; margin: 0; overflow: hidden; }

            .orb { position: absolute; border-radius: 50%; filter: blur(80px); opacity: 0.3; z-index: -1; animation: float 10s infinite alternate; }
            .orb-1 { width: 400px; height: 400px; background: var(--secondary); top: -10%; left: -10%; }
            .orb-2 { width: 300px; height: 300px; background: var(--primary); bottom: -10%; right: -10%; animation-delay: 2s; }
            @keyframes float { 0% { transform: translate(0,0); } 100% { transform: translate(30px, 30px); } }

            .header { padding: 10px 15px; background: var(--header); border-bottom: 1px solid var(--border); z-index: 10; display: flex; flex-direction: column; gap: 8px; }
            .top { display: flex; justify-content: space-between; align-items: center; }
            .brand { font-weight: 700; font-size: 18px; display: flex; gap: 10px; align-items: center; }
            .dot { width: 8px; height: 8px; background: var(--primary); border-radius: 50%; box-shadow: 0 0 10px var(--primary); animation: pulse 2s infinite; }
            
            .model-select { background: rgba(0,0,0,0.3); border: 1px solid var(--border); border-radius: 20px; color: #aaa; padding: 5px 15px; font-size: 12px; cursor: pointer; display: flex; align-items: center; gap: 5px; transition: 0.2s; user-select: none; }
            .model-select:hover { border-color: var(--primary); color: white; background: rgba(255,255,255,0.05); }

            .dt-toggle { font-size: 11px; color: #666; display: flex; align-items: center; gap: 8px; cursor: pointer; margin-left: 2px; width: fit-content; transition: 0.3s; padding: 4px 8px; border-radius: 12px; user-select: none; }
            .dt-box { width: 14px; height: 14px; border: 1px solid #444; border-radius: 3px; display: flex; align-items: center; justify-content: center; transition: 0.3s; background: #111; }
            .dt-toggle:hover { color: #888; background: rgba(255,255,255,0.02); }
            .dt-toggle.active { color: #ffd700; background: rgba(255, 215, 0, 0.05); }
            .dt-toggle.active .dt-box { background: #ffd700; border-color: #ffd700; color: #000; box-shadow: 0 0 8px #ffd700; }

            .chat { flex-grow: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; }
            .msg { max-width: 85%; padding: 12px 16px; border-radius: 18px; font-size: 15px; line-height: 1.5; word-wrap: break-word; animation: pop 0.3s ease; position: relative; }
            .user { align-self: flex-end; background: linear-gradient(135deg, var(--primary), #00a8a2); color: #000; font-weight: 500; border-bottom-right-radius: 4px; }
            .ai { align-self: flex-start; background: rgba(255,255,255,0.05); border: 1px solid var(--border); border-bottom-left-radius: 4px; }
            @keyframes pop { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

            .ai pre { position: relative; background: rgba(0,0,0,0.5); padding: 15px; border-radius: 12px; overflow-x: auto; margin: 10px 0; border: 1px solid rgba(255,255,255,0.1); }
            .ai code { font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #e0e0e0; }
            .copy-btn { position: absolute; top: 5px; right: 5px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.1); color: #aaa; padding: 4px 8px; border-radius: 6px; cursor: pointer; font-size: 10px; display: flex; align-items: center; gap: 5px; transition: 0.2s; }
            .copy-btn:hover { background: rgba(0, 242, 234, 0.2); color: var(--primary); border-color: var(--primary); }

            .input-area { padding: 15px; background: var(--header); border-top: 1px solid var(--border); display: flex; gap: 10px; align-items: flex-end; }
            .txt-box { flex-grow: 1; }
            textarea { width: 100%; background: rgba(0,0,0,0.4); border: 1px solid var(--border); padding: 12px 15px; border-radius: 20px; color: #fff; font-size: 16px; resize: none; height: 48px; max-height: 120px; transition: 0.3s; font-family: inherit; }
            textarea:focus { border-color: var(--primary); box-shadow: 0 0 15px rgba(0,242,234,0.2); }
            .icon-btn { width: 48px; height: 48px; border-radius: 50%; border: 1px solid var(--border); background: rgba(255,255,255,0.05); color: #aaa; font-size: 18px; display: flex; align-items: center; justify-content: center; cursor: pointer; flex-shrink: 0; transition: 0.2s; }
            .icon-btn:hover { color: var(--primary); border-color: var(--primary); }
            .send-btn { background: var(--primary); color: #000; border: none; }
            .tts-btn { position: absolute; bottom: -25px; right: 0; background: rgba(255,255,255,0.1); color: #aaa; border: none; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; cursor: pointer; font-size: 10px; }

            .modal { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 200; display: none; align-items: center; justify-content: center; backdrop-filter: blur(5px); }
            .modal-content { background: #1a1a20; border: 1px solid var(--border); border-radius: 20px; padding: 20px; width: 90%; max-width: 400px; max-height: 80vh; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }
            .modal-item { padding: 12px; border-radius: 12px; background: rgba(255,255,255,0.05); cursor: pointer; display: flex; justify-content: space-between; align-items: center; }
            .modal-item:hover { background: rgba(0,242,234,0.05); }
            .modal-item.selected { background: rgba(0,242,234,0.15); border: 1px solid var(--primary); }
            .tag { font-size: 10px; padding: 2px 6px; border-radius: 4px; background: #333; color: #aaa; text-transform: uppercase; }
            .tag.fast { color: #00ff00; background: rgba(0,255,0,0.1); }
            .tag.best { color: #ffd700; background: rgba(255,215,0,0.1); }
            .tag.opus { color: #7000ff; background: rgba(112, 0, 255, 0.1); }
            .close-btn { align-self: flex-end; cursor: pointer; color: #aaa; font-size: 20px; }

            .call-vis { display: flex; gap: 5px; height: 50px; align-items: center; margin-bottom: 40px; }
            .bar { width: 6px; background: var(--primary); border-radius: 3px; animation: wave 1s infinite ease-in-out; height: 10px; }
            .bar:nth-child(1) { animation-delay: 0s; } .bar:nth-child(2) { animation-delay: 0.1s; } .bar:nth-child(3) { animation-delay: 0.2s; }
            @keyframes wave { 0%, 100% { height: 10px; opacity: 0.5; } 50% { height: 40px; opacity: 1; } }

            #fileInput, #previewContainer { display: none; }
            #previewContainer { position: absolute; bottom: 60px; left: 15px; }
            #imageUploadPreview { width: 60px; height: 60px; border-radius: 10px; object-fit: cover; border: 2px solid var(--primary); }
            .img-wrapper { position: relative; display: inline-block; max-width: 100%; border-radius: 12px; overflow: hidden; margin-top: 10px; }
            .img-wrapper img { width: 100%; height: auto; display: block; }
            .download-btn { position: absolute; bottom: 8px; right: 8px; background: rgba(0,0,0,0.6); color: white; border: 1px solid rgba(255,255,255,0.2); width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; cursor: pointer; backdrop-filter: blur(4px); }
        </style>
    </head>
    <body>

        <div class="orb orb-1"></div><div class="orb orb-2"></div>

        <div class="header">
            <div class="top">
                <div class="brand"><div class="dot"></div> Omni-Chat</div>
                <!-- MODEL BUTTON: INLINE ONCLICK RESTORED -->
                <div class="model-select" onclick="openModelModal()">
                    <span id="currentModelDisplay">Gemini 3.0</span> <i class="fa-solid fa-chevron-down"></i>
                </div>
            </div>
            <!-- DIRECTOR TOGGLE: INLINE ONCLICK RESTORED -->
            <div class="dt-toggle" id="dtToggle" onclick="toggleDT()">
                <div class="dt-box"><i class="fa-solid fa-check" style="display:none" id="dtCheck"></i></div> Director Mode (Ensemble)
            </div>
        </div>

        <div class="chat" id="chat"><div class="msg ai">Online. Director Mode uses GPT-5.2 Pro + Claude Opus + Gemini 3.</div></div>

        <div class="input-area">
            <input type="file" id="fileInput" accept="image/*" onchange="handleFile(this)">
            <div id="previewContainer"><img id="imageUploadPreview"></div>
            
            <!-- BUTTONS: INLINE ONCLICK RESTORED -->
            <button class="icon-btn" onclick="openImgModal()"><i class="fa-solid fa-palette"></i></button>
            <button class="icon-btn" onclick="openVideoModal()"><i class="fa-solid fa-film"></i></button>
            <button class="icon-btn" onclick="document.getElementById('fileInput').click()"><i class="fa-solid fa-paperclip"></i></button>
            
            <div class="txt-box"><textarea id="prompt" placeholder="Message..." rows="1"></textarea></div>
            
            <button class="icon-btn" onclick="startLiveCall()"><i class="fa-solid fa-microphone"></i></button>
            <button class="icon-btn send-btn" onclick="sendText()"><i class="fa-solid fa-arrow-up"></i></button>
        </div>

        <!-- MODEL MODAL -->
        <div class="modal" id="modelModal">
            <div class="modal-content">
                <div style="display:flex; justify-content:space-between; align-items:center;"><h3>Select Chat Model</h3><div class="close-btn" onclick="closeModelModal()">&times;</div></div>
                <div id="chatModelList"></div>
            </div>
        </div>

        <!-- IMAGE MODAL -->
        <div class="modal" id="imgModal">
            <div class="modal-content">
                <div style="display:flex; justify-content:space-between; align-items:center;"><h3>Image Model</h3><div class="close-btn" onclick="closeImgSettings()">&times;</div></div>
                <div id="imgModelList"></div>
            </div>
        </div>

        <!-- VIDEO MODAL -->
        <div class="modal" id="videoModal">
            <div class="modal-content">
                <div style="display:flex; justify-content:space-between; align-items:center;"><h3>Video Generation</h3><div class="close-btn" onclick="closeVideoModal()">&times;</div></div>
                <div style="text-align:center; padding:20px;">
                    <p style="color:#aaa; margin-bottom:20px;">Generate videos using SKYREELS</p>
                    <button onclick="generateVideo()" style="background:var(--primary); color:#000; border:none; padding:15px 30px; border-radius:30px; font-weight:bold; cursor:pointer;">Generate Video</button>
                </div>
            </div>
        </div>

        <!-- LIVE CALL MODAL -->
        <div class="modal" id="callModal">
            <div style="text-align:center; color:white">
                <h2>Live Call</h2>
                <p id="callStatus">Connecting...</p>
                <div class="call-vis"><div class="bar"></div><div class="bar"></div><div class="bar"></div></div>
                <p id="callSub" style="color:#aaa; font-size:12px; height:20px"></p>
                <button onclick="endCall()" style="background:#ff0055; padding:15px 30px; border-radius:30px; border:none; color:white; font-weight:bold; margin-top:20px">End Call</button>
            </div>
        </div>

        <audio id="audioPlayer" style="display:none"></audio>

        <script>
            // --- DATA CONFIGURATION ---
            const chatModels = [
                {id: "gemini-3-flash-preview", name: "Gemini 3.0", tag: "⚡ GOOGLE"},
                {id: "gemma-3-27b-it", name: "Gemma 3 27B", tag: "🔓 GOOGLE"},
                {id: "gemini-2.5-flash", name: "Gemini 2.5 Flash", tag: "⚡ GOOGLE"},
                {id: "gemini-2.5-flash-lite", name: "Gemini 2.5 Flash Lite", tag: "⚡ GOOGLE"},
                {id: "gemini-2.5-flash-tts", name: "Gemini 2.5 Flash TTS", tag: "🎤 GOOGLE"},
                {id: "gemini-robotics-er-1.5-preview", name: "Gemini Robotics ER 1.5", tag: "🤖 GOOGLE"},
                {id: "gemma-3-1b", name: "Gemma 3 1B", tag: "💎 GOOGLE"},
                {id: "gemma-3-2b", name: "Gemma 3 2B", tag: "💎 GOOGLE"},
                {id: "gemma-3-4b", name: "Gemma 3 4B", tag: "💎 GOOGLE"},
                {id: "gemma-3-12b", name: "Gemma 3 12B", tag: "💎 GOOGLE"},
                {id: "gemini-embedding-1.0", name: "Gemini Embedding 1.0", tag: "📊 GOOGLE"},
                {id: "gemini-2.5-flash-native-audio-dialog", name: "Gemini 2.5 Flash Native Audio", tag: "🎤 GOOGLE"}
            ];

            const imgModels = [];

            // State
            let selectedChatModel = "gemini-3-flash-preview"; 
            let selectedImgModel = "black-forest-labs/FLUX.1-schnell";
            let dtEnabled = false; 
            let imgBase64 = null;
            let chatHistory = [];

            // --- UI FUNCTIONS ---
            function toggleDT() {
                dtEnabled = !dtEnabled;
                const el = document.getElementById("dtToggle");
                const icon = document.getElementById("dtCheck");
                if (dtEnabled) { 
                    el.classList.add("active"); 
                    icon.style.display = "block"; 
                    addMsg("Director Mode: Enabled (Ensemble: GPT-5.2 Pro, Claude Opus 4.5, Gemini 3 Pro).", "ai");
                } else { 
                    el.classList.remove("active"); 
                    icon.style.display = "none"; 
                    addMsg("Director Mode: Disabled.", "ai");
                }
            }

            function renderList(list, containerId, currentVal, onClick) {
                const c = document.getElementById(containerId);
                c.innerHTML = "";
                list.forEach(m => {
                    let div = document.createElement("div");
                    div.className = `modal-item ${m.id === currentVal ? 'selected' : ''}`;
                    let tagClass = m.tag.includes('OPUS') ? 'opus' : (m.tag.includes('GOOGLE')?'fast':'best');
                    div.innerHTML = `<span>${m.name}</span> <span class="tag ${tagClass}">${m.tag}</span>`;
                    div.onclick = () => onClick(m.id, m.name);
                    c.appendChild(div);
                });
            }

            function openModelModal() {
                renderList(chatModels, "chatModelList", selectedChatModel, (id, name) => {
                    selectedChatModel = id;
                    document.getElementById("currentModelDisplay").innerText = name;
                    closeModelModal();
                });
                document.getElementById("modelModal").style.display = "flex";
            }
            function closeModelModal() { document.getElementById('modelModal').style.display='none'; }

            function openImgModal() {
                renderList(imgModels, "imgModelList", selectedImgModel, (id, name) => {
                    selectedImgModel = id;
                    closeImgSettings();
                });
                document.getElementById("imgModal").style.display = "flex";
            }
            function closeImgSettings() { document.getElementById('imgModal').style.display='none'; }

            function openVideoModal() {
                document.getElementById("videoModal").style.display = "flex";
            }
            function closeVideoModal() { document.getElementById('videoModal').style.display='none'; }

            async function generateVideo() {
                let t = txtIn.value.trim();
                if(!t) {
                    addMsg("Please enter a prompt for video generation.", "ai");
                    return;
                }
                
                addLoading("Generating video with SKYREELS...");
                try {
                    // Prepare reference images if any are uploaded
                    let refImages = [];
                    if (imgBase64) {
                        // Convert base64 to data URL for reference image
                        refImages.push(`data:image/jpeg;base64,${imgBase64}`);
                    }
                    
                    const response = await fetch("/generate_video", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({
                            prompt: t,
                            ref_images: refImages,
                            duration: 5,
                            aspect_ratio: "16:9"
                        })
                    });
                    
                    const result = await response.json();
                    removeLoading();
                    
                    if(result.status === "success" && result.video_url) {
                        let div = document.createElement("div");
                        div.className = "img-wrapper";
                        let video = document.createElement("video");
                        video.src = result.video_url;
                        video.controls = true;
                        video.style.width = "100%";
                        video.style.height = "auto";
                        video.style.borderRadius = "12px";
                        div.appendChild(video);
                        
                        // Add metadata info
                        let info = document.createElement("div");
                        info.style.fontSize = "12px";
                        info.style.color = "#888";
                        info.style.marginTop = "5px";
                        info.style.textAlign = "center";
                        info.innerHTML = `Duration: ${result.duration}s | Resolution: ${result.resolution || 'Unknown'} | Cost: ${result.cost_credits || 'Unknown'} credits`;
                        div.appendChild(info);
                        
                        let dl = document.createElement("a");
                        dl.className = "download-btn";
                        dl.innerHTML = '<i class="fa-solid fa-download"></i>';
                        dl.href = result.video_url;
                        dl.download = "ai-video.mp4";
                        div.appendChild(dl);
                        
                        addMsg(div, "ai");
                    } else {
                        addMsg("Video generation failed: " + (result.error || "Unknown error"), "ai");
                    }
                } catch(e) {
                    removeLoading();
                    addMsg("Video generation error: " + e, "ai");
                }
            }

            // --- HELPERS ---
            function extractText(res) {
                if (typeof res === 'string') return res;
                if (res?.message?.content) {
                    if (Array.isArray(res.message.content)) return res.message.content[0].text;
                    return res.message.content;
                }
                return JSON.stringify(res);
            }

            function addCopyBtns(element) {
                element.querySelectorAll('pre').forEach(pre => {
                    if(pre.querySelector('.copy-btn')) return;
                    let btn = document.createElement('button');
                    btn.className = 'copy-btn';
                    btn.innerHTML = '<i class="fa-regular fa-copy"></i> Copy';
                    btn.onclick = () => {
                        let code = pre.querySelector('code');
                        if(code) {
                            navigator.clipboard.writeText(code.innerText);
                            btn.innerHTML = '<i class="fa-solid fa-check"></i> Copied';
                            setTimeout(()=> btn.innerHTML = '<i class="fa-regular fa-copy"></i> Copy', 2000);
                        }
                    };
                    pre.appendChild(btn);
                });
            }

            function addMsg(content, type, isHtml=false) {
                let d = document.createElement("div");
                d.className = "msg " + type;
                if(typeof content === 'string') {
                    let cDiv = document.createElement("div");
                    cDiv.innerHTML = content; // marked.parse output
                    d.appendChild(cDiv);
                    if(type === 'ai') addCopyBtns(cDiv);
                } else d.appendChild(content);
                
                if (type === "ai") {
                    let b = document.createElement("button"); 
                    b.className="tts-btn"; b.innerHTML='<i class="fa-solid fa-volume-high"></i>';
                    b.onclick=()=>playTTS(d.innerText); d.appendChild(b);
                }
                
                document.getElementById("chat").appendChild(d);
                document.getElementById("chat").scrollTop = document.getElementById("chat").scrollHeight;
            }

            function addLoading(t="Thinking...") {
                let d = document.createElement("div"); d.className="msg ai loading"; d.id="load";
                d.innerHTML = `${t} <div class="spinner"></div>`;
                document.getElementById("chat").appendChild(d);
                document.getElementById("chat").scrollTop = document.getElementById("chat").scrollHeight;
            }
            function removeLoading() { let e=document.getElementById("load"); if(e) e.remove(); }

            // --- DIRECTOR MODE (ENSEMBLE) ---
            async function runDirectorMode(prompt) {
                addLoading("Consulting Experts (GPT 5.2, Claude 4.5, Gemini 3)...");
                const experts = [
                    { model: 'gpt-5.2-pro', name: 'GPT-5.2 Pro' },
                    { model: 'claude-opus-4-5', name: 'Claude Opus 4.5' },
                    { model: 'gemini-3-pro-preview', name: 'Gemini 3 Pro' }
                ];

                try {
                    // Parallel Requests
                    const promises = experts.map(exp => 
                        puter.ai.chat(prompt, { model: exp.model })
                            .then(res => `--- Expert: ${exp.name} ---\\n${extractText(res)}\\n`) // ESCAPED NEWLINES
                            .catch(err => `--- Expert: ${exp.name} ---\\nFailed: ${err}\\n`)
                    );
                    
                    const results = await Promise.all(promises);
                    const rawData = results.join("\\n"); // ESCAPED NEWLINE

                    // Synthesis
                    removeLoading();
                    addLoading("Synthesizing Final Answer...");
                    
                    // Template literal safe because we use escaped newlines in rawData
                    const finalPrompt = `
                        USER QUERY: ${prompt}
                        
                        EXPERTS OPINIONS:
                        ${rawData}
                        
                        INSTRUCTION: Combine these opinions into one perfect response. Do not mention the experts. Just write the answer.
                    `;

                    const synthesis = await puter.ai.chat(finalPrompt, { model: 'gemini-3-pro-preview' });
                    removeLoading();
                    addMsg(marked.parse(extractText(synthesis)), "ai");
                    
                } catch (e) {
                    removeLoading();
                    addMsg("Director Mode Failed: " + e, "ai");
                }
            }

            // --- MAIN SEND LOGIC ---
            const txtIn = document.getElementById("prompt");
            txtIn.addEventListener("keydown", function(e) { 
                if(e.key === "Enter" && !e.shiftKey) { 
                    e.preventDefault(); 
                    sendText(); 
                } 
            });

            async function sendText() {
                let t = txtIn.value.trim();
                if(!t && !imgBase64) return;
                
                addMsg(t, "user");
                txtIn.value = "";
                

                // 2. Image Generation
                if (t.toLowerCase().startsWith("/image") || t.toLowerCase().includes("generate image")) {
                    addLoading("Painting...");
                    try {
                        let prompt = t.replace("/image", "").trim();
                        let img = await puter.ai.txt2img(prompt, { model: selectedImgModel });
                        removeLoading();
                        
                        let div = document.createElement("div"); div.className="img-wrapper";
                        img.style.width="100%"; div.appendChild(img);
                        let dl = document.createElement("a"); dl.className="download-btn"; dl.innerHTML='<i class="fa-solid fa-download"></i>';
                        dl.href = img.src; dl.download="ai-image.png"; div.appendChild(dl);
                        addMsg(div, "ai");
                    } catch(e) { removeLoading(); addMsg("Error: "+e, "ai"); }
                    return;
                }

                // 2. Director Mode (Ensemble)
                if (dtEnabled) {
                    await runDirectorMode(t);
                    return;
                }

                addLoading();
                
                // 3. Normal Routing
                const serverModels = ["gemini-3-flash-preview", "gemma-3-27b-it", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-flash-tts", "gemini-robotics-er-1.5-preview", "gemma-3-1b", "gemma-3-2b", "gemma-3-4b", "gemma-3-12b", "gemini-embedding-1.0", "gemini-2.5-flash-native-audio-dialog"];
                
                if (serverModels.includes(selectedChatModel)) {
                    // Python Server
                    let p = { prompt: t, history: [], model: selectedChatModel };
                    if(imgBase64) { p.image = imgBase64; imgBase64 = null; document.getElementById('previewContainer').style.display='none'; }
                    
                    fetch("/process_text", {
                        method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(p)
                    }).then(r=>r.json()).then(d => {
                        removeLoading();
                        // Marked.js handles parsing
                        addMsg(marked.parse(d.text || "Error"), "ai"); 
                    });
                } else {
                    // Puter.js
                    try {
                        let response;
                        if (imgBase64) {
                             response = await puter.ai.chat(t, "data:image/jpeg;base64," + imgBase64, { model: selectedChatModel });
                             imgBase64 = null; document.getElementById('previewContainer').style.display='none';
                        } else {
                             response = await puter.ai.chat(t, { model: selectedChatModel });
                        }
                        
                        removeLoading();
                        let text = extractText(response);
                        addMsg(marked.parse(text), "ai"); 

                    } catch(e) {
                        removeLoading();
                        addMsg("Error: " + e, "ai");
                    }
                }
            }

            function playTTS(text) {
                fetch("/generate_tts", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({text}) })
                .then(r=>r.json()).then(d => { if(d.audio) { let a=new Audio("data:audio/mp3;base64,"+d.audio); a.play(); } });
            }

            // --- LIVE CALL & FILE HANDLING ---
            let ws, audioCtx, mediaRecorder;
            async function startLiveCall() {
                document.getElementById('callModal').style.display = 'flex';
                document.getElementById('callStatus').innerText = "Connecting...";
                try {
                    audioCtx = new (window.AudioContext||window.webkitAudioContext)({sampleRate:24000});
                    let stream = await navigator.mediaDevices.getUserMedia({audio:{sampleRate:16000, channelCount:1}});
                    let proto = location.protocol==='https:'?'wss:':'ws:';
                    ws = new WebSocket(`${proto}//${location.host}/ws/live`);
                    
                    ws.onopen = () => {
                        document.getElementById('callStatus').innerText = "Live";
                        mediaRecorder = new MediaRecorder(stream, {mimeType:'audio/webm'});
                        mediaRecorder.ondataavailable = e => {
                            if(e.data.size>0 && ws.readyState===1) {
                                let r=new FileReader(); r.onload=()=>{ ws.send(JSON.stringify({audio:r.result.split(',')[1]})); }; r.readAsDataURL(e.data);
                            }
                        };
                        mediaRecorder.start(100);
                    };
                    ws.onmessage = e => {
                        let d=JSON.parse(e.data);
                        if(d.audio) playPCM(d.audio);
                        if(d.text) document.getElementById('callSub').innerText=d.text;
                    };
                    ws.onclose = endCall;
                } catch(e) { alert(e); endCall(); }
            }

            function playPCM(b64) {
                let bin=atob(b64), len=bin.length, bytes=new Uint8Array(len);
                for(let i=0; i<len; i++) bytes[i]=bin.charCodeAt(i);
                let float32=new Float32Array(new Int16Array(bytes.buffer).length);
                let buf=audioCtx.createBuffer(1, float32.length, 24000);
                buf.getChannelData(0).set(float32);
                let src=audioCtx.createBufferSource(); src.buffer=buf; src.connect(audioCtx.destination); src.start();
            }

            function endCall() {
                if(ws) ws.close(); if(mediaRecorder) mediaRecorder.stop(); if(audioCtx) audioCtx.close();
                document.getElementById('callModal').style.display='none';
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
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
