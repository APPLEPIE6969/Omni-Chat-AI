import os
import json
import base64
import asyncio
import requests
import markdown2
import io
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

# --- SERVER-SIDE MODEL CHAINS (Native Audio Only) ---
# Note: Text generation has moved primarily to Puter.js (Client) for free access.
# These server chains are kept for the "Live Call" feature which requires direct socket access.
MODEL_CHAINS = {
    "NATIVE_AUDIO": ["gemini-2.0-flash-exp"], 
    "NEURAL_TTS": ["gemini-2.5-flash-tts"]
}

# --- SERVER-SIDE MARKDOWN PARSING ---
# We keep this for any server-side rendered text, though mostly handled by JS now.
def parse_markdown(text):
    try:
        return markdown2.markdown(text, extras=["tables", "fenced-code-blocks", "strike", "break-on-newline"])
    except: return text

# --- HELPER: GTTS GENERATION ---
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- WEBSOCKET LIVE CALL (Server-Side) ---
@sock.route('/ws/live')
def live_socket(ws):
    client = genai.Client(api_key=GEMINI_KEY, http_options={'api_version': 'v1alpha'})
    
    LIVE_MODEL = MODEL_CHAINS["NATIVE_AUDIO"][0]
    
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"], 
        output_audio_transcription=types.AudioTranscriptionConfig()
    )
    
    async def session_loop():
        try:
            async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
                print(">> Connected to Gemini Live")
                
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
        <title>Omni-Chat</title>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
        <meta name="theme-color" content="#050508">
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        
        <!-- Puter.js -->
        <script src="https://js.puter.com/v2/"></script>
        <!-- Marked.js -->
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

        <style>
            :root { --bg: #050508; --header: rgba(20,20,30,0.95); --border: rgba(255,255,255,0.1); --primary: #00f2ea; --secondary: #7000ff; --text: #fff; }
            * { box-sizing: border-box; }
            body { background: var(--bg); color: var(--text); font-family: 'Outfit', sans-serif; height: 100dvh; display: flex; flex-direction: column; margin: 0; overflow: hidden; }

            .orb { position: absolute; border-radius: 50%; filter: blur(80px); opacity: 0.3; z-index: -1; animation: float 10s infinite alternate; }
            .orb-1 { width: 400px; height: 400px; background: var(--secondary); top: -10%; left: -10%; }
            .orb-2 { width: 300px; height: 300px; background: var(--primary); bottom: -10%; right: -10%; animation-delay: 2s; }
            @keyframes float { 0% { transform: translate(0,0); } 100% { transform: translate(30px, 30px); } }

            .header { padding: 10px 15px; background: var(--header); border-bottom: 1px solid var(--border); z-index: 10; display: flex; flex-direction: column; gap: 5px; }
            .top { display: flex; justify-content: space-between; align-items: center; }
            .brand { font-weight: 700; font-size: 18px; display: flex; gap: 10px; align-items: center; }
            .dot { width: 8px; height: 8px; background: var(--primary); border-radius: 50%; box-shadow: 0 0 10px var(--primary); animation: pulse 2s infinite; }
            
            .model-select { background: rgba(0,0,0,0.3); border: 1px solid var(--border); border-radius: 20px; color: #aaa; padding: 5px 15px; font-size: 12px; cursor: pointer; display: flex; align-items: center; gap: 5px; transition: 0.2s; }
            .model-select:hover { border-color: var(--primary); color: white; }

            /* DIRECTOR MODE TOGGLE */
            .dt-toggle { font-size: 11px; color: #666; display: flex; align-items: center; gap: 6px; cursor: pointer; margin-left: 20px; width: fit-content; transition: 0.3s; }
            .dt-box { width: 14px; height: 14px; border: 1px solid #444; border-radius: 3px; display: flex; align-items: center; justify-content: center; transition: 0.3s; background: #111; }
            .dt-toggle.active { color: #ffd700; text-shadow: 0 0 10px rgba(255, 215, 0, 0.3); }
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
                <div class="model-select" onclick="openModelModal()">
                    <span id="currentModelDisplay">Gemini 3.0</span> <i class="fa-solid fa-chevron-down"></i>
                </div>
            </div>
            <!-- DIRECTOR MODE TOGGLE -->
            <div class="dt-toggle" id="dtToggle" onclick="toggleDT()">
                <div class="dt-box"><i class="fa-solid fa-check" style="display:none" id="dtCheck"></i></div> Director Mode (Ensemble)
            </div>
        </div>

        <div class="chat" id="chat"><div class="msg ai">Online. Director Mode uses GPT-5.2 Pro, Claude Opus 4.5, and Gemini 3 Pro.</div></div>

        <div class="input-area">
            <input type="file" id="fileInput" accept="image/*" onchange="handleFile(this)">
            <div id="previewContainer"><img id="imageUploadPreview"></div>
            <button class="icon-btn" onclick="openImgModal()"><i class="fa-solid fa-palette"></i></button>
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
            // --- STATE ---
            let selectedChatModel = "gemini-3-flash-preview"; 
            let selectedImgModel = "black-forest-labs/FLUX.1-schnell";
            let dtEnabled = false; 
            let imgBase64 = null;
            let chatHistory = [];

            // --- MODEL LISTS (All Puter + Your Google) ---
            const chatModels = [
                // Server (Native)
                {id: "gemini-3-flash-preview", name: "Gemini 3.0", tag: "⚡ GOOGLE"},
                {id: "gemma-3-27b-it", name: "Gemma 3 27B", tag: "🔓 OPEN"},
                
                // Puter - OpenAI
                {id: "gpt-5.2-codex", name: "GPT 5.2 Codex", tag: "💻 CODE"},
                {id: "gpt-5.2-pro", name: "GPT 5.2 Pro", tag: "👑 ULTIMATE"},
                {id: "gpt-5-nano", name: "GPT-5 Nano", tag: "⚡ FAST"},
                {id: "o3", name: "OpenAI o3", tag: "🤯 REASON"},
                {id: "gpt-4o", name: "GPT-4o", tag: "🔥 BEST"},
                
                // Puter - Claude Suite (NEW)
                {id: "claude-opus-4-5", name: "Claude Opus 4.5", tag: "💎 OPUS"},
                {id: "claude-sonnet-4-5", name: "Claude Sonnet 4.5", tag: "📚 SONNET"},
                {id: "claude-haiku-4-5", name: "Claude Haiku 4.5", tag: "💨 HAIKU"},
                {id: "claude-opus-4-1", name: "Claude Opus 4.1", tag: "💎 OPUS"},
                {id: "claude-sonnet-4", name: "Claude Sonnet 4", tag: "📚 SONNET"},
                
                // Puter - Gemini
                {id: "gemini-3-pro-preview", name: "Gemini 3 Pro", tag: "🧠 PUTER"},
                {id: "gemini-2.5-pro", name: "Gemini 2.5 Pro", tag: "⭐ PUTER"},
                {id: "gemini-2.5-flash-lite", name: "Gemini 2.5 Lite", tag: "⚡ PUTER"},
                {id: "gemini-2.5-flash", name: "Gemini 2.5 Flash", tag: "⚡ PUTER"}
            ];

            const imgModels = [
                {id: "black-forest-labs/FLUX.1-schnell", name: "Flux Schnell", tag: "🚀 FAST"},
                {id: "black-forest-labs/FLUX.1.1-pro", name: "Flux 1.1 Pro", tag: "⭐ BEST"},
                {id: "dall-e-3", name: "DALL-E 3", tag: "🧠 SMART"},
                {id: "ideogram/ideogram-3.0", name: "Ideogram 3", tag: "🔤 TEXT"},
                {id: "google/imagen-4.0-ultra", name: "Imagen 4 Ultra", tag: "⭐ GOOGLE"},
                {id: "gpt-image-1", name: "GPT Image", tag: "🤖 GPT"}
            ];

            // --- UI LOGIC ---
            function toggleDT() {
                dtEnabled = !dtEnabled;
                const el = document.getElementById("dtToggle");
                const icon = document.getElementById("dtCheck");
                if (dtEnabled) { el.classList.add("active"); icon.style.display = "block"; }
                else { el.classList.remove("active"); icon.style.display = "none"; }
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
            function closeModelModal() { document.getElementById("modelModal").style.display = "none"; }

            function openImgModal() {
                renderList(imgModels, "imgModelList", selectedImgModel, (id, name) => {
                    selectedImgModel = id;
                    closeImgSettings();
                });
                document.getElementById("imgModal").style.display = "flex";
            }
            function closeImgSettings() { document.getElementById("imgModal").style.display = "none"; }

            // --- HELPER: SAFE TEXT EXTRACTION FROM PUTER ---
            function extractText(res) {
                // Claude sometimes returns array, sometimes string
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
                    if(isHtml) cDiv.innerHTML = content; else cDiv.innerText = content;
                    d.appendChild(cDiv);
                    if(type === 'ai') addCopyBtns(cDiv);
                } else d.appendChild(content);
                
                if (type === "ai" && typeof content === 'string') {
                    let b = document.createElement("button"); b.className="tts-btn"; b.innerHTML='<i class="fa-solid fa-volume-high"></i>'; 
                    b.onclick=()=>playTTS(d.innerText); d.appendChild(b);
                }
                
                let c = document.getElementById("chat");
                c.appendChild(d); c.scrollTop = c.scrollHeight;
            }

            function addLoading(text="Thinking...") {
                let d = document.createElement("div"); d.className="msg ai loading"; d.id="load";
                d.innerHTML = `${text} <div class="spinner"></div>`;
                document.getElementById("chat").appendChild(d);
                document.getElementById("chat").scrollTop = document.getElementById("chat").scrollHeight;
            }
            function removeLoading() { let e=document.getElementById("load"); if(e) e.remove(); }

            // --- CLIENT-SIDE DIRECTOR MODE (ENSEMBLE) ---
            async function runDirectorMode(prompt) {
                addLoading("Consulting Experts...");
                
                const experts = [
                    { model: 'gpt-5.2-pro', name: 'GPT-5.2 Pro' },
                    { model: 'claude-opus-4-5', name: 'Claude Opus 4.5' },
                    { model: 'gemini-3-pro-preview', name: 'Gemini 3 Pro' }
                ];

                try {
                    // 1. Parallel Expert Calls
                    const promises = experts.map(exp => 
                        puter.ai.chat(prompt, { model: exp.model })
                            .then(res => `--- Expert: ${exp.name} ---\n${extractText(res)}\n`)
                            .catch(err => `--- Expert: ${exp.name} ---\nFailed: ${err}\n`)
                    );
                    
                    const results = await Promise.all(promises);
                    const rawData = results.join("\n");

                    // 2. Synthesis (Gemini 3 Pro)
                    removeLoading();
                    addLoading("Synthesizing Final Answer...");
                    
                    const finalPrompt = `
                        USER QUERY: ${prompt}
                        
                        EXPERTS OPINIONS:
                        ${rawData}
                        
                        INSTRUCTION: You are the Director. Combine these expert opinions into one single, masterfully written, perfect response. 
                        Do not mention 'Draft 1' or 'the experts'. Just write the best answer possible.
                    `;

                    const synthesis = await puter.ai.chat(finalPrompt, { model: 'gemini-3-pro-preview' });
                    const finalText = extractText(synthesis);

                    removeLoading();
                    addMsg(marked.parse(finalText), "ai", true);
                    
                } catch (e) {
                    removeLoading();
                    addMsg("Director Mode Failed: " + e, "ai");
                }
            }

            // --- MAIN CHAT LOGIC ---
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
                
                // 1. Image Generation
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

                // 2. Director Mode (Client-Side Ensemble)
                if (dtEnabled) {
                    await runDirectorMode(t);
                    return;
                }

                addLoading();
                
                // 3. Normal Routing
                const serverModels = ["gemini-3-flash-preview", "gemma-3-27b-it"];
                
                if (serverModels.includes(selectedChatModel)) {
                    // Python Server (Your Key)
                    let p = { prompt: t, history: chatHistory, model: selectedChatModel };
                    if(imgBase64) { p.image = imgBase64; imgBase64 = null; document.getElementById('previewContainer').style.display='none'; }
                    
                    fetch("/process_text", {
                        method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(p)
                    }).then(r=>r.json()).then(d => {
                        removeLoading();
                        addMsg(d.html || d.text, "ai", true); 
                    });
                } else {
                    // Puter.js (Client Keyless)
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
                        addMsg(marked.parse(text), "ai", true); 

                    } catch(e) {
                        removeLoading();
                        addMsg("Puter Error: " + e, "ai");
                    }
                }
            }

            function playTTS(text) {
                fetch("/generate_tts", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({text}) })
                .then(r=>r.json()).then(d => {
                    if(d.audio) { let a=new Audio("data:audio/mp3;base64,"+d.audio); a.play(); }
                });
            }

            // --- LIVE CALL ---
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
                let int16=new Int16Array(bytes.buffer);
                for(let i=0; i<int16.length; i++) float32[i]=int16[i]/32768;
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

@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.json
    history = data.get('history', [])
    prompt = data.get('prompt')
    m = data.get('model')
    img = data.get('image')
    
    # Server side text processing handles server models only now
    from flask import current_app
    
    # Simple direct call to server models
    def try_chain(model_id, payload):
        # Determine chain (Gemini or Gemma)
        models = ["gemini-3-flash-preview"] # Default fallback
        if "gemma" in model_id: models = ["gemma-3-27b-it", "gemma-3-12b-it"]
        elif "gemini" in model_id: models = ["gemini-3-flash-preview", "gemini-2.5-flash"]
        
        for m in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={GEMINI_KEY}"
            try:
                r = requests.post(url, json=payload)
                if r.status_code==200: return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            except: continue
        return "Error: Server models unavailable."

    if image_data := img:
        history.append({"role": "user", "parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": image_data}}]})
    else:
        history.append({"role": "user", "parts": [{"text": prompt}]})

    res = try_chain(m, {"contents": history})
    return jsonify({"text": res, "html": parse_markdown(res)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
