import os
import requests
import json
import threading
import time
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- CONFIGURATION ---
GEMINI_KEY = os.environ.get("GEMINI")

# --- THE FULL ROSTER ---
MODELS = {
    "DIRECTOR": "gemini-3-flash",              # Final Code Review
    "SEARCH": "gemini-1.5-pro",                # Research/Facts
    "ROBOTICS": "gemini-robotics-er-1.5-preview", # Physics constraints
    "DIALOG": "gemini-2.5-flash-native-audio-dialog", # Speech patterns
    "AUDIO": "gemini-2.5-flash-tts",           # SoundService logic
    "CREATIVE": "gemma-3-27b-it",              # Lore/Story
    "WORKER": "gemini-2.5-flash",              # Builder/Scripter
    "SCOUT": "gemini-2.5-flash-lite"           # Routing
}

# Global State
current_status = {
    "message": "System Ready",
    "logs": [],
    "final_code": None,
    "is_processing": False
}

def log_event(text, highlight=False):
    prefix = "✨ " if highlight else ">> "
    print(f"{prefix}{text}")
    current_status["logs"].append(f"{prefix}{text}")
    current_status["message"] = text

# --- ROBUST AI CALLER ---
def call_ai(primary_role, prompt, system_instruction):
    primary_model = MODELS.get(primary_role, MODELS["WORKER"])
    fallback_model = MODELS["WORKER"]

    def _send(model_name, p, s):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_KEY}"
        payload = { "contents": [{ "parts": [{ "text": f"{s}\n\nTask: {p}" }] }] }
        return requests.post(url, json=payload)

    # 1. Try Primary
    try:
        response = _send(primary_model, prompt, system_instruction)
        data = response.json()
        if "candidates" in data:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        
        err = data.get("error", {}).get("message", "Unknown Error")
        log_event(f"⚠️ {primary_role} Busy: {err}")
        log_event(f"🔄 Switching to Fallback Worker...")

    except Exception as e:
        log_event(f"❌ Connection Fail: {e}")

    # 2. Try Fallback
    try:
        response = _send(fallback_model, prompt, system_instruction)
        data = response.json()
        if "candidates" in data:
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except:
        log_event("💀 CRITICAL: All agents failed.")
    
    return None

# --- AGENT FUNCTIONS ---
def run_router(prompt):
    log_event(f"🔍 [SCOUT] Analyzing request...")
    instruction = (
        "Return JSON booleans: "
        "{ \"needs_search\": bool, \"needs_physics\": bool, \"needs_lore\": bool, "
        "\"needs_dialog\": bool, \"needs_sound\": bool, \"needs_build\": bool }"
    )
    res = call_ai("SCOUT", prompt, instruction)
    try:
        return json.loads(res.replace("```json", "").replace("```", ""))
    except:
        return {"needs_build": True}

def run_specialist(role, name, prompt, context=""):
    log_event(f"⚡ [{name}] Working...")
    inst = f"You are a Specialist. Context: {context}. Return ONLY Lua code."
    return call_ai(role, prompt, inst)

# --- TRUE DEEP THINK (Critique -> Fix) ---
def run_boss_review(code, prompt):
    log_event(f"🧐 [DIRECTOR] Analyzing code structure...")
    
    # Step 1: Critique
    critique_inst = (
        "You are a Senior Code Director. Look at this Lua code. "
        "Does it have syntax errors, logic flaws, or deprecated methods? "
        "If it is perfect, reply ONLY: 'PERFECT'. "
        "If it has errors, list them."
    )
    critique = call_ai("DIRECTOR", f"Request: {prompt}\nCode:\n{code}", critique_inst)

    if critique and "PERFECT" in critique.upper():
        log_event("✅ [DIRECTOR] Code is Flawless.")
        return code
    
    # Step 2: Fix
    log_event(f"🔧 [DIRECTOR] Fixing issues found...")
    fix_inst = (
        f"You are the Director. Fix these specific errors:\n{critique}\n"
        "Return ONLY the corrected Lua code."
    )
    return call_ai("DIRECTOR", f"Original Code:\n{code}", fix_inst)

# --- WEB SERVER ---
code_queue = []

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Neural Architect 3.0</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #050508;
                --card-bg: rgba(20, 20, 30, 0.6);
                --glass-border: rgba(255, 255, 255, 0.08);
                --primary: #00f2ea;
                --primary-glow: rgba(0, 242, 234, 0.4);
                --secondary: #7000ff;
                --secondary-glow: rgba(112, 0, 255, 0.4);
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
                justify-content: center;
                align-items: center;
                overflow-x: hidden;
                position: relative;
                padding: 20px;
            }

            .orb { position: absolute; border-radius: 50%; filter: blur(100px); opacity: 0.4; z-index: -1; animation: float 10s infinite alternate ease-in-out; }
            .orb-1 { width: 500px; height: 500px; background: var(--secondary); top: -10%; left: -10%; }
            .orb-2 { width: 400px; height: 400px; background: var(--primary); bottom: -10%; right: -10%; animation-delay: 2s; }
            @keyframes float { 0% { transform: translate(0,0); } 100% { transform: translate(30px, 30px); } }

            .interface {
                width: 100%; max-width: 550px;
                background: var(--card-bg);
                backdrop-filter: blur(24px);
                -webkit-backdrop-filter: blur(24px);
                border: 1px solid var(--glass-border);
                border-radius: 24px;
                padding: 40px;
                box-shadow: 0 20px 50px rgba(0,0,0,0.4);
                transition: transform 0.3s;
            }

            h1 {
                font-size: 24px; font-weight: 700; margin-bottom: 8px; letter-spacing: -0.5px;
                background: linear-gradient(135deg, #fff 0%, #aaa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                display: flex; align-items: center; gap: 12px;
            }
            .status-indicator { width: 8px; height: 8px; background: var(--primary); border-radius: 50%; box-shadow: 0 0 10px var(--primary); animation: pulse 2s infinite; }
            @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
            
            p.subtitle { color: var(--text-dim); font-size: 14px; margin-bottom: 32px; font-weight: 300; }

            .input-wrapper { position: relative; margin-bottom: 24px; }
            input[type="text"] {
                width: 100%; background: rgba(0,0,0,0.3); border: 2px solid var(--glass-border); padding: 18px;
                border-radius: 16px; color: #fff; font-size: 16px; font-family: 'Outfit', sans-serif; transition: all 0.3s ease;
            }
            input:focus { border-color: var(--primary); box-shadow: 0 0 20px var(--primary-glow); transform: translateY(-2px); }
            input::placeholder { color: rgba(255,255,255,0.2); }

            .controls { display: flex; align-items: center; justify-content: space-between; margin-bottom: 30px; background: rgba(255,255,255,0.03); padding: 12px 16px; border-radius: 14px; border: 1px solid var(--glass-border); }
            .label-group { display: flex; flex-direction: column; }
            .label-title { font-size: 14px; font-weight: 500; color: #fff; }
            .label-desc { font-size: 12px; color: var(--text-dim); margin-top: 2px; }

            .switch { position: relative; width: 50px; height: 28px; }
            .switch input { opacity: 0; width: 0; height: 0; }
            .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #333; transition: .4s; border-radius: 34px; }
            .slider:before { position: absolute; content: ""; height: 20px; width: 20px; left: 4px; bottom: 4px; background-color: white; transition: .4s; border-radius: 50%; box-shadow: 0 2px 5px rgba(0,0,0,0.3); }
            input:checked + .slider { background-color: var(--secondary); }
            input:checked + .slider:before { transform: translateX(22px); }
            input:checked ~ .slider { box-shadow: 0 0 15px var(--secondary-glow); }

            button {
                width: 100%; padding: 18px; border: none; border-radius: 16px;
                background: linear-gradient(135deg, var(--primary) 0%, #00c2bb 100%);
                color: #000; font-size: 16px; font-weight: 700; cursor: pointer;
                transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                box-shadow: 0 10px 20px rgba(0, 242, 234, 0.2);
            }
            button:hover { transform: translateY(-3px) scale(1.02); box-shadow: 0 15px 30px rgba(0, 242, 234, 0.4); }
            button:active { transform: scale(0.95); }
            button.loading { background: #333; color: #666; cursor: not-allowed; pointer-events: none; box-shadow: none; }

            .console-window {
                margin-top: 30px; background: var(--code-bg); border-radius: 12px; padding: 15px; height: 160px;
                overflow-y: auto; font-family: 'JetBrains Mono', monospace; font-size: 12px;
                border: 1px solid var(--glass-border); position: relative;
            }
            .console-window::-webkit-scrollbar { width: 5px; }
            .console-window::-webkit-scrollbar-thumb { background: #333; border-radius: 10px; }
            
            .log-entry { margin-bottom: 6px; display: flex; gap: 8px; }
            .log-msg { color: #aaa; }

            .code-panel {
                margin-top: 20px; background: var(--code-bg); border: 1px solid var(--primary); border-radius: 12px; padding: 20px;
                display: none; animation: slideUp 0.5s ease; position: relative; box-shadow: 0 0 30px rgba(0, 242, 234, 0.1);
            }
            .code-header { display: flex; justify-content: space-between; margin-bottom: 10px; color: var(--primary); font-size: 14px; font-weight: bold; text-transform: uppercase; }
            pre { margin: 0; white-space: pre-wrap; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #aaffaa; max-height: 300px; overflow-y: auto; }
            .copy-btn { position: absolute; top: 15px; right: 15px; width: auto; padding: 5px 12px; background: rgba(255,255,255,0.1); border-radius: 6px; font-size: 11px; color: white; border: 1px solid rgba(255,255,255,0.2); }
            .copy-btn:hover { background: rgba(255,255,255,0.2); transform: none; box-shadow: none; }

            @keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

        </style>
    </head>
    <body>

        <div class="orb orb-1"></div>
        <div class="orb orb-2"></div>

        <div class="interface">
            <h1><div class="status-indicator"></div> Neural Architect 3.0</h1>
            <p class="subtitle">Powered by Gemini Swarm (8-Core)</p>

            <div class="input-wrapper">
                <input type="text" id="prompt" placeholder="Describe your build..." autocomplete="off">
            </div>

            <div class="controls">
                <div class="label-group">
                    <span class="label-title">Deep Think Mode</span>
                    <span class="label-desc" id="modeDesc">Review by Gemini 3.0</span>
                </div>
                <label class="switch">
                    <input type="checkbox" id="deepThink" onchange="toggleGlow()">
                    <span class="slider"></span>
                </label>
            </div>

            <button id="genBtn" onclick="send()">Initialize Generation</button>

            <div class="console-window" id="console">
                <div class="log-entry"><span class="log-msg">System Ready...</span></div>
            </div>

            <div class="code-panel" id="codePanel">
                <div class="code-header">
                    <span>Generated Lua Source</span>
                    <button class="copy-btn" onclick="copyCode()">COPY</button>
                </div>
                <pre id="codeContent"></pre>
            </div>
        </div>

        <script>
            function log(txt) { 
                let c = document.getElementById("console");
                let html = `<div class="log-entry"><span class="log-msg">${txt}</span></div>`;
                c.innerHTML += html;
                c.scrollTop = c.scrollHeight;
            }

            function toggleGlow() {
                let check = document.getElementById("deepThink").checked;
                let card = document.querySelector(".interface");
                let desc = document.getElementById("modeDesc");
                
                if(check) {
                    card.style.boxShadow = "0 20px 60px rgba(112, 0, 255, 0.3)";
                    card.style.borderColor = "rgba(112, 0, 255, 0.3)";
                    desc.innerText = "Director Mode (Gemini 3.0) Active";
                    desc.style.color = "#d4b3ff";
                } else {
                    card.style.boxShadow = "0 20px 50px rgba(0,0,0,0.4)";
                    card.style.borderColor = "rgba(255, 255, 255, 0.08)";
                    desc.innerText = "Standard Swarm Mode";
                    desc.style.color = "#8888aa";
                }
            }

            function send() {
                let p = document.getElementById("prompt");
                let dt = document.getElementById("deepThink").checked;
                let btn = document.getElementById("genBtn");
                
                if(!p.value) return;

                btn.innerText = "Processing...";
                btn.classList.add("loading");
                document.getElementById("codePanel").style.display = "none";
                document.getElementById("codeContent").innerText = "";
                
                fetch("/process", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({prompt: p.value, deep_think: dt})
                }).then(r => {
                    p.value = "";
                    setTimeout(() => {
                        btn.innerText = "Initialize Generation";
                        btn.classList.remove("loading");
                    }, 2000);
                });
            }

            function copyCode() {
                let text = document.getElementById("codeContent").innerText;
                navigator.clipboard.writeText(text);
                alert("Copied!");
            }

            setInterval(() => {
                fetch("/status").then(r=>r.json()).then(d => {
                    if(d.logs.length > 0) {
                        let c = document.getElementById("console");
                        c.innerHTML = "";
                        d.logs.forEach(l => {
                            c.innerHTML += `<div class="log-entry"><span class="log-msg">${l}</span></div>`;
                        });
                        c.scrollTop = c.scrollHeight;
                    }

                    if(d.final_code && !d.is_processing) {
                        let panel = document.getElementById("codePanel");
                        if(panel.style.display === "none" || panel.style.display === "") {
                            document.getElementById("codeContent").innerText = d.final_code;
                            panel.style.display = "block";
                        }
                    }
                });
            }, 1000);
        </script>
    </body>
    </html>
    '''

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    prompt = data.get('prompt')
    deep_think = data.get('deep_think', False)
    
    current_status["logs"] = []
    current_status["final_code"] = None
    current_status["is_processing"] = True

    def task():
        log_event(f"🚀 Job Started: {prompt}")
        final_code = ""
        context = ""

        # 1. SCOUT
        plan = run_router(prompt)
        
        # 2. SEARCH
        if plan.get("needs_search"):
            context = call_ai("SEARCH", prompt, "You are a Research Engine. Brief technical facts.")
            log_event(f"🌍 Context Found: {len(context)} chars")

        # 3. BUILD
        if plan.get("needs_physics"):
            final_code += f"\n--[PHYSICS]\n{run_specialist('ROBOTICS', 'Robotics-ER', prompt, context)}\n"
        elif plan.get("needs_build"):
            final_code += f"\n--[BUILD]\n{run_specialist('WORKER', 'Architect', prompt, context)}\n"

        # 4. EXTRAS
        if plan.get("needs_lore"):
            final_code += f"\n--[LORE]\n{run_specialist('CREATIVE', 'Gemma', prompt)}\n"
        
        if plan.get("needs_dialog"):
            final_code += f"\n--[DIALOG]\n{run_specialist('DIALOG', 'Dialog-Coach', prompt)}\n"
            
        if plan.get("needs_sound"):
            final_code += f"\n--[SOUND]\n{run_specialist('AUDIO', 'Audio-Eng', prompt)}\n"
        
        # 5. LOGIC
        final_code += f"\n--[LOGIC]\n{run_specialist('WORKER', 'Scripter', prompt, f'Context: {context}. Object built.')}\n"

        # 6. BOSS REVIEW
        if deep_think:
            final_code = run_boss_review(final_code, prompt)

        clean_code = final_code.replace("```lua", "").replace("```", "")
        code_queue.append(clean_code)
        
        current_status["final_code"] = clean_code
        current_status["is_processing"] = False
        log_event("✅ DONE. Code available.")

    threading.Thread(target=task).start()
    return jsonify({"success": True})

@app.route('/status')
def status():
    return jsonify(current_status)

@app.route('/get_latest_code', methods=['GET'])
def get_code():
    if code_queue: return jsonify({"has_code": True, "code": code_queue.pop(0)})
    return jsonify({"has_code": False})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
