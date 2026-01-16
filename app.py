import os
import requests
import json
import threading
import time
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- CONFIGURATION ---
GEMINI_KEY = os.environ.get("GEMINI")
MODEL_ROSTER = ["gemini-1.5-flash", "gemini-2.5-flash"]

# Status Tracking
current_status = {
    "message": "System Ready",
    "active_model": "None",
    "agent": "Idle",
    "logs": []
}

def log_event(text):
    print(text)
    current_status["logs"].append(text)
    current_status["message"] = text

# --- THE CORE AI ENGINE ---
def call_gemini(prompt, system_role):
    for model in MODEL_ROSTER:
        current_status["active_model"] = model
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}"
        payload = { "contents": [{ "parts": [{ "text": f"{system_role}\n\nTask: {prompt}" }] }] }
        try:
            r = requests.post(url, json=payload)
            data = r.json()
            if "candidates" in data: return data["candidates"][0]["content"]["parts"][0]["text"]
        except: continue
    return None

# --- THE SELF-REFLECTION LOOP ---
def generate_with_reflection(prompt, role, deep_think=False):
    # 1. First Draft
    log_event("📝 Generating Initial Draft...")
    draft_code = call_gemini(prompt, role)
    
    if not deep_think or not draft_code:
        return draft_code

    # 2. The Critique (The AI looks at its own work)
    log_event("🤔 Deep Think: Reviewing code for errors...")
    critique_prompt = (
        f"You are a Senior Roblox Code Reviewer. Look at this Lua code:\n\n{draft_code}\n\n"
        f"The user asked for: '{prompt}'.\n"
        "Identify any logic errors, missing 'end' statements, or physics issues. "
        "If the code is 100% perfect, reply ONLY with the word 'PERFECT'. "
        "If it has issues, describe them briefly."
    )
    critique = call_gemini(critique_prompt, "You are a critical code reviewer.")

    if "PERFECT" in critique.upper():
        log_event("✅ Deep Think: Code looks good!")
        return draft_code
    
    # 3. The Refinement (The AI fixes the mistakes)
    log_event(f"🔧 Deep Think: Fixing issues found: {critique[:50]}...")
    fix_prompt = (
        f"Original Request: {prompt}\n"
        f"Draft Code: {draft_code}\n"
        f"Reviewer Feedback: {critique}\n\n"
        "Rewrite the Lua code completely to fix these issues. Return ONLY the code."
    )
    final_code = call_gemini(fix_prompt, role)
    
    return final_code

# --- THE AGENTS ---
def run_architect(prompt, deep_think):
    current_status["agent"] = "Architect"
    instruction = (
        "You are a Roblox Builder. You cannot upload meshes. "
        "Build the requested object using ONLY Instance.new('Part'). "
        "Set Size, Position, Color, Anchored = true. Group into a Model. Return ONLY Lua."
    )
    return generate_with_reflection(prompt, instruction, deep_think)

def run_scripter(prompt, context, deep_think):
    current_status["agent"] = "Scripter"
    instruction = (
        f"You are a Roblox Scripter. Write valid Lua code. Context: {context}. "
        "Ensure all variables are defined. Return ONLY Lua."
    )
    return generate_with_reflection(prompt, instruction, deep_think)

# --- WEB SERVER ---
code_queue = []

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>AI Architect Pro</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">
        <style>
            :root { --accent: #00ff88; --bg: #0f0f0f; --panel: #1e1e1e; }
            body { background: var(--bg); color: white; font-family: 'Inter', sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
            .container { width: 400px; background: var(--panel); padding: 30px; border-radius: 20px; box-shadow: 0 10px 40px rgba(0,0,0,0.5); border: 1px solid #333; }
            h1 { font-size: 20px; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
            .status-dot { width: 10px; height: 10px; background: var(--accent); border-radius: 50%; box-shadow: 0 0 10px var(--accent); }
            
            /* Input Area */
            input[type="text"] { width: 100%; padding: 15px; background: #000; border: 1px solid #333; color: white; border-radius: 10px; box-sizing: border-box; font-size: 16px; outline: none; transition: 0.3s; }
            input[type="text"]:focus { border-color: var(--accent); }

            /* Toggle Switch */
            .toggle-container { display: flex; align-items: center; justify-content: space-between; margin: 20px 0; background: #252525; padding: 10px 15px; border-radius: 10px; }
            .switch { position: relative; display: inline-block; width: 50px; height: 26px; }
            .switch input { opacity: 0; width: 0; height: 0; }
            .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #444; transition: .4s; border-radius: 34px; }
            .slider:before { position: absolute; content: ""; height: 20px; width: 20px; left: 3px; bottom: 3px; background-color: white; transition: .4s; border-radius: 50%; }
            input:checked + .slider { background-color: var(--accent); }
            input:checked + .slider:before { transform: translateX(24px); }
            .label-text { font-size: 14px; color: #ccc; }

            /* Console */
            #console { height: 150px; overflow-y: auto; font-family: monospace; font-size: 12px; color: #888; margin-top: 20px; border-top: 1px solid #333; padding-top: 10px; }
            .log-new { color: var(--accent); }

            button { width: 100%; padding: 15px; background: linear-gradient(135deg, var(--accent), #00cc6a); border: none; border-radius: 10px; color: black; font-weight: bold; cursor: pointer; margin-top: 10px; font-size: 16px; }
            button:active { transform: scale(0.98); }
        </style>
    </head>
    <body>
        <div class="container">
            <h1><div class="status-dot"></div> Roblox Architect</h1>
            
            <input type="text" id="prompt" placeholder="Describe your build...">
            
            <div class="toggle-container">
                <span class="label-text">🧠 Deep Think (Self-Correction)</span>
                <label class="switch">
                    <input type="checkbox" id="deepThink">
                    <span class="slider"></span>
                </label>
            </div>

            <button onclick="send()">Generate</button>
            <div id="console">System Ready...</div>
        </div>

        <script>
            function log(txt) { 
                let c = document.getElementById("console");
                c.innerHTML += `<div class="log-new">> ${txt}</div>`;
                c.scrollTop = c.scrollHeight;
            }

            function send() {
                let p = document.getElementById("prompt").value;
                let dt = document.getElementById("deepThink").checked;
                if(!p) return;
                
                log(dt ? "Sending (Deep Think ON)..." : "Sending (Fast Mode)...");
                
                fetch("/process", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({prompt: p, deep_think: dt})
                });
                document.getElementById("prompt").value = "";
            }

            setInterval(() => {
                fetch("/status").then(r=>r.json()).then(d => {
                    if(d.logs.length > 0) d.logs.forEach(l => log(l));
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
    deep_think = data.get('deep_think', False) # Read the switch
    current_status["logs"] = []

    def task():
        # 1. Build
        build_code = run_architect(prompt, deep_think)
        # 2. Script
        script_code = run_scripter(prompt, "Object is built.", deep_think)
        
        final = ""
        if build_code: final += f"\n{build_code}\n"
        if script_code: final += f"\n{script_code}\n"
        
        clean = final.replace("```lua", "").replace("```", "")
        code_queue.append(clean)
        log_event("✨ Process Complete.")

    threading.Thread(target=task).start()
    return jsonify({"success": True})

@app.route('/status')
def status():
    l = list(current_status["logs"])
    current_status["logs"] = []
    return jsonify({"logs": l})

@app.route('/get_latest_code', methods=['GET'])
def get_code():
    if code_queue: return jsonify({"has_code": True, "code": code_queue.pop(0)})
    return jsonify({"has_code": False})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
