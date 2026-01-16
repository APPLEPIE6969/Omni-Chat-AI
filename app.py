from flask import Flask, request, jsonify

app = Flask(__name__)

# The storage for commands
command_queue = []

@app.route('/')
def home():
    return '''
    <html>
        <head>
            <title>Roblox AI Commander</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { font-family: sans-serif; background: #1a1a1a; color: #fff; text-align: center; padding: 20px; }
                input { padding: 15px; width: 80%; max-width: 400px; border-radius: 8px; border: none; font-size: 16px;}
                button { padding: 15px 30px; background: #00d26a; color: white; border: none; cursor: pointer; border-radius: 8px; font-size: 16px; margin-top: 10px; font-weight: bold;}
                button:active { background: #00a855; }
                #status { margin-top: 20px; color: #aaa; }
            </style>
        </head>
        <body>
            <h1>🤖 AI Builder</h1>
            <input type="text" id="cmd" placeholder="E.g., Make a giant gold statue...">
            <br>
            <button onclick="send()">Send to Roblox</button>
            <p id="status"></p>

            <script>
                function send() {
                    let txt = document.getElementById("cmd").value;
                    let btn = document.querySelector("button");
                    btn.innerText = "Sending...";
                    
                    fetch("/add_command", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({prompt: txt})
                    }).then(r => r.text()).then(t => {
                        document.getElementById("status").innerText = t;
                        btn.innerText = "Send to Roblox";
                        document.getElementById("cmd").value = "";
                    });
                }
            </script>
        </body>
    </html>
    '''

@app.route('/add_command', methods=['POST'])
def add_command():
    data = request.json
    prompt = data.get('prompt')
    if prompt:
        command_queue.append(prompt)
        return "Command Sent! Watch Roblox Studio."
    return "Error: Empty prompt"

@app.route('/get_latest', methods=['GET'])
def get_latest():
    if len(command_queue) > 0:
        cmd = command_queue.pop(0)
        return jsonify({"has_command": True, "prompt": cmd})
    else:
        return jsonify({"has_command": False})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
