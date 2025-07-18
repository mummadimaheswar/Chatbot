# app.py

from flask import Flask, render_template_string, request, jsonify
from chatbot import get_response

app = Flask(__name__)

# Simple HTML UI template
HTML_PAGE = """
<!doctype html>
<title>AI Chatbot</title>
<h1>AI-Powered Chatbot 🤖</h1>
<div style="max-width:600px;">
  <input id="user_input" type="text" style="width:80%;" placeholder="Ask something..." autofocus>
  <button onclick="send()">Send</button>
  <div id="chatbox" style="margin-top:20px;"></div>
</div>

<script>
function send() {
  var user_input = document.getElementById('user_input').value;
  fetch("/get_response", {
    method: "POST",
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({"message": user_input})
  })
  .then(res => res.json())
  .then(data => {
    var chat = document.getElementById('chatbox');
    chat.innerHTML += "<b>You:</b> " + user_input + "<br>";
    chat.innerHTML += "<b>Bot:</b> " + data.response + "<br><br>";
    document.getElementById('user_input').value = "";
  });
}
</script>
"""

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

@app.route("/get_response", methods=['POST'])
def respond():
    user_input = request.json['message']
    bot_response = get_response(user_input)
    return jsonify({'response': bot_response})

if __name__ == "__main__":
    app.run(debug=True)
