from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import os
from chatbot import Chatbot

app = Flask(__name__, static_folder='../build', static_url_path='/')
CORS(app)

bot = Chatbot()

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/home')
def home():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/AppDevelopment')
def AppDevelopment():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/CloudServices')
def CloudServices():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/InteractiveDashboards')
def InteractiveDashboards():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/AIservices')
def AIservices():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/about')
def about():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/services')
def services():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/career')
def career():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/contact')
def contact():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    if request.is_json:
        data = request.get_json()
        user_message = data.get('message', '')
        bot_response = bot.process_input(user_message)
        return jsonify({'response': bot_response})
    else:
        return jsonify({'error': 'Invalid request format. JSON expected.'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
