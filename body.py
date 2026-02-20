from flask import Flask, request, jsonify
from naoqi import ALProxy, ALModule, ALBroker
import time

app = Flask(__name__)

nao = True
nao_IP = "172.20.10.4"
nao_port = 9559
sleep_time = 0.01

tts = ALProxy("ALTextToSpeech", nao_IP, nao_port)
tts.setVolume(1.0)
tts.setLanguage("Italian")
animatedSpeech = ALProxy("ALAnimatedSpeech", nao_IP, nao_port)
behaviorManager = ALProxy("ALBehaviorManager", nao_IP, nao_port)
motion = ALProxy("ALMotion", nao_IP, nao_port)
posture = ALProxy("ALRobotPosture", nao_IP, nao_port)


class AudioCaptureModule(ALModule):
    def __init__(self, name):
        ALModule.__init__(self, name)
        self.audio_device = ALProxy("ALAudioDevice", nao_IP, nao_port)
        self.is_listening = False
        self.buffers = []

    def start_listening(self):
        self.audio_device.setClientPreferences(self.getName(), 16000, 3, 0)
        self.audio_device.subscribe(self.getName())
        self.is_listening = True

    def stop_listening(self):
        self.audio_device.unsubscribe(self.getName())
        self.is_listening = False

    def processRemote(self, nbOfChannels, nbOfSamplesByChannel, timeStamp, inputBuffer):
        print("received audio data from NAO with the following parameters: nbOfChannels = " + str(nbOfChannels) +
              ", nbOfSamplesByChannel = " + str(nbOfSamplesByChannel) +
              ", timeStamp = " + str(timeStamp[0]) + " sec " + str(timeStamp[1]) + " musec" +
              ", length of inputBuffer = " + str(len(inputBuffer)))
        if self.is_listening:
            self.buffers.append(inputBuffer)

    def get_audio_chunk(self):
        if self.buffers:
            return self.buffers.pop(0)
        else:
            print("no audio data available")
            return None


try:
    pythonBroker = ALBroker("pythonBroker", "0.0.0.0", 0, nao_IP, nao_port)
    global AudioCapture
    AudioCapture = AudioCaptureModule("AudioCapture")
    print("AudioCapture module initialized")
except RuntimeError:
    print("Error initializing broker!")
    exit(1)


# server endpoints ====================================================================================================

@app.route("/talk", methods=["POST"])
def talk():
    print("Received a request to talk")
    message = request.json.get("message")
    animatedSpeech.say(str(message))
    return jsonify(success=True)


@app.route("/run_behavior", methods=["POST"])
def run_behavior():
    behavior_name = str(request.json.get("behavior_name"))
    print("Received a request to run behavior: " + behavior_name)
    if not behavior_name:
        return jsonify(success=False, error="No behavior_name provided"), 400
    try:
        if behaviorManager.isBehaviorInstalled(behavior_name):
            behaviorManager.runBehavior(behavior_name)
            print("Behavior '" + behavior_name + "' executed successfully")
            return jsonify(success=True)
        else:
            print("Behavior '" + behavior_name + "' is not installed on the robot")
            return jsonify(success=False, error="Behavior not found"), 404
    except Exception as e:
        print("Error running behavior: " + str(e))
        return jsonify(success=False, error=str(e)), 500


@app.route("/move", methods=["POST"])
def move():
    data = request.json
    x     = float(data.get("x", 0.0))
    y     = float(data.get("y", 0.0))
    theta = float(data.get("theta", 0.0))
    print("Received move request: x=" + str(x) + " y=" + str(y) + " theta=" + str(theta))
    try:
        motion.setStiffnesses("Body", 1.0)
        posture.goToPosture("StandInit", 0.5)
        motion.moveTo(x, y, theta)
        print("Move completed successfully")
        return jsonify(success=True)
    except Exception as e:
        print("Error during move: " + str(e))
        return jsonify(success=False, error=str(e)), 500


@app.route("/list_behaviors", methods=["GET"])
def list_behaviors():
    print("Received a request to list installed behaviors")
    try:
        behaviors = behaviorManager.getInstalledBehaviors()
        return jsonify(behaviors=sorted(behaviors))
    except Exception as e:
        print("Error listing behaviors: " + str(e))
        return jsonify(success=False, error=str(e)), 500


@app.route("/start_listening", methods=["POST"])
def start_listening():
    print("Received a request to start listening, current length of server buffer: " + str(len(AudioCapture.buffers)))
    AudioCapture.start_listening()
    return jsonify(success=True)


@app.route("/stop_listening", methods=["POST"])
def stop_listening():
    print("Received a request to stop listening, current length of server buffer: " + str(len(AudioCapture.buffers)))
    AudioCapture.stop_listening()
    return jsonify(success=True)


@app.route("/get_audio_chunk", methods=["GET"])
def get_audio_chunk():
    print("Received a request to get an audio chunk, current length of server buffer: " + str(len(AudioCapture.buffers)))
    audio_data = AudioCapture.get_audio_chunk()
    if audio_data is not None:
        return audio_data
    else:
        print("Server buffer is empty, waiting for audio data...")
        while audio_data is None:
            audio_data = AudioCapture.get_audio_chunk()
            time.sleep(sleep_time)
        return audio_data


@app.route("/get_server_buffer_length", methods=["GET"])
def get_server_buffer_length():
    print("Received a request to print the length of the server buffer, current length of server buffer: " + str(len(AudioCapture.buffers)))
    return jsonify(length=len(AudioCapture.buffers))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004)