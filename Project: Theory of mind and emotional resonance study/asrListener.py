import speech_recognition as sr
import pyttsx3
import os 

class Listener():
    def __init__(self):
        self.is_listening = True
        self.recognizer = sr.Recognizer()
        self.engine_speaker = pyttsx3.init()
        self.engine_speaker.setProperty('rate', self.engine_speaker.getProperty('rate')-50)
        self.engine_speaker.setProperty('voice', "english")

    def speak(self, text = "Hello world!"):
        self.engine_speaker.say(text)
        self.engine_speaker.runAndWait()

    def _testVoices(self):
        voices = self.engine_speaker.getProperty('voices')
     
        for voice in voices:
            self.engine_speaker.setProperty('voice', voice.id)
            print("voice_id {}".format(voice.id))
            self.engine_speaker.say('With great powers comes great responsibilities')
        self.engine_speaker.runAndWait()

    def recognize_mic(self):
        message = ""
        self.is_listening = True
        while self.is_listening:
            print("try")
            message = self.listen()
        return message

    def listen(self, speak_it = True, reproduce_audio = False):
        self.is_listening = True
        
        # audio input taken directly from the microphone (online manner)
        with sr.Microphone() as audioSource:
            text = ""
            # adjust the energy threshold dynamically
            print("Calibrating for background noise ...")
            self.recognizer.adjust_for_ambient_noise(audioSource, duration=1)
            print("Calibration completed, now you can speak.")
            # set the minimum length of silence after having spoken, it means the end of registration
            self.recognizer.pause_threshold = 0.5
            self.recognizer.energy_threshold = 400
            self.recognizer.dynamic_energy_threshold = True  
            # sample the audio source
            audio= self.recognizer.listen(audioSource)
            try:
                # set as recognizer the google one
                text = self.recognizer.recognize_google(audio)
                text = text.lower()
                print("audio input: {}".format(text))
            except:
                print("audio input not recognized, try again please")
                self.is_listening = False
        
        if text != "" and reproduce_audio: self.speak(text)
        return text
    
    
    def recognize_audio(self, name, path="", reproduce_audio = False):
        path_to_file = os.path.join(path,name)
        audio_file = sr.AudioFile(path_to_file)
        with audio_file as audioSource:
            audio = self.recognizer.record(audioSource)
            
        text = self.recognizer.recognize_google(audio)
        # print(text)
        if reproduce_audio: self.speak(text)
        return text 
    

new = Listener()
# new.speak()
new.recognize_audio("test.wav")
