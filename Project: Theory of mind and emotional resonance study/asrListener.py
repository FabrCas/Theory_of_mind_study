"""
check the library to use, also for extracting test from static file and not only from a real-time vocal stream of data
"""


import speech_recognition as sr
from colorama import init, Fore, Back, Style

class Listener():
    def __init__(self):
        self.ok = True

    def startListening(self):
        cmd = ""
        self.ok = True
        while self.ok:
            cmd = self.listen()
        return cmd


    def listen(self):
        self.ok = True
        recognizer = sr.Recognizer()
        # audio input taken directly from the microphone (online manner)
        with sr.Microphone() as audioSource:
            print("Speak now: ")
            text = ""
            # adjust the energy threshold dynamically
            recognizer.adjust_for_ambient_noise(audioSource, duration=0.5)
            # set the minimum length of silence after having spoken, it means the end of registration
            recognizer.pause_threshold = 1
            # sample the audio source
            audio= recognizer.listen(audioSource)
            try:
                # set as recognizer the google one
                text = recognizer.recognize_google(audio)
                print(Fore.LIGHTGREEN_EX +"audio input: {}".format(text))
            except:
                print(Fore.LIGHTRED_EX + "audio input not recognized, try again please")
                self.ok = False

        return text
