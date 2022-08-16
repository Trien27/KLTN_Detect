import winsound
from playsound import playsound
from gtts import gTTS
import os
def play_audio(path_of_audio):
    playsound(path_of_audio)
def convert_to_audio(text):
    audio = gTTS(text, lang="vi")
    audio.save('sound1.mp3')
    play_audio('sound1.mp3')
convert_to_audio("Tốc độ tối đa 120 km/h")