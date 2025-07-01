from subprocess import call
import speech_recognition as sr
import serial
import RPi.GPIO as GPIO
#GPIO Mode (BOARD / BCM)
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
Motor1A = 26
Motor1B = 19
Motor2A = 13
Motor2B = 6

			# GPIO Numbering
GPIO.setup(Motor1A,GPIO.OUT)  # All pins as Outputs
GPIO.setup(Motor1B,GPIO.OUT)
GPIO.setup(Motor2A,GPIO.OUT)  # All pins as Outputs
GPIO.setup(Motor2B,GPIO.OUT)
def sound():
    r = sr.Recognizer()
    print("Please talk...")
    with sr.Microphone() as source:
        #read the audio data from the default microphone
        audio_data = r.record(source, duration=5)
        print("Recognizing...",audio_data)
        #os.save("v.mp3")
        # convert speech to text
        data = r.recognize_google(audio_data)
        print("Recognised Speech:"+data)
        if data =='forward':
            GPIO.output(Motor1A,GPIO.LOW)
            GPIO.output(Motor1B,GPIO.HIGH)
            GPIO.output(Motor2A,GPIO.HIGH)
            GPIO.output(Motor2B,GPIO.LOW)
        if (data == 'backward'):
            GPIO.output(Motor1A,GPIO.HIGH)
            GPIO.output(Motor1B,GPIO.LOW)
            GPIO.output(Motor2A,GPIO.LOW)
            GPIO.output(Motor2B,GPIO.HIGH)
        if (data == 'right'):
            GPIO.output(Motor1A,GPIO.LOW)
            GPIO.output(Motor1B,GPIO.HIGH)
            GPIO.output(Motor2A,GPIO.LOW)
            GPIO.output(Motor2B,GPIO.LOW)
        if (data == 'left'):
            GPIO.output(Motor1A,GPIO.LOW)
            GPIO.output(Motor1B,GPIO.LOW)
            GPIO.output(Motor2A,GPIO.HIGH)
            GPIO.output(Motor2B,GPIO.LOW)
        if (data == 'stop'):
            GPIO.output(Motor1A,GPIO.LOW)
            GPIO.output(Motor1B,GPIO.LOW)
            GPIO.output(Motor2A,GPIO.LOW)
            GPIO.output(Motor2B,GPIO.LOW)
        
        
while True:
    sound()
    
    
        

