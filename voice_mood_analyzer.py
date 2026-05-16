# Voice-enabled Mood Analyzer — speech recognition + sentiment analysis + Tkinter GUI
# Made By Riza, Habil, Rihan

import speech_recognition as sr
import tkinter as tk
from textblob import TextBlob
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

mood_data = {'Happy': 0, 'Sad': 0, 'Neutral': 0}
# Store (mood_label, sentiment_score) tuples so we can average numerically
weekly_mood = []  # list of (label, score) tuples

def update_mood(text):
    global mood_data
    global weekly_mood

    analysis = TextBlob(text)
    sentiment = analysis.sentiment.polarity

    if sentiment > 0:
        mood = 'Happy'
    elif sentiment < 0:
        mood = 'Sad'
    else:
        mood = 'Neutral'

    mood_data[mood] += 1
    weekly_mood.append((mood, sentiment))  # store tuple for numeric averaging

def get_weekly_mood():
    """Compute the overall mood for the current session from stored (label, score) tuples."""
    if not weekly_mood:
        return "No data"

    # Average the numeric sentiment scores — no datetime comparison needed
    scores = [score for _, score in weekly_mood]
    avg_score = sum(scores) / len(scores)

    if avg_score > 0:
        return 'Happy'
    elif avg_score < 0:
        return 'Sad'
    else:
        return 'Neutral'

def record_and_analyze():
    record_button.config(state='disabled', text='Listening…')
    root.update_idletasks()
    user_input = speech_to_text()
    if not user_input.startswith("Speech not recognized") and not user_input.startswith("Could not"):
        update_mood(user_input)
        status_label.config(text=f'Heard: "{user_input}"')
    else:
        status_label.config(text=f'⚠️ {user_input}')
    update_gui()
    record_button.config(state='normal', text='Record and Analyze')

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something about your day...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Speech not recognized"
    except sr.RequestError as e:
        return f"Could not request results; {e}"

def update_gui():
    pie_chart.clear()
    # Only draw the pie if we have any data; avoids a matplotlib warning on empty data
    if sum(mood_data.values()) > 0:
        pie_chart.pie(mood_data.values(), labels=mood_data.keys(), autopct='%1.1f%%', startangle=140)
        pie_chart.axis('equal')
    canvas.draw()
    weekly_label.config(text="Session Overall Mood: " + get_weekly_mood())

root = tk.Tk()
root.title("Mood Analyzer")

record_button = tk.Button(root, text="Record and Analyze", command=record_and_analyze)
record_button.pack(pady=6)

# Status label — shows what was heard or any speech errors
status_label = tk.Label(root, text='Press the button and speak about your day.', wraplength=320, fg='gray')
status_label.pack()

chart_frame = ttk.Frame(root)
chart_frame.pack()

fig = Figure(figsize=(4, 4))
pie_chart = fig.add_subplot(111)

canvas = FigureCanvasTkAgg(fig, chart_frame)
canvas.get_tk_widget().pack()

# Weekly label is updated dynamically after each recording
weekly_label = tk.Label(root, text="Session Overall Mood: " + get_weekly_mood())
weekly_label.pack()

root.mainloop()
