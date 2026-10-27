import os, subprocess, asyncio, tempfile
import numpy as np
import discord
from discord.ext import commands
from PIL import Image, ImageOps
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import speech_recognition as sr
from tf_keras.models import load_model

TOKEN = ""  # Discord bot token

model = load_model("model/keras_model.h5", compile=False)
labels = [l.strip().split(" ", 1)[1] for l in open("model/labels.txt")]

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


# ---------- model ----------
def classify(path):
    img = ImageOps.fit(Image.open(path).convert("RGB"), (224, 224), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1
    pred = model.predict(arr[None], verbose=0)[0]
    i = int(pred.argmax())
    return labels[i], float(pred[i])


# ---------- scipy sound feedback ----------
# Higher confidence = higher motor whine. Low confidence = low hum.
def make_motor_sound(confidence, path):
    rate, dur = 44100, 1.5
    t = np.linspace(0, dur, int(rate * dur), endpoint=False)
    freq = (150 + confidence * 700) * (0.5 + 0.5 * t / dur)
    phase = 2 * np.pi * np.cumsum(freq) / rate
    wave = 0.6 * np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
    b, a = butter(4, 3000 / (rate / 2))
    wave = lfilter(b, a, wave)
    wave *= np.minimum(1, t * 10) * np.minimum(1, (dur - t) * 5)
    wavfile.write(path, rate, (wave / np.abs(wave).max() * 32767 * 0.8).astype(np.int16))


# ---------- speech recognition ----------
def transcribe(path):
    wav = path + ".wav"
    subprocess.run(["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", wav],
                   check=True, capture_output=True)
    r = sr.Recognizer()
    with sr.AudioFile(wav) as src:
        audio = r.record(src)
    return r.recognize_google(audio, language="en-US")


def respond(text):
    if "range" in text:
        return "Use `!range <battery_Wh> <Wh_per_km>` to estimate range."
    if "identify" in text or "what bike" in text:
        return "Send me a photo with `!identify`."
    return "Try saying 'range' or 'identify'."


# ---------- bot ----------
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")


# !identify + attach a photo
@bot.command()
async def identify(ctx):
    if not ctx.message.attachments:
        return await ctx.send("Attach a photo, rider \U0001F3CD\uFE0F\u26A1")
    with tempfile.TemporaryDirectory() as d:
        img_path, snd_path = f"{d}/in.jpg", f"{d}/out.wav"
        await ctx.message.attachments[0].save(img_path)
        label, conf = await asyncio.to_thread(classify, img_path)
        make_motor_sound(conf, snd_path)
        await ctx.send(f"That's a **{label}** ({conf:.0%} sure)",
                       file=discord.File(snd_path, "feedback.wav"))


# !listen + attach a voice message or audio file
@bot.command()
async def listen(ctx):
    if not ctx.message.attachments:
        return await ctx.send("Attach a voice clip.")
    with tempfile.TemporaryDirectory() as d:
        src = f"{d}/clip"
        await ctx.message.attachments[0].save(src)
        try:
            text = await asyncio.to_thread(transcribe, src)
        except sr.UnknownValueError:
            return await ctx.send("Couldn't understand that, too much wind noise \U0001F4A8")
        await ctx.send(f"You said: *{text}*\n{respond(text.lower())}")


# !range <battery_Wh> <Wh_per_km>
@bot.command(name="range")
async def range_cmd(ctx, battery_wh: float, wh_per_km: float):
    km = battery_wh / wh_per_km * 0.9
    await ctx.send(f"Estimated range: **{km:.0f} km** (with 10% reserve)")


bot.run(TOKEN)
