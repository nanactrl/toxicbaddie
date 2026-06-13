import os
import discord
from discord.ext import commands
import joblib
import sqlite3
import re
from datetime import timedelta
from dotenv import load_dotenv
import hashlib
from sklearn.utils.validation import check_is_fitted
from collections import defaultdict
import time

# ================= CONFIG =================
load_dotenv()

BASE_DIR = r"C:\FYPPP2\toxicbaddie"
MODEL_PATH = os.path.join(BASE_DIR, "models", "toxic_model.pkl")
DB_PATH = os.path.join(BASE_DIR, "toxicbaddie.db")

TOKEN = os.getenv("DISCORD_BOT_TOKEN")

if not TOKEN or TOKEN.strip() == "" or TOKEN.startswith("YOUR_") or len(TOKEN) < 50:
    print("❌ Invalid or missing DISCORD_BOT_TOKEN!")
    exit()

print("✅ Token loaded successfully!")

# ================= SECURE MODEL LOADING =================
def load_safe_model(model_path):
    """Secure ML Model Loader with Hash Verification"""
    if not os.path.exists(model_path):
        print("⚠️ Model file not found!")
        return None
    
    print(f"🔒 Loading model from: {model_path}")
    
    try:
        with open(model_path, "rb") as f:
            model_bytes = f.read()
        
        current_hash = hashlib.sha256(model_bytes).hexdigest()
        
        # ←←← Expected Hash (Update only when you change the model) ←←←
        expected_hash = "dbdb3d65657cf284040cd6f3364f90440ea44584d6a380d35cabc0e2b05ffe2d"
        
        # Hash Verification
        if current_hash != expected_hash:
            print("⚠️ Model hash mismatch! (Tamper warning)")
            # raise ValueError("Model tampered!")   # Uncomment this in final/production version
        else:
            print("✅ Model integrity check passed")
            
    except Exception as e:
        print(f"❌ Error reading model file: {e}")
        return None

    # Load the actual model
    try:
        model = joblib.load(model_path)
        check_is_fitted(model)
        print("✅ Hybrid Model (TF-IDF + LinearSVC) loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
# ================= BOT SETUP =================
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

# ================= LOAD MODEL =================
model = load_safe_model(MODEL_PATH) 

# ================= RATE LIMITING =================
message_cooldown = defaultdict(lambda: {"count": 0, "last_time": 0})

# ================= DATABASE =================
conn = sqlite3.connect(DB_PATH)
conn.execute("""
    CREATE TABLE IF NOT EXISTS user_warnings (
        user_id TEXT, guild_id TEXT, warning_count INTEGER DEFAULT 0,
        last_warning DATETIME DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (user_id, guild_id)
    )
""")
conn.execute("""
    CREATE TABLE IF NOT EXISTS toxic_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT, guild_id TEXT, username TEXT,
        message TEXT, channel TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()

# ================= HELPERS =================
def normalize_text(text):
    if not text or len(text) > 1000:
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s.,?!'\"]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_toxic_word(text):
    text_norm = normalize_text(text)
    if not text_norm:
        return False, None

    toxic_words = {
        "sial", "bodoh", "babi", "anjing", "kimak", "pukimak", "lancau", 
        "lanjiao", "kontol", "bangang", "fuck", "shit", "bitch", "asshole", 
        "cunt", "stupid", "idiot", "dumb", "noob", "retard", "wtf", "wth", 
        "wtaf", "tf", "stfu", "gtfo", "lmfao", "fck", "gampang"
    }

    words = set(text_norm.split())
    for word in toxic_words:
        if word in words or re.search(rf"\b{word}\b", text_norm):
            return True, word
    return False, None


# ================= EVENTS =================
@bot.event
async def on_ready():
    print(f"✅ Bot online as {bot.user}")


@bot.event
async def on_message(message):
    if message.author.bot or not message.content.strip():
        return

    # Rate Limiting
    current_time = time.time()
    guild_id = str(message.guild.id)
    user_id = str(message.author.id)
    user_key = f"{guild_id}_{user_id}"

    if current_time - message_cooldown[user_key]["last_time"] > 60:
        message_cooldown[user_key] = {"count": 0, "last_time": current_time}

    message_cooldown[user_key]["count"] += 1
    if message_cooldown[user_key]["count"] > 20:
        return

    # Toxicity Detection
    text = message.content
    clean_text = normalize_text(text)

    is_toxic = False
    reason = ""

    # Layer 1: Keyword Filter
    toxic_detected, trigger = contains_toxic_word(text)
    if toxic_detected:
        is_toxic = True
        reason = f"Toxic word: **{trigger}**"

    # Layer 2: Hybrid ML Model (TF-IDF + LinearSVC)
    elif model:                                 # ←←← HYBRID MODEL USED HERE
        try:
            # This line uses your trained pipeline:
            # TfidfVectorizer + LinearSVC
            prediction = model.predict([clean_text])[0]   # ←←← HYBRID MODEL LINE
            if prediction == 1:
                is_toxic = True
                reason = "🧠 ML Model (TF-IDF + LinearSVC) detected toxicity"
        except Exception as e:
            print(f"Model prediction error: {e}")

    if is_toxic:
        await handle_toxic_message(message, guild_id, user_id, text, reason)
        return

    await bot.process_commands(message)


# ================= CORE MODERATION =================
async def handle_toxic_message(message, guild_id, user_id, text, reason):
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO toxic_messages (user_id, guild_id, username, message, channel)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, guild_id, str(message.author), text[:500], f"#{message.channel.name}"))

    cursor.execute("""
        INSERT INTO user_warnings (user_id, guild_id, warning_count, last_warning)
        VALUES (?, ?, 1, CURRENT_TIMESTAMP)
        ON CONFLICT(user_id, guild_id)
        DO UPDATE SET warning_count = warning_count + 1,
                      last_warning = CURRENT_TIMESTAMP
    """, (user_id, guild_id))

    conn.commit()

    cursor.execute("SELECT warning_count FROM user_warnings WHERE user_id = ? AND guild_id = ?",
                   (user_id, guild_id))
    new_count = cursor.fetchone()[0]

    print(f"🚨 Toxic: {message.author} | {new_count}/10")

    embed = discord.Embed(
        title="🚨 TOXIC MESSAGE DETECTED",
        description=f"{message.author.mention} (**{new_count}/10 warnings**)",
        color=0xff0000
    )
    embed.add_field(name="Reason", value=reason, inline=False)
    embed.add_field(name="Message", value=text[:300] + "..." if len(text) > 300 else text, inline=False)

    await message.channel.send(embed=embed)

    try:
        await message.delete()
    except:
        pass

    # Auto-moderation
    try:
        if new_count >= 10:
            await message.author.kick(reason="10 toxicity warnings")
            await message.channel.send(f"🚪 {message.author.mention} was kicked.")
        elif new_count >= 5:
            until = discord.utils.utcnow() + timedelta(hours=12)
            await message.author.timeout(until, reason="Repeated toxicity")
            await message.channel.send(f"🔇 {message.author.mention} timed out (12h).")
        elif new_count >= 3:
            until = discord.utils.utcnow() + timedelta(hours=2)
            await message.author.timeout(until, reason="Toxic behavior")
            await message.channel.send(f"⚠️ {message.author.mention} timed out (2h).")
    except discord.Forbidden:
        await message.channel.send("❌ Missing `Moderate Members` permission.")
    except Exception as e:
        print(f"Moderation error: {e}")


# ================= COMMANDS =================
@bot.command()
@commands.has_permissions(administrator=True)
async def warnings(ctx, member: discord.Member = None):
    member = member or ctx.author
    cursor = conn.cursor()
    cursor.execute("SELECT warning_count FROM user_warnings WHERE user_id = ? AND guild_id = ?",
                   (str(member.id), str(ctx.guild.id)))
    row = cursor.fetchone()
    count = row[0] if row else 0
    await ctx.send(f"⚠️ {member.display_name}: **{count}/10 warnings**")


@bot.command()
@commands.has_permissions(administrator=True)
async def resetwarnings(ctx, member: discord.Member):
    cursor = conn.cursor()
    cursor.execute("UPDATE user_warnings SET warning_count = 0 WHERE user_id = ? AND guild_id = ?",
                   (str(member.id), str(ctx.guild.id)))
    conn.commit()
    await ctx.send(f"✅ Reset warnings for {member.display_name}")


# ================= RUN BOT =================
bot.run(TOKEN)