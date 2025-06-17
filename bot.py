import discord
from discord.ext import commands
import os
import re
from dotenv import load_dotenv

from transformers import T5Tokenizer, T5ForConditionalGeneration

from translator import translate_text
from summarizer import summarize_text
from qa import answer_question
from moderation import MessageModerator

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

# Load shared FLAN-T5 model and tokenizer
print("Loading models...")
flan_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
flan_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

print("Models loaded.")

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Zalogowano jako {bot.user}')


@bot.command(name="komendy")
async def commands_list(ctx):
    help_text = """
**Dostępne komendy:**

`!tlumacz [z] [do] [tekst]` - Tłumaczy tekst z języka `z` na `do`.  
Przykład: `!tlumacz pl en Cześć!`
`!tlumacz` obsługuje następujące języki:
```json
        "en": "English",
        "pl": "Polish",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
```
`!podsumuj [limit]` - Podsumowuje ostatnie wiadomości z kanału. Domyślnie 50.
Przykład: `!podsumuj 30`

`!pytaj [pytanie]` - Odpowiada na pytanie na podstawie historii rozmowy.
Przykład: `!pytaj Co omawialiśmy wcześniej?`

`!komendy` - Wyświetla listę dostępnych komend.
"""
    await ctx.send(help_text)


@bot.command(name="tlumacz")
async def translate(ctx, source_lang: str, target_lang: str, *, text: str):
    command_text = f"!tlumacz {source_lang} {target_lang} {text}"
    try:
        translation = translate_text(command_text, flan_model, flan_tokenizer)
    except Exception as e:
        await ctx.send(f"Błąd tłumaczenia: {str(e)}")
        return
    await ctx.send(f"Tłumaczenie ({source_lang} → {target_lang}): {translation}")

@bot.command(name="podsumuj")
async def summarize(ctx, limit: int = 50):
    messages = list(reversed([
        f"{msg.content}"
        async for msg in ctx.channel.history(limit=limit)
        if msg.author != bot.user and not msg.content.startswith("!")
    ]))

    content = " ".join(messages)
    summary = summarize_text(content, flan_model, flan_tokenizer)
    await ctx.send(f"Podsumowanie:\n{summary}")

@bot.command(name="pytaj")
async def ask(ctx, *, question: str):
    messages = list(reversed([
            f"{msg.author.display_name}: {msg.content}"
            async for msg in ctx.channel.history(limit=100)
            if msg.author != bot.user and not msg.content.startswith("!")
        ]))
    context = " ".join(messages)
    answer = answer_question(question, context, flan_model, flan_tokenizer)
    await ctx.send(f"Odpowiedź: {answer}")

# moderation setup

moderator = MessageModerator(spam_threshold=0.6, similarity_threshold=0.6)

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    content = message.content.strip()
    exempt_commands = [
        r"!podsumuj(\s+\d{1,3})?",
        r"!ustaw_spam\s+\d*\.?\d+",
        r"!ustaw_similarity\s+\d*\.?\d+"
    ]
    if any(re.fullmatch(cmd, content) for cmd in exempt_commands):
        await bot.process_commands(message)
        return

    is_bad, reason = await moderator.is_inappropriate(message)
    if is_bad:
        await message.delete()
        await message.channel.send(f"Usunięto wiadomość ({reason}).")
        return

    await bot.process_commands(message)

@bot.command(name="ustaw_spam")
async def set_spam_threshold(ctx, value: float):
    if ctx.channel.name != "komendy":
        await ctx.send("Ta komenda może być używana tylko w kanale #komendy.")
        return
    if not 0.0 <= value <= 1.0:
        await ctx.send("Wartość musi być między 0.0 a 1.0.")
        return
    moderator.spam_threshold = value
    await ctx.send(f"Ustawiono próg spamu na {value:.2f}.")

@bot.command(name="ustaw_similarity")
async def set_similarity_threshold(ctx, value: float):
    if ctx.channel.name != "komendy":
        await ctx.send("Ta komenda może być używana tylko w kanale #komendy.")
        return
    if not 0.0 <= value <= 1.0:
        await ctx.send("Wartość musi być między 0.0 a 1.0.")
        return
    moderator.similarity_threshold = value
    await ctx.send(f"Ustawiono próg podobieństwa na {value:.2f}.")


bot.run(TOKEN)
