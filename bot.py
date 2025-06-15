import discord
from discord.ext import commands
import os
from dotenv import load_dotenv

from translator import translate_text
from summarizer import summarize_text
from qa import answer_question
from moderation import is_inappropriate

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Zalogowano jako {bot.user}')

@bot.command(name="tlumacz")
async def translate(ctx, lang: str, *, text: str):
    translated = translate_text(text, lang)
    await ctx.send(f"Tłumaczenie ({lang}): {translated}")

@bot.command(name="podsumuj")
async def summarize(ctx, limit: int = 50):
    messages = [msg async for msg in ctx.channel.history(limit=limit)]
    content = " ".join([msg.content for msg in messages if msg.content])
    summary = summarize_text(content)
    await ctx.send(f"Podsumowanie:\n{summary}")

@bot.command(name="pytaj")
async def ask(ctx, *, question: str):
    messages = [msg async for msg in ctx.channel.history(limit=100)]
    context = " ".join([msg.content for msg in messages])
    answer = answer_question(question, context)
    await ctx.send(f"Odpowiedź: {answer}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if is_inappropriate(message.content):
        await message.delete()
        await message.channel.send("Usunięto nieodpowiednią wiadomość.")
        return

    await bot.process_commands(message)

bot.run(TOKEN)
