import os
from dotenv import load_dotenv
import discord
from discord.ext import commands
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from peft import PeftModel, PeftConfig

load_dotenv()
TOKEN = os.getenv("TOKEN")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"


# Dialogpt
BASE = "microsoft/DialoGPT-medium"
ADAPT = "rigby_v3"
tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, pad_token_id=tokenizer.eos_token_id)
model = PeftModel.from_pretrained(model, ADAPT, pad_token_id=tokenizer.eos_token_id)
model = model.to(DEVICE)

chat_history = []

def generate_dialog(input):
    global chat_history
    new_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt').to(DEVICE)
    #all_input_ids = torch.cat([chat_history, new_input_ids], dim=-1) if len(chat_history) != 0 else new_input_ids
    #chat_history = model.generate(all_input_ids, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id, do_sample=True, top_k=3)

    return (tokenizer.decode(chat_history[:, all_input_ids.shape[-1]:][0], skip_special_tokens=True))

def clear_history()->None:
    global chat_history
    del chat_history
    chat_history = []

def roll(number):
    return random.randint(1, int(number)) if number.isdigit() and number != "0" else "Cannot Roll"


intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')

@bot.command()
async def diag(ctx, *, arg):
    response = generate_dialog(arg)
    with open('chat_history.txt', 'a') as file:
        file.write(str(ctx.author) + ',' + arg + '\n')
        file.write(str(bot.user) + ',' + response + '\n')
    await ctx.send(response)

@bot.command()
async def clear(ctx):
    clear_history()
    await ctx.send('Chat history has been cleared.')

@bot.command()
async def roll(ctx, arg: roll):
    await ctx.send(arg)


bot.run(TOKEN)