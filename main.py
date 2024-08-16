import os
from dotenv import load_dotenv
from discord import Intents, Client, Message
import discord
from discord.ext import commands
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline, FlaxStableDiffusionPipeline
import random


load_dotenv()
TOKEN = os.getenv("TOKEN")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# Meta-Llama-3-8B-Instruct
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype = torch.bfloat16,
    device_map = "auto"
)

# Dialogpt
gpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
gpt_model = AutoModelForCausalLM.from_pretrained("rigby_v2", pad_token_id=gpt_tokenizer.eos_token_id).to(DEVICE)

def generate_dialog(input):
    _input = gpt_tokenizer.encode(input + gpt_tokenizer.eos_token, return_tensors="pt").to(DEVICE)
    output = gpt_model.generate(_input, max_length=200, pad_token_id=gpt_tokenizer.eos_token_id, do_sample=True, top_k=3)
    return (gpt_tokenizer.decode(output[:, _input.shape[-1]:][0], skip_special_tokens=True))


def generate_response(input):
    messages = [
        {"role": "system", "content": "You are a serious chatbot who gives concise responses"},
        {"role": "user", "content": input}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=128,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )

    response = outputs[0][input_ids.shape[-1]:]
    return (tokenizer.decode(response, skip_special_tokens=True))


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
async def chat(ctx, *, arg: generate_response):
    await ctx.send(arg)

@bot.command()
async def diag(ctx, *, arg: generate_dialog):
    await ctx.send(arg)

@bot.command()
async def roll(ctx, arg: roll):
    await ctx.send(arg)


bot.run(TOKEN)