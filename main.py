import os
from dotenv import load_dotenv
from discord import Intents, Client, Message
import discord
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


load_dotenv()
TOKEN = os.getenv("TOKEN")

DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

CHECKPOINT = "microsoft/DialoGPT-large"
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, pad_token_id=tokenizer.eos_token_id).to(DEVICE)

def generate_response(input):
    _input = tokenizer.encode(input + tokenizer.eos_token, return_tensors="pt").to(DEVICE)
    output = model.generate(_input, max_length=200, pad_token_id=tokenizer.eos_token_id)
    return (tokenizer.decode(output[:, _input.shape[-1]:][0], skip_special_tokens=True))

class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')

    

    async def on_message(self, message):
        if message.author.id == self.user.id:
            return

        if message.content.startswith('!'):
            user_message = str(message.content)
            user_message = user_message[1:]
            bot_response = generate_response(user_message)
            await message.channel.send(bot_response)


intents = discord.Intents.default()
intents.message_content = True

client = MyClient(intents=intents)
client.run(TOKEN)