import os
from dotenv import load_dotenv
import discord
from discord.ext import commands
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Langchain
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()
# Global Variables
TOKEN = os.getenv("TOKEN")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Llama 3.2 1B
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

# Huggingface Pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Chat History
chat_history = [
    {"role": "system", "content": "You are a cheerful conversational chatbot."}
]

# Generation
def generate_response(input):

    global chat_history

    chat_history.append({"role": "user", "content": str(input)})

    outputs = pipe(
        chat_history,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.6,
        top_k=50,
    )

    response = outputs[0]["generated_text"][-1] 

    chat_history.append(response) 

    return (response["content"]) 


# Discord API
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})') 
    print('------')

# Discord Chat Command: (!c)
@bot.command()
async def c(ctx, *, arg): 
    output = generate_response(arg)
    with open('chat_history.txt', 'a') as file:
        file.write(str(ctx.author) + ',' + arg + '\n')
        file.write(str(bot.user) + ',' + output + '\n')
    await ctx.send(output)

bot.run(TOKEN) 