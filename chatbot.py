import os
from dotenv import load_dotenv
import discord
from discord.ext import commands
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()
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

def generate_response(input):
    messages = [
        {"role": "system", "content": "You are a conversational chatbot."},
        {"role": "user", "content": str(input)}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device) # type: ignore

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.2,
        top_p=0.9
    )

    response = outputs[0][input_ids.shape[-1]:]
    return (tokenizer.decode(response, skip_special_tokens=True))

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})') # type: ignore
    print('------')

@bot.command()
async def c(ctx, *, arg: generate_response): # type: ignore
    await ctx.send(arg)

bot.run(TOKEN) # type: ignore