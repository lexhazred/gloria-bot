import os
import discord
from datetime import datetime, timedelta
from typing import List
from termcolor import colored


from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from discord.ext import commands

load_dotenv()

USER_NAME = "Student" # The name you want to use when interviewing the agent.
LLM = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0,max_tokens=1500) # Can be any LLM you want.
from langchain.experimental.generative_agents import GenerativeAgent, GenerativeAgentMemory

import math
import faiss

def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)

def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {}, relevance_score_fn=relevance_score_fn)
    return TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=15)

tommies_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=8 # we will give this a relatively low number to show how reflection works
)

tommie = GenerativeAgent(name="Occult Master", 
              age=100000,
              traits="esoteric, loves greek and sanskrit riddles, brilliant", # You can add more persistent traits here 
              status="studying the inner self", # When connected to a virtual world, we can have the characters update their status
              memory_retriever=create_new_memory_retriever(),
              llm=LLM,
              memory=tommies_memory
             )

print(tommie.get_summary())

# We can add memories directly to the memory object
tommie_observations = [
    "Occult Master recalls deciphering an ancient Greek riddle that revealed the location of a hidden temple",
    "Occult Master reminisces about the time they performed a complex Sanskrit ritual that brought rain to a drought-stricken village",
    "Occult Master observes the alignment of the stars and predicts a major shift in the spiritual realm",
    "Occult Master notices a dark energy emanating from a nearby haunted house",
    "Occult Master hears the whispers of spirits and communicates with them to gain insight on the future",
    "Occult Master hungers for knowledge and delves deeper into the secrets of the universe",
    "Occult Master meditates to connect with their higher self and gain clarity on their path.",
]
for observation in tommie_observations:
    tommie.memory.add_memory(observation)

# Now that Tommie has 'memories', their self-summary is more descriptive, though still rudimentary.
# We will see how this summary updates after more observations to create a more rich description.
# print(tommie.get_summary(force_refresh=True))

def interview_agent(agent: GenerativeAgent, message: str) -> str:
    """Help the notebook user interact with the agent."""
    new_message = f"{message}"
    return agent.generate_dialogue_response(new_message)[1]


# Set up the Discord bot
intents = discord.Intents.all()
intents.typing = False
intents.presences = False

bot = commands.Bot(command_prefix="!", intents=intents)

# Start the bot
@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")

# Basic chat command
@bot.command()
async def summon(ctx, *, message):
    await ctx.send("Thinking...")

    #########################################################
    # AGENT GOES HERE - REPLACE RESPONSE WITH OUTPUT
    response = interview_agent(tommie, message)
    #########################################################

    await ctx.send(response)


# Run the bot
bot.run(os.getenv("DISCORD_BOT_TOKEN"))