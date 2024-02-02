import discord
from discord.ext import commands
from dotenv import load_dotenv, find_dotenv
import os
from langchain.prompts import SystemMessagePromptTemplate, PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.schema import HumanMessage

os.environ["OPENAI_API_KEY"] = ""

load_dotenv(find_dotenv())


loader = TextLoader("./warriors.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=100)
texts = text_splitter.split_documents(documents)


embeddings = OpenAIEmbeddings()
retriever = Chroma.from_documents(texts, embeddings).as_retriever()
chat = ChatOpenAI(temperature=0)

prompt_template = """You are a helpful dicord bot that knows about Warrior Cats.
{context}
Please provide the most suitable response for the users question.
Answer:"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context"]
)


system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)


intents = discord.Intents.default()
intents.message_content = True


bot = commands.Bot(command_prefix='!', intents=intents)

chathistory = []  # Store a list of previous questions and responses
MAX_HISTORY_LENGTH = 10  # Adjust this to control how much context is kept

@bot.command()
async def question(ctx, *, question):
    global chathistory
    global MAX_HISTORY_LENGTH
    try:
        chathistory.append({"content": question})
        chathistory = chathistory[-MAX_HISTORY_LENGTH:]  # Keep only the last X items

        # Retrieve relevant documents from the entire chat history
        docs = retriever.get_relevant_documents(" ".join([item["content"] for item in chathistory]))

        formatted_prompt = system_message_prompt.format(context=docs)
        messages = [formatted_prompt] + [HumanMessage(content=item["content"]) for item in chathistory]

        result = chat(messages)
        await ctx.send(result.content)

    except Exception as e:
        print(f"Error occurred: {e}")
        await ctx.send("Sorry, I was unable to process your question.")


bot.run(os.environ.get("DISCORD_TOKEN"))

