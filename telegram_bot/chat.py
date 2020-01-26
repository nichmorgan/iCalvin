from IA.model import model
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import logging
import json

TOKEN = json.load(open('telegram_bot/token.json'))['token']

updater = Updater(token=TOKEN, use_context=True)
dispatcher = updater.dispatcher

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)

def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text='Olá, sou o iKnox. Fique a vontade para perguntar.')

def reply(update, context):
    question = update.message.text
    answer = model.predict(question)
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=answer
    )

start_handler = CommandHandler('start', start)
dispatcher.add_handler(start_handler)

reply_handler = MessageHandler(Filters.text, reply)
dispatcher.add_handler(reply_handler)

updater.start_polling()