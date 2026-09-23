import os

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
# The one Telegram user this bot answers. Everyone else is ignored and logged.
ALLOWED_TELEGRAM_USER_ID = os.environ.get("ALLOWED_TELEGRAM_USER_ID", "")
