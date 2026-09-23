import logging

from telegram import Update

from app.channels.telegram_channel import build_application

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("orchestrator")


def main() -> None:
    app = build_application()
    logger.info("Starting Telegram polling")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
