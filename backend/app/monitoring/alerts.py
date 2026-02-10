import smtplib
from email.message import EmailMessage

import httpx
import structlog

from app.config import settings

logger = structlog.get_logger()


async def send_telegram_message(message: str) -> bool:
    """Send alert via Telegram bot."""
    if not settings.telegram_bot_token or not settings.telegram_chat_ids:
        logger.warning("telegram.not_configured")
        return False

    chat_ids = [c.strip() for c in settings.telegram_chat_ids.split(",")]

    try:
        async with httpx.AsyncClient() as client:
            for chat_id in chat_ids:
                resp = await client.post(
                    f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": message,
                        "parse_mode": "HTML",
                    },
                    timeout=10.0,
                )
                resp.raise_for_status()
        logger.info("telegram.sent", recipients=len(chat_ids))
        return True
    except Exception:
        logger.exception("telegram.send_error")
        return False


async def send_signal_message(message: str) -> bool:
    """Send alert via Signal messenger using signal-cli-rest-api."""
    if not settings.signal_sender_number or not settings.signal_recipient_numbers:
        logger.warning("signal.not_configured")
        return False

    recipients = [n.strip() for n in settings.signal_recipient_numbers.split(",")]

    try:
        async with httpx.AsyncClient() as client:
            for recipient in recipients:
                resp = await client.post(
                    f"{settings.signal_cli_rest_api_url}/v2/send",
                    json={
                        "message": message,
                        "number": settings.signal_sender_number,
                        "recipients": [recipient],
                    },
                    timeout=10.0,
                )
                resp.raise_for_status()
        logger.info("signal.sent", recipients=len(recipients))
        return True
    except Exception:
        logger.exception("signal.send_error")
        return False


async def send_email_alert(subject: str, body: str) -> bool:
    """Send alert email via SMTP."""
    if not settings.smtp_user or not settings.alert_email_to:
        logger.warning("email.not_configured")
        return False

    try:
        msg = EmailMessage()
        msg["Subject"] = f"[AI Trader] {subject}"
        msg["From"] = settings.smtp_from or settings.smtp_user
        msg["To"] = settings.alert_email_to
        msg.set_content(body)

        if settings.smtp_port == 465:
            with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port) as server:
                server.login(settings.smtp_user, settings.smtp_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
                server.starttls()
                server.login(settings.smtp_user, settings.smtp_password)
                server.send_message(msg)

        logger.info("email.sent", subject=subject)
        return True
    except Exception:
        logger.exception("email.send_error")
        return False


async def send_alert(title: str, message: str, critical: bool = False) -> None:
    """Send alert via all configured channels. Email is used for critical alerts."""
    full_message = f"🚨 {title}\n\n{message}" if critical else f"📊 {title}\n\n{message}"
    await send_telegram_message(full_message)
    await send_signal_message(full_message)

    if critical:
        await send_email_alert(title, message)
