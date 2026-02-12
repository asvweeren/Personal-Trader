import asyncio
import smtplib
from email.message import EmailMessage

import httpx
import structlog

from app.config import settings

logger = structlog.get_logger()

MAX_RETRIES = 3
RETRY_DELAYS = [1.0, 3.0, 8.0]


async def send_telegram_message(message: str) -> bool:
    """Send alert via Telegram bot with retry on transient failures."""
    if not settings.telegram_bot_token or not settings.telegram_chat_ids:
        logger.warning("telegram.not_configured")
        return False

    chat_ids = [c.strip() for c in settings.telegram_chat_ids.split(",")]

    for attempt in range(MAX_RETRIES):
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
                    # Don't retry on client errors (bad token, bad chat_id)
                    if 400 <= resp.status_code < 500:
                        logger.error(
                            "telegram.client_error",
                            status=resp.status_code,
                            body=resp.text[:200],
                        )
                        return False
                    resp.raise_for_status()
            logger.info("telegram.sent", recipients=len(chat_ids))
            return True
        except httpx.HTTPStatusError as e:
            if e.response is not None and 400 <= e.response.status_code < 500:
                logger.error("telegram.client_error", status=e.response.status_code)
                return False
            logger.warning(
                "telegram.retry",
                attempt=attempt + 1,
                error=str(e),
            )
        except Exception as e:
            logger.warning(
                "telegram.retry",
                attempt=attempt + 1,
                error=str(e),
            )

        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_DELAYS[attempt])

    logger.exception("telegram.send_failed_after_retries", attempts=MAX_RETRIES)
    return False


async def send_email_alert(subject: str, body: str) -> bool:
    """Send alert email via SMTP with retry on transient failures."""
    if not settings.smtp_user or not settings.alert_email_to:
        logger.warning("email.not_configured")
        return False

    for attempt in range(MAX_RETRIES):
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
        except smtplib.SMTPAuthenticationError:
            logger.error("email.auth_error")
            return False
        except (ConnectionError, smtplib.SMTPServerDisconnected, OSError) as e:
            logger.warning(
                "email.retry",
                attempt=attempt + 1,
                error=str(e),
            )
        except Exception as e:
            logger.warning(
                "email.retry",
                attempt=attempt + 1,
                error=str(e),
            )

        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_DELAYS[attempt])

    logger.exception("email.send_failed_after_retries", attempts=MAX_RETRIES)
    return False


async def send_alert(title: str, message: str, critical: bool = False) -> None:
    """Send alert via all configured channels. Email is used for critical alerts."""
    full_message = f"🚨 {title}\n\n{message}" if critical else f"📊 {title}\n\n{message}"
    await send_telegram_message(full_message)

    if critical:
        await send_email_alert(title, message)
