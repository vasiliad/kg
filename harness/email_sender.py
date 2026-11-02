"""
Secure SMTP Email Sender for Gmail.
Dispatches system health reports and daily intelligence digests.
Uses TLS encryption on port 587.
"""

import os
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email.utils import formatdate, make_msgid
from typing import Dict, Any, Optional, Tuple

from harness import redaction

# A recipient that survives validation: one address, a dot in the domain,
# no separators. The previous check was `"@" in recipient`, which accepted
# "a@b", "one@x, two@y" and an address with a newline in it — the last one
# is header injection.
ADDRESS_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9]([A-Za-z0-9\-]*[A-Za-z0-9])?(\.[A-Za-z0-9]([A-Za-z0-9\-]*[A-Za-z0-9])?)+$")


def validate_recipient(address: str) -> Tuple[bool, str]:
    """Whether the address may be used as the To: of a letter, and why not."""
    text = (address or "").strip()
    if not text:
        return False, "адрес не задан"
    if any(ch in text for ch in "\r\n"):
        return False, "адрес содержит перевод строки"
    if "," in text or ";" in text or " " in text:
        return False, "адрес содержит разделитель или пробел (нужен ровно один адрес)"
    if text.count("@") != 1:
        return False, "в адресе не ровно один символ @"
    local, _, domain = text.partition("@")
    if not local:
        return False, "пустая часть до @"
    if "." not in domain:
        return False, "в домене нет точки"
    if not ADDRESS_RE.match(text):
        return False, "адрес не похож на почтовый"
    return True, ""

class EmailSender:
    def __init__(
        self,
        sender_email: Optional[str] = None,
        app_password: Optional[str] = None,
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587
    ):
        self.sender_email = sender_email or os.getenv("GMAIL_SENDER", "")
        self.app_password = app_password or os.getenv("GMAIL_APP_PASSWORD", os.getenv("GMAIL_PASSWORD", ""))
        if not self.sender_email:
            print("[!] WARNING: GMAIL_SENDER not set! Email dispatch will be skipped.")
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port

    def is_configured(self) -> bool:
        """Returns True if sender email and app password are provided."""
        return bool(self.sender_email and self.app_password)

    def send_email(
        self,
        recipient_email: str,
        subject: str,
        html_content: str,
        plain_text_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Sends a multipart email (HTML + Plain Text fallback) via Gmail SMTP TLS.
        """
        valid, reason = validate_recipient(recipient_email)
        if not valid:
            msg = (
                f"[-] EmailSender: no valid recipient configured for email '{subject}' ({reason}). "
                "Set DISPATCH_RECIPIENT_EMAIL (GitHub secret) or change it from Telegram (/setemail)."
            )
            print(msg)
            return {"status": "SKIPPED_NO_RECIPIENT", "message": msg}

        if not (html_content or "").strip() and not (plain_text_content or "").strip():
            msg = f"[-] EmailSender: refusing to send '{subject}' — the body is empty."
            print(msg)
            return {"status": "ERROR", "error": msg}

        if not self.is_configured():
            msg = (
                f"[-] EmailSender: GMAIL_APP_PASSWORD not set. Email '{subject}' to "
                f"{redaction.recipient_tag(recipient_email)} was generated but not dispatched via SMTP."
            )
            print(msg)
            return {"status": "SKIPPED_NOT_CONFIGURED", "message": msg}

        # Clean spaces in app password if user pasted with spaces
        clean_password = self.app_password.replace(" ", "").strip()

        print(f"[*] Dispatching email '{subject}' to {redaction.recipient_tag(recipient_email)} via {self.smtp_host}:{self.smtp_port}...")

        # The MIME assembly is INSIDE the try: a body that cannot be
        # encoded returns ERROR instead of raising out of the dispatcher
        # and taking the second letter down with it.
        try:
            # Build message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = Header(subject, "utf-8")
            # The signature is a runtime setting (data/.bot_settings.json), so the
            # operator can change it from Telegram without touching the code and
            # without a redeploy. Env GMAIL_SENDER_NAME overrides the default.
            try:
                from harness import runtime_settings
                sender_name = runtime_settings.get_sender_name()
            except Exception:
                sender_name = os.getenv("GMAIL_SENDER_NAME", "Автономный комплекс KG")
            msg["From"] = f"{sender_name} <{self.sender_email}>"
            msg["To"] = recipient_email
            msg["Date"] = formatdate(localtime=True)
            msg["Message-ID"] = make_msgid()

            # Fallback plain text
            if plain_text_content:
                part_text = MIMEText(plain_text_content, "plain", "utf-8")
                msg.attach(part_text)

            # HTML part
            part_html = MIMEText(html_content, "html", "utf-8")
            msg.attach(part_html)

            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=20) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self.sender_email, clean_password)
                server.sendmail(self.sender_email, [recipient_email], msg.as_string())
            
            print(f"[+] Email successfully delivered to {redaction.recipient_tag(recipient_email)}!")
            return {"status": "DELIVERED", "recipient": redaction.recipient_tag(recipient_email), "subject": subject}
        except smtplib.SMTPAuthenticationError as e:
            err = f"Authentication failed: {e}. Ensure 16-char Google App Password is used."
            print(f"[-] {err}")
            return {"status": "AUTH_ERROR", "error": err}
        except Exception as e:
            err = f"Failed to deliver email: {e}"
            print(f"[-] {err}")
            return {"status": "NETWORK_ERROR", "error": err}

if __name__ == "__main__":
    sender = EmailSender()
    print("EmailSender configuration status: configured =", sender.is_configured())
