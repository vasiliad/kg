"""
Secure SMTP Email Sender for Gmail.
Dispatches system health reports and daily intelligence digests.
Uses TLS encryption on port 587.

TWO SENDER ACCOUNTS, ONE PROMISE (2026-11-04):

The reports channel used to hang on ONE Google account: if Google refused
its app password (a password change, a 2FA change, an anti-abuse block —
the measured ways free mail dies), BOTH daily letters died with it. Now a
letter is delivered by the PRIMARY account (GMAIL_SENDER +
GMAIL_APP_PASSWORD) or — exactly once, only on a transport/auth failure —
by the BACKUP account (GMAIL_SENDER_2 + GMAIL_APP_PASSWORD_2). A content
refusal (no recipient, empty body) never burns the backup: the problem is
the letter, not the wire.

Every result records WHICH account served, as a fingerprint («via=
sha256:…») — the log proves the failover works without publishing an
address.

THE ARCHIVE COPY: when ARCHIVE_RECIPIENT_EMAIL is set, every letter rides
the same SMTP session to that address too — the complex's own mailbox
becomes an archive that depends on neither the operator's inbox nor on
GitHub. A failed copy is recorded, never fatal: the operator's letter
comes first.
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


# The failures that describe the WIRE, not the letter: exactly the ones a
# second account can cure. Anything else (no recipient, empty body) is the
# letter's own problem and must not consume the backup attempt.
_WIRE_FAILURES = ("AUTH_ERROR", "NETWORK_ERROR")


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
        # The backup account: the same kind of secret, a different identity.
        self.backup_sender_email = os.getenv("GMAIL_SENDER_2", "").strip()
        self.backup_app_password = os.getenv("GMAIL_APP_PASSWORD_2", "").strip()
        # The archive mailbox: every letter's silent second recipient.
        self.archive_recipient = os.getenv("ARCHIVE_RECIPIENT_EMAIL", "").strip()
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port

    def is_configured(self) -> bool:
        """Returns True if sender email and app password are provided."""
        return bool(self.sender_email and self.app_password)

    def backup_configured(self) -> bool:
        """Returns True if the SECOND sender account may take the wire."""
        return bool(self.backup_sender_email and self.backup_app_password)

    def _deliver(
        self,
        sender_email: str,
        password: str,
        recipient_email: str,
        subject: str,
        html_content: str,
        plain_text_content: Optional[str],
    ) -> Dict[str, Any]:
        """One SMTP attempt through ONE account, archive copy riding along."""
        clean_password = (password or "").replace(" ", "").strip()

        print(f"[*] Dispatching email '{subject}' to {redaction.recipient_tag(recipient_email)} "
              f"via {redaction.fingerprint(sender_email)} ({self.smtp_host}:{self.smtp_port})...")

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
            # Measured 2026-11-04: `f"{name} <{addr}>"` put the ADDRESS inside
            # the RFC2047 encoded word — a malformed From that strict
            # receivers cannot parse. The name is encoded, the addr-spec stays
            # bare — the only form RFC 5322 allows for a non-ASCII name.
            from email.header import Header as _Header
            from email.utils import formataddr as _formataddr
            msg["From"] = _formataddr((str(_Header(sender_name, "utf-8")), sender_email))
            msg["To"] = recipient_email
            msg["Date"] = formatdate(localtime=True)
            msg["Message-ID"] = make_msgid()
            # NO Bcc header on purpose: the archive address rides the RCPT
            # list only, so a header can never publish it.

            # Fallback plain text
            if plain_text_content:
                part_text = MIMEText(plain_text_content, "plain", "utf-8")
                msg.attach(part_text)

            # HTML part
            part_html = MIMEText(html_content, "html", "utf-8")
            msg.attach(part_html)

            recipients = [recipient_email]
            archive_copy = "not configured"
            archive_valid, _ = validate_recipient(self.archive_recipient)
            if self.archive_recipient and archive_valid:
                recipients.append(self.archive_recipient)
                archive_copy = "sent"
            elif self.archive_recipient:
                archive_copy = "invalid address"

            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=20) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(sender_email, clean_password)
                server.sendmail(sender_email, recipients, msg.as_string())

            print(f"[+] Email successfully delivered to {redaction.recipient_tag(recipient_email)} "
                  f"(archive copy: {archive_copy})!")
            return {
                "status": "DELIVERED",
                "recipient": redaction.recipient_tag(recipient_email),
                "subject": subject,
                "via": redaction.fingerprint(sender_email),
                "archive_copy": archive_copy,
            }
        except smtplib.SMTPAuthenticationError as e:
            err = f"Authentication failed: {e}. Ensure 16-char Google App Password is used."
            print(f"[-] {err}")
            return {"status": "AUTH_ERROR", "error": err}
        except Exception as e:
            err = f"Failed to deliver email: {e}"
            print(f"[-] {err}")
            return {"status": "NETWORK_ERROR", "error": err}

    def send_email(
        self,
        recipient_email: str,
        subject: str,
        html_content: str,
        plain_text_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Sends a multipart email (HTML + Plain Text fallback) via Gmail SMTP TLS.

        The wire has TWO accounts: the primary serves by default; a wire
        failure (auth refused, connection dropped) hands the letter to the
        backup ONCE. The result says which account served — as a
        fingerprint, never as an address.
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

        result = self._deliver(
            self.sender_email, self.app_password,
            recipient_email, subject, html_content, plain_text_content,
        )

        if result.get("status") in _WIRE_FAILURES and self.backup_configured():
            print(f"[!] The primary sender failed ({result.get('status')}) — "
                  "handing the letter to the BACKUP account...")
            backup_result = self._deliver(
                self.backup_sender_email, self.backup_app_password,
                recipient_email, subject, html_content, plain_text_content,
            )
            backup_result["failed_over"] = True
            backup_result["primary_status"] = result.get("status")
            return backup_result

        return result

if __name__ == "__main__":
    sender = EmailSender()
    print("EmailSender configuration status: configured =", sender.is_configured(),
          "| backup =", sender.backup_configured(),
          "| archive =", bool(sender.archive_recipient))
