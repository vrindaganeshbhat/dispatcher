import imaplib
import email
from email.header import decode_header
from datetime import datetime
import logging
import os
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('email_reader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmailReader:
    def __init__(self):
        # System info
        self.current_user = "vrindaganeshbhat"
        self.start_time = "2025-02-06 12:26:16"
        
        # Email configuration
        self.email = os.getenv('EMAIL_USERNAME')
        if not self.email:
            raise ValueError("EMAIL_USERNAME not found in environment variables")
            
        self.app_password = os.getenv('EMAIL_APP_PASSWORD')
        if not self.app_password:
            raise ValueError("EMAIL_APP_PASSWORD not found in environment variables")
        
        # Server settings for Gmail
        self.imap_server = "imap.gmail.com"
        
        # Connection object
        self.imap = None
        
        # Initialize connection
        self.connect_imap()

    def connect_imap(self):
        """Establish IMAP connection"""
        if self.imap is None:
            try:
                self.imap = imaplib.IMAP4_SSL(self.imap_server)
                self.imap.login(self.email, self.app_password)
                logger.info("IMAP connection established")
            except Exception as e:
                logger.error(f"IMAP connection failed: {str(e)}")
                self.imap = None
                raise

    def get_latest_emails(self, num_emails: int = 10) -> List[Dict]:
        """Fetch the latest emails"""
        emails = []
        try:
            # Select inbox
            self.imap.select('INBOX')
            
            # Search for all emails and get the latest ones
            _, messages = self.imap.search(None, 'ALL')
            email_ids = messages[0].split()
            
            # Get the last n email IDs
            latest_emails = email_ids[-num_emails:] if len(email_ids) > num_emails else email_ids
            
            # Fetch each email
            for email_id in reversed(latest_emails):
                try:
                    _, msg_data = self.imap.fetch(email_id, '(RFC822)')
                    email_body = msg_data[0][1]
                    email_msg = email.message_from_bytes(email_body)
                    
                    # Extract email content
                    subject = self.decode_email_header(email_msg['subject'])
                    sender = self.decode_email_header(email_msg['from'])
                    date = email_msg['date']
                    body = self.get_email_body(email_msg)
                    
                    # Store email data
                    email_data = {
                        'id': email_id.decode(),
                        'subject': subject,
                        'from': sender,
                        'date': date,
                        'body': body[:500] + '...' if len(body) > 500 else body  # Truncate long bodies
                    }
                    
                    emails.append(email_data)
                    logger.info(f"Retrieved email: {subject}")
                    
                except Exception as e:
                    logger.error(f"Error processing email {email_id}: {str(e)}")
                    continue
            
            return emails
            
        except Exception as e:
            logger.error(f"Error fetching emails: {str(e)}")
            raise
        
    def decode_email_header(self, header: str) -> str:
        """Decode email header"""
        if not header:
            return ""
        try:
            decoded_header = decode_header(header)[0][0]
            if isinstance(decoded_header, bytes):
                return decoded_header.decode()
            return decoded_header
        except:
            return header

    def get_email_body(self, email_msg) -> str:
        """Extract email body"""
        if email_msg.is_multipart():
            for part in email_msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        return part.get_payload(decode=True).decode()
                    except:
                        continue
            return "No text content available"
        else:
            try:
                return email_msg.get_payload(decode=True).decode()
            except:
                return "Could not decode email body"

    def close(self):
        """Close the IMAP connection"""
        if self.imap:
            try:
                self.imap.close()
                self.imap.logout()
                logger.info("IMAP connection closed")
            except Exception as e:
                logger.error(f"Error closing IMAP connection: {str(e)}")

def main():
    try:
        # Create email reader
        reader = EmailReader()
        
        # Log startup
        logger.info("Starting Email Reader")
        logger.info(f"Current User: {reader.current_user}")
        logger.info(f"Start Time: {reader.start_time}")
        logger.info(f"Email account: {reader.email}")
        
        # Get latest emails
        emails = reader.get_latest_emails(10)
        
        # Display emails
        print("\nLatest 10 Emails:")
        print("================\n")
        
        for i, email_data in enumerate(emails, 1):
            print(f"Email {i}:")
            print(f"From: {email_data['from']}")
            print(f"Subject: {email_data['subject']}")
            print(f"Date: {email_data['date']}")
            print(f"Body Preview: {email_data['body'][:200]}...")
            print("-" * 80 + "\n")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
    finally:
        reader.close()

if __name__ == "__main__":
    main()