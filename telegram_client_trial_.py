import asyncio
import aiohttp
from datetime import datetime

# System information with exact format
CURRENT_TIME = "2025-02-06 12:34:20"
CURRENT_USER = "vrindaganeshbhat"

class TelegramReader:
    def __init__(self):
        # Replace with your Telegram bot token
        self.bot_token = "7453011390:AAH2oUfD2XGI8dv9qEjEsxXDcjZE3ORDBoA"
        # Replace with your chat ID
        self.chat_id = "6090719217"
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    async def get_updates(self, limit=10):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/getUpdates?limit={limit}") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('result', [])
                    else:
                        print(f"Error: {response.status}")
                        return []
        except Exception as e:
            print(f"Connection error: {str(e)}")
            print("\nTo fix this:")
            print("1. Replace YOUR_BOT_TOKEN with your Telegram Bot Token")
            print("2. Replace YOUR_CHAT_ID with your Chat ID")
            print("3. To get these:")
            print("   - Bot Token: Message @BotFather on Telegram")
            print("   - Chat ID: Message @userinfobot on Telegram")
            return []

    def format_message(self, message, index):
        sender = message.get('from', {}).get('username', 'Unknown')
        text = message.get('text', 'No text')
        date = datetime.fromtimestamp(message.get('date', 0)).strftime('%Y-%m-%d %H:%M:%S')
        
        return (
            f"Message {index}:\n"
            f"From: @{sender}\n"
            f"Date: {date}\n"
            f"Text: {text}\n"
            f"{'-' * 70}\n"
        )

async def main():
    # Print system information in exact format
    print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {CURRENT_TIME}")
    print(f"Current User's Login: {CURRENT_USER}")
    print()

    reader = TelegramReader()
    print("Fetching latest messages from Telegram...\n")
    
    messages = await reader.get_updates(10)
    
    if messages:
        for i, update in enumerate(reversed(messages), 1):
            if 'message' in update:
                print(reader.format_message(update['message'], i))
    else:
        print("No messages found or error occurred")

if __name__ == "__main__":
    asyncio.run(main())