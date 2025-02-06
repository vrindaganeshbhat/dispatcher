from datetime import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # Fixed import
from nltk.tokenize import word_tokenize
import numpy as np
from collections import defaultdict
import json
import re

class EnhancedLayer2Processor:
    def __init__(self):
        print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted)")
        print(f"Current User's Login\n")
        
        # Download required NLTK data
        try:
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            print(f"Error downloading NLTK data: {e}")

        # Initialize components
        try:
            self.sia = SentimentIntensityAnalyzer()  # Fixed initialization
        except Exception as e:
            print(f"Error initializing SentimentIntensityAnalyzer: {e}")
            
        self.processed_messages = []
        self.dispatch_queue = {
            'urgent': [],
            'high': [],
            'normal': [],
            'low': []
        }
        
        # Enhanced patterns for better extraction
        self.patterns = {
            'location': r'(?i)zone [a-z]|sector \d+|area [a-z0-9]+|location: [^,\.]+',
            'time': r'(?i)(\d{1,2}:\d{2}(?:\s*[ap]m)?)|(\d{1,2}\s*[ap]m)|tomorrow|today|morning|afternoon|evening',
            'driver': r'(?i)driver[s]?\s+([a-z]+)',
            'vehicle': r'(?i)(truck|van|vehicle)\s+([a-z0-9]+)',
            'load': r'(?i)(kg|tons?|packages?|items?|units?)\s+(\d+)',
        }
        
        # Priority keywords with weights
        self.priority_keywords = {
            'urgent': 5,
            'emergency': 5,
            'asap': 4,
            'priority': 4,
            'quick': 3,
            'fast': 3,
            'immediate': 5,
            'critical': 5
        }

    def process_message(self, message):
        """Enhanced message processing with dispatch focus"""
        try:
            # Basic validation
            if not self._validate_message(message):
                raise ValueError("Invalid message format")

            # Extract dispatch info
            dispatch_info = self._extract_dispatch_info(message['text'])
            
            # NLP Analysis
            nlp_results = self._enhanced_nlp_analysis(message['text'])
            
            # Calculate priority
            priority_info = self._calculate_dispatch_priority(message['text'])
            
            # Generate actions
            actions = self._generate_action_items(message['text'])

            # Combine results
            result = {
                'message_id': len(self.processed_messages) + 1,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'original_text': message['text'],
                'extracted_data': dispatch_info,
                'nlp_analysis': nlp_results,
                'dispatch_priority': priority_info,
                'action_items': actions
            }

            # Add to dispatch queue
            self._add_to_dispatch_queue(result)
            
            # Store result
            self.processed_messages.append(result)
            return result

        except Exception as e:
            print(f"Error processing message: {str(e)}")
            return None

    def _enhanced_nlp_analysis(self, text):
        """Enhanced NLP analysis"""
        try:
            tokens = word_tokenize(text.lower())
            sentiment = self.sia.polarity_scores(text)
            
            return {
                'tokens': tokens,
                'sentiment': sentiment,
                'word_count': len(tokens),
                'key_terms': self._extract_key_terms(text)
            }
        except Exception as e:
            print(f"NLP analysis error: {str(e)}")
            return None

    def _extract_key_terms(self, text):
        """Extract key terms from text"""
        key_terms = []
        text_lower = text.lower()
        
        # Check for important terms
        for term in ['urgent', 'delivery', 'pickup', 'driver', 'location']:
            if term in text_lower:
                key_terms.append(term)
                
        return key_terms

    def _extract_dispatch_info(self, text):
        """Extract key dispatch information"""
        info = {}
        
        # Extract using patterns
        for key, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                info[key] = matches[0] if isinstance(matches[0], str) else matches[0][0]
        
        return info

    def _calculate_dispatch_priority(self, text):
        """Calculate dispatch priority"""
        text_lower = text.lower()
        priority_score = 0
        
        # Check priority keywords
        for keyword, weight in self.priority_keywords.items():
            if keyword in text_lower:
                priority_score += weight
                
        # Normalize score to 0-10
        priority_score = min(10, priority_score)
        
        return {
            'score': priority_score,
            'level': self._get_priority_level(priority_score),
            'requires_immediate_action': priority_score >= 7
        }

    def _get_priority_level(self, score):
        """Convert score to priority level"""
        if score >= 8: return 'urgent'
        if score >= 6: return 'high'
        if score >= 4: return 'normal'
        return 'low'

    def _generate_action_items(self, text):
        """Generate action items"""
        actions = []
        info = self._extract_dispatch_info(text)
        
        if 'delivery' in text.lower():
            actions.append({
                'type': 'delivery',
                'location': info.get('location', 'Unknown'),
                'time': info.get('time', 'Not specified')
            })
            
        return actions

    def _add_to_dispatch_queue(self, result):
        """Add to dispatch queue"""
        priority = result['dispatch_priority']['level']
        self.dispatch_queue[priority].append({
            'message_id': result['message_id'],
            'timestamp': result['timestamp'],
            'actions': result['action_items']
        })

    def _validate_message(self, message):
        """Validate message format"""
        return isinstance(message, dict) and 'text' in message

    def get_dispatch_queue_status(self):
        """Get queue status"""
        return {
            priority: {
                'count': len(queue),
                'items': queue
            }
            for priority, queue in self.dispatch_queue.items()
        }

def main():
    # System information
    CURRENT_TIME = "2025-02-06 13:25:36"
    CURRENT_USER = "vrindaganeshbhat"
    
    # Initialize processor
    processor = EnhancedLayer2Processor()
    
    # Test messages
    test_messages = [
        {
            'text': "URGENT: Delivery needed in Zone A, 500 kg load. Driver John required ASAP. Fragile items.",
            'timestamp': CURRENT_TIME
        },
        {
            'text': "Regular pickup from Sector 3, tomorrow morning at 10 AM. Driver Mike assigned.",
            'timestamp': CURRENT_TIME
        }
    ]
    
    # Process messages
    print("Processing messages...\n")
    for msg in test_messages:
        result = processor.process_message(msg)
        if result:
            print(f"\nProcessed Message {result['message_id']}:")
            print(json.dumps(result, indent=2))
            print("-" * 80)
    
    # Show queue status
    print("\nDispatch Queue Status:")
    print(json.dumps(processor.get_dispatch_queue_status(), indent=2))

if __name__ == "__main__":
    main()