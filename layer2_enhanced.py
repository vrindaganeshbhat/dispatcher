from datetime import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk import ne_chunk, pos_tag
import re
import json
from collections import defaultdict

class AdvancedInfoExtractor:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
        except Exception as e:
            print(f"Error downloading NLTK data: {e}")

        self.sia = SentimentIntensityAnalyzer()
        
        # Advanced patterns for specific information
        self.patterns = {
            # Location patterns
            'location': {
                'zone': r'(?i)zone\s+([a-z0-9]+)',
                'sector': r'(?i)sector\s+(\d+)',
                'area': r'(?i)area\s+([a-z0-9\s]+)',
                'address': r'(?i)(?:at|in|from|to)\s+([^,.]+(?:road|street|ave|avenue|blvd|boulevard))[^,]*'
            },
            
            # Time patterns
            'time': {
                'exact_time': r'(?i)(?:at\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm|hrs))',
                'relative_time': r'(?i)(in\s+\d+\s+(?:minutes?|hours?|days?))',
                'date_time': r'(?i)(?:on|for|by)\s+((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?)',
                'day_part': r'(?i)(morning|afternoon|evening|night)'
            },
            
            # Load patterns
            'load': {
                'weight': r'(?i)(\d+(?:\.\d+)?\s*(?:kg|kilos?|tons?|t))',
                'volume': r'(?i)(\d+(?:\.\d+)?\s*(?:m3|cubic\s+meters?))',
                'quantity': r'(?i)(\d+)\s*(?:items?|pieces?|packages?|boxes?|units?)',
                'dimensions': r'(?i)(\d+(?:\.\d+)?\s*x\s*\d+(?:\.\d+)?\s*x\s*\d+(?:\.\d+)?\s*(?:cm|m|ft|feet|meters?)?)'
            },
            
            # Contact patterns
            'contact': {
                'phone': r'(?i)(?:call|phone|tel|contact)\s*(?::|at)?\s*(\+?\d[\d\s-]{8,})',
                'email': r'[\w\.-]+@[\w\.-]+\.\w+',
                'person': r'(?i)(?:contact|ask\s+for|attention)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
            },
            
            # Vehicle patterns
            'vehicle': {
                'type': r'(?i)(truck|van|lorry|vehicle)\s*(?:#|number|no\.?)?\s*([a-z0-9]+)?',
                'capacity': r'(?i)(\d+(?:\.\d+)?\s*(?:ton|t))\s*(?:capacity|truck|vehicle)',
                'registration': r'(?i)(?:reg|registration|number)\s*(?::|#)?\s*([a-z0-9\s-]+)'
            }
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

    def extract_info(self, text):
        """
        Extract information using multiple techniques
        """
        extracted_info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'entities': self._extract_entities(text),
            'patterns': self._extract_patterns(text),
            'priority': self._calculate_priority(text),
            'sentiment': self.sia.polarity_scores(text),
            'metadata': self._extract_metadata(text)
        }
        return extracted_info

    def _extract_entities(self, text):
        """Extract named entities using NLTK"""
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        named_entities = ne_chunk(pos_tags)
        
        entities = defaultdict(list)
        for chunk in named_entities:
            if hasattr(chunk, 'label'):
                entities[chunk.label()].append(' '.join([token for token, pos in chunk.leaves()]))
        
        return dict(entities)

    def _extract_patterns(self, text):
        """Extract information using regex patterns"""
        extracted = {}
        
        for category, patterns in self.patterns.items():
            if isinstance(patterns, dict):
                extracted[category] = {}
                for subcategory, pattern in patterns.items():
                    matches = re.findall(pattern, text)
                    if matches:
                        extracted[category][subcategory] = matches
            else:
                matches = re.findall(patterns, text)
                if matches:
                    extracted[category] = matches
                    
        return extracted

    def _calculate_priority(self, text):
        """Calculate priority score and level"""
        text_lower = text.lower()
        priority_score = 0
        matched_keywords = []

        # Check priority keywords
        for keyword, weight in self.priority_keywords.items():
            if keyword in text_lower:
                priority_score += weight
                matched_keywords.append(keyword)

        # Check time sensitivity
        if any(word in text_lower for word in ['today', 'now', 'immediately']):
            priority_score += 2
            matched_keywords.append('time_sensitive')

        # Check special handling
        if any(word in text_lower for word in ['fragile', 'perishable', 'dangerous']):
            priority_score += 2
            matched_keywords.append('special_handling')

        # Normalize score to 0-10
        priority_score = min(10, priority_score)

        return {
            'score': priority_score,
            'level': self._get_priority_level(priority_score),
            'matched_keywords': matched_keywords
        }

    def _get_priority_level(self, score):
        """Convert priority score to level"""
        if score >= 8:
            return 'urgent'
        elif score >= 6:
            return 'high'
        elif score >= 4:
            return 'medium'
        return 'low'

    def _extract_metadata(self, text):
        """Extract metadata about the message"""
        tokens = word_tokenize(text)
        return {
            'word_count': len(tokens),
            'character_count': len(text),
            'average_word_length': sum(len(word) for word in tokens) / len(tokens) if tokens else 0
        }

def main():
    # Initialize extractor
    extractor = AdvancedInfoExtractor()
    
    # Test messages
    test_messages = [
        """URGENT: Delivery needed at Zone A-12, 500 kg fragile load. 
        Driver John (contact: +1-555-0123) required ASAP. 
        Delivery address: 123 Main Street, Building 4.
        Must deliver by 2:30 PM today.""",
        
        """Regular pickup scheduled for tomorrow morning at Sector 3.
        3 packages, total weight 200 kg.
        Contact Mike at warehouse entrance."""
    ]
    
    print("Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted):", 
          datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("Current User's Login: vrindaganeshbhat\n")
    
    # Process each message
    for i, message in enumerate(test_messages, 1):
        print(f"\nProcessing Message {i}:")
        print("-" * 50)
        print("Original Message:", message)
        print("\nExtracted Information:")
        results = extractor.extract_info(message)
        print(json.dumps(results, indent=2))
        print("=" * 80)

if __name__ == "__main__":
    main()