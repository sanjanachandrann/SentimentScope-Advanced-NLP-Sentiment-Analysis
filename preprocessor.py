"""
Text preprocessing utilities for sentiment analysis
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from textblob import TextBlob
import contractions

class TextPreprocessor:
    """Advanced text preprocessing class for sentiment analysis"""
    
    def __init__(self, language='english'):
        self.language = language
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Download required NLTK data
        nltk_downloads = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        for item in nltk_downloads:
            try:
                nltk.data.find(f'tokenizers/{item}')
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{item}')
                except LookupError:
                    try:
                        nltk.data.find(f'taggers/{item}')
                    except LookupError:
                        nltk.download(item)
        
        self.stop_words = set(stopwords.words(language))
        
        # Common emoticons and their meanings
        self.emoticons = {
            ':)': 'happy', ':-)': 'happy', '(:': 'happy', '(-:': 'happy',
            ':(': 'sad', ':-(': 'sad', ':\'(': 'sad', ':\'-(' : 'sad',
            ':D': 'happy', ':-D': 'happy', 'xD': 'happy', 'XD': 'happy',
            ':P': 'playful', ':-P': 'playful', ':p': 'playful', ':-p': 'playful',
            ';)': 'wink', ';-)': 'wink', ':o': 'surprised', ':-o': 'surprised',
            ':O': 'surprised', ':-O': 'surprised', ':|': 'neutral', ':-|': 'neutral'
        }
        
        # Slang dictionary for normalization
        self.slang_dict = {
            'u': 'you', 'ur': 'your', 'r': 'are', 'n': 'and', 'b4': 'before',
            'gr8': 'great', 'luv': 'love', 'gud': 'good', 'nite': 'night',
            'omg': 'oh my god', 'lol': 'laugh out loud', 'rofl': 'rolling on floor laughing',
            'wtf': 'what the fuck', 'ttyl': 'talk to you later', 'brb': 'be right back',
            'thx': 'thanks', 'ty': 'thank you', 'np': 'no problem', 'imo': 'in my opinion'
        }
    
    def clean_text(self, text):
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (but keep the text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        return text
    
    def expand_contractions(self, text):
        """Expand contractions like don't -> do not"""
        try:
            return contractions.fix(text)
        except:
            # Fallback manual expansion
            contractions_dict = {
                "ain't": "is not", "aren't": "are not", "can't": "cannot",
                "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
                "don't": "do not", "hadn't": "had not", "hasn't": "has not",
                "haven't": "have not", "he'd": "he would", "he'll": "he will",
                "he's": "he is", "i'd": "i would", "i'll": "i will",
                "i'm": "i am", "i've": "i have", "isn't": "is not",
                "it'd": "it would", "it'll": "it will", "it's": "it is",
                "let's": "let us", "mightn't": "might not", "mustn't": "must not",
                "shan't": "shall not", "she'd": "she would", "she'll": "she will",
                "she's": "she is", "shouldn't": "should not", "that's": "that is",
                "there's": "there is", "they'd": "they would", "they'll": "they will",
                "they're": "they are", "they've": "they have", "we'd": "we would",
                "we're": "we are", "we've": "we have", "weren't": "were not",
                "what's": "what is", "where's": "where is", "who's": "who is",
                "won't": "will not", "wouldn't": "would not", "you'd": "you would",
                "you'll": "you will", "you're": "you are", "you've": "you have"
            }
            
            for contraction, expansion in contractions_dict.items():
                text = text.replace(contraction, expansion)
            
            return text
    
    def handle_emoticons(self, text):
        """Replace emoticons with their text meanings"""
        for emoticon, meaning in self.emoticons.items():
            text = text.replace(emoticon, f' {meaning} ')
        return text
    
    def normalize_slang(self, text):
        """Normalize internet slang and abbreviations"""
        words = text.split()
        normalized_words = []
        
        for word in words:
            if word.lower() in self.slang_dict:
                normalized_words.append(self.slang_dict[word.lower()])
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def remove_special_chars(self, text):
        """Remove special characters and digits"""
        # Keep only alphabets and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def handle_repeated_chars(self, text):
        """Handle repeated characters like 'sooooo' -> 'so'"""
        # Replace 3+ repeated characters with 2
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        return text
    
    def tokenize_text(self, text):
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from tokens"""
        return [token for token in tokens if token.lower() not in self.stop_words and len(token) > 2]
    
    def lemmatize_tokens(self, tokens):
        """Lemmatize tokens to their root form"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def stem_tokens(self, tokens):
        """Stem tokens to their root form"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def spell_correction(self, text):
        """Basic spell correction using TextBlob"""
        try:
            blob = TextBlob(text)
            return str(blob.correct())
        except:
            return text
    
    def preprocess_pipeline(self, text, steps=None):
        """
        Complete preprocessing pipeline
        
        Args:
            text (str): Input text
            steps (list): List of preprocessing steps to apply
        
        Returns:
            str: Preprocessed text
        """
        if steps is None:
            steps = [
                'clean_text',
                'expand_contractions',
                'handle_emoticons',
                'normalize_slang',
                'handle_repeated_chars',
                'remove_special_chars',
                'tokenize',
                'remove_stopwords',
                'lemmatize'
            ]
        
        processed_text = text
        
        for step in steps:
            if step == 'clean_text':
                processed_text = self.clean_text(processed_text)
            elif step == 'expand_contractions':
                processed_text = self.expand_contractions(processed_text)
            elif step == 'handle_emoticons':
                processed_text = self.handle_emoticons(processed_text)
            elif step == 'normalize_slang':
                processed_text = self.normalize_slang(processed_text)
            elif step == 'handle_repeated_chars':
                processed_text = self.handle_repeated_chars(processed_text)
            elif step == 'spell_correction':
                processed_text = self.spell_correction(processed_text)
            elif step == 'remove_special_chars':
                processed_text = self.remove_special_chars(processed_text)
            elif step == 'tokenize':
                tokens = self.tokenize_text(processed_text)
                continue  # Don't join tokens yet
            elif step == 'remove_stopwords':
                tokens = self.remove_stopwords(tokens)
            elif step == 'lemmatize':
                tokens = self.lemmatize_tokens(tokens)
            elif step == 'stem':
                tokens = self.stem_tokens(tokens)
        
        # Join tokens back to string if tokenization was performed
        if 'tokenize' in steps:
            processed_text = ' '.join(tokens)
        
        return processed_text
    
    def get_word_features(self, text):
        """Extract word-based features from text"""
        tokens = self.tokenize_text(text.lower())
        
        features = {
            'word_count': len(tokens),
            'avg_word_length': sum(len(word) for word in tokens) / len(tokens) if tokens else 0,
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_count': sum(1 for char in text if char.isupper()),
            'punctuation_count': sum(1 for char in text if char in string.punctuation)
        }
        
        return features
    
    def extract_ngrams(self, text, n=2):
        """Extract n-grams from text"""
        tokens = self.tokenize_text(text.lower())
        ngrams = []
        
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
        
        return ngrams

# Usage example and testing
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    # Test text
    test_text = "I'm sooooo happy!!! This is gr8 :) Can't wait to use it!!! #amazing"
    
    print("Original text:", test_text)
    print("Preprocessed text:", preprocessor.preprocess_pipeline(test_text))
    print("Word features:", preprocessor.get_word_features(test_text))
    print("Bigrams:", preprocessor.extract_ngrams(test_text, 2))