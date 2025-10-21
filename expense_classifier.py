from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpenseClassifier:
    def __init__(self, model_path='my_distilbert_model'):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the trained DistilBERT model and tokenizer"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load model and tokenizer
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
            
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
            logger.info(f"Model has {self.model.config.num_labels} categories")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
            self.tokenizer = None
    
    def predict(self, description, confidence_threshold=0.7):
        """
        Predict expense category using DistilBERT model
        Returns: (category, confidence_score)
        """
        # If model failed to load, return None to use rule-based fallback
        if self.model is None or self.tokenizer is None:
            return None, 0.0
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                description,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            
            # Move to device
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class_id = predictions.argmax().item()
                confidence = predictions.max().item()
                category = self.model.config.id2label[predicted_class_id]
            
            # Only return model prediction if confidence is above threshold
            if confidence >= confidence_threshold:
                return category, confidence
            else:
                return None, confidence
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None, 0.0

# Global classifier instance
classifier = ExpenseClassifier()