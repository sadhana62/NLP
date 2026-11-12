import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging

logger = logging.getLogger("pageecho.sentiment")

# Ensure vader_lexicon is available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

class SentimentService:
    def __init__(self):
        try:
            self.analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.error(f"Failed to initialize VADER analyzer: {e}")
            self.analyzer = None

    def analyze(self, text: str) -> dict:
        """Analyze text sentiment and return scores + emotion badge label."""
        if not text or not self.analyzer:
            return {
                "compound": 0.0,
                "pos": 0.0,
                "neg": 0.0,
                "neu": 1.0,
                "label": "Neutral",
                "color": "#94a3b8"
            }

        scores = self.analyzer.polarity_scores(text)
        compound = scores.get("compound", 0.0)

        if compound >= 0.5:
            label = "Joyful / Enthusiastic"
            color = "#10b981"  # Emerald green
        elif compound >= 0.1:
            label = "Positive / Warm"
            color = "#3b82f6"  # Blue
        elif compound <= -0.5:
            label = "Distressed / Melancholic"
            color = "#ef4444"  # Red
        elif compound <= -0.1:
            label = "Concerned / Tense"
            color = "#f59e0b"  # Amber
        else:
            label = "Calm / Neutral"
            color = "#8b5cf6"  # Purple / Indigo accent

        return {
            "compound": round(compound, 3),
            "pos": round(scores.get("pos", 0.0), 3),
            "neg": round(scores.get("neg", 0.0), 3),
            "neu": round(scores.get("neu", 0.0), 3),
            "label": label,
            "color": color
        }
