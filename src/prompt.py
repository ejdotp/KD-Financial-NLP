# src/prompt.py
# Llama-3.1-8B prompt function for pseudo-label generation
# Part of: Black-Box KD for Financial Sentiment and Urgency Classification

import json
import time
from groq import Groq


def get_pseudo_labels(sentence, client, retries=3):
    """
    Send one financial sentence to Llama-3.1-8B via Groq API
    and return sentiment + urgency labels with confidence score.

    Args:
        sentence: Financial news sentence (string)
        client  : Groq client instance
        retries : Number of retry attempts on failure

    Returns:
        dict with keys: sentiment, urgency, confidence
        None if all retries exhausted
    """
    prompt = f"""You are a financial analyst. Classify the sentence below.

Sentence: "{sentence}"

Instructions:
1. Sentiment: choose exactly one of: positive, negative, neutral
2. Urgency: choose exactly one of: urgent, non-urgent
   - urgent = immediate action required, regulatory enforcement, crisis,
               major loss, legal action, market-moving announcement
   - non-urgent = general reporting, historical data, routine update
3. Confidence: your confidence in BOTH labels combined (0.00 to 1.00)

Respond ONLY with valid JSON. No explanation. No extra text.
Format: {{"sentiment": "...", "urgency": "...", "confidence": 0.XX}}"""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model       = "llama-3.1-8b-instant",
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = 60,
                temperature = 0.0
            )
            raw    = response.choices[0].message.content.strip()
            raw    = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)

            assert result["sentiment"]  in ["positive", "negative", "neutral"]
            assert result["urgency"]    in ["urgent", "non-urgent"]
            assert 0.0 <= float(result["confidence"]) <= 1.0

            return result

        except (json.JSONDecodeError, AssertionError, KeyError):
            time.sleep(1)
            continue
        except Exception as e:
            print(f"  API error on attempt {attempt+1}: {str(e)[:60]}")
            time.sleep(5)
            continue

    return None


def get_pseudo_labels_positive_focused(sentence, client, retries=3):
    """
    Stronger prompt for positive vs neutral distinction.
    Use for re-labelling positive misses after initial run.
    """
    prompt = f"""You are a senior financial analyst with 20 years of experience.

Sentence: "{sentence}"

Key instruction: Financial positive sentences often use SUBTLE language.
These count as POSITIVE:
- Any growth, increase, improvement, or exceeding expectations
- Expansion, new contracts, partnerships, or market gains
- Profits, margins improving even slightly
- Forward guidance that is optimistic

These count as NEUTRAL:
- Pure factual statements with no directional implication
- Administrative announcements with no financial impact

Classify:
1. Sentiment: positive, negative, or neutral
2. Urgency: urgent or non-urgent
3. Confidence: 0.00 to 1.00

Respond ONLY in JSON: {{"sentiment": "...", "urgency": "...", "confidence": 0.XX}}"""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model       = "llama-3.1-8b-instant",
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = 60,
                temperature = 0.0
            )
            raw    = response.choices[0].message.content.strip()
            raw    = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)

            assert result["sentiment"]  in ["positive", "negative", "neutral"]
            assert result["urgency"]    in ["urgent", "non-urgent"]
            assert 0.0 <= float(result["confidence"]) <= 1.0

            return result
        except:
            time.sleep(2)
            continue

    return None