import os
import sys
import json
import re

cache_file = r"b:\Sadhana\NLP\backend\cache\books\dbe81a1c57b9ffa5bc3803f7dfbc72bb85ea47930b799864a81038ab1dfa819b.json"

with open(cache_file, "r", encoding="utf-8") as f:
    book_data = json.load(f)

text = book_data["cleaned_text"]
char_name = "Mrs. Elm"
core_name = char_name.split()[-1] # "Elm"

print(f"Searching quotes for {char_name} (core: {core_name})...")

# Patterns for dialogue
# 1. 'Quote' said Mrs Elm / Mrs Elm said 'Quote'
quotes = []
# Find all quotes
pattern1 = rf'[\'\"“]([^\'\"”\n\r]{{10,250}})[\'\"”]\s*(?:,|\.)?\s*(?:said|replied|asked|smiled|warned|murmured|explained|noted)?\s*(?:Mrs\.?\s*Elm|Elm)'
pattern2 = rf'(?:Mrs\.?\s*Elm|Elm)\s+(?:said|replied|asked|smiled|warned|murmured|explained|noted)[:,\s]+[\'\"“]([^\'\"”\n\r]{{10,250}})[\'\"”]'

matches1 = re.findall(pattern1, text, flags=re.IGNORECASE)
matches2 = re.findall(pattern2, text, flags=re.IGNORECASE)

print(f"Matches pattern 1 (quote then said Elm): {len(matches1)}")
for m in matches1[:5]:
    print(" -", m.strip())

print(f"\nMatches pattern 2 (Elm said quote): {len(matches2)}")
for m in matches2[:5]:
    print(" -", m.strip())
