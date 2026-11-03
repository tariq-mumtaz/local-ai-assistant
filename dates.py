"""
Date handling: detect date questions and compute date facts in Python,
so the model doesn't have to do calendar maths. Pure Python, no model.
"""

import calendar
import re
from datetime import datetime, timedelta


ARABIC_MONTHS = {
    # Gulf Arabic
    "يناير": 1, "فبراير": 2, "مارس": 3,
    "أبريل": 4, "مايو": 5, "يونيو": 6,
    "يوليو": 7, "أغسطس": 8, "سبتمبر": 9,
    "أكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12,
    # Levantine Arabic
    "كانون الثاني": 1, "شباط": 2, "آذار": 3,
    "نيسان": 4, "أيار": 5, "حزيران": 6,
    "تموز": 7, "آب": 8, "أيلول": 9,
    "تشرين الأول": 10, "تشرين الثاني": 11, "كانون الأول": 12
}

ARABIC_DAYS = {
    "Monday": "الاثنين",
    "Tuesday": "الثلاثاء",
    "Wednesday": "الأربعاء",
    "Thursday": "الخميس",
    "Friday": "الجمعة",
    "Saturday": "السبت",
    "Sunday": "الأحد"
}


def detect_language(text: str) -> str:
    """Return "arabic" if more than 30% of characters are Arabic, else "english"."""
    arabic_chars = sum(1 for c in text if '؀' <= c <= 'ۿ')
    total_chars = len(text.replace(" ", ""))
    if total_chars == 0:
        return "english"
    return "arabic" if (arabic_chars / total_chars) > 0.3 else "english"

def convert_arabic_numerals(text: str) -> str:
    """Convert Arabic-Indic numerals ٠١٢٣ to Western 0123"""
    arabic_indic = "٠١٢٣٤٥٦٧٨٩"
    western = "0123456789"
    for a, w in zip(arabic_indic, western):
        text = text.replace(a, w)
    return text

def is_date_question(question: str) -> bool:
    """Detect questions about dates (keywords in English/Arabic, or a year 20xx)."""
    keywords = [
        # English
        "what day", "which day", "what date",
        "day is", "day will", "day was",
        "when is", "when will", "when was",
        "how many days", "days until", "days between",
        "days from", "days ago", "next week",
        "next month", "next year", "tomorrow",
        "yesterday", "day after", "day before",
        "falls on", "what is the day",
        # Arabic
        "ما هو اليوم", "أي يوم", "كم يوم",
        "متى", "غداً", "غدا", "أمس",
        "الأسبوع القادم", "الشهر القادم", "السنة القادمة"
    ]

    has_year = bool(re.search(r'\b20\d{2}\b', question))
    has_date_word = any(kw in question.lower() for kw in keywords)

    return has_date_word or has_year


def extract_date_with_regex(question: str):
    """Fast regex extraction before trying model"""

    # Normalize Arabic numerals first
    question = convert_arabic_numerals(question)

    months_en = {
        "january": 1, "february": 2, "march": 3,
        "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9,
        "october": 10, "november": 11, "december": 12
    }

    # "30th April 2026" or "30 April 2026"
    pattern = r'(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})'
    match = re.search(pattern, question, re.IGNORECASE)

    if match:
        try:
            day = int(match.group(1))
            month = months_en[match.group(2).lower()]
            year = int(match.group(3))
            return datetime(year, month, day)
        except ValueError:
            pass  # invalid date, e.g. 31 February

    # "April 30 2026" or "April 30th 2026"
    pattern2 = r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\s+(\d{4})'
    match2 = re.search(pattern2, question, re.IGNORECASE)
    if match2:
        try:
            month = months_en[match2.group(1).lower()]
            day = int(match2.group(2))
            year = int(match2.group(3))
            return datetime(year, month, day)
        except ValueError:
            pass

    # dd/mm/yyyy or dd-mm-yyyy (day first)
    pattern3 = r'(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})'
    match3 = re.search(pattern3, question)
    if match3:
        try:
            return datetime(int(match3.group(3)), int(match3.group(2)), int(match3.group(1)))
        except ValueError:
            pass

    return None


def calculate_date_facts(question: str, model_extractor=None, today=None) -> str:
    """
    Python calculates all date facts — never wrong
    Returns a string of facts to inject into prompt
    """
    today = today or datetime.now() # tests can pass a fixed date
    
    lang = detect_language(question)
    result = []

    # Today as reference
    if lang == "arabic":
        result.append(f"اليوم هو {ARABIC_DAYS[today.strftime('%A')]}، {today.strftime('%d %B %Y')}.")
    else:
        result.append(f"Today is {today.strftime('%A, %d %B %Y')}.")


    # Regex first (fast); ask the model only if regex finds nothing and a model is available
    target_date = extract_date_with_regex(question)
    if target_date:
        print(f"[Date Extractor] Date found using Regex : {target_date.strftime('%d/%m/%Y')}")
    elif model_extractor:
        target_date = model_extractor(question)
        
    if target_date:
        day_name = target_date.strftime("%A")
        diff = (target_date.date() - today.date()).days
        formatted = target_date.strftime("%d %B %Y")

        if lang == "arabic":
            arabic_day = ARABIC_DAYS.get(day_name, day_name)
            result.append(f"{formatted} يقع في يوم {arabic_day}.")
            if diff > 0:
                result.append(f"بعد {diff} يوم من اليوم.")
            elif diff < 0:
                result.append(f"مضت {abs(diff)} يوم.")
            else:
                result.append("هذا هو اليوم.")
        else:
            result.append(f"{formatted} falls on a {day_name}.")
            if diff > 0:
                result.append(f"That is {diff} day{'s' if diff != 1 else ''} from today.")
            elif diff < 0:
                result.append(f"That was {abs(diff)} day{'s' if diff != 1 else ''} ago.")
            else:
                result.append("That is today.")

    # Handle relative date keywords
    q = question.lower()

    if "tomorrow" in q or "غداً" in question or "غدا" in question:
        t = today + timedelta(days=1)
        if lang == "arabic":
            result.append(f"غداً هو {ARABIC_DAYS[t.strftime('%A')]}، {t.strftime('%d %B %Y')}.")
        else:
            result.append(f"Tomorrow is {t.strftime('%A, %d %B %Y')}.")

    if "yesterday" in q or "أمس" in question:
        y = today - timedelta(days=1)
        if lang == "arabic":
            result.append(f"أمس كان {ARABIC_DAYS[y.strftime('%A')]}، {y.strftime('%d %B %Y')}.")
        else:
            result.append(f"Yesterday was {y.strftime('%A, %d %B %Y')}.")

    if "next week" in q or "الأسبوع القادم" in question:
        nw = today + timedelta(weeks=1)
        if lang == "arabic":
            result.append(f"هذا اليوم الأسبوع القادم: {ARABIC_DAYS[nw.strftime('%A')]}، {nw.strftime('%d %B %Y')}.")
        else:
            result.append(f"This day next week is {nw.strftime('%A, %d %B %Y')}.")

    if "next year" in q or "السنة القادمة" in question:
        try:
            ny = today.replace(year=today.year + 1)
        except ValueError:
            # Handle February 29 on non-leap years
            ny = today.replace(year=today.year + 1, day=28)
            
        if lang == "arabic":
            result.append(f"هذا اليوم السنة القادمة: {ARABIC_DAYS[ny.strftime('%A')]}، {ny.strftime('%d %B %Y')}.")
        else:
            result.append(f"This day next year is {ny.strftime('%A, %d %B %Y')}.")

    if "next month" in q or "الشهر القادم" in question:
        # Calculate Year and Month for next month; shortened if that month is shorter (31 Jan → 28/29 Feb)
        year, month = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)

        # Calculate Day for: Clamp to next month's last day, e.g. 31 Jan → 28 Feb (29 in leap years)
        day = min(today.day, calendar.monthrange(year, month)[1])

        nm = today.replace(year=year, month=month, day=day)

        if lang == "arabic":
            result.append(f"هذا اليوم الشهر القادم: {ARABIC_DAYS[nm.strftime('%A')]}، {nm.strftime('%d %B %Y')}.")
        else:
            result.append(f"This day next month is {nm.strftime('%A, %d %B %Y')}.")

    return " ".join(result)
