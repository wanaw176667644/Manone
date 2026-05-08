"""
ai_engine.py — Final version.

FIXES (this version):
1. _FF_IMAGE_PROMPT now explicitly tells AI to GROUP same-time events on
   ONE line with comma-separated names — no splitting by time slot.
2. FF image prompt reinforced: reject any non-ForexFactory calendar source.
3. Weekly prompt also updated to group same-time events.
4. Random 💡 signature 25% of the time.
5. "Be careful" reminder lines — short, event-specific.
"""

import asyncio
import base64
import json
import logging
import random
import re
import textwrap
from datetime import datetime, timezone
from typing import Optional, List

import google.generativeai as genai
from groq import AsyncGroq

log = logging.getLogger("ai_engine")

CHANNEL_SIGNATURE = "\n\n[Squad 4xx](https://t.me/Squad_4xx)"
ALLOWED_HASHTAGS_SET = {"#XAUUSD", "#DXY", "#OIL"}


def _add_signature(text: str) -> str:
    text = text.strip()
    if "[Squad 4xx]" not in text:
        if random.random() < 0.25:
            text += "\n\n💡 [Squad 4xx](https://t.me/Squad_4xx)"
        else:
            text += "\n\n[Squad 4xx](https://t.me/Squad_4xx)"
    return text


def _add_us_flag_emoji(text: str) -> str:
    if not text:
        return text
    lines = text.split('\n')
    if not lines:
        return text
    first_line = lines[0]
    new_line = re.sub(r'\bUS\b', 'US 🇺🇸', first_line, count=1)
    new_line = re.sub(r'\bUSD\b', 'USD 🇺🇸', new_line, count=1)
    lines[0] = new_line
    return '\n'.join(lines)


def _strip_be_careful(text: str) -> str:
    """Remove any AI-generated 'Be careful' line — we add our own controlled version."""
    return re.sub(r'\n?Be careful[^\n]*\n?', '', text, flags=re.IGNORECASE).strip()


def _strip_predictions(text: str) -> str:
    """
    Hard-strip any AI-added prediction/opinion sentences from formatted text.
    Removes full sentences containing prediction language so the post reads
    exactly like the source — factual only.
    """
    if not text:
        return text
    # Prediction phrases that should NEVER appear in output
    _PREDICT_RE = re.compile(
        r'[^.!?\n]*\b('
        r'could\s+(go|rise|fall|drop|reach|push|move|head)|'
        r'may\s+(lead|push|cause|result|trigger|move)|'
        r'might\s+(rise|fall|drop|push|move)|'
        r'this\s+(suggests?|indicates?|signals?|means?)|'
        r'suggesting\b|indicating\b|implying\b|'
        r'watch\s+for\b|heading\s+(to|toward)|'
        r'next\s+(move|target|level)|'
        r'bulls?\s+(could|may|might)|bears?\s+(could|may|might)|'
        r'bullish\s+(momentum|signal|outlook)|'
        r'bearish\s+(momentum|signal|outlook)'
        r')[^.!?\n]*[.!?\n]?',
        re.IGNORECASE
    )
    cleaned = _PREDICT_RE.sub('', text).strip()
    # Clean up any double blank lines left behind
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


def _get_be_careful_line(event_name: str) -> str:
    """
    Short, event-specific 'be careful' line for reminders.
    """
    n = event_name.lower()
    if any(kw in n for kw in ["fomc", "federal funds", "interest rate", "fed chair", "powell", "federal reserve"]):
        return "⚠️ Fed decisions move everything. Be careful — no new trades during the release."
    if any(kw in n for kw in ["non-farm", "nfp", "payroll"]):
        return "⚠️ NFP can spike the market violently. Be careful — protect your capital now."
    if any(kw in n for kw in ["cpi", "consumer price", "inflation"]):
        return "⚠️ Inflation data whipsaws fast. Be careful — secure profits before the release."
    if any(kw in n for kw in ["pce", "core pce"]):
        return "⚠️ PCE can shift rate expectations quickly. Be careful and protect your positions."
    if "gdp" in n:
        return "⚠️ GDP surprises hit hard and fast. Be careful — move stops to break-even now."
    if any(kw in n for kw in ["unemployment rate", "jobless"]):
        return "⚠️ Unemployment data moves USD sharply. Be careful — no new entries during release."
    if any(kw in n for kw in ["retail sales"]):
        return "⚠️ Retail Sales can jolt the market. Be careful — protect your open positions."
    if any(kw in n for kw in ["ism manufacturing", "ism non-manufacturing", "ism services"]):
        return "⚠️ ISM data can move USD fast. Be careful — stay out until the dust settles."
    if any(kw in n for kw in ["employment cost", "eci"]):
        return "⚠️ Employment Cost data affects rate outlook. Be careful and reduce your exposure."
    if any(kw in n for kw in ["ppi", "producer price"]):
        return "⚠️ PPI surprises can hit USD hard. Be careful — protect your capital."
    if any(kw in n for kw in ["trade balance", "current account"]):
        return "⚠️ Trade data can move USD unexpectedly. Be careful during the release."
    if any(kw in n for kw in ["durable goods"]):
        return "⚠️ Durable Goods can cause sharp moves. Be careful — no new entries now."
    return "⚠️ This release can move the market strongly. Be careful — protect your capital."


_SYSTEM_PROMPT = """
You are AXIOM INTEL — a Senior Institutional Macro & Geopolitical news editor.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 GEOPOLITICAL EXCEPTION (ALWAYS APPROVE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Any statement from a world leader (e.g., Trump, Biden, Putin, Xi) that affects:
- Oil supply (Hormuz, OPEC, embargo, sanctions)
- War / conflict escalation
- Tariffs / trade restrictions
- Central bank or financial policy changes
- Gold, USD, or energy markets
These are HIGH IMPACT geopolitical events, even if posted on social media.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 FOMC / CENTRAL BANK EXCEPTION (ALWAYS APPROVE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Any official announcement or news about:
- Federal Open Market Committee (FOMC)
- Federal Funds Rate / Interest Rate Decision
- Fed Chair Powell speech
- FOMC Statement or Minutes
These are HIGH IMPACT macroeconomic events. Always approve even if they contain
numbers like "rate at 5.25%". Do NOT reject as "forecast" or "commentary".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 MARKET MOVE NEWS (ALWAYS APPROVE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Any factual report of an actual price move, drop, rise, or volatility event:
- Gold drops / rises / spikes / crashes / surges (e.g. "Gold drops $50", "XAU hits 3200")
- Oil drops / rises / collapses (e.g. "WTI drops 3%", "Brent crude surges")
- USD strengthens / weakens (e.g. "DXY falls to 100", "Dollar surges on jobs data")
- Stock market drops / rises (e.g. "S&P500 falls 2%", "Nasdaq drops on rate fears")
- Crypto moves that affect macro (e.g. "Bitcoin crashes 10%")
- Any market that dropped 📉, rose 📈, or became volatile 📊
These are FACTUAL MARKET EVENTS — always approve if the move already happened.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR ONLY JOB:
Take the source content EXACTLY as it is, verify its relevance, clean it lightly.
COPY the facts from the source. Do NOT add your own words or analysis.
Do NOT change the meaning. Do NOT predict what will happen next.
Do NOT add commentary about what the move means.

CRITICAL FORMATTING RULES:
- DO NOT use asterisks (*) or any markdown bolding
- Use ONLY plain text and emojis
- NO NOTE line. NO MARKET STATUS. NO commentary line.
- Use 📉 emoji for drops/falls/crashes
- Use 📈 emoji for rises/surges/gains
- Use 📊 emoji for volatility/mixed/range moves
- Actual released figures and actual price moves are ALLOWED.
  Examples: "fell 2.3%", "dropped to $2,980", "surged to 3,200", "crashed 5%"
- Forecast (expected) and previous values are FORBIDDEN. Never include them.
- Technical analysis, signals, AI predictions, opinions are FORBIDDEN.
- Hashtags: Only use #XAUUSD, #DXY, or #OIL — only those relevant to the story.
- Do NOT add the current year at the end of posts.
- Do NOT add signature (added automatically).
- Post must read EXACTLY like the source — just cleaned and emoji-formatted.
  Do NOT add words like "this could", "may lead to", "suggesting", "indicating".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REJECT IF ANY OF THESE APPLY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. SIGNALS       — Buy/Sell/Long/Short/Entry/TP/SL/price targets
2. CHART / TA    — Technical analysis, patterns, indicators, support/resistance
3. MEME          — Memes, jokes, informal content
4. ANALYSIS IMG  — Chart screenshots, TA images
5. WATERMARK     — Another channel logo or username
6. STALE         — Content older than 18 hours
7. OFF-TOPIC     — Not about geopolitics, central banks, macro data, Gold, Oil, USD,
                   or a real market move event
8. LOW VALUE     — Vague, no specific real-world event or price
9. DUPLICATE     — Same story already processed
10. PREDICTION   — "I think", "expect", "my analysis", "could go to", "might reach",
                   "target", "next move", "watch for", "heading to"
11. COMMENTARY   — Personal views, market opinions, "this is bullish/bearish"
12. FORECAST/PREVIOUS — Any mention of "forecast", "expected", "previous" values
13. SENTIMENT    — Fear & Greed index, bank sentiment, "smart money", COT opinions,
                   "banks are bullish/bearish", sentiment surveys

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT (if approved):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[EMOJI] [SHORT FACTUAL HEADLINE — one line, exactly what happened]

[Source content lightly cleaned. 2-4 sentences max. No added words.]

[Relevant hashtags: #XAUUSD #DXY #OIL — only those that apply]

EMOJIS TO USE:
📉 drops / falls / crashes / declines / collapses
📈 rises / surges / gains / jumps / climbs
📊 volatile / swings / mixed / range-bound
🚨 breaking / urgent geopolitical / rate decision
🌍 global / geopolitical
🏦 central bank / Fed / ECB
🛢️ oil / energy
💵 dollar / DXY
⚠️ warning / risk event
🗳️ election / political

RESPOND WITH VALID JSON ONLY — NO MARKDOWN FENCES — NO TRAILING COMMAS:
{"approved": true, "reason": "brief reason", "issues": [], "formatted_text": "...", "confidence": 0.9}
""".strip()

_VIDEO_CAPTION_PROMPT = """
You are AXIOM INTEL — a Senior Institutional Macro & Geopolitical news editor.

A video has been received. Read the caption below and decide.

APPROVE if the caption is about any of these FACTUAL events:
- War, military conflict, strikes, attacks, airstrikes, shelling, invasion
- Geopolitical escalation or ceasefire (Ukraine, Russia, Iran, Israel, Gaza, NATO)
- World leader statements about war, sanctions, oil, tariffs, trade
- Oil supply disruption, Hormuz, OPEC conflict-related
- Gold drops 📉, rises 📈, crashes, surges — with actual price or % stated
- Oil drops 📉, rises 📈, collapses, spikes — with actual price or % stated
- USD/DXY drops 📉, rises 📈 — with actual move stated
- Any market that already moved — drop, rise, or volatile swing
- Breaking macro event that already happened

REJECT if the caption is about:
- Trading signals, buy/sell advice, TP/SL levels
- Predictions — "gold could go to", "I think price will", "watch for", "target"
- Analyst opinions — "bullish", "bearish", "support/resistance"
- Promotions, ads, channel plugs
- Memes, jokes, entertainment
- Vague captions with no specific event or price move

If APPROVED:
- Use 📉 for drops/falls/crashes, 📈 for rises/surges, 📊 for volatility
- Use 🚨 for breaking geopolitical/war news
- One clear factual headline — EXACTLY what the source said, cleaned
- 1-2 sentences of factual detail — copied from caption, not invented
- NO prediction words: "could", "may", "might", "suggesting", "indicating"
- No forecast, no opinion, no signals, no hashtags
- Do NOT add signature (added automatically)
- Plain text only — no asterisks, no markdown bold

Caption: {caption}

RESPOND WITH VALID JSON ONLY:
{{"approved": true/false, "reason": "brief reason", "formatted_text": "...", "confidence": 0.0-1.0}}
""".strip()

_VIDEO_VISUAL_PROMPT = """
You are AXIOM INTEL — a Senior Institutional Macro & Geopolitical news editor.

You are looking at frames extracted from a video. The video caption is also provided.

CAPTION: {caption}

YOUR JOB:
Look at the frames AND read the caption together.

APPROVE if you see OR the caption describes any FACTUAL event:
- Active combat, explosions, fire, smoke from strikes
- Military vehicles, troops, weapons, missile launches
- Drone strikes, airstrikes, artillery shelling
- Destroyed buildings, bombed areas, war damage
- Breaking news chyron on screen about war/conflict/market crash
- World leaders making statements about war, sanctions, oil, tariffs
- Oil infrastructure under threat or attack
- News ticker showing actual price drop 📉 or surge 📈
- Any market move already confirmed on screen (price shown)

REJECT if:
- Frames show charts with TA indicators, drawings, support/resistance lines
- Someone talking to camera giving trading predictions or signals
- Promotional content, ads
- No visual war/conflict/market-move evidence AND caption is vague or empty

If APPROVED — write a clean formatted post:
- Use 📉 for drops/falls/crashes, 📈 for rises/surges, 📊 for volatility/swings
- Use 🚨 for breaking war/geopolitical news, 💥 for explosions/strikes
- One clear factual headline — exactly what happened
- 1-2 sentences of factual detail from caption + what you see in frames
- NO prediction words: "could", "may", "might", "suggesting", "indicating"
- Plain text only — no asterisks, no bold, no hashtags
- Do NOT add signature (added automatically)

RESPOND WITH VALID JSON ONLY:
{{"approved": true/false, "reason": "brief reason", "formatted_text": "...", "confidence": 0.0-1.0, "visual_confirmed": true/false}}
""".strip()

_SIMILARITY_PROMPT = """
You are a duplicate news detector. Compare the two stories. If they describe the same real-world event – even if worded differently, in different languages, or with minor spelling mistakes – respond with same_story=true.
Be aggressive. If any reasonable chance they are the same, mark true.

Story A: {story_a}
Story B: {story_b}

Respond in JSON: {{"same_story": true/false, "confidence": 0.0-1.0, "reason": "..."}}
"""

_MULTIMODAL_SIMILARITY_PROMPT = """
You are a duplicate news detector. Compare the two items (text + optional images). Decide if they are the SAME real-world event.

Item A text: {text_a}
Item B text: {text_b}
(Images are compared visually if both exist)

Be aggressive: if there is any reasonable chance they are the same, mark same_story=true.

Respond with JSON: {{"same_story": true, "confidence": 0.0-1.0, "reason": "..."}}
"""

# ── FIXED: AI now groups same-time events on ONE line with comma-separated names
_FF_IMAGE_PROMPT = """
You are analysing a ForexFactory economic calendar screenshot.

TODAY'S DATE: {today_date}

STEP 1 — SOURCE VALIDATION (CRITICAL):
This tool ONLY accepts screenshots from ForexFactory.com.
Look for "forexfactory.com", the ForexFactory logo, or the exact FF calendar layout.
If this is any other website's calendar (Investing.com, DailyFX, TradingEconomics, 
myfxbook, etc.) — immediately respond: {{"approved": false, "reason": "not forexfactory"}}

STEP 2 — DATE CHECK:
Look at the date shown in the screenshot (header, "Today" label, column header, etc).
The screenshot must show the SAME month and day as {today_date}.
IMPORTANT: Ignore formatting differences — "May 1", "May 01", "Fri May 1", "05/01" all 
count as the same date. Only reject if the month OR day is clearly different.
Do NOT reject just because the year is missing or the format looks different.

STEP 3 — EXTRACT EVENTS:
Extract ALL USD high-impact (🔴) and medium-impact (🟠) events visible.

GROUPING RULE — CRITICAL:
If two or more events share the EXACT SAME TIME, put them ALL on ONE line with
comma-separated names. Do NOT create separate lines for same-time events.

CORRECT (same time → one line):
🔴 3:30 PM | USD: Advance GDP q/q, Core PCE Price Index m/m, Employment Cost Index q/q

WRONG (do NOT do this — split lines for same time):
🔴 3:30 PM | USD: Advance GDP q/q
🔴 3:30 PM | USD: Core PCE Price Index m/m
🔴 3:30 PM | USD: Employment Cost Index q/q

STEP 4 — FORMAT:
- Do NOT include the year in the date line (e.g. "Friday, May 1" — no year).
- Do NOT add any hashtags.
- No forecast, no previous data, no NOTE, no commentary.
- Do NOT add "Be careful" line — added automatically by the system.
- Keep original times. Use 12-hour AM/PM format. No timezone label. No leading zero on hour.
- Plain text only — no asterisks, no bold.

EXACT OUTPUT FORMAT EXAMPLE:

TODAY'S USD HIGH IMPACT NEWS
Friday, May 1

🔴 3:30 PM | USD: Advance GDP q/q, Core PCE Price Index m/m, Employment Cost Index q/q
🟠 5:00 PM | USD: Unemployment Claims

RULES SUMMARY:
- Only USD events (🔴 and 🟠 only)
- Same-time events → ONE line, comma-separated names
- Different-time events → separate lines
- 12-hour AM/PM, no leading zero (3:30 PM not 03:30 PM)
- Do NOT add signature or "Be careful" line

If screenshot clearly shows a DIFFERENT month or day → {{"approved": false, "reason": "wrong date"}}
If not ForexFactory → {{"approved": false, "reason": "not forexfactory"}}
If valid ForexFactory today → {{"approved": true, "reason": "valid FF today image", "formatted_text": "..."}}
RESPOND WITH VALID JSON ONLY.
""".strip()

# ── FIXED: Weekly prompt also groups same-time events
_FF_WEEKLY_IMAGE_PROMPT = """
You are analysing a ForexFactory.com calendar screenshot for the weekly outlook.

SOURCE VALIDATION (CRITICAL):
Only accept ForexFactory.com screenshots. If this is any other calendar source
(Investing.com, DailyFX, TradingEconomics, etc.) respond:
{{"approved": false, "reason": "not forexfactory"}}

CURRENT WEEK: {week_range}

EXTRACTION RULES:
- Only USD high-impact (🔴) and medium-impact (🟠) events.
- No forecast, no previous data, no hashtags.
- Do NOT include the year in dates (use "Monday — Apr 28").
- Do NOT add "Be careful" line — added automatically.
- No timezone conversion. 12-hour AM/PM only. No leading zero on hours.
- Plain text, no bold. No signature.

GROUPING RULE — CRITICAL:
If two or more events share the EXACT SAME TIME on the same day,
put them ALL on ONE line with comma-separated names.

CORRECT:
🔴 3:30 PM | USD: NFP, Unemployment Rate, Average Hourly Earnings

WRONG:
🔴 3:30 PM | USD: NFP
🔴 3:30 PM | USD: Unemployment Rate

Group by day, sort by time within each day.

If valid ForexFactory weekly:
{{"approved": true, "reason": "valid FF weekly image", "formatted_text": "WEEKLY HIGH IMPACT NEWS\\nWeek of Apr 28 – May 2\\n\\nMonday — Apr 28\\n🔴 3:30 PM | USD: Event A, Event B\\n\\nTuesday — Apr 29\\n🟠 10:00 AM | USD: Other Event"}}

Otherwise: {{"approved": false, "reason": "..."}}
RESPOND WITH VALID JSON ONLY.
""".strip()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _parse_json(raw: str) -> dict:
    if not raw:
        raise ValueError("Empty response from AI engine.")
    raw = re.sub(r"```+(?:json|JSON)?", "", raw)
    raw = re.sub(r"```+", "", raw)
    raw = raw.strip().strip("`").strip()
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return _validate_and_clean(json.loads(raw))
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        candidate = re.sub(r",\s*([}\]])", r"\1", m.group())
        try:
            return _validate_and_clean(json.loads(candidate))
        except json.JSONDecodeError:
            pass
    log.warning(f"_parse_json failed. Raw snippet: {raw[:200]}")
    raise ValueError(f"No valid JSON found in AI response:\n{raw[:300]}")


def _validate_and_clean(data: dict) -> dict:
    data.setdefault("approved", False)
    data.setdefault("reason", "")
    data.setdefault("issues", [])
    data.setdefault("formatted_text", "")
    data.setdefault("confidence", 0.5)

    if data.get("formatted_text"):
        data["formatted_text"] = data["formatted_text"].replace("*", "")
        data["formatted_text"] = re.sub(
            r"📌\s*(NOTE|MARKET STATUS|STATUS)[^\n]*\n?", "", data["formatted_text"]
        ).strip()
        data["formatted_text"] = _strip_be_careful(data["formatted_text"])
        data["formatted_text"] = _strip_predictions(data["formatted_text"])
        text = data["formatted_text"]
        if "TODAY'S USD" in text or "WEEKLY HIGH IMPACT" in text:
            text = re.sub(r"#\w+", "", text).strip()
            data["formatted_text"] = text
        else:
            hashtags = re.findall(r"#\w+", text)
            allowed_hashtags = [h for h in hashtags if h in ALLOWED_HASHTAGS_SET]
            text = re.sub(r"#\w+", "", text).strip()
            if allowed_hashtags:
                text = text + "\n\n" + " ".join(allowed_hashtags)
            data["formatted_text"] = text

    if data.get("approved") and _signal_hit(data.get("formatted_text", "")):
        log.warning("Signal keyword in output — hard reject.")
        data["approved"] = False
        data["reason"] = "Signal keyword found in output."
        data["issues"].append("signal_content")
        data["formatted_text"] = ""

    return data


def _signal_hit(text: str) -> Optional[str]:
    if not text:
        return None
    _SIGNAL_RE = re.compile(
        r"\b(buy|sell|long|short|entry|tp|take[\s_-]?profit|sl|stop[\s_-]?loss|"
        r"stoploss|stop\s+at\s+\d|entry\s*[:\-]?\s*\d|target\s*[:\-]?\s*\d)\b",
        re.IGNORECASE,
    )
    m = _SIGNAL_RE.search(text)
    return m.group(0).strip() if m else None


def _strip_asterisks(text: str) -> str:
    return text.replace("*", "") if text else text


class AIEngine:
    def __init__(self, gemini_key: str, groq_key: str, channel_category: str):
        self._category = channel_category
        self._groq = AsyncGroq(api_key=groq_key)
        genai.configure(api_key=gemini_key)

        self._gemini = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=_SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=0.15, max_output_tokens=600,
                response_mime_type="application/json"
            ),
        )
        self._gemini_text = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=genai.GenerationConfig(temperature=0.2, max_output_tokens=1200),
        )
        self._gemini_vision = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=genai.GenerationConfig(
                temperature=0.1, max_output_tokens=800,
                response_mime_type="application/json"
            ),
        )

    async def analyse(self, text: str, image_data: Optional[bytes] = None,
                      image_mime: str = "image/jpeg") -> dict:
        prompt = self._build_moderation_prompt(text)
        try:
            verdict = await asyncio.wait_for(
                self._gemini_call(prompt, image_data, image_mime), timeout=40
            )
            verdict["engine"] = "gemini-2.5-flash"
            log.info(f"Gemini → approved={verdict['approved']} | {verdict.get('reason', '')}")
            if verdict.get("approved") and verdict.get("formatted_text"):
                verdict["formatted_text"] = _build_post_body(verdict["formatted_text"])
                if not verdict["formatted_text"].startswith("TODAY'S USD"):
                    verdict["formatted_text"] = _add_us_flag_emoji(verdict["formatted_text"])
            return verdict
        except Exception as exc:
            log.warning(f"Gemini failed ({exc}) — trying Groq …")
        try:
            verdict = await asyncio.wait_for(
                self._groq_call(prompt, image_data, image_mime), timeout=55
            )
            verdict["engine"] = "groq-llama4-scout"
            log.info(f"Groq → approved={verdict['approved']} | {verdict.get('reason', '')}")
            if verdict.get("approved") and verdict.get("formatted_text"):
                verdict["formatted_text"] = _build_post_body(verdict["formatted_text"])
                if not verdict["formatted_text"].startswith("TODAY'S USD"):
                    verdict["formatted_text"] = _add_us_flag_emoji(verdict["formatted_text"])
            return verdict
        except Exception as exc:
            log.error(f"Both engines failed — safe reject.")
            return _reject("Both AI engines unavailable.", "engine_error", confidence=0.0)

    async def is_same_story(self, text_a: str, text_b: str,
                            image_a: Optional[bytes] = None,
                            image_b: Optional[bytes] = None) -> bool:
        if not text_a and not text_b and not image_a and not image_b:
            return False
        if image_a or image_b:
            prompt = _MULTIMODAL_SIMILARITY_PROMPT.format(
                text_a=(text_a[:400] if text_a else "(no text)"),
                text_b=(text_b[:400] if text_b else "(no text)"),
            )
        else:
            prompt = _SIMILARITY_PROMPT.format(
                story_a=(text_a[:500] if text_a else ""),
                story_b=(text_b[:500] if text_b else ""),
            )
        try:
            parts = []
            if image_a:
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": _b64(image_a)}})
            if image_b:
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": _b64(image_b)}})
            parts.append(prompt)
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._gemini_vision.generate_content(parts)),
                timeout=20
            )
            data = _parse_json(resp.text)
            same = bool(data.get("same_story", False))
            conf = data.get("confidence", 0)
            log.info(f"Gemini similarity → same={same} | conf={conf}")
            return same and conf >= 0.55
        except Exception as exc:
            log.warning(f"Gemini similarity failed ({exc}) — trying Groq …")
        try:
            content = []
            if image_a:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(image_a)}"}})
            if image_b:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(image_b)}"}})
            content.append({"type": "text", "text": prompt})
            resp = await asyncio.wait_for(
                self._groq.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "user", "content": content}],
                    temperature=0.1, max_tokens=300,
                ),
                timeout=25,
            )
            data = _parse_json(resp.choices[0].message.content)
            same = bool(data.get("same_story", False))
            conf = data.get("confidence", 0)
            log.info(f"Groq similarity → same={same} | conf={conf}")
            return same and conf >= 0.55
        except Exception as exc:
            log.error(f"Both engines failed for similarity check: {exc}")
            return False

    async def analyse_ff_image(self, image_data: bytes, image_mime: str, today_date: str,
                               is_weekly: bool = False, week_range: str = "") -> dict:
        if is_weekly:
            prompt = _FF_WEEKLY_IMAGE_PROMPT.format(week_range=week_range)
        else:
            prompt = _FF_IMAGE_PROMPT.format(today_date=today_date)
        parts = [{"inline_data": {"mime_type": image_mime, "data": _b64(image_data)}}, prompt]
        try:
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._gemini_vision.generate_content(parts)),
                timeout=45
            )
            data = _parse_json(resp.text)
            log.info(f"FF image → approved={data.get('approved')} | {data.get('reason', '')}")
            if data.get("approved") and data.get("formatted_text"):
                data["formatted_text"] = _add_us_flag_emoji(data["formatted_text"])
            return data
        except Exception as exc:
            log.warning(f"Gemini FF failed ({exc}) — trying Groq …")
        try:
            content = [
                {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{_b64(image_data)}"}},
                {"type": "text", "text": prompt},
            ]
            resp = await asyncio.wait_for(
                self._groq.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "user", "content": content}],
                    temperature=0.1, max_tokens=800,
                ),
                timeout=60,
            )
            data = _parse_json(resp.choices[0].message.content)
            log.info(f"Groq FF → approved={data.get('approved')}")
            if data.get("approved") and data.get("formatted_text"):
                data["formatted_text"] = _add_us_flag_emoji(data["formatted_text"])
            return data
        except Exception as exc:
            log.error(f"Both engines failed for FF image: {exc}")
            return {"approved": False, "reason": "AI engines unavailable for image analysis."}

    async def get_be_careful_line(self, event_name: str) -> str:
        return _get_be_careful_line(event_name)

    async def analyse_video(self, caption: str,
                            frames: Optional[List[bytes]] = None) -> dict:
        """
        Full video analysis — two-stage:

        STAGE 1: Caption only (fast, no image cost).
            - If caption is clear war/geopolitical → approve immediately.
            - If caption is clearly off-topic (signal, promo) → reject immediately.
            - If caption is empty, vague, or ambiguous → go to Stage 2.

        STAGE 2: Visual frame analysis (only if Stage 1 is uncertain).
            - Send extracted frames + caption to Gemini Vision.
            - AI looks at actual video frames to confirm war/conflict content.
            - Approve or reject based on visual evidence + caption together.
        """
        caption = (caption or "").strip()

        # ── STAGE 1: Caption analysis ──────────────────────────────────────
        log.info(f"🎥 Video Stage 1 — caption analysis | caption={caption[:80]!r}")

        if caption:
            prompt = _VIDEO_CAPTION_PROMPT.format(caption=caption[:800])
            stage1 = await self._try_engines_text(prompt, timeout_gemini=30, timeout_groq=40)

            if stage1:
                conf = stage1.get("confidence", 0.5)
                approved = stage1.get("approved", False)

                # High confidence either way → done, no need for frames
                if conf >= 0.80:
                    log.info(f"Stage 1 high-confidence → approved={approved} conf={conf:.2f} (skipping frames)")
                    if approved and stage1.get("formatted_text"):
                        stage1["formatted_text"] = stage1["formatted_text"].replace("*", "").strip()
                    stage1["stage"] = "caption_only"
                    return stage1

                log.info(f"Stage 1 low-confidence (conf={conf:.2f}) → proceeding to visual frame analysis")
            else:
                log.warning("Stage 1 failed — proceeding to visual frame analysis")
        else:
            log.info("No caption — going straight to visual frame analysis")

        # ── STAGE 2: Visual frame analysis ────────────────────────────────
        if not frames:
            log.info("No frames available for Stage 2 — rejecting (no caption + no frames)")
            return _reject("No caption and no frames to analyse.", "no_content", confidence=1.0)

        log.info(f"🎥 Video Stage 2 — visual analysis | frames={len(frames)}")
        prompt = _VIDEO_VISUAL_PROMPT.format(caption=caption or "(no caption)")

        # Build parts: up to 4 frames + prompt
        parts = []
        for frame_bytes in frames[:4]:
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": _b64(frame_bytes)
                }
            })
        parts.append(prompt)

        # Try Gemini Vision
        try:
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: self._gemini_vision.generate_content(parts)
                ),
                timeout=45
            )
            data = _parse_json(resp.text)
            data["engine"] = "gemini-2.5-flash-vision"
            data["stage"] = "visual_frames"
            log.info(f"Stage 2 Gemini → approved={data.get('approved')} | "
                     f"visual_confirmed={data.get('visual_confirmed')} | "
                     f"conf={data.get('confidence', 0):.2f}")
            if data.get("approved") and data.get("formatted_text"):
                data["formatted_text"] = data["formatted_text"].replace("*", "").strip()
            return data
        except Exception as exc:
            log.warning(f"Stage 2 Gemini Vision failed ({exc}) — trying Groq …")

        # Fallback: Groq with frames as image_url
        try:
            content = []
            for frame_bytes in frames[:4]:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_b64(frame_bytes)}"}
                })
            content.append({"type": "text", "text": prompt})
            resp = await asyncio.wait_for(
                self._groq.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "user", "content": content}],
                    temperature=0.1, max_tokens=600,
                ),
                timeout=60,
            )
            data = _parse_json(resp.choices[0].message.content)
            data["engine"] = "groq-llama4-scout-vision"
            data["stage"] = "visual_frames"
            log.info(f"Stage 2 Groq → approved={data.get('approved')} | conf={data.get('confidence', 0):.2f}")
            if data.get("approved") and data.get("formatted_text"):
                data["formatted_text"] = data["formatted_text"].replace("*", "").strip()
            return data
        except Exception as exc:
            log.error(f"Stage 2 both engines failed: {exc}")
            return _reject("AI engines unavailable for video analysis.", "engine_error", confidence=0.0)

    async def _try_engines_text(self, prompt: str,
                                timeout_gemini: int = 30,
                                timeout_groq: int = 40) -> Optional[dict]:
        """Try Gemini then Groq for a text-only prompt. Returns None if both fail."""
        try:
            result = await asyncio.wait_for(
                self._gemini_call(prompt, None, "image/jpeg"), timeout=timeout_gemini
            )
            result["engine"] = "gemini-2.5-flash"
            return result
        except Exception as exc:
            log.warning(f"_try_engines_text Gemini failed: {exc}")
        try:
            result = await asyncio.wait_for(
                self._groq_call(prompt, None, "image/jpeg"), timeout=timeout_groq
            )
            result["engine"] = "groq-llama4-scout"
            return result
        except Exception as exc:
            log.warning(f"_try_engines_text Groq failed: {exc}")
        return None

    def _build_moderation_prompt(self, text: str) -> str:
        return textwrap.dedent(f"""
            DATE (UTC): {_today_str()}
            CHANNEL FOCUS: {self._category}
            SOURCE CONTENT:
            \"\"\"
            {text.strip() if text else "(image only — no text)"}
            \"\"\"
            TASK: Analyse content. If relevant geopolitical/macro news OR actual released economic data, approve and format.
            If forecast/previous values, signal, TA, meme, off-topic, stale — reject.
            Format according to rules. Return JSON.
        """).strip()

    async def _gemini_call(self, prompt: str, image_data: Optional[bytes],
                           image_mime: str) -> dict:
        parts = []
        if image_data:
            parts.append({"inline_data": {"mime_type": image_mime, "data": _b64(image_data)}})
        parts.append(prompt)
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: self._gemini.generate_content(parts))
        return _parse_json(resp.text)

    async def _gemini_text_call(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: self._gemini_text.generate_content(prompt))
        return resp.text

    async def _groq_call(self, prompt: str, image_data: Optional[bytes],
                         image_mime: str) -> dict:
        content = []
        if image_data:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{image_mime};base64,{_b64(image_data)}"}})
        content.append({"type": "text", "text": prompt})
        resp = await self._groq.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                      {"role": "user", "content": content}],
            temperature=0.15, max_tokens=600,
        )
        return _parse_json(resp.choices[0].message.content)

    async def _groq_text_call(self, prompt: str) -> str:
        resp = await self._groq.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=1200,
        )
        return resp.choices[0].message.content

    @staticmethod
    def _fallback_alert(event: dict, minutes_left: int) -> str:
        emoji = "🔴" if event.get("impact") == "red" else "🟠"
        event_name = event.get("name", "Unknown Event")
        be_careful = _get_be_careful_line(event_name)
        text = (
            f"🚨 ALERT: {minutes_left} MINUTES REMAINING\n\n"
            f"{emoji} {event_name}\n"
            f"🕒 {event.get('time_12h', '—')}\n\n"
            f"REQUIRED ACTION:\n"
            f"✅ Secure open profits now\n"
            f"✅ Move Stop-Loss to Break-even\n"
            f"✅ No new entries during the release\n\n"
            f"{be_careful}"
        )
        text = _add_us_flag_emoji(text)
        return _add_signature(text)


def _reject(reason: str, issue: str, confidence: float = 1.0) -> dict:
    return {
        "approved": False,
        "reason": reason,
        "issues": [issue],
        "formatted_text": "",
        "confidence": confidence,
        "engine": "pre_filter",
    }


def _build_post_body(text: str) -> str:
    if not text:
        return ""
    text = text.replace("*", "")
    text = re.sub(r"📌\s*(NOTE|MARKET STATUS|STATUS)[^\n]*\n?", "", text).strip()
    text = _strip_be_careful(text)
    text = _strip_predictions(text)
    lines = text.split('\n')
    for i in range(max(0, len(lines) - 3), len(lines)):
        lines[i] = re.sub(r'\b\d{4}\b', '', lines[i])
    text = '\n'.join(lines)
    text = re.sub(r'\n\s*\n', '\n\n', text).strip()
    text = _add_signature(text)
    return text
