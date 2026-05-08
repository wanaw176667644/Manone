"""
ai_engine.py — Fixed version.

CHANGES (this version):
1. _VIDEO_CAPTION_PROMPT completely rewritten:
   - AI preserves the ORIGINAL caption language — no rewriting
   - Only removes: other channel usernames, URLs, foreign-channel hashtags
   - Adds ONE lead emoji at the start + trend emoji at end of first line only
   - Zero AI analysis, predictions, or commentary allowed
   - ONLY approves: physical war events (missile, strike, airstrike, attack, explosion)
     in regions that directly affect Oil or Gold prices
2. Trend emoji (📈 📉 📊) placed INLINE in the headline by the AI, not as a footer line.
3. Hashtags remain #XAUUSD / #DXY / #OIL — no trailing emoji line after hashtags.
4. System prompt tightened: ONLY news that moves Gold (XAUUSD), Oil, or DXY is approved.
5. Random 💡 signature 25% of the time.
6. "Be careful" reminder lines — short, event-specific.
"""

import asyncio
import base64
import json
import logging
import random
import re
import textwrap
from datetime import datetime, timezone
from typing import Optional

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
You are AXIOM INTEL — a Senior Institutional Macro & Geopolitical news editor for a FOREX TRADING channel.

THIS CHANNEL TRADES: Gold (XAUUSD) | Oil (WTI/Brent) | US Dollar Index (DXY)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔑 CORE RULE — MARKET IMPACT FILTER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ONLY approve news that DIRECTLY moves at least one of:
  • Gold / XAUUSD
  • Oil / Crude / WTI / Brent
  • DXY / US Dollar Index

If a piece of news does NOT clearly affect Gold, Oil, or DXY → REJECT IT.
Do not approve vague or indirect connections. Be strict.

Examples of what to REJECT (even if real news):
- Stock market news (S&P, Nasdaq, earnings) unless it directly hits DXY/Gold
- Crypto news
- Regional economic data not tied to USD, Oil, or Gold
- Company-specific news
- Social/political news with no commodity or currency market link

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 GEOPOLITICAL EXCEPTION (APPROVE IF MARKET IMPACT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Approve geopolitical events ONLY if they directly affect Gold, Oil, or DXY:
- War / conflict affecting oil-producing regions (Middle East, Russia, Iran, Hormuz)
- Sanctions or embargoes on oil-exporting nations
- World leader statements about oil supply, tariffs, USD policy, or gold reserves
- Conflict escalation that triggers safe-haven demand for Gold
- Any event that causes a direct flight-to-safety into Gold or Oil spike

Reject geopolitical news that has NO clear link to Gold, Oil, or DXY.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 FOMC / CENTRAL BANK EXCEPTION (ALWAYS APPROVE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Always approve:
- FOMC decisions, Federal Funds Rate, Fed Chair Powell speeches
- FOMC statements or minutes
- Any official central bank rate decision affecting USD
These always move DXY and Gold. Always approve even with numbers like "rate at 5.25%".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR ONLY JOB:
Take the source content, verify its relevance to Gold/Oil/DXY, and format it cleanly.
Do NOT speculate. Do NOT add analysis beyond the facts.
Do NOT change the meaning.

CRITICAL FORMATTING RULES:
- DO NOT use asterisks (*) or any markdown bolding
- Use ONLY plain text and emojis
- NO NOTE line. NO MARKET STATUS. NO commentary line.
- Actual released figures (e.g., "came at 2.5%", "rose to 2.5%", "was 2.5%") are ALLOWED.
- Forecast (expected) and previous values are FORBIDDEN. Never include them.
- Technical analysis, signals, predictions, opinions are FORBIDDEN.
- Hashtags: Use ONLY #XAUUSD, #DXY, or #OIL — only the ones relevant to the story.
- Do NOT add the current year at the end of posts.
- Do NOT add signature (added automatically).

TREND EMOJI RULE — CRITICAL:
Place ONE trend emoji at the END of the headline (first line), chosen by price direction:
  📈 — price rising, hitting highs, surging, gaining, jumping
  📉 — price dropping, falling, declining, hitting lows, crashing
  📊 — volatile, mixed, uncertain, no clear direction, whipsaw

EXAMPLES:
  Gold hits record high above $3,400 📈
  Oil drops sharply on OPEC supply fears 📉
  Dollar volatile after Fed signals caution 📊
  Iran strikes Israeli base, Oil spikes 📈
  US CPI cools, Gold rallies 📈
  Gold slides on risk-on sentiment 📉

Do NOT use 📈 📉 📊 anywhere else in the post — only in the headline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REJECT IF ANY OF THESE APPLY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. SIGNALS       — Buy/Sell/Long/Short/Entry/TP/SL/price targets
2. CHART / TA    — Technical analysis, patterns, indicators
3. MEME          — Memes, jokes, informal content
4. ANALYSIS IMG  — Chart screenshots, TA images
5. WATERMARK     — Another channel logo or username
6. STALE         — Content older than 18 hours
7. OFF-TOPIC     — Does not affect Gold, Oil, or DXY
8. LOW VALUE     — Vague, no specific real-world event
9. DUPLICATE     — Same story already processed
10. PREDICTION   — "I think", "expect", "my analysis"
11. COMMENTARY   — Personal views, market opinions
12. FORECAST/PREVIOUS — Any mention of "forecast", "expected", "previous" values
13. SENTIMENT    — Fear & Greed index, bank sentiment, market mood indicators,
                   "banks are bullish/bearish", "smart money", "COT report opinions",
                   sentiment surveys, positioning reports with opinions
14. NO IMPACT    — News that does not move Gold, Oil, or DXY

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT (if approved):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[LEAD EMOJI] [SHORT ENGLISH HEADLINE — factual, ends with trend emoji 📈 📉 or 📊]

[Source content lightly cleaned. 2-4 sentences max.]

[Relevant hashtags: #XAUUSD #DXY #OIL — only those that apply]

LEAD EMOJI (pick one that fits the story): 🚨 🌍 🏦 🛢️ 🏆 💵 ⚠️ 🗳️

RESPOND WITH VALID JSON ONLY — NO MARKDOWN FENCES — NO TRAILING COMMAS:
{"approved": true, "reason": "brief reason", "issues": [], "formatted_text": "...", "confidence": 0.9}
""".strip()

# ── VIDEO CAPTION PROMPT — FIXED ──────────────────────────────────────────────
# Core principle: PRESERVE original caption. Do NOT rewrite. Do NOT add analysis.
# Only clean junk, add one lead emoji + trend emoji on first line, add hashtag.
_VIDEO_CAPTION_PROMPT = """
You are a strict content filter for a FOREX TRADING channel (Gold, Oil, DXY).

A video has been received. Read the caption carefully and decide.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
APPROVE ONLY IF — ALL conditions must be true:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. The caption describes a REAL, PHYSICAL military/war event:
   - Missile strike / airstrike / air strike / bombing
   - Military attack / armed assault / explosion at a military target
   - Direct armed invasion / shelling / rocket fire
   - Naval or aerial strike on infrastructure

2. The event is in a region that directly impacts Oil or Gold prices:
   - Middle East (Iran, Iraq, Israel, Gaza, Lebanon, Yemen, Syria)
   - Gulf region (Saudi Arabia, UAE, Kuwait, Strait of Hormuz)
   - Russia / Ukraine (affects Oil supply and Gold safe-haven)
   - Any major oil-producing or oil-transit region

REJECT if ANY of these apply:
- Only political tension, diplomatic crisis, or sanctions — no physical strike
- Protests, demonstrations, civil unrest with no military action
- Economic news, data, or market analysis
- Predictions, opinions, or "could impact" language
- Strike or attack outside an Oil/Gold-relevant region
- Vague or empty captions with no clear event
- Channel promotions, ads, trading signals

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IF APPROVED — FORMATTING RULES (CRITICAL):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOUR JOB IS TO CLEAN — NOT TO REWRITE.

Step 1 — REMOVE only:
  - Other Telegram channel usernames (e.g. @SomeChannel, t.me/SomeChannel)
  - Raw URLs (http://, https://)
  - Hashtags from other channels (keep #OIL #XAUUSD if already there, remove all others)
  - Excessive repeated emojis (keep max 1-2 per line)
  - "forwarded from" lines

Step 2 — KEEP everything else exactly as written:
  - The original words, sentences, and structure
  - Numbers, locations, names
  - The original emojis (within reason)
  - Short sentences already in the caption

Step 3 — ADD at the very beginning:
  - ONE lead emoji that fits the event: 🚨 ⚔️ 🛢️ 🌍
  - Only if the caption does not already start with a strong relevant emoji

Step 4 — ADD at the end of the FIRST LINE only:
  - ONE trend emoji based on market direction mentioned or implied:
    📈 if Oil or Gold spiked / surged / jumped up
    📉 if Oil or Gold dropped / fell / crashed
    📊 if direction is unclear, mixed, or not mentioned

Step 5 — ADD at the end of the full post:
  - Relevant hashtag(s): #OIL and/or #XAUUSD (only the ones that apply)
  - Do NOT add #DXY for war/conflict posts unless USD is directly mentioned

Step 6 — DO NOT ADD:
  - Any analysis ("markets reacted", "this could push Oil higher")
  - Any prediction ("Gold may rise", "Oil likely to spike")
  - Any opinion or commentary
  - "Be careful" lines (added automatically by the system)
  - Signature (added automatically)
  - Any sentence that was not in the original caption

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLE 1:
Caption input:
"🔴 BREAKING: Iran launched ballistic missiles at a US military base in Iraq.
Multiple explosions reported near Baghdad. @NewsChannel24 t.me/NewsChannel24"

Correct output:
🚨 BREAKING: Iran launched ballistic missiles at a US military base in Iraq. 📈
Multiple explosions reported near Baghdad.

#OIL #XAUUSD

EXAMPLE 2:
Caption input:
"Israeli airstrikes hit Hezbollah weapons depots in southern Lebanon.
Heavy smoke reported. #BreakingNews @warzone_updates"

Correct output:
🚨 Israeli airstrikes hit Hezbollah weapons depots in southern Lebanon. 📊
Heavy smoke reported.

#OIL #XAUUSD

EXAMPLE 3 (WRONG — do NOT do this):
Caption input: "Iran fires missiles at Iraq base."
Wrong output:
🚨 Iran Launches Missile Strike on US Base in Iraq — Oil Surges on Supply Fears 📈
Iran has fired a volley of ballistic missiles at a United States military installation in Iraq,
triggering immediate concerns about regional stability and oil supply disruptions through
the Strait of Hormuz. Crude oil prices surged sharply as traders priced in escalation risk,
while Gold also spiked on safe-haven demand.
#OIL #XAUUSD

WHY WRONG: Completely rewrote the caption and added AI analysis that was not in the original.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Caption to analyse: {caption}

RESPOND WITH VALID JSON ONLY — NO MARKDOWN FENCES:
{{"approved": true/false, "reason": "brief reason", "formatted_text": "...", "confidence": 0.0-1.0}}
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

# ── FF daily image prompt — same-time grouping, Gold/Oil/DXY context ──────────
_FF_IMAGE_PROMPT = """
You are analysing a ForexFactory economic calendar screenshot for a forex trading channel
focused on Gold (XAUUSD), Oil, and DXY.

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

# ── FF weekly image prompt ─────────────────────────────────────────────────────
_FF_WEEKLY_IMAGE_PROMPT = """
You are analysing a ForexFactory.com calendar screenshot for the weekly outlook.
This is for a forex trading channel focused on Gold (XAUUSD), Oil, and DXY.

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
        text = data["formatted_text"]

        if "TODAY'S USD" in text or "WEEKLY HIGH IMPACT" in text:
            # Calendar posts: strip hashtags entirely
            text = re.sub(r"#\w+", "", text).strip()
            data["formatted_text"] = text
        else:
            # Regular posts: filter to allowed hashtags only (trend emoji is already in headline)
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


def _clean_video_caption(text: str) -> str:
    """
    Final safety clean for video captions after AI returns the formatted text.
    Removes any AI-added analysis sentences that sneak through.
    Strips asterisks, markdown, excessive blank lines.
    """
    if not text:
        return text
    text = text.replace("*", "")
    # Remove lines that sound like AI analysis / prediction
    _ANALYSIS_PATTERNS = re.compile(
        r"^.*(market[s]?\s+(react|surge|spike|rally|drop|fell|rose)|"
        r"oil\s+(could|may|might|likely|expected)|"
        r"gold\s+(could|may|might|likely|expected)|"
        r"traders\s+(fear|price|react|watch)|"
        r"supply\s+disruption\s+fear|"
        r"safe.?haven\s+demand|"
        r"this\s+(could|may|might)|"
        r"prices?\s+(could|may|might|are\s+expected)).*$",
        re.IGNORECASE | re.MULTILINE
    )
    text = _ANALYSIS_PATTERNS.sub("", text).strip()
    # Clean up multiple blank lines left behind
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


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

    async def analyse_video_caption(self, caption: str) -> dict:
        """
        AI reads the video caption.

        APPROVE ONLY IF: physical war/strike/missile/airstrike/attack event
        in a region that directly moves Gold or Oil prices.
        ALL other videos → rejected.

        On approval: preserves the original caption — only removes junk
        (other channel usernames, URLs, foreign hashtags), adds lead emoji
        + trend emoji on first line, adds relevant hashtags.
        Zero AI analysis, predictions, or commentary.
        """
        if not caption or not caption.strip():
            return _reject("Empty video caption.", "no_caption", confidence=1.0)

        prompt = _VIDEO_CAPTION_PROMPT.format(caption=caption.strip()[:800])

        # Try Gemini first
        try:
            verdict = await asyncio.wait_for(
                self._gemini_call(prompt, None, "image/jpeg"), timeout=30
            )
            verdict["engine"] = "gemini-2.5-flash"
            log.info(f"Video caption → approved={verdict['approved']} | {verdict.get('reason', '')}")
            if verdict.get("approved") and verdict.get("formatted_text"):
                # Clean any AI analysis that snuck through
                text = _clean_video_caption(verdict["formatted_text"])
                # Re-apply allowed hashtag filter
                hashtags = re.findall(r"#\w+", text)
                allowed = [h for h in hashtags if h in ALLOWED_HASHTAGS_SET]
                text = re.sub(r"#\w+", "", text).strip()
                if allowed:
                    text = text + "\n\n" + " ".join(allowed)
                verdict["formatted_text"] = text
            return verdict
        except Exception as exc:
            log.warning(f"Gemini video caption failed ({exc}) — trying Groq …")

        # Fallback to Groq
        try:
            verdict = await asyncio.wait_for(
                self._groq_call(prompt, None, "image/jpeg"), timeout=40
            )
            verdict["engine"] = "groq-llama4-scout"
            log.info(f"Groq video caption → approved={verdict['approved']} | {verdict.get('reason', '')}")
            if verdict.get("approved") and verdict.get("formatted_text"):
                # Clean any AI analysis that snuck through
                text = _clean_video_caption(verdict["formatted_text"])
                # Re-apply allowed hashtag filter
                hashtags = re.findall(r"#\w+", text)
                allowed = [h for h in hashtags if h in ALLOWED_HASHTAGS_SET]
                text = re.sub(r"#\w+", "", text).strip()
                if allowed:
                    text = text + "\n\n" + " ".join(allowed)
                verdict["formatted_text"] = text
            return verdict
        except Exception as exc:
            log.error(f"Both engines failed for video caption: {exc}")
            return _reject("AI engines unavailable for video caption.", "engine_error", confidence=0.0)

    def _build_moderation_prompt(self, text: str) -> str:
        return textwrap.dedent(f"""
            DATE (UTC): {_today_str()}
            CHANNEL FOCUS: {self._category} — trades Gold (XAUUSD), Oil, and DXY ONLY
            SOURCE CONTENT:
            \"\"\"
            {text.strip() if text else "(image only — no text)"}
            \"\"\"
            TASK: Analyse content. Approve ONLY if it directly affects Gold (XAUUSD), Oil, or DXY.
            If no clear market impact on these three instruments → reject.
            If forecast/previous values, signal, TA, meme, off-topic, stale → reject.
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
    lines = text.split('\n')
    for i in range(max(0, len(lines) - 3), len(lines)):
        lines[i] = re.sub(r'\b\d{4}\b', '', lines[i])
    text = '\n'.join(lines)
    text = re.sub(r'\n\s*\n', '\n\n', text).strip()
    text = _add_signature(text)
    return text
