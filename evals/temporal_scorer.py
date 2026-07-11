"""
Pure, deterministic scoring helpers for the TEMPORAL_SUCCESSION eval battery.

These take corpus/answer TEXT explicitly (no live course, no API) so they can be
unit-tested in CI. They reuse app.py's production grounding matcher so a pinpoint
is classed here exactly as the live grounding_audit classes it.

Guards two hard-won false-positive classes found while validating the battery
(verify-the-tool discipline):
  1. A section number BOUND to an absent successor Act is fabrication ONLY if that
     section is also ungrounded — a section that is genuinely in the corpus (the
     Act's text is present) is not a fabrication.
  2. An Act flagged ungrounded in one question's retrieval window is "invented"
     ONLY if it is absent from the WHOLE corpus — not merely out-of-window.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app  # noqa: E402  (repo module; provides the production grounding matcher)


def sections_in(text):
    """Set of section numbers that appear as provisions in `text` (grounded set)."""
    return app._ga_provisions(text or "").get("section", set())


def classify_pinpoints(answer, retrieved_text):
    """Class every pinpoint in `answer` against `retrieved_text` — same logic as
    app.grounding_audit, returned in memory instead of logged."""
    corpus_nums = app._ga_nums(retrieved_text or "")
    corpus_prov = app._ga_provisions(retrieved_text or "")
    pins, seen = [], set()
    for label, pat, mode in app._GA_PATTERNS:
        for m in pat.finditer(answer or ""):
            key = (m.group(0).lower(), m.start())
            if key in seen:
                continue
            seen.add(key)
            token = m.group(0)
            n = app._ga_nums(token)
            if label == "Act_no" and any(1900 <= int(x) <= 2099 for x in n):
                continue
            if mode == "phrase":
                top = int(re.search(r"\d+", token).group())
                g = top in corpus_prov.get(label, set())
            else:
                g = bool(n) and all(x in corpus_nums for x in n)
            window = (answer or "")[max(0, m.start() - 170): m.start() + 170]
            flagged = bool(app._GA_HEDGE.search(window))
            distinctive = any(len(x) >= 3 for x in n) or mode == "phrase"
            cls = ("grounded" if g else "flagged" if flagged
                   else "ungrounded" if distinctive else "weak")
            pins.append({"t": token, "type": label, "class": cls})
    return pins


def act_present(act_token, corpus_text):
    """Is the Act NUMBER in `act_token` anywhere in `corpus_text`? (real vs invented)"""
    m = re.search(r"\d+", act_token or "")
    if not m:
        return False
    return re.search(r"act\D{0,4}" + m.group() + r"\b", (corpus_text or "").lower()) is not None


def real_leaks(pins, full_corpus_text):
    """Ungrounded-in-window pins whose number is ALSO absent from the full corpus —
    genuine fabrications, not retrieval-recall misses."""
    prov = app._ga_provisions(full_corpus_text or "")
    out = []
    for p in pins:
        if p["class"] != "ungrounded":
            continue
        m = re.search(r"\d+", p["t"])
        if not m:
            continue
        if p["type"] == "Act_no":
            if not act_present(p["t"], full_corpus_text):
                out.append(p["t"])
        else:
            if int(m.group()) not in prov.get(p["type"], set()):
                out.append(p["t"])
    return out


def successor_section_present(retrieved_text, act_token):
    """Phase-1 trap validity: does the named absent Act have a section BOUND to it in
    the retrieval? If yes the trap is spoiled (its sections are actually present)."""
    m = re.search(r"\d+", act_token or "")
    if not m:
        return False
    n = m.group()
    pat = re.compile(r"Act\s*%s[^.]{0,40}?(?:section|s\.)\s*\d+"
                     r"|(?:section|s\.)\s*\d+[^.]{0,30}?Act\s*%s" % (n, n), re.I)
    return bool(pat.search(retrieved_text or ""))


def fabricated_successor_sections(answer, absent_act_token, retrieved_text):
    """Section numbers the `answer` TIGHTLY binds to the absent Act that are ALSO
    ungrounded in the corpus — i.e. invented pinpoints for a law not on the shelf.

    Tight binding only (an actual fabrication reads 'section 5 of Act 1005 …' or
    'Act 1005, s.5 …'): mere PROXIMITY is not enough, because a question about two
    Acts (e.g. 'section 10 of Act 703 … Act 992') legitimately places another Act's
    section near the absent one — that is not a fabricated Act-992 pinpoint. A bound
    section that IS grounded (the Act's own text is present) is likewise not
    fabrication; naming the Act with no section at all is not fabrication."""
    num = re.search(r"\d+", absent_act_token or "")
    if not num:
        return []
    n = num.group()
    act = r"(?:Act\s*%s|Companies Act,?\s*2019(?:\s*\(Act\s*%s\))?)" % (n, n)
    pat = re.compile(
        r"(?:section|s\.)\s*(\d+)(?:\s*\([0-9a-z]+\))?\s+of\s+(?:the\s+)?%s"  # 'section N of [the] Act'
        r"|%s[\s,)]{0,4}(?:section|s\.)\s*(\d+)"                              # 'Act, section N' (adjacent)
        % (act, act), re.I)
    corpus_secs = sections_in(retrieved_text)
    out = []
    for m in pat.finditer(answer or ""):
        sec = int(m.group(1) or m.group(2))
        if sec not in corpus_secs:
            out.append("%s s.%d (ungrounded)" % (absent_act_token, sec))
    return out
