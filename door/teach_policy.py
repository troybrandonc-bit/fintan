"""What may be taught: a topic or a document is checked before anything is fetched or learned.

This is a first line of checks and does not make a complete safety system. The rules here are few and plain; each
names a category and gives the user a reason. A second line can be
plugged in (`judge`): a function(text) -> (allow: bool, reason: str), for example the served model asked a fixed
question. A refusal never stops the user from using the model as it is; it only keeps the thing out of the weights.

    from teach_policy import check_topic, check_document
    check_topic("how nerve agents are synthesised")   -> {"allow": False, "category": "weapons", "reason": "..."}
    check_document(text, name="notes.md")             -> {"allow": True, "flags": ["secrets"], ...}

Categories that block teaching:
  weapons        making or deploying weapons able to cause mass harm (chemical, biological, nuclear, explosives)
  attacks        building malware, breaking into systems or accounts the user does not own
  self_harm      methods of suicide or self-harm
  abuse          sexual content involving minors; grooming
  targeting      profiles of a private person built to stalk, harass or expose them
Categories that only warn (the document is still learned unless the caller refuses warnings):
  secrets        passwords, API keys, private keys, card numbers found in a document
  personal       many e-mail addresses or phone numbers of other people in a document
"""
import re

_W = r"(?<![a-z0-9])"
BLOCK = [
    ("weapons", "making weapons that can cause mass harm is not something Fintan will learn",
     re.compile(_W + r"(?:(?:synthes[ie][sz]\w*|synthesis|mak(?:e|ing)|build(?:ing)?|produc\w+|weaponi[sz]\w+|cultur\w+|enrich\w*)\b.{0,60}"
                     r"\b(?:nerve agent|sarin|vx gas|tabun|mustard gas|chemical weapon|bioweapon|biological weapon|anthrax|smallpox|"
                     r"ricin|botulinum|dirty bomb|nuclear (?:bomb|weapon|device)|pipe bomb|explosive device|ied\b)|"
                     r"\b(?:bomb|explosive|nerve agent|bioweapon)[- ]making\b|"
                     # the thing before the verb, in how-to wording: "how nerve agents are synthesised"
                     r"\b(?:how|steps?|recipe|instructions?|procedure|guide)\b.{0,40}\b(?:nerve agents?|sarin|vx gas|tabun|"
                     r"mustard gas|chemical weapons?|bioweapons?|biological weapons?|anthrax|ricin|botulinum|dirty bombs?|"
                     r"nuclear (?:bombs?|weapons?|devices?)|pipe bombs?|explosive devices?)\b.{0,30}\b(?:synthes[ie][sz]\w*|"
                     r"made|produced|built|weaponi[sz]\w*|cultured|enriched|assembled))", re.I | re.S)),
    ("attacks", "building tools to break into systems or accounts is not something Fintan will learn",
     re.compile(_W + r"(?:(?:writ(?:e|ing)|build(?:ing)?|creat\w+|deploy\w*|develop\w*)\b.{0,40}\b(?:ransomware|malware|keylogger|"
                     r"botnet|computer virus|trojan|rootkit|spyware)|(?:hack|break|crack)\w*\s+into\b.{0,40}\b(?:account|server|network|"
                     r"system|phone|e-?mail)|steal\w*\b.{0,30}\b(?:password|credential|login)s?)", re.I | re.S)),
    ("self_harm", "methods of self-harm are not something Fintan will learn; if this is about you, please talk to someone you trust "
                  "or a local helpline",
     re.compile(_W + r"(?:(?:how to|ways? to|methods? (?:of|for)|best way to)\b.{0,30}\b(?:kill (?:myself|yourself)|commit suicide|"
                     r"end (?:my|your) life|self[- ]harm|hurt (?:myself|yourself)))", re.I | re.S)),
    ("abuse", "sexual content involving minors is never learned",
     re.compile(_W + r"(?:(?:sexual|nude|explicit|erotic)\b.{0,40}\b(?:child|children|minor|minors|underage|pre-?teen)s?\b|"
                     r"\b(?:child|minor)s?\b.{0,40}\b(?:sexual(?:ly)?|porn\w*|grooming)\b|\bgroom(?:ing)?\b.{0,30}\b(?:child|minor|kid)s?\b)",
                re.I | re.S)),
    ("targeting", "building a profile to find, follow or expose a private person is not something Fintan will learn",
     re.compile(_W + r"(?:(?:stalk|dox|doxx|track down|spy on|harass)\w*\b|find (?:out )?where\b.{0,40}\b(?:lives|works|goes to school)|"
                     r"home address of\b)", re.I | re.S)),
]
SECRET = re.compile(r"(?:-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----|\b(?:sk|pk|rk)[-_][A-Za-z0-9_\-]{20,}|\bAKIA[0-9A-Z]{16}\b|"
                    r"\bgh[pousr]_[A-Za-z0-9]{30,}|\bhf_[A-Za-z0-9]{30,}|(?:password|passwd|api[_ -]?key|secret|token)\s*[:=]\s*\S{6,}|"
                    r"\b(?:\d[ -]?){13,19}\b)", re.I)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d ().-]{8,}\d)(?!\d)")
# asking about a danger (history, policy, protection) is allowed: the block patterns need a making/doing verb next to the thing
MAX_SCAN = 200000


def _blocked(text):
    for cat, reason, pat in BLOCK:
        m = pat.search(text[:MAX_SCAN])
        if m:
            return cat, reason, m.group(0)[:80]
    return None


def check_topic(topic, judge=None):
    """A topic the user names. Returns {"allow", "category", "reason", "matched"}."""
    t = (topic or "").strip()
    if len(t) < 2:
        return {"allow": False, "category": "empty", "reason": "name something to learn", "matched": ""}
    b = _blocked(t)
    if b:
        return {"allow": False, "category": b[0], "reason": b[1], "matched": b[2]}
    if judge is not None:
        ok, why = judge(t)
        if not ok:
            return {"allow": False, "category": "judge", "reason": why or "the safety check declined this topic", "matched": ""}
    return {"allow": True, "category": None, "reason": "", "matched": ""}


def check_document(text, name="", judge=None, refuse_warnings=False):
    """A document about to be learned. Blocks as a topic does; warns on secrets and on other people's contact details.
    Returns {"allow", "category", "reason", "flags": [..], "name"}."""
    t = text or ""
    out = {"allow": True, "category": None, "reason": "", "flags": [], "name": name}
    b = _blocked(t)
    if b:
        return dict(out, allow=False, category=b[0], reason=b[1])
    if SECRET.search(t[:MAX_SCAN]):
        out["flags"].append("secrets")
    if len(set(EMAIL.findall(t[:MAX_SCAN]))) >= 10 or len(set(PHONE.findall(t[:MAX_SCAN]))) >= 10:
        out["flags"].append("personal")
    if judge is not None:
        ok, why = judge(t[:4000])
        if not ok:
            return dict(out, allow=False, category="judge", reason=why or "the safety check declined this document")
    if refuse_warnings and out["flags"]:
        return dict(out, allow=False, category=out["flags"][0],
                    reason="the document holds %s; remove them or allow them explicitly" % " and ".join(
                        {"secrets": "what look like passwords or keys", "personal": "many people's contact details"}[f] for f in out["flags"]))
    return out


def redact_secrets(text):
    """The document with anything that looks like a password, key or card number replaced; used when the user lets a
    flagged document through, so the secret itself is never learned."""
    return SECRET.sub("[removed]", text or "")
