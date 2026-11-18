"""
Hand-crafted features for phishing detection (Task 3)
=====================================================
TF-IDF captures WHICH WORDS an email uses. These features capture HOW the email
is built: suspicious links, urgency language, shouting, HTML, etc. Combining
both usually works better than either alone, and the hand-crafted features are
easy to explain to a human ("flagged because it contains an IP-address link").
"""
import re
from urllib.parse import urlparse

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"'\)\]]+", re.I)
IP_RE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")
SHORTENERS = ("bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
              "buff.ly", "rebrand.ly", "cutt.ly")
SUSPICIOUS_TLDS = (".xyz", ".top", ".tk", ".click", ".work", ".gq", ".ml", ".cf", ".zip")
URGENT_WORDS = (
    "urgent", "immediately", "verify", "suspend", "locked", "confirm your",
    "password", "click here", "limited time", "act now", "security alert",
    "unusual activity", "update your", "expire", "winner", "prize", "bank",
    "login", "account", "within 24 hours", "final notice",
)

FEATURE_NAMES = [
    "n_urls", "has_ip_url", "has_url_shortener", "has_at_in_url", "max_url_length",
    "max_host_dots", "max_host_hyphens", "has_plain_http_url", "has_suspicious_tld",
    "n_urgent_keywords", "n_exclamations", "uppercase_ratio", "contains_html",
    "log_length",
]


def extract_urls(text):
    return URL_RE.findall(text)


def url_host(url):
    if not url.lower().startswith("http"):
        url = "http://" + url
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:          # malformed URL such as a broken IPv6 bracket
        return ""


def email_features(text):
    """Return a list of numbers, one per name in FEATURE_NAMES."""
    text = str(text)
    low = text.lower()
    urls = extract_urls(text)
    hosts = [url_host(u) for u in urls]
    letters = [c for c in text if c.isalpha()]
    return [
        len(urls),
        int(any(IP_RE.fullmatch(h) for h in hosts)),
        int(any(h == s or h.endswith("." + s) for h in hosts for s in SHORTENERS)),
        int(any("@" in u for u in urls)),
        max((len(u) for u in urls), default=0),
        max((h.count(".") for h in hosts), default=0),
        max((h.count("-") for h in hosts), default=0),
        int(any(u.lower().startswith("http://") for u in urls)),
        int(any(h.endswith(t) for h in hosts for t in SUSPICIOUS_TLDS)),
        sum(low.count(k) for k in URGENT_WORDS),
        text.count("!"),
        (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
        int("<html" in low or "<a href" in low),
        float(np.log1p(len(text))),
    ]


def explain(text):
    """Non-zero hand-crafted features: used to justify a prediction."""
    return {n: round(v, 3) for n, v in zip(FEATURE_NAMES, email_features(text)) if v}


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """scikit-learn transformer: list of email texts -> numeric matrix."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([email_features(t) for t in X], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES)
