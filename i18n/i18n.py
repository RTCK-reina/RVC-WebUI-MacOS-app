import json
import locale
import os
from configs import singleton_variable


def _locale_path(language: str) -> str:
    """Resolve locale/<language>.json path.

    Prefer RVC_BASE_DIR (set by the .app launcher) to avoid cwd-dependence;
    fall back to the path relative to this file so `python rpc_server.py`
    from any cwd still works.
    """
    base = os.environ.get("RVC_BASE_DIR")
    if base:
        candidate = os.path.join(base, "i18n", "locale", f"{language}.json")
        if os.path.exists(candidate):
            return candidate
    # Relative to this i18n/i18n.py file.
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "locale", f"{language}.json")
    if os.path.exists(candidate):
        return candidate
    # Legacy: cwd-relative.
    return f"./i18n/locale/{language}.json"


def load_language_list(language):
    with open(_locale_path(language), "r", encoding="utf-8") as f:
        language_list = json.load(f)
    return language_list


@singleton_variable
class I18nAuto:
    def __init__(self, language=None):
        if language in ["Auto", None]:
            try:
                language = locale.getdefaultlocale(
                    envvars=("LANG", "LC_ALL", "LC_CTYPE", "LANGUAGE")
                )[0]
            except Exception:
                language = "en_US"
        if language is None:
            language = "en_US"
        if not os.path.exists(_locale_path(language)):
            language = "en_US"
        self.language = language
        self.language_map = load_language_list(language)

    def __call__(self, key):
        return self.language_map.get(key, key)

    def __repr__(self):
        return "Language: " + self.language
