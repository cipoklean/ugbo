"""Quick terminal check: python -m ugbo.cli Nsukka maize cassava --lang pcm"""
import argparse

from .plan import facts_summary, make_plan
from .season import CROPS


def main():
    ap = argparse.ArgumentParser(description="Ugbo weekly field plan")
    ap.add_argument("town")
    ap.add_argument("crops", nargs="+", choices=CROPS)
    ap.add_argument("--lang", default="pcm", choices=["en", "pcm", "ha", "yo", "ig"])
    ap.add_argument("--ask", default="", help="a question from the farmer")
    ap.add_argument("--show-prompt", action="store_true")
    a = ap.parse_args()
    p = make_plan(a.town, a.crops, a.lang, a.ask)
    print(facts_summary(p), "\n")
    if a.show_prompt:
        print(p.prompt, "\n")
    print(p.text)


if __name__ == "__main__":
    main()
