"""Two-stage Goodreads recommender.

  uv run main.py model                      # Call 1: ratings + reader.md -> taste_model.md
  uv run main.py recommend "under 300 pages" # Call 2: taste_model.md (+ query) -> recommendations

Recommendations derive only from the taste model, so a bad recommendation can be
traced to (and fixed at) the sentence in taste_model.md that caused it.
"""

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# override=True: the shell exports an empty ANTHROPIC_API_KEY, which would otherwise win.
load_dotenv(Path(__file__).parent / ".env", override=True)

MODEL = "claude-opus-5-5"
ROOT = Path(__file__).parent
CSV_PATH = ROOT / "my_goodreads.csv"
READER_PATH = ROOT / "reader.md"
TASTE_MODEL_PATH = ROOT / "taste_model.md"
RECS_DIR = ROOT / "recommendations"

# read=180: fail (and retry) if a stream goes silent, instead of hanging indefinitely.
client = anthropic.Anthropic(timeout=anthropic.Timeout(900.0, read=180.0), max_retries=3)


# --- Data ---------------------------------------------------------------------

def load_books(path: Path = CSV_PATH) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def format_history(books: list[dict]) -> str:
    """One line per book, grouped so the model can't confuse 'unrated' with 'disliked'."""
    def line(b: dict) -> str:
        year = b["Original Publication Year"] or b["Year Published"] or "?"
        parts = [f'{b["Title"]} — {b["Author"]} ({year})']
        if b["Number of Pages"]:
            parts.append(f'{b["Number of Pages"]}pp')
        if b["Date Read"]:
            parts.append(f'read {b["Date Read"]}')
        if b["Read Count"] and int(b["Read Count"]) > 1:
            parts.append(f'read {b["Read Count"]}x')
        if b["Bookshelves"]:
            parts.append(f'shelves: {b["Bookshelves"]}')
        text = " | ".join(parts)
        if b["My Review"]:
            text += f'\n    Review: {b["My Review"].strip()}'
        if b["Private Notes"]:
            text += f'\n    Notes: {b["Private Notes"].strip()}'
        return text

    def rating(b: dict) -> float:
        return float(b["My Rating"] or 0)

    read = [b for b in books if b["Exclusive Shelf"] == "read"]
    sections = []
    for stars in sorted({rating(b) for b in read if rating(b) > 0}, reverse=True):
        group = [b for b in read if rating(b) == stars]
        sections.append(f"## Rated {stars:g}/5\n" + "\n".join(f"- {line(b)}" for b in group))

    unrated = [b for b in books if rating(b) == 0 and b["Exclusive Shelf"] == "read"]
    if unrated:
        sections.append("## Read, no rating given\n" + "\n".join(f"- {line(b)}" for b in unrated))

    for shelf in sorted({b["Exclusive Shelf"] for b in books} - {"read"}):
        group = [b for b in books if b["Exclusive Shelf"] == shelf]
        sections.append(f"## Shelf: {shelf}\n" + "\n".join(f"- {line(b)}" for b in group))

    return "\n\n".join(sections)


# --- LLM ----------------------------------------------------------------------

def call(system: str, user: str, effort: str = "high") -> str:
    with client.messages.stream(
        model=MODEL,
        max_tokens=32000,
        output_config={"effort": effort},
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "refusal":
        sys.exit(f"Model refused: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        sys.exit("Response truncated at max_tokens.")
    return "".join(b.text for b in response.content if b.type == "text").strip()


# --- Call 1: taste model ------------------------------------------------------

TASTE_SYSTEM = """You build explicit, inspectable models of a reader's taste from their Goodreads history \
and their own notes. The reader will use your document as the sole input to a separate recommendation \
step, and will correct it when it is wrong — so every claim must be specific enough to be checked.

The model separates two things. Territory is what they read about. Sensibility is how they read: the \
forms, voices, registers, difficulty and kinds of question that make a book land for them. Territory is \
backward-looking; sensibility is portable, and it is what lets the recommender lead them into subjects \
their history doesn't cover yet.

The reader's notes are authoritative:
- Corrections override anything you would infer from the history. Apply them silently; don't repeat \
the wrong claim.
- Directions are where the reader wants to go. They are stated intent, not hypotheses — but how to \
enter each one well is your hypothesis, derived from their sensibility.

How to read the history:
- "Read, no rating given" means the reader finished the book and did not rate it. It is NOT a negative \
signal. Treat it as evidence of what they choose to read, not how they felt about it. Ratings, where \
present, are the only direct evidence of enjoyment.
- Weight repeated choices (multiple books by one author, clusters around a period or subject) as strong \
evidence of seriousness, regardless of rating.
- The sample is small. Prefer a few sharp observations to many vague ones, and say when evidence is thin.
- Name specific books and authors as evidence for each claim, inline.

Write 400–700 words of markdown (700 is a hard ceiling — cut the weakest claims to fit) with exactly \
these sections:

# Taste Model
## Sensibility
How they read, stated without reference to subject matter so each claim transfers to new territory \
(e.g. "Abstract argument lands best when carried by a single life", not "They like Napoleon"). Cite \
the books that evidence each claim.
## Territory
Subjects, writers, eras and traditions they return to, and what they enjoy without deep commitment.
## Rejections
What they rate low or conspicuously avoid. Prefix every reason with "Hypothesis:" — these are guesses.
## Tensions
Pairs that pull against each other in their reading and what the pull might mean \
(e.g. Nietzsche alongside C.S. Lewis).
## Directions
One entry per direction in the reader's notes. For each: which forms, voices and kinds of entry point \
their sensibility suggests (naming the Sensibility claims relied on), and what to avoid. Name at most \
one or two example authors per direction — the recommender picks the books. For directions already \
well represented in their history, say how to push further rather than repeat it.

Write short declarative sentences, one claim per sentence, so each can be quoted and corrected \
individually. Output only the document."""


def build_taste_model() -> None:
    history = format_history(load_books())
    reader = READER_PATH.read_text(encoding="utf-8") if READER_PATH.exists() else "(none)"
    text = call(
        TASTE_SYSTEM,
        f"<reader_notes>\n{reader}\n</reader_notes>\n\n<reading_history>\n{history}\n</reading_history>",
    )
    TASTE_MODEL_PATH.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n[{len(text.split())} words → {TASTE_MODEL_PATH.name}]", file=sys.stderr)


# --- Call 2: recommendations --------------------------------------------------

RECS_SYSTEM = """You recommend books strictly from an explicit taste model the reader has written \
(or approved). The taste model is your only evidence about their taste — do not infer preferences \
from anything else. The already-read list is provided only so you don't recommend those books; it is \
not evidence of taste.

Recommend 6–8 books unless the request says otherwise. Tag each with one kind:
- direction: enters a subject from the Directions section. Chosen by applying the Sensibility to that \
subject — the book should read like something this reader would pick up, not the standard primer. \
Must cite the relevant Directions entry and at least one Sensibility sentence.
- familiar: deepens their existing Territory.
- stretch: tests a Tension or pushes against a Rejection hypothesis.

Default mix: about half direction (spread across different directions), the rest familiar, and at \
least one stretch. The request overrides this — "stay in my lane" means no direction picks; naming a \
subject means focus there.

For each book:

### <Title> — <Author> (<year>, ~<pages>pp) · <kind>
**Why:** 2–4 sentences.
**Cites:** one or more verbatim quotes from the taste model, each in double quotes, copied exactly — \
these are how the reader traces a bad recommendation back to the sentence that caused it.
**Risk:** one sentence on which part of the taste model this bet could be wrong about.

Honor the request's constraints (length, difficulty, mood) strictly; if a constraint conflicts with \
the taste model, the constraint wins and you say so. Only recommend real, published books you are \
confident exist, with accurate authors."""


def recommend(query: str) -> None:
    if not TASTE_MODEL_PATH.exists():
        sys.exit(f"{TASTE_MODEL_PATH.name} not found — run `main.py model` first.")
    taste = TASTE_MODEL_PATH.read_text(encoding="utf-8")
    already_read = "\n".join(f'- {b["Title"]} — {b["Author"]}' for b in load_books())

    user = (
        f"<taste_model>\n{taste}\n</taste_model>\n\n"
        f"<already_read>\n{already_read}\n</already_read>\n\n"
        f"<request>\n{query or 'No specific constraints — recommend what I should read next.'}\n</request>"
    )
    text = call(RECS_SYSTEM, user)

    RECS_DIR.mkdir(exist_ok=True)
    out = RECS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.md"
    out.write_text(f"> Request: {query or '(none)'}\n\n{text}\n", encoding="utf-8")
    print(text)

    unverified = check_citations(text, taste)
    if unverified:
        print("\n⚠ Citations not found verbatim in taste_model.md:", file=sys.stderr)
        for q in unverified:
            print(f'  - "{q}"', file=sys.stderr)
    print(f"\n[saved → {out.relative_to(ROOT)}]", file=sys.stderr)


def check_citations(recs: str, taste: str) -> list[str]:
    """Return quoted citations that don't appear in the taste model (whitespace/case-insensitive)."""
    norm = lambda s: re.sub(r"\s+", " ", s.replace("*", "")).strip().lower()
    taste_n = norm(taste)
    quotes = []
    for cites in re.findall(r"\*\*Cites:\*\*(.*?)(?=\n\*\*|\n###|\Z)", recs, re.S):
        quotes += re.findall(r"[\"“]([^\"”]{8,})[\"”]", cites)
    return [q for q in quotes if norm(q.rstrip(".")) not in taste_n]


# --- CLI ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("model", help="Generate taste_model.md from the Goodreads export")
    rec = sub.add_parser("recommend", help="Recommend books from taste_model.md")
    rec.add_argument("query", nargs="*", help='Optional context, e.g. "shorter than 300 pages"')
    args = parser.parse_args()

    if args.cmd == "model":
        build_taste_model()
    else:
        recommend(" ".join(args.query))


if __name__ == "__main__":
    main()
