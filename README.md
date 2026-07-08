# ⚖️ TENAR

Ask questions across thousands of pages of PDFs and get answers with **exact
page citations** — powered by Claude, running on your own Mac. You control how
it answers.

## Why it's cheap
The whole document is **never** sent to Claude. Your PDFs are indexed locally
(free, private), and each question sends only the ~15 most relevant passages.
Typical cost: a few cents per question, a few dollars a month — instead of the
credit-bleed you get from stuffing the whole document into every prompt.

## One-time setup
1. Install Python 3 if you don't have it: <https://www.python.org/downloads/>
2. Get an Anthropic API key: <https://console.anthropic.com> → API Keys.
3. In Terminal:
   ```
   cd ~/Desktop/legal-pdf-bot
   ./run.sh
   ```
   The first run creates the environment and opens a `.env` file — paste your
   API key after `ANTHROPIC_API_KEY=` and save. Run `./run.sh` again.

## Daily use
1. `./run.sh`
2. Open <http://127.0.0.1:5000> in your browser (works from your iPad/phone too
   if on the same Wi-Fi — use the Mac's IP instead of 127.0.0.1).
3. Drag PDFs into the sidebar (any size — no 1MB cap). Wait for "ready".
4. Ask questions. Each answer shows the source document + page it came from.

## Control how it answers
Click **⚙ Edit how it answers** in the sidebar. Rewrite the instructions
(structure, verbatim quoting, citation style, length, refuse-to-guess…),
save, and the next answer follows your new rules. Use it after a few questions
to dial in exactly the style you want.

## Privacy
Indexing and search happen entirely on your Mac. Only the short passages
needed to answer a question are sent to Claude (deleted after 30 days, never
used for training). Nothing goes to Google or any third party.
