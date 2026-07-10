# TENAR "Add PDF to course" browser extension

Push the PDF you're viewing straight into a TENAR course. The **browser** does the
download, so it works on sites the server can't reach — modern TLS (parliament.gh),
JavaScript-rendered pages, logged-in pages, and bot-blocked sites (IAEA, ICJ).
Scanned PDFs are auto-OCR'd on arrival.

## One-time setup

1. **Load the extension** (no terminal):
   - Chrome/Edge: open `chrome://extensions`, turn on **Developer mode** (top-right),
     click **Load unpacked**, and select this `browser-extension` folder.
   - (Firefox: `about:debugging` → This Firefox → Load Temporary Add-on → pick
     `manifest.json`.)
2. **Get your token:** with TENAR running and you logged in as owner, open
   `http://localhost:5000/api/extension/token` in the browser — copy the `token`.
3. Click the TENAR extension icon → **Setup** → paste the token (and the TENAR
   address if not `http://localhost:5000`) → **Save & load courses**.

## Use

1. Open a PDF, **or a page that links to several PDFs** (e.g. GRA's Acts list, a UN
   treaty page, parliament.gh), in a browser tab.
2. Click the TENAR extension icon. It scans the page and lists **every PDF on it**,
   all ticked by default (use *all / none / rescan* to adjust).
3. Pick the **course**, untick any you don't want, and hit **Add selected**.
4. Each is fetched by your browser, uploaded, OCR'd if it's a scan, and re-indexed —
   citeable in answers shortly. You'll see ✓/✗ next to each.

After changing any extension file, click the **reload** ↻ icon on the extension's card
in `chrome://extensions` so Chrome picks up the new version.

The token is stored only in your browser and checked by your local TENAR; it keeps
the upload endpoint from being open to anyone.
