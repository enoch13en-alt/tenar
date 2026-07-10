const $ = s => document.querySelector(s);
const setStatus = m => { $("#status").textContent = m; };
let PDFS = [];

async function loadCfg() {
  return new Promise(res => chrome.storage.local.get(["server", "token"], res));
}
async function saveCfg(o) {
  return new Promise(res => chrome.storage.local.set(o, res));
}
function server() { return $("#server").value.trim().replace(/\/+$/, ""); }
function token() { return $("#token").value.trim(); }

async function loadCourses() {
  if (!server() || !token()) return;
  try {
    const r = await fetch(server() + "/api/extension/courses?token=" + encodeURIComponent(token()));
    const j = await r.json();
    if (j.courses && j.courses.length)
      $("#course").innerHTML = j.courses.map(c => `<option>${c.replace(/</g, "&lt;")}</option>`).join("");
    else setStatus(j.error || "no courses / bad token");
  } catch (e) { setStatus("can't reach TENAR at " + server()); }
}

// Injected into the page: collect every PDF (the page itself if it's a PDF, plus
// every <a> that points at a .pdf). Returns [{url, title}], de-duplicated.
function _collect() {
  const out = [];
  try {
    const ct = document.contentType || "";
    if (ct.indexOf("pdf") >= 0 || /\.pdf($|\?|#)/i.test(location.href))
      out.push({ url: location.href, title: document.title || location.href });
  } catch (e) {}
  document.querySelectorAll("a[href]").forEach(a => {
    const href = a.href;
    if (/\.pdf($|\?|#)/i.test(href))
      out.push({ url: href, title: (a.textContent || "").trim().slice(0, 140)
                 || (href.split("/").pop() || "").split("?")[0] });
  });
  const seen = new Set();
  return out.filter(x => { if (seen.has(x.url)) return false; seen.add(x.url); return true; });
}

async function scanPage() {
  const tab = window._tab;
  $("#pdfList").innerHTML = ""; setStatus("");
  let found = [];
  try {
    const res = await chrome.scripting.executeScript({ target: { tabId: tab.id }, func: _collect });
    found = (res && res[0] && res[0].result) || [];
  } catch (e) {
    // injection blocked (e.g. Chrome's built-in PDF viewer) — use the tab URL
    if (tab.url && /\.pdf($|\?)/i.test(tab.url))
      found = [{ url: tab.url, title: tab.title || tab.url.split("/").pop() }];
  }
  PDFS = found;
  $("#count").textContent = found.length + " PDF" + (found.length === 1 ? "" : "s") + " on this page";
  if (!found.length) {
    $("#pdfList").innerHTML = "<div style='color:#e0b050'>No PDFs found on this page. Open a PDF (or a page that links to PDFs) and hit rescan.</div>";
    return;
  }
  $("#pdfList").innerHTML = found.map((p, i) =>
    '<div class="pdf"><input type="checkbox" data-i="' + i + '" checked>'
    + '<div class="t">' + (p.title || p.url).replace(/</g, "&lt;")
    + '<div class="u">' + p.url.replace(/</g, "&lt;") + '</div>'
    + '<div class="st" data-st="' + i + '"></div></div></div>').join("");
}

async function init() {
  const cfg = await loadCfg();
  $("#server").value = cfg.server || "http://localhost:5000";
  $("#token").value = cfg.token || "";
  const tabs = await new Promise(res => chrome.tabs.query({ active: true, currentWindow: true }, res));
  window._tab = tabs && tabs[0];
  if (cfg.token) { loadCourses(); scanPage(); }
  else { setStatus("Open Setup and paste your TENAR token once."); $("details").open = true; }
}

$("#save").onclick = async () => {
  await saveCfg({ server: server(), token: token() });
  setStatus("saved — loading…"); loadCourses(); scanPage();
};
$("#rescan").onclick = scanPage;
$("#selAll").onclick = () => $("#pdfList").querySelectorAll("input").forEach(c => c.checked = true);
$("#selNone").onclick = () => $("#pdfList").querySelectorAll("input").forEach(c => c.checked = false);

$("#add").onclick = async () => {
  const course = $("#course").value;
  if (!token()) { setStatus("set your token in Setup first"); return; }
  if (!course || course.startsWith("—")) { setStatus("pick a course"); return; }
  const picks = [...$("#pdfList").querySelectorAll("input:checked")].map(c => +c.dataset.i);
  if (!picks.length) { setStatus("tick at least one PDF"); return; }
  $("#add").disabled = true;
  let ok = 0;
  for (const i of picks) {                        // sequential — don't hammer the server
    const p = PDFS[i];
    const st = $('[data-st="' + i + '"]');
    st.textContent = " · fetching…"; st.style.color = "#8fb4ff";
    try {
      const resp = await fetch(p.url, { credentials: "include" });
      if (!resp.ok) { st.textContent = " · ✗ " + resp.status; st.style.color = "#e06f6f"; continue; }
      const blob = await resp.blob();
      const fname = ((p.url.split("/").pop() || "document").split("?")[0]) || "document.pdf";
      const title = (p.title || fname).replace(/\.pdf$/i, "").trim();
      const fd = new FormData();
      fd.append("token", token()); fd.append("course", course); fd.append("title", title);
      fd.append("file", blob, fname.endsWith(".pdf") ? fname : fname + ".pdf");
      st.textContent = " · sending…";
      const r = await fetch(server() + "/api/extension/add", { method: "POST", body: fd });
      const j = await r.json();
      if (j.ok) { st.textContent = " · ✓ added" + (j.ocr ? " (OCR)" : ""); st.style.color = "#6fcf97"; ok++; }
      else { st.textContent = " · ✗ " + (j.error || "failed"); st.style.color = "#e06f6f"; }
    } catch (e) { st.textContent = " · ✗ " + e.message; st.style.color = "#e06f6f"; }
  }
  $("#add").disabled = false;
  setStatus("Done — added " + ok + "/" + picks.length + ". Re-indexing; citeable shortly.");
};

init();
