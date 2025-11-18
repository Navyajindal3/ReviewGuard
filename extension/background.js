// Routes classification requests from content scripts to your FastAPI
const DEFAULTS = {
  apiUrl: "http://127.0.0.1:8000/predict",
  threshold: 0.65,
  enabled: true
};

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    if (msg?.type === "FRG_CLASSIFY") {
      try {
        const opts = await new Promise(res =>
          chrome.storage.sync.get(DEFAULTS, res)
        );
        if (!opts.enabled) {
          sendResponse({ ok: true, results: [], skipped: true });
          return;
        }
        const r = await fetch(opts.apiUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ texts: msg.texts })
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        sendResponse({ ok: true, results: data.results, threshold: data.threshold ?? opts.threshold });
      } catch (err) {
        console.error("FRG classify error:", err);
        sendResponse({ ok: false, error: String(err) });
      }
    }
  })();
  return true; // keep the message channel open for async sendResponse
});
