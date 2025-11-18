const DEFAULTS = {
  apiUrl: "http://127.0.0.1:8000/predict",
  threshold: 0.65,
  enabled: true
};

const $ = (id) => document.getElementById(id);
const apiUrl = $("apiUrl");
const threshold = $("threshold");
const thVal = $("thVal");
const enabled = $("enabled");
const saveBtn = $("save");

function renderThVal() {
  thVal.textContent = threshold.value;
}

chrome.storage.sync.get(DEFAULTS, (opts) => {
  apiUrl.value = opts.apiUrl;
  threshold.value = opts.threshold;
  enabled.checked = !!opts.enabled;
  renderThVal();
});

threshold.addEventListener("input", renderThVal);

saveBtn.addEventListener("click", () => {
  const opts = {
    apiUrl: apiUrl.value.trim(),
    threshold: Number(threshold.value),
    enabled: enabled.checked
  };
  chrome.storage.sync.set(opts, () => {
    saveBtn.textContent = "Saved ✓";
    setTimeout(() => (saveBtn.textContent = "Save"), 1200);
  });
});
