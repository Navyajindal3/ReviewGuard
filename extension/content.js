// content.js — Single-score per review card + Grammarly-like hover popup

/* ---- run-once guard ---- */
if (window.__frg_active) {
  console.log("[FRG] already active, skipping");
} else {
  window.__frg_active = true;

  (function () {
    const DEFAULTS = { threshold: 0.65, enabled: true };

    // Prefer full review cards on Amazon; fall back only if cards are absent.
    const CARD_SELECTORS = ['div[data-hook="review"]'];        // Amazon card
    const BODY_FALLBACKS = [
      '.review-text-content',
      '[data-hook="review-body"]',
      '.review-content', '.review-text',
      '.ugc-review', '.ugc-comment'
    ];

    const seenCard = new WeakSet();          // processed cards
    const cache = new Map();                 // hash(text) -> result
    let enabled = DEFAULTS.enabled;
    let currentThreshold = DEFAULTS.threshold;
    const log = (...a) => console.log("[FRG]", ...a);
    const sleep = (ms) => new Promise(r => setTimeout(r, ms));

    /* ---------- utils ---------- */
    const norm = s => (s || "").trim().replace(/\s+/g, " ");
    function hashText(s) { let h = 2166136261>>>0; for (let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619);} return (h>>>0).toString(16); }

    function getCardText(card) {
      const title = card.querySelector('a[data-hook="review-title"] span')?.textContent || "";
      const body = card.querySelector('span[data-hook="review-body"] span, .review-text-content span, .review-text-content')?.textContent || "";
      const txt = norm([title, body].filter(Boolean).join(" — "));
      return txt.length >= 40 ? txt : ""; // ignore tiny fragments
    }

    function collectCards(root=document) {
      const cards = Array.from(root.querySelectorAll(CARD_SELECTORS.join(',')));
      if (cards.length) return cards;

      // Fallback: no explicit cards on this site
      const nodes = [];
      BODY_FALLBACKS.forEach(sel => nodes.push(...root.querySelectorAll(sel)));
      // Deduplicate by closest block element
      return Array.from(new Set(nodes.map(n => n.closest('article, section, div')))).filter(Boolean);
    }

    /* ---------- UI injection (chip + hover popover) ---------- */
    function injectUI(card, verdict, proba) {
  if (card.querySelector(':scope .frg-chip')) return;

  const header = card.querySelector('a[data-hook="review-title"]') || card.firstElementChild || card;
  const isFake = /fake/i.test(verdict);
  const pct = Math.round(proba * 100);

  // CHIP
  const chip = document.createElement('span');
  chip.className = `frg-chip ${isFake ? "fake" : "genuine"}`;
  chip.innerHTML = `<span class="frg-dot"></span>${isFake ? "Likely fake" : "Likely genuine"} (${pct}%)`;
  header.appendChild(chip);

  // POPOVER (fixed so it's never clipped)
  const pop = document.createElement('div');
  pop.className = `frg-pop ${isFake ? "fake" : "genuine"}`;
  pop.innerHTML = `
    <div class="frg-row"><span class="frg-title">${isFake ? "Likely fake" : "Likely genuine"}</span><span class="frg-subtle">${pct}%</span></div>
    <div class="frg-progress"><i style="width:${pct}%"></i></div>
    <div class="frg-subtle">Source: title + body • Threshold: ${currentThreshold}</div>
  `;
  // fixed positioning container
  Object.assign(pop.style, {
    position: "fixed",
    display: "none",
    zIndex: "2147483647",
  });
  document.body.appendChild(pop); // append to body to avoid clipping

  function placePop() {
    const r = chip.getBoundingClientRect();
    // Default below-left of chip; keep in viewport
    const top = Math.min(window.innerHeight - 12 - pop.offsetHeight, r.bottom + 8);
    const left = Math.min(window.innerWidth - 12 - pop.offsetWidth, Math.max(12, r.left));
    pop.style.top = `${Math.max(12, top)}px`;
    pop.style.left = `${left}px`;
  }

  let showTimer, hideTimer;
  const open = () => {
    clearTimeout(hideTimer);
    pop.style.display = "block";
    pop.setAttribute("aria-live", "on");
    // wait for layout to get proper size then place
    requestAnimationFrame(() => { placePop(); });
    console.log("[FRG] pop opened");
  };
  const close = () => {
    clearTimeout(showTimer);
    pop.style.display = "none";
    pop.removeAttribute("aria-live");
  };

  // Hover open/close (Grammarly-like)
  chip.addEventListener('mouseenter', () => { showTimer = setTimeout(open, 120); });
  chip.addEventListener('mouseleave', () => { hideTimer = setTimeout(close, 160); });
  pop.addEventListener('mouseenter', () => { clearTimeout(hideTimer); });
  pop.addEventListener('mouseleave', () => { hideTimer = setTimeout(close, 160); });

  // ALSO open/close on click (helps if hover is finicky)
  chip.addEventListener('click', (e) => {
    e.stopPropagation();
    if (pop.style.display === "none") open(); else close();
  });

  // Reposition on scroll/resize
  document.addEventListener('scroll', () => { if (pop.style.display !== "none") placePop(); }, { passive: true });
  window.addEventListener('resize', () => { if (pop.style.display !== "none") placePop(); }, { passive: true });

  // Keyboard
  chip.tabIndex = 0;
  chip.addEventListener('keydown', (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); (pop.style.display === "none" ? open() : close()); }
    if (e.key === "Escape") close();
  });
  document.addEventListener('keydown', (e) => { if (e.key === "Escape") close(); });
  document.addEventListener('click', (e) => { if (!pop.contains(e.target) && e.target !== chip) close(); }, { capture: true });
}


    /* ---------- background messaging ---------- */
    function classifyBatch(texts) {
      return new Promise(resolve => {
        chrome.runtime.sendMessage({ type: "FRG_CLASSIFY", texts }, resp => resolve(resp || { ok:false, error:"no response"}));
      });
    }

    /* ---------- main pass ---------- */
    async function process() {
      if (!enabled) return;

      const cards = collectCards();
      const toScore = [];
      const mapIdx = [];

      // Build request list, score ONCE per card
      for (let i=0; i<cards.length; i++) {
        const card = cards[i];
        if (seenCard.has(card)) continue;

        const text = getCardText(card);
        if (!text) continue;

        const h = hashText(text);
        if (cache.has(h)) {
          const r = cache.get(h);
          injectUI(card, r.label, r.proba_fake);
          seenCard.add(card);
        } else {
          toScore.push(text);
          mapIdx.push(i);
          // mark early to avoid double-queue on mutation bursts
          seenCard.add(card);
          card.dataset.frgProcessed = "1";
        }
      }

      if (!toScore.length) return;

      // Batch politely
      const BATCH = 20;
      for (let i=0; i<toScore.length; i+=BATCH) {
        const slice = toScore.slice(i, i+BATCH);
        const resp  = await classifyBatch(slice);
        if (!resp?.ok) { console.warn("[FRG] classify failed:", resp?.error); continue; }
        currentThreshold = resp.threshold ?? currentThreshold;

        resp.results.forEach((r, k) => {
          const card = cards[mapIdx[i+k]];
          if (!r || !card) return;
          const text = getCardText(card);
          if (!text) return;
          cache.set(hashText(text), r);
          injectUI(card, r.label, r.proba_fake);
        });

        await sleep(120);
      }
    }

    /* ---------- boot ---------- */
    function bootstrap() {
      chrome.storage.sync.get(DEFAULTS, (opts) => {
        enabled = !!opts.enabled;
        currentThreshold = Number(opts.threshold) || DEFAULTS.threshold;

        process();

        const mo = new MutationObserver((mutations) => {
          // ignore our own UI mutations
          for (const m of mutations) {
            if (m.target?.closest?.('.frg-chip, .frg-pop')) return;
          }
          clearTimeout(bootstrap._t);
          bootstrap._t = setTimeout(process, 250);
        });
        mo.observe(document.documentElement, { childList:true, subtree:true });
      });
    }

    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", bootstrap);
    else bootstrap();
  })();
}
