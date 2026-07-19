/*
  script.js
  ---------
  Runs the whole model in the browser: no backend, no API call.

  New in this rebuild: clicking a sent message bubble reveals which
  words pushed the model toward spam or ham. The math for this reuses
  exactly what predictSpamProbability() already computes -- each
  matched vocabulary term's (tfidf_normalized * coefficient) value --
  the only change is that we keep those per-term numbers instead of
  only summing them.
*/

let MODEL = null;
let CURRENT_LANG = "en";
let messageSeq = 0;

const MODEL_PATHS = {
  en: "js/model_en.json",
  fa: "js/model_fa.json",
};

const UI_STRINGS = {
  en: {
    placeholder: "Type or paste an SMS message…",
    loading: "loading model…",
    ready: "ready — runs on-device",
    failed: "model failed to load",
    emptyThread: "Messages you test will appear here, scored as ham or spam.",
    emptyThreadHint: "Tap any message afterward to see why.",
    tapHint: "tap to see why",

    eyebrow: "on-device text classifier",
    heroTitle: "Spam Sense",
    heroSub: "Paste any SMS. It's scored the instant you hit send — no server, no network call, just a model running in this tab.",

    phoneName: "Unknown Sender",

    panelTitleHow: "How to read a result",
    legendHamLabel: "Ham",
    legendHamCopy: "A legitimate message. Rendered as a normal received bubble.",
    legendSpamLabel: "Spam",
    legendSpamCopy: "Unwanted or scam text. Flagged with a red receipt line.",
    legendExplainLabel: "Tap a message",
    legendExplainCopy: "See exactly which words pushed the model's decision.",
    panelTitleSample: "Try a sample",
    panelTitleModel: "Model",
    statsTypeLabel: "Type",
    statsTypeValue: "TF-IDF + Logistic Regression",
    statsAccuracyLabel: "Test accuracy",
    statsAccuracyValue: "98.5%",
    statsF1Label: "Test F1 (spam)",
    statsF1Value: "0.94",
    statsRunsLabel: "Runs",
    statsRunsValue: "entirely in your browser",

    footer: "English model trained on the SMS Spam Collection dataset (Almeida & Hidalgo, UCI / Kaggle). Persian model trained on a separately collected Persian SMS dataset. Built as a learning project — not a production spam filter.",

    explainSpamPushing: "Pushed toward spam",
    explainHamPushing: "Pushed toward ham",
    explainEmpty: "No familiar words found in this message.",
    explainMismatch: "Switch back to this message's language to see its explanation.",

    samples: [
      { label: "Free cruise winner", text: "Congratulations! You have been selected to win a free cruise. Call 09061234567 now to claim your prize!" },
      { label: "Dinner tonight?", text: "Hey, are we still on for dinner tonight at 7?" },
      { label: "Account suspended", text: "URGENT: Your account has been suspended. Click the link to verify your identity immediately." },
      { label: "Pick up milk", text: "Can you pick up milk on your way home? Thanks!" },
    ],
  },
  fa: {
    placeholder: "پیامک خود را اینجا بنویسید یا paste کنید…",
    loading: "در حال بارگذاری مدل…",
    ready: "آماده — کاملاً روی همین مرورگر اجرا می‌شود",
    failed: "بارگذاری مدل ناموفق بود",
    emptyThread: "پیام‌هایی که تست می‌کنید، اینجا به‌عنوان اسپم یا سالم نمایش داده می‌شوند.",
    emptyThreadHint: "روی هر پیام بزنید تا دلیل آن را ببینید.",
    tapHint: "لمس کنید تا دلیل را ببینید",

    eyebrow: "طبقه‌بند متن، اجرا روی همین دستگاه",
    heroTitle: "اسپم‌یاب",
    heroSub: "هر پیامکی را وارد کنید، همان لحظه بررسی می‌شود — بدون سرور، بدون درخواست شبکه، فقط یک مدل که همین‌جا در مرورگر اجرا می‌شود.",

    phoneName: "فرستنده ناشناس",

    panelTitleHow: "راهنمای خواندن نتیجه",
    legendHamLabel: "سالم",
    legendHamCopy: "یک پیام معتبر. به‌صورت حباب دریافتی معمولی نمایش داده می‌شود.",
    legendSpamLabel: "اسپم",
    legendSpamCopy: "پیام ناخواسته یا کلاهبرداری. با یک خط قرمز رسید مشخص می‌شود.",
    legendExplainLabel: "روی پیام بزنید",
    legendExplainCopy: "دقیقاً ببینید کدام کلمات تصمیم مدل را رقم زده‌اند.",
    panelTitleSample: "یک نمونه را امتحان کنید",
    panelTitleModel: "مدل",
    statsTypeLabel: "نوع مدل",
    statsTypeValue: "TF-IDF + رگرسیون لجستیک",
    statsAccuracyLabel: "دقت روی داده آزمایشی",
    statsAccuracyValue: "۱۰۰٪",
    statsF1Label: "F1 روی داده آزمایشی (اسپم)",
    statsF1Value: "۱٫۰۰",
    statsRunsLabel: "محل اجرا",
    statsRunsValue: "کاملاً در همین مرورگر",

    footer: "مدل انگلیسی روی مجموعه‌داده SMS Spam Collection (المیدا و هیدالگو، UCI / Kaggle) آموزش دیده است. مدل فارسی روی یک مجموعه‌داده فارسی جداگانه آموزش دیده است. این پروژه یک پروژه یادگیری است — نه یک فیلتر اسپم آماده تولید.",

    explainSpamPushing: "به‌سمت اسپم",
    explainHamPushing: "به‌سمت سالم",
    explainEmpty: "کلمه آشنایی در این پیام پیدا نشد.",
    explainMismatch: "برای دیدن دلیل این پیام، به زبان همان پیام برگردید.",

    samples: [
      { label: "برنده جایزه", text: "تبریک! شما برنده یک جایزه نقدی شده‌اید. برای دریافت همین الان با شماره زیر تماس بگیرید." },
      { label: "قرار امشب", text: "امشب ساعت هشت بریم بیرون؟" },
      { label: "حساب مسدود شد", text: "حساب شما مسدود شده است. برای فعال‌سازی روی لینک زیر کلیک کنید." },
      { label: "خرید نان", text: "میای سر راه یه نون بگیری؟" },
    ],
  },
};

/* ==================== Text cleaning: English ==================== */
function cleanTextEn(text) {
  let t = text.toLowerCase();
  t = t.replace(/(https?:\/\/\S+|www\.\S+)/g, " urltoken ");
  t = t.replace(/\S+@\S+/g, " emailtoken ");
  t = t.replace(/\b\d{5,}\b/g, " numtoken ");
  t = t.replace(/[!"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~]/g, "");
  t = t.replace(/\s+/g, " ").trim();
  return t;
}

/* ==================== Text cleaning: Persian ==================== */
const ARABIC_TO_PERSIAN = {
  "ي": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه",
  "أ": "ا", "إ": "ا", "آ": "ا", "ؤ": "و", "ئ": "ی",
};
const DIGIT_MAP = {
  "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
  "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
  "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
  "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
};
const ZWNJ = "\u200c";
const PERSIAN_PUNCT_RE = /[،؛؟«»٫٬ـ!"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~]/g;

function cleanTextFa(text) {
  let t = text.toLowerCase();
  for (const [arabicChar, persianChar] of Object.entries(ARABIC_TO_PERSIAN)) {
    t = t.split(arabicChar).join(persianChar);
  }
  for (const [srcDigit, asciiDigit] of Object.entries(DIGIT_MAP)) {
    t = t.split(srcDigit).join(asciiDigit);
  }
  t = t.split(ZWNJ).join("");
  t = t.replace(/(https?:\/\/\S+|www\.\S+)/g, " urltoken ");
  t = t.replace(/\S+@\S+/g, " emailtoken ");
  t = t.replace(/\b\d{5,}\b/g, " numtoken ");
  t = t.replace(PERSIAN_PUNCT_RE, "");
  t = t.replace(/\s+/g, " ").trim();
  return t;
}

const CLEANERS = { en: cleanTextEn, fa: cleanTextFa };

function tokenize(cleaned) {
  const words = cleaned.split(" ").filter((w) => w.length >= 2);
  const terms = [...words];
  for (let i = 0; i < words.length - 1; i++) {
    terms.push(words[i] + " " + words[i + 1]);
  }
  return terms;
}

/* ====================================================================
   analyzeMessage: TF-IDF -> L2 normalize -> logistic regression, but
   keeps each matched term's individual (weight * coef) contribution
   instead of only summing them into z. That per-term list is exactly
   what powers the explain panel.
==================================================================== */
function analyzeMessage(rawText, lang) {
  const cleaned = CLEANERS[lang](rawText);
  const terms = tokenize(cleaned);

  const counts = new Map();
  for (const term of terms) {
    const idx = MODEL.vocabulary[term];
    if (idx === undefined) continue;
    counts.set(idx, (counts.get(idx) || 0) + 1);
  }

  const weighted = new Map();
  let sumSquares = 0;
  for (const [idx, count] of counts.entries()) {
    const w = count * MODEL.idf[idx];
    weighted.set(idx, w);
    sumSquares += w * w;
  }
  const norm = Math.sqrt(sumSquares) || 1;

  let z = MODEL.intercept;
  const contributions = []; // { term, contribution }
  const idxToTerm = {};
  for (const [term, idx] of Object.entries(MODEL.vocabulary)) idxToTerm[idx] = term;

  for (const [idx, w] of weighted.entries()) {
    const contribution = (w / norm) * MODEL.coef[idx];
    z += contribution;
    contributions.push({ term: idxToTerm[idx], contribution });
  }

  const probSpam = 1 / (1 + Math.exp(-z));
  return { probSpam, contributions };
}

/* ==================== Model loading ==================== */
async function loadModel(lang) {
  const statusEl = document.getElementById("modelStatus");
  const strings = UI_STRINGS[lang];
  document.getElementById("sendBtn").disabled = true;
  statusEl.classList.remove("ready");
  statusEl.textContent = strings.loading;

  try {
    const res = await fetch(MODEL_PATHS[lang]);
    MODEL = await res.json();
    statusEl.textContent = strings.ready;
    statusEl.classList.add("ready");
    document.getElementById("sendBtn").disabled = false;
  } catch (err) {
    statusEl.textContent = strings.failed;
    console.error(err);
  }
}

/* ==================== Explain panel rendering ==================== */
function renderChips(container, items, kind) {
  container.innerHTML = "";
  items.forEach((item, i) => {
    const chip = document.createElement("span");
    chip.className = `chip chip-${kind}`;
    chip.style.setProperty("--i", i);
    chip.textContent = item.term;
    container.appendChild(chip);
  });
}

function populateExplainPanel(group, rawText, lang) {
  const emptyEl = group.querySelector(".explain-empty");
  const mismatchEl = group.querySelector(".explain-mismatch");
  const columnsEl = group.querySelector(".explain-columns");

  emptyEl.hidden = true;
  mismatchEl.hidden = true;
  columnsEl.style.display = "";

  if (lang !== CURRENT_LANG) {
    mismatchEl.hidden = false;
    columnsEl.style.display = "none";
    return;
  }

  const { contributions } = analyzeMessage(rawText, lang);

  if (contributions.length === 0) {
    emptyEl.hidden = false;
    columnsEl.style.display = "none";
    return;
  }

  const spamPushing = contributions
    .filter((c) => c.contribution > 0)
    .sort((a, b) => b.contribution - a.contribution)
    .slice(0, 5);
  const hamPushing = contributions
    .filter((c) => c.contribution < 0)
    .sort((a, b) => a.contribution - b.contribution)
    .slice(0, 5);

  renderChips(group.querySelector(".chip-row-spam"), spamPushing, "spam");
  renderChips(group.querySelector(".chip-row-ham"), hamPushing, "ham");

  if (spamPushing.length === 0 && hamPushing.length === 0) {
    emptyEl.hidden = false;
    columnsEl.style.display = "none";
  }
}

function toggleExplain(group, rawText, lang) {
  const wrap = group.querySelector(".explain-wrap");
  const bubble = group.querySelector(".bubble");
  const isOpen = wrap.classList.contains("open");

  if (isOpen) {
    wrap.classList.remove("open");
    bubble.setAttribute("aria-expanded", "false");
    return;
  }

  populateExplainPanel(group, rawText, lang);
  wrap.classList.add("open");
  bubble.setAttribute("aria-expanded", "true");
}

/* ==================== Message rendering ==================== */
function appendResult(rawText, probSpam, lang) {
  const thread = document.getElementById("thread");
  const empty = document.getElementById("threadEmpty");
  if (empty) empty.remove();

  const isSpam = probSpam >= 0.5;
  const confidence = isSpam ? probSpam : 1 - probSpam;
  const label = isSpam ? "spam" : "ham";
  const strings = UI_STRINGS[CURRENT_LANG];

  const template = document.getElementById("bubbleTemplate");
  const node = template.content.cloneNode(true);
  const group = node.querySelector(".bubble-group");
  const bubble = group.querySelector(".bubble");
  const receipt = group.querySelector(".receipt");

  const msgId = `msg-${++messageSeq}`;
  group.dataset.msgId = msgId;
  group.dataset.lang = lang;

  bubble.textContent = rawText;
  bubble.id = msgId;

  receipt.classList.add(label);
  receipt.querySelector(".tag").textContent = label.toUpperCase();
  receipt.querySelector(".confidence-text").textContent = `${(confidence * 100).toFixed(1)}%`;
  receipt.querySelector(".explain-hint").textContent = strings.tapHint;

  const fill = receipt.querySelector(".confidence-fill");
  fill.classList.add(label);

  applyTranslationsWithin(group, strings);

  thread.appendChild(node);

  // Animate the confidence fill on the next frame so the width
  // transition (a Tween, per the skill) actually has a 0% starting
  // point to animate from rather than snapping straight to target.
  requestAnimationFrame(() => {
    fill.style.width = `${(confidence * 100).toFixed(0)}%`;
  });

  bubble.addEventListener("click", () => {
    const groupEl = thread.querySelector(`[data-msg-id="${msgId}"]`);
    if (!groupEl) return; // node was removed (e.g. thread reset on language switch)
    toggleExplain(groupEl, rawText, lang);
  });

  thread.scrollTop = thread.scrollHeight;
}

function applyTranslationsWithin(root, strings) {
  root.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.dataset.i18n;
    if (strings[key] !== undefined) el.textContent = strings[key];
  });
}

/* ==================== General UI wiring ==================== */
function resetThread() {
  const thread = document.getElementById("thread");
  const strings = UI_STRINGS[CURRENT_LANG];
  thread.innerHTML = `
    <div class="thread-empty" id="threadEmpty">
      <p>${strings.emptyThread}</p>
      <p class="thread-empty-hint">${strings.emptyThreadHint}</p>
    </div>`;
}

function renderSamples() {
  const container = document.getElementById("samples");
  container.innerHTML = "";
  UI_STRINGS[CURRENT_LANG].samples.forEach((sample) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "sample-chip";
    chip.textContent = sample.label;
    chip.addEventListener("click", () => {
      document.getElementById("messageInput").value = sample.text;
      classifyCurrentInput();
    });
    container.appendChild(chip);
  });
}

function applyLanguageChrome(lang) {
  const strings = UI_STRINGS[lang];

  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.dataset.i18n;
    if (strings[key] !== undefined) el.textContent = strings[key];
  });

  document.getElementById("messageInput").placeholder = strings.placeholder;
  document.documentElement.lang = lang;

  const isRtl = lang === "fa";
  document.querySelectorAll("[data-rtl-aware]").forEach((el) => {
    el.dir = isRtl ? "rtl" : "ltr";
  });
  document.body.classList.toggle("lang-fa", isRtl);

  document.querySelectorAll(".lang-toggle button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.lang === lang);
  });
}

async function switchLanguage(lang) {
  if (lang === CURRENT_LANG) return;
  CURRENT_LANG = lang;
  applyLanguageChrome(lang);
  resetThread();
  renderSamples();
  await loadModel(lang);
}

function classifyCurrentInput() {
  const input = document.getElementById("messageInput");
  const text = input.value.trim();
  if (!text || !MODEL) return;

  const { probSpam } = analyzeMessage(text, CURRENT_LANG);
  appendResult(text, probSpam, CURRENT_LANG);

  input.value = "";
  input.style.height = "auto";
}

document.addEventListener("DOMContentLoaded", () => {
  applyLanguageChrome(CURRENT_LANG);
  renderSamples();
  loadModel(CURRENT_LANG);

  document.getElementById("composerForm").addEventListener("submit", (e) => {
    e.preventDefault();
    classifyCurrentInput();
  });

  const input = document.getElementById("messageInput");
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 90) + "px";
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      classifyCurrentInput();
    }
  });

  document.querySelectorAll(".lang-toggle button").forEach((btn) => {
    btn.addEventListener("click", () => switchLanguage(btn.dataset.lang));
  });
});
