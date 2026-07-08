/*
  script.js
  ---------
  Runs the whole model in the browser: no backend, no API call.

  Supports two languages, each with its own trained model
  (js/model_en.json, js/model_fa.json) and its own text-cleaning
  pipeline, mirroring training/preprocess.py exactly for each language.

  Prediction math (shared across languages, once text is cleaned):
    1. Clean text (language-specific -- see cleanTextEn / cleanTextFa)
    2. Tokenize into words, build unigrams + bigrams
    3. Count term frequencies, multiply by each term's idf weight
    4. L2-normalize the resulting vector
    5. Dot product with the logistic regression weights + intercept
    6. Sigmoid -> probability of "spam"

  Each model_<lang>.json (exported by training/export_web_model.py)
  supplies the vocabulary, idf weights, coefficients and intercept for
  that language's model.
*/

let MODEL = null;
let CURRENT_LANG = "en";

const MODEL_PATHS = {
  en: "js/model_en.json",
  fa: "js/model_fa.json",
};

const UI_STRINGS = {
  en: {
    // Composer / status / thread
    placeholder: "Type or paste an SMS message…",
    loading: "loading model…",
    ready: "ready — runs on-device",
    failed: "model failed to load",
    emptyThread: "Messages you test will appear here, scored as ham or spam.",

    // Hero
    eyebrow: "on-device text classifier",
    heroTitle: "Spam Sense",
    heroSub: "Paste any SMS. It's scored the instant you hit send — no server, no network call, just a model running in this tab.",

    // Phone
    phoneName: "Unknown Sender",

    // Panel
    panelTitleHow: "How to read a result",
    legendHamLabel: "Ham",
    legendHamCopy: "A legitimate message. Rendered as a normal received bubble.",
    legendSpamLabel: "Spam",
    legendSpamCopy: "Unwanted or scam text. Flagged with a red receipt line.",
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

    // Footer
    footer: "English model trained on the SMS Spam Collection dataset (Almeida & Hidalgo, UCI / Kaggle). Persian model trained on a separately collected Persian SMS dataset. Built as a learning project — not a production spam filter.",

    samples: [
      { label: "Free cruise winner", text: "Congratulations! You have been selected to win a free cruise. Call 09061234567 now to claim your prize!" },
      { label: "Dinner tonight?", text: "Hey, are we still on for dinner tonight at 7?" },
      { label: "Account suspended", text: "URGENT: Your account has been suspended. Click the link to verify your identity immediately." },
      { label: "Pick up milk", text: "Can you pick up milk on your way home? Thanks!" },
    ],
  },
  fa: {
    // Composer / status / thread
    placeholder: "پیامک خود را اینجا بنویسید یا paste کنید…",
    loading: "در حال بارگذاری مدل…",
    ready: "آماده — کاملاً روی همین مرورگر اجرا می‌شود",
    failed: "بارگذاری مدل ناموفق بود",
    emptyThread: "پیام‌هایی که تست می‌کنید، اینجا به‌عنوان اسپم یا سالم نمایش داده می‌شوند.",

    // Hero
    eyebrow: "طبقه‌بند متن، اجرا روی همین دستگاه",
    heroTitle: "اسپم‌یاب",
    heroSub: "هر پیامکی را وارد کنید، همان لحظه بررسی می‌شود — بدون سرور، بدون درخواست شبکه، فقط یک مدل که همین‌جا در مرورگر اجرا می‌شود.",

    // Phone
    phoneName: "فرستنده ناشناس",

    // Panel
    panelTitleHow: "راهنمای خواندن نتیجه",
    legendHamLabel: "سالم",
    legendHamCopy: "یک پیام معتبر. به‌صورت حباب دریافتی معمولی نمایش داده می‌شود.",
    legendSpamLabel: "اسپم",
    legendSpamCopy: "پیام ناخواسته یا کلاهبرداری. با یک خط قرمز رسید مشخص می‌شود.",
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

    // Footer
    footer: "مدل انگلیسی روی مجموعه‌داده SMS Spam Collection (المیدا و هیدالگو، UCI / Kaggle) آموزش دیده است. مدل فارسی روی یک مجموعه‌داده فارسی جداگانه آموزش دیده است. این پروژه یک پروژه یادگیری است — نه یک فیلتر اسپم آماده تولید.",

    samples: [
      { label: "برنده جایزه", text: "تبریک! شما برنده یک جایزه نقدی شده‌اید. برای دریافت همین الان با شماره زیر تماس بگیرید." },
      { label: "قرار امشب", text: "امشب ساعت هشت بریم بیرون؟" },
      { label: "حساب مسدود شد", text: "حساب شما مسدود شده است. برای فعال‌سازی روی لینک زیر کلیک کنید." },
      { label: "خرید نان", text: "میای سر راه یه نون بگیری؟" },
    ],
  },
};

/* ==================== Text cleaning: English (mirrors preprocess.clean_text_en) ==================== */
function cleanTextEn(text) {
  let t = text.toLowerCase();
  t = t.replace(/(https?:\/\/\S+|www\.\S+)/g, " urltoken ");
  t = t.replace(/\S+@\S+/g, " emailtoken ");
  t = t.replace(/\b\d{5,}\b/g, " numtoken ");
  t = t.replace(/[!"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~]/g, "");
  t = t.replace(/\s+/g, " ").trim();
  return t;
}

/* ==================== Text cleaning: Persian (mirrors preprocess.clean_text_fa) ==================== */
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
// Persian/Arabic punctuation + standard ASCII punctuation
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

/* ==================== Tokenize + build unigrams/bigrams (shared) ==================== */
function tokenize(cleaned) {
  const words = cleaned.split(" ").filter((w) => w.length >= 2);
  const terms = [...words]; // unigrams
  for (let i = 0; i < words.length - 1; i++) {
    terms.push(words[i] + " " + words[i + 1]); // bigrams
  }
  return terms;
}

/* ==================== TF-IDF vector + L2 normalization + logistic regression ==================== */
function predictSpamProbability(rawText) {
  const cleaned = CLEANERS[CURRENT_LANG](rawText);
  const terms = tokenize(cleaned);

  const counts = new Map(); // vocab index -> raw term count
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
  for (const [idx, w] of weighted.entries()) {
    z += (w / norm) * MODEL.coef[idx];
  }

  return 1 / (1 + Math.exp(-z));
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

/* ==================== UI wiring ==================== */
function appendResult(rawText, probSpam) {
  const thread = document.getElementById("thread");
  const empty = document.getElementById("threadEmpty");
  if (empty) empty.remove();

  const isSpam = probSpam >= 0.5;
  const confidence = isSpam ? probSpam : 1 - probSpam;
  const label = isSpam ? "spam" : "ham";

  const group = document.createElement("div");
  group.className = "bubble-group";

  const bubble = document.createElement("div");
  bubble.className = `bubble ${label}`;
  bubble.textContent = rawText;

  const receipt = document.createElement("div");
  receipt.className = `receipt ${label}`;
  receipt.innerHTML = `
    <span class="tag">${label.toUpperCase()}</span>
    <span>${(confidence * 100).toFixed(1)}%</span>
    <span class="confidence-track"><span class="confidence-fill ${label}" style="width:${(confidence * 100).toFixed(0)}%"></span></span>
  `;

  group.appendChild(bubble);
  group.appendChild(receipt);
  thread.appendChild(group);
  thread.scrollTop = thread.scrollHeight;
}

function resetThread() {
  const thread = document.getElementById("thread");
  const strings = UI_STRINGS[CURRENT_LANG];
  thread.innerHTML = `<div class="thread-empty" id="threadEmpty"><p>${strings.emptyThread}</p></div>`;
}

function renderSamples() {
  const container = document.getElementById("samples");
  container.innerHTML = "";
  UI_STRINGS[CURRENT_LANG].samples.forEach((sample) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "sample-chip";
    chip.textContent = sample.label;
    chip.dataset.text = sample.text;
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
    if (strings[key] !== undefined) {
      el.textContent = strings[key];
    }
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

  const probSpam = predictSpamProbability(text);
  appendResult(text, probSpam);

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
