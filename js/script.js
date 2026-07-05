/*
  script.js
  ---------
  Runs the whole model in the browser: no backend, no API call.

  This reimplements, in JavaScript, exactly what scikit-learn's
  TfidfVectorizer + LogisticRegression do at prediction time:

    1. Clean the text the same way preprocess.py does
    2. Tokenize into words, build unigrams + bigrams
    3. Count term frequencies, multiply by each term's idf weight
    4. L2-normalize the resulting vector
    5. Dot product with the logistic regression weights + intercept
    6. Sigmoid -> probability of "spam"

  model.json (exported by export_web_model.py) supplies the vocabulary,
  idf weights, coefficients and intercept -- i.e. everything the trained
  model learned, as plain data.
*/

let MODEL = null;

async function loadModel() {
  const statusEl = document.getElementById("modelStatus");
  try {
    const res = await fetch("js/model.json");
    MODEL = await res.json();
    statusEl.textContent = "ready — runs on-device";
    statusEl.classList.add("ready");
    document.getElementById("sendBtn").disabled = false;
  } catch (err) {
    statusEl.textContent = "model failed to load";
    console.error(err);
  }
}

/* ---- Text cleaning (mirrors preprocess.clean_text) ---- */
function cleanText(text) {
  let t = text.toLowerCase();
  t = t.replace(/(https?:\/\/\S+|www\.\S+)/g, " urltoken ");
  t = t.replace(/\S+@\S+/g, " emailtoken ");
  t = t.replace(/\b\d{5,}\b/g, " numtoken ");
  t = t.replace(/[!"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~]/g, "");
  t = t.replace(/\s+/g, " ").trim();
  return t;
}

/* ---- Tokenize + build unigrams/bigrams (mirrors TfidfVectorizer defaults) ---- */
function tokenize(cleaned) {
  const words = cleaned.split(" ").filter((w) => w.length >= 2);
  const terms = [...words]; // unigrams
  for (let i = 0; i < words.length - 1; i++) {
    terms.push(words[i] + " " + words[i + 1]); // bigrams
  }
  return terms;
}

/* ---- TF-IDF vector (sparse) + L2 normalization, then logistic regression ---- */
function predictSpamProbability(rawText) {
  const cleaned = cleanText(rawText);
  const terms = tokenize(cleaned);

  const counts = new Map(); // vocab index -> raw term count
  for (const term of terms) {
    const idx = MODEL.vocabulary[term];
    if (idx === undefined) continue;
    counts.set(idx, (counts.get(idx) || 0) + 1);
  }

  // tf-idf weight per present feature
  const weighted = new Map();
  let sumSquares = 0;
  for (const [idx, count] of counts.entries()) {
    const w = count * MODEL.idf[idx];
    weighted.set(idx, w);
    sumSquares += w * w;
  }
  const norm = Math.sqrt(sumSquares) || 1;

  // dot product with logistic regression coefficients
  let z = MODEL.intercept;
  for (const [idx, w] of weighted.entries()) {
    z += (w / norm) * MODEL.coef[idx];
  }

  const probSpam = 1 / (1 + Math.exp(-z));
  return probSpam;
}

/* ---------------- UI wiring ---------------- */
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
    <span>${(confidence * 100).toFixed(1)}% confidence</span>
    <span class="confidence-track"><span class="confidence-fill ${label}" style="width:${(confidence * 100).toFixed(0)}%"></span></span>
  `;

  group.appendChild(bubble);
  group.appendChild(receipt);
  thread.appendChild(group);
  thread.scrollTop = thread.scrollHeight;
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
  document.getElementById("sendBtn").disabled = true;
  loadModel();

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

  document.querySelectorAll(".sample-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.getElementById("messageInput").value = chip.dataset.text;
      classifyCurrentInput();
    });
  });
});
