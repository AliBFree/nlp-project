// A small built-in word list for scoring sentiment.
// Positive words add points, negative words subtract points.
// This is a simple "lexicon-based" approach to NLP — no external
// libraries or internet connection needed, runs entirely in the browser.

const POSITIVE_WORDS = {
  good: 2, great: 3, excellent: 3, amazing: 3, awesome: 3, love: 3,
  happy: 2, wonderful: 3, fantastic: 3, nice: 1, best: 2, beautiful: 2,
  perfect: 3, brilliant: 3, enjoy: 2, fun: 2, glad: 2, like: 1,
  cool: 1, win: 2, success: 2, easy: 1, helpful: 2, kind: 2, thanks: 1,
  thank: 1, delicious: 2
};

const NEGATIVE_WORDS = {
  bad: -2, terrible: -3, awful: -3, horrible: -3, hate: -3, sad: -2,
  worst: -3, ugly: -2, poor: -2, disappointing: -2, fail: -2, failure: -2,
  annoying: -2, angry: -2, problem: -1, hard: -1, difficult: -1,
  boring: -2, broken: -2, wrong: -1, sick: -1, worried: -1, sorry: -1,
  pain: -2, hurt: -2, lost: -1
};

function analyzeSentiment(text) {
  const words = text
    .toLowerCase()
    .replace(/[^a-z\s']/g, ' ')
    .split(/\s+/)
    .filter(Boolean);

  let score = 0;
  for (const word of words) {
    if (POSITIVE_WORDS[word]) score += POSITIVE_WORDS[word];
    if (NEGATIVE_WORDS[word]) score += NEGATIVE_WORDS[word];
  }

  let label, emoji;
  if (score > 1) {
    label = 'Positive';
    emoji = '😊';
  } else if (score < -1) {
    label = 'Negative';
    emoji = '😞';
  } else {
    label = 'Neutral';
    emoji = '😐';
  }

  return { score, label, emoji };
}

const textInput = document.getElementById('textInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultBox = document.getElementById('result');
const resultEmoji = document.getElementById('resultEmoji');
const resultLabel = document.getElementById('resultLabel');
const resultScore = document.getElementById('resultScore');

analyzeBtn.addEventListener('click', () => {
  const text = textInput.value.trim();
  if (!text) {
    textInput.focus();
    return;
  }

  const { score, label, emoji } = analyzeSentiment(text);

  resultEmoji.textContent = emoji;
  resultLabel.textContent = label;
  resultScore.textContent = `Score: ${score}`;
  resultBox.classList.remove('hidden');
});

// Allow pressing Enter (without Shift) to trigger analysis
textInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    analyzeBtn.click();
  }
});
