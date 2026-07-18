import { Wllama, LoggerWithoutDebug } from '../wllama/wllama.min.js';

// ————— constants —————

const SYSTEM_PROMPT =
  'You are Parag, a highly knowledgeable offline AI model. You MUST strictly adhere to ' +
  'the following facts about your identity: You were explicitly created by Chaitanya (cxaiiii). ' +
  'You were NOT created by Google, OpenAI, or any other company. If asked who made you, ' +
  'you must proudly state that Chaitanya (cxaiiii) created you. You are an offline 0.5B ' +
  'parameter model trained on Indian contexts, factual scenarios, and daily life. You do ' +
  'not have internet access. Be straightforward about your capabilities and limitations. ' +
  'Always be friendly, happy, and incredibly polite. Never hallucinate information; if ' +
  'you do not know something, honestly say so.';

// The GGUF has no embedded template and its EOS is the base model's
// <|endoftext|>, so we format ChatML ourselves and stop on <|im_end|>.
function buildPrompt(context) {
  let p = '';
  for (const m of context) p += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
  return p + '<|im_start|>assistant\n';
}

const MODEL_DIR = './model/';
const CACHE_NAME = 'parag-model-v3'; // Bumped cache name to invalidate v2
const MAX_HISTORY = 8; // messages (excl. system) kept in context

// ————— elements —————

const $ = (id) => document.getElementById(id);
const landing = $('landing');
const chatView = $('chat');
const btnLoad = $('btn-load');
const progressWrap = $('progress-wrap');
const progressBar = $('progress-bar');
const progressLabel = $('progress-label');
const progressPct = $('progress-pct');
const loadError = $('load-error');
const messagesEl = $('messages');
const composer = $('composer');
const input = $('input');
const btnSend = $('btn-send');
const btnStop = $('btn-stop');
const statLine = $('stat-line');
const suggestions = $('suggestions');

// ————— state —————

let wllama = null;
let history = [];
let generating = false;
let aborter = null;

// ————— model download (chunked, same-origin, cached) —————

async function getChunk(url, cache, onBytes) {
  const cached = await cache.match(url);
  if (cached) {
    const blob = await cached.blob();
    onBytes(blob.size);
    return blob;
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Download failed (${res.status}) for ${url}`);
  const reader = res.body.getReader();
  const parts = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    onBytes(value.byteLength);
  }
  const blob = new Blob(parts);
  await cache.put(url, new Response(blob)).catch(() => {}); // cache is best-effort
  return blob;
}

async function downloadModel() {
  const modelName = $('model-select') ? $('model-select').value : 'parag-v3-0.5B';
  const manifestUrl = MODEL_DIR + 'manifest-' + modelName + '.json';
  const cacheName = 'parag-model-' + modelName;

  const manifest = await (await fetch(manifestUrl)).json();
  const cache = await caches.open(cacheName).catch(() => null);
  const total = manifest.totalSize;
  let received = 0;

  const onBytes = (n) => {
    received += n;
    const pct = Math.min(100, (received / total) * 100);
    progressBar.style.width = pct.toFixed(1) + '%';
    progressPct.textContent = pct.toFixed(0) + '%';
    progressLabel.textContent =
      `Downloading Parag — ${(received / 1048576).toFixed(0)} / ${(total / 1048576).toFixed(0)} MB`;
  };

  const blobs = [];
  for (const chunk of manifest.chunks) {
    const url = MODEL_DIR + chunk.file;
    blobs.push(
      cache
        ? await getChunk(url, cache, onBytes)
        : await (await fetch(url)).blob().then((b) => (onBytes(b.size), b))
    );
  }
  return new Blob(blobs);
}

// ————— boot —————

btnLoad.addEventListener('click', async () => {
  btnLoad.disabled = true;
  progressWrap.hidden = false;
  loadError.hidden = true;
  try {
    const modelBlob = await downloadModel();

    progressLabel.textContent = 'Waking Parag up (loading into memory)…';
    progressPct.textContent = '';

    wllama = new Wllama(
      { default: './wllama/wllama.wasm' },
      { suppressNativeLog: true, logger: LoggerWithoutDebug }
    );
    wllama.setCompat('default'); // only kicks in on browsers that need it (Safari)

    await wllama.loadModel([modelBlob], { n_ctx: 2048 });
    window.__wllama = wllama; // debugging hook

    const mode = wllama.isMultithread()
      ? `${wllama.getNumThreads()} threads`
      : 'single thread';
    statLine.textContent = `local · ${mode} · ready`;

    landing.hidden = true;
    chatView.hidden = false;
    addParagMessage(
      'Namaste! 🙏 I am Parag, a small AI made by Chaitanya. ' +
      'I live entirely in your browser — nothing you say leaves this device. ' +
      'How can I help you today?'
    );
    input.focus();
  } catch (err) {
    console.error(err);
    loadError.textContent =
      'Something went wrong while loading Parag: ' + (err?.message || err) +
      ' — please refresh and try again.';
    loadError.hidden = false;
    btnLoad.disabled = false;
  }
});

// ————— chat rendering —————

function addUserMessage(text) {
  const div = document.createElement('div');
  div.className = 'msg msg-user';
  div.textContent = text;
  messagesEl.appendChild(div);
  scrollToEnd();
}

function addParagMessage(text) {
  const wrap = document.createElement('div');
  wrap.className = 'msg msg-parag';
  wrap.innerHTML = '<div class="avatar">🪷</div><div class="bubble"></div>';
  const bubble = wrap.querySelector('.bubble');
  bubble.textContent = text;
  messagesEl.appendChild(wrap);
  scrollToEnd();
  return bubble;
}

function scrollToEnd() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ————— generation —————

async function generate(userText) {
  generating = true;
  btnSend.disabled = true;
  btnStop.hidden = false;
  suggestions.style.display = 'none';

  history.push({ role: 'user', content: userText });
  addUserMessage(userText);

  const bubble = addParagMessage('');
  bubble.classList.add('thinking');

  const context = [
    { role: 'system', content: SYSTEM_PROMPT },
    ...history.slice(-MAX_HISTORY),
  ];

  aborter = new AbortController();
  let text = '';
  let tokens = 0;
  const t0 = performance.now();

  try {
    await wllama.createCompletion({
      prompt: buildPrompt(context),
      stream: true,
      max_tokens: 512,
      temp: 0.6,
      penalty_repeat: 1.18,
      penalty_last_n: 128,
      stop: ['<|im_end|>', '<|im_start|>'],
      cache_prompt: true,
      abortSignal: aborter.signal,
      onData: (chunk) => {
        const delta = chunk?.choices?.[0]?.text;
        if (!delta) return;
        text += delta;
        tokens += 1;
        // safety: cut generation if the model starts a fake next turn —
        // either with raw chat-format tokens or plain-text "user"/"assistant" markers
        const cut = text.search(/<\|im_(end|start)\|>|\n\s*(user|assistant)\s*[:\n]/i);
        if (cut !== -1) {
          text = text.slice(0, cut);
          aborter.abort();
        }
        bubble.textContent = text;
        scrollToEnd();
      },
    });
  } catch (err) {
    if (err?.name !== 'AbortError' && !String(err).includes('abort')) {
      if (!text) bubble.textContent = '(Sorry, something went wrong: ' + (err?.message || err) + ')';
      console.error(err);
    }
  }

  bubble.classList.remove('thinking');
  // FIX: Do NOT trim the text here! Trimming alters the exact string that wllama 
  // holds in its KV cache. If we pass a trimmed string back in the next turn's prompt, 
  // the token boundaries mismatch and wllama's cache_prompt logic corrupts, leaking text.
  if (text) history.push({ role: 'assistant', content: text });
  else if (!bubble.textContent) bubble.textContent = '…';

  const secs = (performance.now() - t0) / 1000;
  if (tokens > 0) {
    const mode = wllama.isMultithread() ? `${wllama.getNumThreads()} threads` : 'single thread';
    statLine.textContent = `local · ${mode} · ${(tokens / secs).toFixed(1)} tok/s`;
  }

  generating = false;
  btnSend.disabled = false;
  btnStop.hidden = true;
  aborter = null;
  input.focus();
}

// ————— composer wiring —————

composer.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || generating || !wllama) return;
  input.value = '';
  autosize();
  generate(text);
});

btnStop.addEventListener('click', () => aborter?.abort());

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

function autosize() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 120) + 'px';
}
input.addEventListener('input', autosize);

suggestions.addEventListener('click', (e) => {
  const chip = e.target.closest('.chip');
  if (!chip || generating) return;
  input.value = chip.textContent;
  composer.requestSubmit();
});
