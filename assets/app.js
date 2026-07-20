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

const MODEL_NAME = 'parag-v3-0.5B'; // only model we ship
const DOWNLOAD_CONCURRENCY = 6;     // parallel chunk fetches

async function downloadModel() {
  const manifestUrl = MODEL_DIR + 'manifest-' + MODEL_NAME + '.json';
  const cacheName = 'parag-model-' + MODEL_NAME;

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

  // Fetch chunks with a bounded concurrency pool. The old code awaited each
  // chunk before starting the next, so every network round-trip serialized and
  // the download crawled. A pool keeps several fetches in flight at once; the
  // ordered `blobs` array is filled by index so assembly stays correct.
  const chunks = manifest.chunks;
  const blobs = new Array(chunks.length);
  let cursor = 0;

  async function worker() {
    while (true) {
      const i = cursor++;
      if (i >= chunks.length) return;
      const url = MODEL_DIR + chunks[i].file;
      blobs[i] = cache
        ? await getChunk(url, cache, onBytes)
        : await (await fetch(url)).blob().then((b) => (onBytes(b.size), b));
    }
  }

  const pool = Math.min(DOWNLOAD_CONCURRENCY, chunks.length);
  await Promise.all(Array.from({ length: pool }, worker));
  return new Blob(blobs);
}

// ————— boot —————

btnLoad.addEventListener('click', async () => {
  btnLoad.disabled = true;
  progressWrap.hidden = false;
  loadError.hidden = true;
  progressBar.style.width = '2%';
  progressPct.textContent = '0%';
  progressLabel.textContent = 'Starting download…';
  try {
    const modelBlob = await downloadModel();

    // Model is loading into wasm — this can take 10-30s with no download events,
    // so animate an indeterminate bar instead of freezing at 100%.
    progressLabel.textContent = 'Waking Parag up (loading into memory)…';
    progressPct.textContent = '';
    progressBar.classList.add('indeterminate');

    wllama = new Wllama(
      { default: './wllama/wllama.wasm' },
      { suppressNativeLog: true, logger: LoggerWithoutDebug }
    );
    wllama.setCompat('default'); // only kicks in on browsers that need it (Safari)

    await wllama.loadModel([modelBlob], { n_ctx: 4096 });
    window.__wllama = wllama; // debugging hook

    const mode = wllama.isMultithread()
      ? `${wllama.getNumThreads()} threads`
      : 'single thread';
    statLine.textContent = `local · ${mode} · ready`;

    progressBar.classList.remove('indeterminate');
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
    progressBar.classList.remove('indeterminate');
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

// Cuts the reply at a genuine next-turn marker only. Anchored to the start of a
// line and requires the ChatML role tag or a role word immediately followed by a
// newline/colon at line-start — so ordinary prose like "the assistant said" or a
// line ending before "User guide" is NOT chopped mid-sentence.
const TURN_CUT = /<\|im_(?:end|start)\|>|^\s*(?:user|assistant|system)\s*[:\n]/im;

let genId = 0;          // increments per generation; stale callbacks are ignored
let cancelRequested = false; // set by the Stop button (soft-stop)

async function generate(userText) {
  const myGen = ++genId;
  cancelRequested = false;
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

  let text = '';       // canonical accumulated reply (never read back from DOM)
  let rawText = '';    // debug only: every delta, ignoring TURN_CUT / frozen
  let tokens = 0;
  let frozen = false;  // once true, we stop appending (cut-marker or user stop)
  const t0 = performance.now();

  const promptStr = buildPrompt(context);
  // Set window.__paragDebug = true in the console to see, per turn, the exact
  // prompt sent and the RAW completion before TURN_CUT chops it. If the raw text
  // contains a "\nUser:" / "\nassistant:" style header, the mid-sentence cutoff
  // is TURN_CUT slicing a hallucinated turn boundary — i.e. the model derailed,
  // not the plumbing.
  if (window.__paragDebug) console.log('[parag] PROMPT >>>\n' + promptStr);

  let result = null;
  try {
    // NOTE: we deliberately do NOT abort. wllama's abort is JS-only — it stops
    // the result loop but leaves the C++ generation session live, and the NEXT
    // createCompletion then resumes that stale session, leaking its tokens into
    // the new reply. Instead we let the session wind down naturally (bounded by
    // max_tokens) while keeping input disabled, so a new generation can never
    // start mid-session. `frozen` just stops us displaying the wind-down tail.
    result = await wllama.createCompletion({
      prompt: promptStr,
      stream: true,
      max_tokens: 1024,
      temp: 0.6,
      // A 0.5B with no repetition penalty degenerates into loops
      // ("…The Prime Minister of India is…The Prime Minister of India is…")
      // and mangled tokens. 1.1 is the standard llama.cpp value that
      // suppresses this without flattening normal phrasing. penalty_last_n
      // spans well past the ~8-token loop window we were seeing.
      penalty_repeat: 1.1,
      penalty_last_n: 256,
      cache_prompt: false,
      // No `stop` strings: <|im_end|> is a native EOG token so generation stops
      // on it anyway, and string-stops make wllama hold back a lookahead buffer
      // (~9 chars) that isn't flushed at end-of-turn and bleeds into the NEXT
      // reply's opening (the "zone."/"for clarity." leak). TURN_CUT below still
      // guards against the model emitting a literal role marker as text.
      onData: (chunk) => {
        if (myGen !== genId) return;
        const delta = chunk?.choices?.[0]?.text;
        if (window.__paragDebug && typeof delta === 'string') rawText += delta;
        if (frozen) return;
        if (cancelRequested) { frozen = true; return; }
        if (!delta) return;
        text += delta;
        tokens += 1;
        const m = text.match(TURN_CUT);
        if (m) { text = text.slice(0, m.index); frozen = true; return; }
        bubble.textContent = text;
        scrollToEnd();
      },
    });

    // If we never froze, adopt wllama's final full string when it's longer than
    // what streamed (last token can be held in the C++ batch buffer).
    if (!frozen && myGen === genId && result) {
      const full = result?.choices?.[0]?.text;
      if (typeof full === 'string' && full.length > text.length) text = full;
      const m = text.match(TURN_CUT);
      if (m) text = text.slice(0, m.index);
    }
  } catch (err) {
    if (!text) bubble.textContent = '(Sorry, something went wrong: ' + (err?.message || err) + ')';
    console.error(err);
  }

  if (window.__paragDebug) {
    console.log('[parag] RAW completion (pre-TURN_CUT) >>>\n' + rawText);
    console.log('[parag] displayed (post-cut) >>>\n' + text);
    console.log('[parag] frozen=' + frozen + ' tokens=' + tokens);
  }

  bubble.classList.remove('thinking');
  // The 0.5B sometimes emits a stray leading "." (or similar) as its first
  // token — a cosmetic model quirk, not a leak (rawText is per-turn). Strip a
  // run of leading punctuation so ".Love…" renders as "Love…".
  const finalText = text.trim().replace(/^[.,;:!?]+\s*/, '');
  if (finalText) {
    bubble.textContent = finalText;
    history.push({ role: 'assistant', content: finalText });
  } else {
    bubble.textContent = '…';
  }

  const secs = (performance.now() - t0) / 1000;
  if (tokens > 0) {
    const mode = wllama.isMultithread() ? `${wllama.getNumThreads()} threads` : 'single thread';
    statLine.textContent = `local · ${mode} · ${(tokens / secs).toFixed(1)} tok/s`;
  }

  generating = false;
  btnSend.disabled = false;
  btnStop.hidden = true;
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

btnStop.addEventListener('click', () => {
  // Soft-stop: freeze the visible reply immediately. The engine winds the
  // session down on its own (bounded by max_tokens); input re-enables once it
  // has fully settled, so the next message can't collide with a live session.
  cancelRequested = true;
  btnStop.hidden = true;
  statLine.textContent = 'stopping…';
});

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
