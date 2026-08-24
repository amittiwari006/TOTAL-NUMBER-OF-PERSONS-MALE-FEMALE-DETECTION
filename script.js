// ---------- Element references ----------
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusEl = document.getElementById('status');
const placeholder = document.getElementById('placeholder');
const countBadge = document.getElementById('countBadge');
const totalNum = document.getElementById('totalNum');
const maleNum = document.getElementById('maleNum');
const femaleNum = document.getElementById('femaleNum');
const unclearNum = document.getElementById('unclearNum');
const recDot = document.getElementById('recDot');

// Dashboard card elements
const dashTotal = document.getElementById('dashTotal');
const dashMale = document.getElementById('dashMale');
const dashFemale = document.getElementById('dashFemale');
const dashUnclear = document.getElementById('dashUnclear');

// ---------- Config ----------
const FACE_MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.13/model/';
const GENDER_CONFIDENCE_MIN = 0.65; // below this, treat as "unclear" rather than guess

// ---------- State ----------
let stream = null;
let cocoModel = null;
let modelsReady = false;
let rafId = null;
let running = false;

// ---------- UI helpers ----------
function setStatus(msg, isError) {
  statusEl.textContent = msg || '';
  statusEl.className = 'status' + (isError ? ' error' : '');
}

// ---------- Model loading ----------
async function ensureModels() {
  if (modelsReady) return;

  setStatus('Loading body-detection model…');
  cocoModel = await cocoSsd.load({ base: 'lite_mobilenet_v2' });

  setStatus('Loading face/gender model…');
  await faceapi.nets.tinyFaceDetector.loadFromUri(FACE_MODEL_URL);
  await faceapi.nets.ageGenderNet.loadFromUri(FACE_MODEL_URL);

  modelsReady = true;
}

// ---------- Camera control ----------
async function startCamera() {
  startBtn.disabled = true;
  setStatus('Requesting camera access…');

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 960 } },
      audio: false
    });
  } catch (err) {
    setStatus('Could not access camera: ' + err.message + '. Check browser permissions.', true);
    startBtn.disabled = false;
    return;
  }

  video.srcObject = stream;
  await video.play();

  placeholder.style.display = 'none';
  countBadge.style.display = 'flex';
  recDot.style.display = 'flex';
  stopBtn.disabled = false;

  try {
    await ensureModels();
  } catch (err) {
    setStatus('Could not load detection models. Check your internet connection and try again.', true);
    stopCamera();
    return;
  }

  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  setStatus('Live — detecting bodies…');
  running = true;
  detectLoop();
}

function stopCamera() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  window.removeEventListener('resize', resizeCanvas);

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;

  ctx.clearRect(0, 0, overlay.width, overlay.height);
  placeholder.style.display = 'flex';
  countBadge.style.display = 'none';
  recDot.style.display = 'none';

  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus('');

  // Reset dashboard numbers back to zero
  dashTotal.textContent = '0';
  dashMale.textContent = '0';
  dashFemale.textContent = '0';
  dashUnclear.textContent = '0';
}

function resizeCanvas() {
  overlay.width = video.videoWidth || overlay.clientWidth;
  overlay.height = video.videoHeight || overlay.clientHeight;
}

// ---------- Detection loop ----------
async function detectLoop() {
  if (!running) return;

  if (video.readyState >= 2) {
    try {
      const [persons, faceResults] = await Promise.all([
        cocoModel.detect(video).then(preds => preds.filter(p => p.class === 'person' && p.score > 0.5)),
        faceapi
          .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 }))
          .withAgeAndGender()
      ]);

      const results = matchGenderToPersons(persons, faceResults);
      drawResults(results);
      updateCounts(results);
    } catch (err) {
      // transient frame errors are ignored, keep the loop alive
    }
  }

  rafId = requestAnimationFrame(detectLoop);
}

// ---------- Matching a face's gender to the body box it belongs to ----------
function boxCenter(x, y, w, h) {
  return { cx: x + w / 2, cy: y + h / 2 };
}

function pointInBox(px, py, x, y, w, h) {
  return px >= x && px <= x + w && py >= y && py <= y + h;
}

function matchGenderToPersons(persons, faceResults) {
  return persons.map(p => {
    const [x, y, w, h] = p.bbox;
    let gender = 'unclear';
    let genderProb = 0;

    for (const f of faceResults) {
      const box = f.detection.box;
      const { cx, cy } = boxCenter(box.x, box.y, box.width, box.height);
      if (pointInBox(cx, cy, x, y, w, h)) {
        if (f.genderProbability >= GENDER_CONFIDENCE_MIN) {
          gender = f.gender; // 'male' | 'female'
          genderProb = f.genderProbability;
        }
        break;
      }
    }

    return { x, y, w, h, gender, genderProb };
  });
}

// ---------- Drawing ----------
function colorFor(gender) {
  if (gender === 'male') return '#27ae60';
  if (gender === 'female') return '#e74c3c';
  return '#f0a93a';
}

function drawResults(results) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  results.forEach((r, i) => {
    const color = colorFor(r.gender);
    ctx.lineWidth = Math.max(2, overlay.width * 0.004);
    ctx.strokeStyle = color;
    ctx.strokeRect(r.x, r.y, r.w, r.h);

    const label = r.gender === 'unclear'
      ? `Person ${i + 1}`
      : `Person ${i + 1} · ${r.gender} (${Math.round(r.genderProb * 100)}%)`;

    ctx.font = `${Math.max(13, overlay.width * 0.016)}px Calibri, sans-serif`;
    const textW = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(r.x, Math.max(0, r.y - 22), textW + 12, 22);
    ctx.fillStyle = '#0d1230';
    ctx.fillText(label, r.x + 6, Math.max(16, r.y - 6));
  });
}

// ---------- Count badge ----------
function updateCounts(results) {
  const total = results.length;
  const male = results.filter(r => r.gender === 'male').length;
  const female = results.filter(r => r.gender === 'female').length;
  const unclear = total - male - female;

  totalNum.textContent = total;
  maleNum.textContent = male;
  femaleNum.textContent = female;
  unclearNum.textContent = unclear;

  // Keep the dashboard cards in sync with the same numbers
  dashTotal.textContent = total;
  dashMale.textContent = male;
  dashFemale.textContent = female;
  dashUnclear.textContent = unclear;
}

// ---------- Event bindings ----------
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
