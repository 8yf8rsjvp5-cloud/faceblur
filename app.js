// ============================================================================
// FaceBlur — распознавание лиц (MediaPipe) + замазывание + экспорт видео.
// Всё выполняется локально в браузере, видео никуда не отправляется.
//
// ВАЖНО: библиотека распознавания лиц грузится динамическим import() внутри
// initFaceDetector(), а не статическим import в начале файла. Если бы это
// был статический import и CDN оказался недоступен (слабая сеть, блокировка,
// временный сбой jsdelivr) — упал бы вообще весь модуль, и не работало бы
// ничего, включая ручной режим и загрузку видео. Так — только автопоиск лиц
// становится недоступен, а всё остальное приложение работает как обычно.
// ============================================================================

let FaceDetector, FilesetResolver;

const els = {
  statusBanner: document.getElementById('statusBanner'),
  dropZone: document.getElementById('dropZone'),
  fileInput: document.getElementById('fileInput'),
  uploadPanel: document.getElementById('uploadPanel'),
  editPanel: document.getElementById('editPanel'),
  settingsPanel: document.getElementById('settingsPanel'),
  processPanel: document.getElementById('processPanel'),
  video: document.getElementById('sourceVideo'),
  canvas: document.getElementById('previewCanvas'),
  stage: document.getElementById('stage'),
  manualModeBtn: document.getElementById('manualModeBtn'),
  clearManualBtn: document.getElementById('clearManualBtn'),
  manualHint: document.getElementById('manualHint'),
  faceCountBadge: document.getElementById('faceCountBadge'),
  intensitySlider: document.getElementById('intensitySlider'),
  intensityVal: document.getElementById('intensityVal'),
  confidenceSlider: document.getElementById('confidenceSlider'),
  confidenceVal: document.getElementById('confidenceVal'),
  paddingSlider: document.getElementById('paddingSlider'),
  paddingVal: document.getElementById('paddingVal'),
  processBtn: document.getElementById('processBtn'),
  cancelProcessBtn: document.getElementById('cancelProcessBtn'),
  progressWrap: document.getElementById('progressWrap'),
  progressFill: document.getElementById('progressFill'),
  progressText: document.getElementById('progressText'),
  resultBox: document.getElementById('resultBox'),
  resultVideo: document.getElementById('resultVideo'),
  downloadLink: document.getElementById('downloadLink'),
  startOverBtn: document.getElementById('startOverBtn'),
};

const ctx = els.canvas.getContext('2d', { willReadFrequently: true });

let faceDetector = null;
let faceDetectorAvailable = true;
let currentStyle = 'pixelate';
let manualRegions = []; // {x0,y0,x1,y1} нормализовано 0..1 относительно кадра
let manualModeOn = false;
let drawingRegion = null;
let isProcessing = false;
let mediaRecorder = null;
let recordedChunks = [];
let cancelRequested = false;

function showStatus(msg, kind){
  els.statusBanner.textContent = msg;
  els.statusBanner.className = 'status-banner show';
  if (kind === 'error') els.statusBanner.style.borderColor = 'var(--danger)';
}
function hideStatus(){
  els.statusBanner.className = 'status-banner';
}

// ---------------------------------------------------------------------------
// Загрузка модели распознавания лиц
// ---------------------------------------------------------------------------

async function initFaceDetector(){
  showStatus('Загружаю модель распознавания лиц… это происходит один раз.');
  try {
    if (!FaceDetector || !FilesetResolver){
      const mod = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs");
      FaceDetector = mod.FaceDetector;
      FilesetResolver = mod.FilesetResolver;
    }
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    faceDetector = await FaceDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      minDetectionConfidence: 0.5
    });
    faceDetectorAvailable = true;
    hideStatus();
  } catch(e){
    faceDetectorAvailable = false;
    showStatus('⚠ Не удалось загрузить модель автоматического распознавания лиц (проверь интернет). Автопоиск лиц недоступен, но ручной режим — рисование зон вручную — работает как обычно.', 'error');
  }
}

// ---------------------------------------------------------------------------
// Загрузка файла
// ---------------------------------------------------------------------------

els.dropZone.addEventListener('click', () => els.fileInput.click());
els.dropZone.addEventListener('dragover', (e) => { e.preventDefault(); els.dropZone.classList.add('dragover'); });
els.dropZone.addEventListener('dragleave', () => els.dropZone.classList.remove('dragover'));
els.dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  els.dropZone.classList.remove('dragover');
  if (e.dataTransfer.files[0]) loadVideoFile(e.dataTransfer.files[0]);
});
els.fileInput.addEventListener('change', (e) => {
  if (e.target.files[0]) loadVideoFile(e.target.files[0]);
});

async function loadVideoFile(file){
  // На iOS видео из галереи иногда приходит с пустым file.type — тогда
  // проверяем по расширению файла вместо жёсткого требования video/*.
  const looksLikeVideo = file.type.startsWith('video/')
    || /\.(mp4|mov|m4v|webm|avi|mkv|3gp)$/i.test(file.name || '');
  if (!looksLikeVideo){
    showStatus('Это не похоже на видеофайл. Выбери файл формата mp4, mov, webm и т.п.', 'error');
    return;
  }

  // Видео показываем сразу, не дожидаясь загрузки модели распознавания лиц —
  // на мобильном интернете модель может грузиться долго, и раньше из-за этого
  // казалось, что видео вообще не загружается. Модель теперь подгружается
  // параллельно в фоне (см. вызов ниже), а не блокирует показ видео.
  if (!faceDetector && faceDetectorAvailable){
    initFaceDetector(); // намеренно без await — грузим в фоне
  }

  const url = URL.createObjectURL(file);
  els.video.src = url;
  const loaded = await new Promise((resolve) => {
    els.video.onloadedmetadata = () => resolve(true);
    els.video.onerror = () => resolve(false);
  });
  if (!loaded){
    showStatus('Не удалось прочитать видеофайл — возможно, формат не поддерживается этим браузером.', 'error');
    return;
  }

  els.canvas.width = els.video.videoWidth;
  els.canvas.height = els.video.videoHeight;

  manualRegions = [];
  els.editPanel.style.display = 'block';
  els.settingsPanel.style.display = 'block';
  els.processPanel.style.display = 'block';
  els.resultBox.classList.remove('show');

  els.video.currentTime = 0;
  await new Promise((resolve) => { els.video.onseeked = resolve; });
  renderStaticPreview();
}

// ---------------------------------------------------------------------------
// Статичный предпросмотр первого кадра с рамками лиц
// ---------------------------------------------------------------------------

function renderStaticPreview(){
  ctx.drawImage(els.video, 0, 0, els.canvas.width, els.canvas.height);

  let detections = [];
  if (faceDetector && faceDetectorAvailable){
    try {
      const result = faceDetector.detectForVideo(els.video, performance.now());
      detections = filterByConfidence(result.detections);
    } catch(e){ /* модель может быть ещё не готова к первому вызову — просто пропустим */ }
  }

  // рамки автоматически найденных лиц (для наглядности, не сама маска)
  ctx.strokeStyle = '#5fb8a8';
  ctx.lineWidth = Math.max(2, els.canvas.width * 0.004);
  detections.forEach(d => {
    const box = expandBox(d.boundingBox, getPaddingPercent());
    ctx.strokeRect(box.x, box.y, box.w, box.h);
  });

  // ручные зоны
  ctx.strokeStyle = '#c98a4b';
  manualRegions.forEach(r => {
    const x = r.x0 * els.canvas.width, y = r.y0 * els.canvas.height;
    const w = (r.x1 - r.x0) * els.canvas.width, h = (r.y1 - r.y0) * els.canvas.height;
    ctx.strokeRect(x, y, w, h);
  });

  if (drawingRegion){
    ctx.strokeStyle = '#c98a4b';
    ctx.setLineDash([6,4]);
    ctx.strokeRect(drawingRegion.x, drawingRegion.y, drawingRegion.w, drawingRegion.h);
    ctx.setLineDash([]);
  }

  els.faceCountBadge.textContent = detections.length > 0
    ? `найдено лиц: ${detections.length} (на первом кадре)`
    : 'лиц не обнаружено на первом кадре';
}

function filterByConfidence(detections){
  const threshold = getConfidenceThreshold();
  return (detections || []).filter(d => (d.categories?.[0]?.score ?? 0) >= threshold);
}
function getConfidenceThreshold(){ return parseInt(els.confidenceSlider.value, 10) / 100; }
function getPaddingPercent(){ return parseInt(els.paddingSlider.value, 10) / 100; }
function getIntensity(){ return parseInt(els.intensitySlider.value, 10); }

function expandBox(bbox, paddingPct){
  const padX = bbox.width * paddingPct * 0.5;
  const padY = bbox.height * paddingPct * 0.5;
  let x = bbox.originX - padX;
  let y = bbox.originY - padY;
  let w = bbox.width + padX * 2;
  let h = bbox.height + padY * 2;
  x = Math.max(0, x); y = Math.max(0, y);
  w = Math.min(w, els.canvas.width - x);
  h = Math.min(h, els.canvas.height - y);
  return { x, y, w, h };
}

// ---------------------------------------------------------------------------
// Ручное рисование зон
// ---------------------------------------------------------------------------

els.manualModeBtn.addEventListener('click', () => {
  manualModeOn = !manualModeOn;
  els.manualModeBtn.style.borderColor = manualModeOn ? 'var(--teal)' : 'var(--line)';
  els.manualHint.style.display = manualModeOn ? 'block' : 'none';
});

els.clearManualBtn.addEventListener('click', () => {
  manualRegions = [];
  renderStaticPreview();
});

function canvasCoordsFromEvent(e){
  const rect = els.canvas.getBoundingClientRect();
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;
  const x = (clientX - rect.left) / rect.width * els.canvas.width;
  const y = (clientY - rect.top) / rect.height * els.canvas.height;
  return { x, y };
}

let startPoint = null;
function onPointerDown(e){
  if (!manualModeOn) return;
  e.preventDefault();
  startPoint = canvasCoordsFromEvent(e);
}
function onPointerMove(e){
  if (!manualModeOn || !startPoint) return;
  e.preventDefault();
  const p = canvasCoordsFromEvent(e);
  drawingRegion = {
    x: Math.min(startPoint.x, p.x), y: Math.min(startPoint.y, p.y),
    w: Math.abs(p.x - startPoint.x), h: Math.abs(p.y - startPoint.y)
  };
  renderStaticPreview();
}
function onPointerUp(){
  if (!manualModeOn || !startPoint || !drawingRegion) { startPoint = null; return; }
  if (drawingRegion.w > 8 && drawingRegion.h > 8){
    manualRegions.push({
      x0: drawingRegion.x / els.canvas.width,
      y0: drawingRegion.y / els.canvas.height,
      x1: (drawingRegion.x + drawingRegion.w) / els.canvas.width,
      y1: (drawingRegion.y + drawingRegion.h) / els.canvas.height
    });
  }
  drawingRegion = null;
  startPoint = null;
  renderStaticPreview();
}
els.canvas.addEventListener('mousedown', onPointerDown);
els.canvas.addEventListener('mousemove', onPointerMove);
window.addEventListener('mouseup', onPointerUp);
els.canvas.addEventListener('touchstart', onPointerDown, { passive:false });
els.canvas.addEventListener('touchmove', onPointerMove, { passive:false });
els.canvas.addEventListener('touchend', onPointerUp);

// ---------------------------------------------------------------------------
// Настройки — переключатели стиля и слайдеры
// ---------------------------------------------------------------------------

document.querySelectorAll('.style-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.style-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentStyle = btn.dataset.style;
  });
});

els.intensitySlider.addEventListener('input', () => {
  els.intensityVal.textContent = els.intensitySlider.value + '%';
});
els.confidenceSlider.addEventListener('input', () => {
  els.confidenceVal.textContent = els.confidenceSlider.value + '%';
  if (!isProcessing) renderStaticPreview();
});
els.paddingSlider.addEventListener('input', () => {
  els.paddingVal.textContent = els.paddingSlider.value + '%';
  if (!isProcessing) renderStaticPreview();
});

// ---------------------------------------------------------------------------
// Применение эффекта к области кадра
// ---------------------------------------------------------------------------

function applyEffectToRegion(x, y, w, h){
  if (w <= 0 || h <= 0) return;
  const intensity = getIntensity();
  x = Math.max(0, Math.floor(x)); y = Math.max(0, Math.floor(y));
  w = Math.min(Math.floor(w), els.canvas.width - x);
  h = Math.min(Math.floor(h), els.canvas.height - y);
  if (w <= 0 || h <= 0) return;

  if (currentStyle === 'black'){
    const alpha = 0.35 + (intensity/100) * 0.65;
    ctx.fillStyle = `rgba(10,10,10,${alpha})`;
    ctx.fillRect(x, y, w, h);

  } else if (currentStyle === 'pixelate'){
    const cellCount = Math.max(2, Math.round(24 - (intensity/100) * 20));
    const sw = Math.max(1, Math.floor(w / cellCount));
    const sh = Math.max(1, Math.floor(h / cellCount));
    const tmp = document.createElement('canvas');
    tmp.width = Math.max(1, Math.floor(w / sw));
    tmp.height = Math.max(1, Math.floor(h / sh));
    const tctx = tmp.getContext('2d');
    tctx.drawImage(els.video, x, y, w, h, 0, 0, tmp.width, tmp.height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, tmp.width, tmp.height, x, y, w, h);
    ctx.imageSmoothingEnabled = true;

  } else if (currentStyle === 'blur'){
    const radius = Math.round(3 + (intensity/100) * 30);
    const pad = radius;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.clip();
    ctx.filter = `blur(${radius}px)`;
    const sx = Math.max(0, x - pad), sy = Math.max(0, y - pad);
    const sw2 = Math.min(els.canvas.width, w + pad*2);
    const sh2 = Math.min(els.canvas.height, h + pad*2);
    ctx.drawImage(els.video, sx, sy, sw2, sh2, sx, sy, sw2, sh2);
    ctx.restore();
  }
}

// ---------------------------------------------------------------------------
// Обработка и экспорт видео
// ---------------------------------------------------------------------------

function pickMimeType(){
  const candidates = [
    'video/webm;codecs=vp9,opus',
    'video/webm;codecs=vp8,opus',
    'video/webm',
    'video/mp4'
  ];
  return candidates.find(t => window.MediaRecorder && MediaRecorder.isTypeSupported(t)) || '';
}

els.processBtn.addEventListener('click', startProcessing);
els.cancelProcessBtn.addEventListener('click', () => { cancelRequested = true; });
els.startOverBtn.addEventListener('click', () => location.reload());

async function startProcessing(){
  if (!faceDetector && faceDetectorAvailable){
    await initFaceDetector();
  }
  if (!faceDetectorAvailable && manualRegions.length === 0){
    showStatus('Автопоиск лиц недоступен, и не задано ни одной ручной зоны — нечего замазывать. Добавь хотя бы одну зону вручную или обнови страницу для повторной попытки загрузки модели.', 'error');
    return;
  }
  isProcessing = true;
  cancelRequested = false;
  els.processBtn.disabled = true;
  els.cancelProcessBtn.style.display = 'inline-block';
  els.progressWrap.classList.add('show');
  els.resultBox.classList.remove('show');
  manualModeOn = false;
  els.manualHint.style.display = 'none';

  const mimeType = pickMimeType();
  if (!mimeType){
    showStatus('Браузер не поддерживает запись видео (MediaRecorder). Попробуй обновить браузер.', 'error');
    resetProcessingUI();
    return;
  }

  let combinedStream;
  try {
    const canvasStream = els.canvas.captureStream(30);
    const audioSource = els.video.captureStream ? els.video.captureStream() : els.video.mozCaptureStream();
    const audioTracks = audioSource.getAudioTracks();
    combinedStream = new MediaStream([...canvasStream.getVideoTracks(), ...audioTracks]);
  } catch(e){
    showStatus('Не удалось захватить видео/аудио поток: ' + e.message, 'error');
    resetProcessingUI();
    return;
  }

  recordedChunks = [];
  mediaRecorder = new MediaRecorder(combinedStream, { mimeType });
  mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };

  const finishedPromise = new Promise((resolve) => {
    mediaRecorder.onstop = () => resolve();
  });

  mediaRecorder.start();
  els.video.currentTime = 0;
  await new Promise((resolve) => { els.video.onseeked = resolve; });
  await els.video.play();

  function frameLoop(){
    if (cancelRequested || els.video.ended || els.video.paused){
      if (mediaRecorder.state !== 'inactive') mediaRecorder.stop();
      return;
    }
    renderProcessedFrame();
    const pct = Math.min(100, Math.round((els.video.currentTime / els.video.duration) * 100));
    els.progressFill.style.width = pct + '%';
    els.progressText.textContent = pct + '%' + (cancelRequested ? ' (отмена…)' : '');
    requestAnimationFrame(frameLoop);
  }
  requestAnimationFrame(frameLoop);

  await finishedPromise;
  els.video.pause();

  if (cancelRequested){
    resetProcessingUI();
    return;
  }

  const blob = new Blob(recordedChunks, { type: mimeType.split(';')[0] });
  const url = URL.createObjectURL(blob);
  els.resultVideo.src = url;
  els.downloadLink.href = url;
  const ext = mimeType.includes('mp4') ? 'mp4' : 'webm';
  els.downloadLink.download = 'faceblur-result.' + ext;
  els.resultBox.classList.add('show');
  els.progressText.textContent = 'Готово';

  resetProcessingUI();
}

function renderProcessedFrame(){
  ctx.drawImage(els.video, 0, 0, els.canvas.width, els.canvas.height);

  let detections = [];
  if (faceDetector && faceDetectorAvailable){
    try {
      const result = faceDetector.detectForVideo(els.video, performance.now());
      detections = filterByConfidence(result.detections);
    } catch(e){ /* пропускаем кадр при редкой ошибке детектора */ }
  }

  detections.forEach(d => {
    const box = expandBox(d.boundingBox, getPaddingPercent());
    applyEffectToRegion(box.x, box.y, box.w, box.h);
  });

  manualRegions.forEach(r => {
    const x = r.x0 * els.canvas.width, y = r.y0 * els.canvas.height;
    const w = (r.x1 - r.x0) * els.canvas.width, h = (r.y1 - r.y0) * els.canvas.height;
    applyEffectToRegion(x, y, w, h);
  });
}

function resetProcessingUI(){
  isProcessing = false;
  els.processBtn.disabled = false;
  els.cancelProcessBtn.style.display = 'none';
}

// ---------------------------------------------------------------------------
// Service worker
// ---------------------------------------------------------------------------

if ('serviceWorker' in navigator){
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('sw.js').catch(()=>{});
  });
}
