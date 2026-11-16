// ======== CONFIG ========
const FRAMES_PER_SECOND = 0.5;   // how fast the camera moves to the next image (1 = 1 FPS, 0.2 = every 5 s)
const SCAN_TIME_MS = 600;        // how long "SCANNING" shows before the result
const LOG_SIZE = 4;              // how many past detections to list
// ========================

const frameImg = document.getElementById("frame");
const cat = document.getElementById("cat");
const verdict = document.getElementById("verdict");
const confText = document.getElementById("conf-text");
const confBar = document.getElementById("conf-bar");
const log = document.getElementById("log");

let frames = [];
let lastName = null;
let frameCount = 0;

// running clock, like a real camera
function tick() {
  const now = new Date();
  document.getElementById("clock").textContent =
    now.toISOString().slice(0, 10) + "  " + now.toTimeString().slice(0, 8);
}
setInterval(tick, 1000);
tick();

function showScanning() {
  cat.src = "/static/img/catgeneral.png";
  verdict.textContent = "SCANNING";
  verdict.className = "verdict scanning";
  confText.textContent = "--";
  confBar.style.width = "0";
  confBar.className = "";
}

function showResult(result) {
  const label = result.label;               // "bird" or "drone"
  const percent = Math.round(result.confidence * 100);

  cat.src = `/static/img/cat${label}.png`;
  verdict.textContent = label.toUpperCase();
  verdict.className = `verdict ${label}`;
  confText.textContent = percent + "%";
  confBar.style.width = percent + "%";
  confBar.className = label;

  showFile(label, percent);

  const item = document.createElement("li");
  item.innerHTML = `<span>${new Date().toTimeString().slice(0, 8)}</span>` +
                   `<span class="${label}">${label.toUpperCase()} ${percent}%</span>`;
  log.prepend(item);
  while (log.children.length > LOG_SIZE) log.lastChild.remove();
}

// last analyzed frame as a "case file" with a stamp at a random angle/position
function showFile(label, percent) {
  const random = (min, max) => min + Math.random() * (max - min);

  document.getElementById("file").classList.add("visible");
  document.getElementById("thumb").src = frameImg.src;
  document.getElementById("file-tab").textContent =
    "CASE FILE · " + document.getElementById("frame-id").textContent;
  document.getElementById("file-name").textContent = decodeURIComponent(lastName);
  document.getElementById("file-time").textContent = percent + "%";

  const stamp = document.getElementById("stamp");
  stamp.textContent = label.toUpperCase();
  stamp.className = "stamp " + label;
  stamp.style.left = random(8, 45) + "%";
  stamp.style.top = random(10, 45) + "%";
  stamp.style.setProperty("--angle", random(-28, 28) + "deg");
  void stamp.offsetWidth;            // restart the stamp animation
  stamp.classList.add("hit");
}

async function nextFrame() {
  if (frames.length === 0) return;

  // pick a random frame (never the same one twice in a row)
  let name = frames[Math.floor(Math.random() * frames.length)];
  while (frames.length > 1 && name === lastName) {
    name = frames[Math.floor(Math.random() * frames.length)];
  }
  lastName = name;
  frameCount += 1;

  frameImg.src = `/camsim/${encodeURIComponent(name)}`;
  document.getElementById("frame-id").textContent = "FRM " + String(frameCount).padStart(6, "0");
  showScanning();

  const [result] = await Promise.all([
    fetch(`/api/predict?frame=${encodeURIComponent(name)}`).then(r => r.json()),
    new Promise(resolve => setTimeout(resolve, SCAN_TIME_MS)),
  ]);
  if (!result.error) showResult(result);
}

async function start() {
  frames = await fetch("/api/frames").then(r => r.json());
  if (frames.length === 0) {
    document.getElementById("no-signal").style.display = "flex";
    return;
  }
  nextFrame();
  setInterval(nextFrame, 1000 / FRAMES_PER_SECOND);
}

start();
