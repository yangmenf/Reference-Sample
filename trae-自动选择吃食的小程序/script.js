const foods = [
  "火锅",
  "烤肉",
  "炒饭",
  "面条",
  "汉堡",
  "披萨",
  "寿司",
  "盖浇饭",
  "麻辣烫",
  "炸鸡",
  "煲仔饭",
  "沙拉",
  "饺子",
  "粥",
  "便当",
  "烧烤",
];

const foodDisplay = document.getElementById("foodDisplay");
const startBtn = document.getElementById("startBtn");
let isSelecting = false;
let intervalId = null;

startBtn.addEventListener("click", () => {
  if (isSelecting) {
    stopSelection();
  } else {
    startSelection();
  }
});

function startSelection() {
  isSelecting = true;
  startBtn.textContent = "停止";
  foodDisplay.classList.add("animate");

  intervalId = setInterval(() => {
    const randomIndex = Math.floor(Math.random() * foods.length);
    foodDisplay.textContent = foods[randomIndex];
  }, 100);
}

function stopSelection() {
  isSelecting = false;
  startBtn.textContent = "开始选择";
  foodDisplay.classList.remove("animate");
  clearInterval(intervalId);
}
