// API 基础URL
const API_BASE_URL = "http://localhost:5000/api";

// 获取DOM元素
const taskInput = document.getElementById("taskInput");
const taskPriority = document.getElementById("taskPriority");
const taskDueDate = document.getElementById("taskDueDate");
const addTaskBtn = document.getElementById("addTask");
const taskList = document.getElementById("taskList");
const filterStatus = document.getElementById("filterStatus");
const filterPriority = document.getElementById("filterPriority");
const searchInput = document.getElementById("searchInput");
const clearCompletedBtn = document.getElementById("clearCompleted");
const totalTasksSpan = document.getElementById("totalTasks");
const completedTasksSpan = document.getElementById("completedTasks");
const activeTasksSpan = document.getElementById("activeTasks");

// 存储所有任务
let allTasks = [];

// 获取所有任务
async function fetchTasks() {
  try {
    const response = await fetch(`${API_BASE_URL}/tasks`);
    if (!response.ok) {
      throw new Error("获取任务失败");
    }
    allTasks = await response.json();
    applyFiltersAndRender();
    updateStats();
  } catch (error) {
    console.error("获取任务失败:", error);
    alert("获取任务失败，请检查后端服务是否正常运行");
  }
}

// 添加新任务
async function addTask() {
  const title = taskInput.value.trim();
  if (!title) {
    alert("请输入任务内容");
    return;
  }

  const task = {
    title: title,
    priority: taskPriority.value,
    dueDate: taskDueDate.value || null,
    status: "active",
  };

  try {
    const response = await fetch(`${API_BASE_URL}/tasks`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(task),
    });

    if (!response.ok) {
      throw new Error("添加任务失败");
    }

    taskInput.value = "";
    await fetchTasks();
  } catch (error) {
    console.error("添加任务失败:", error);
    alert("添加任务失败，请检查后端服务是否正常运行");
  }
}

// 更新任务状态
async function updateTaskStatus(taskId, newStatus) {
  const task = allTasks.find((t) => t.id === taskId);
  if (!task) return;

  const updatedTask = {
    ...task,
    status: newStatus,
    title: task.title,
    priority: task.priority,
    dueDate: task.due_date,
  };

  try {
    const response = await fetch(`${API_BASE_URL}/tasks/${taskId}`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(updatedTask),
    });

    if (!response.ok) {
      throw new Error("更新任务状态失败");
    }

    await fetchTasks();
  } catch (error) {
    console.error("更新任务状态失败:", error);
    alert("更新任务状态失败，请检查后端服务是否正常运行");
  }
}

// 删除任务
async function deleteTask(taskId) {
  try {
    const response = await fetch(`${API_BASE_URL}/tasks/${taskId}`, {
      method: "DELETE",
    });

    if (!response.ok) {
      throw new Error("删除任务失败");
    }

    await fetchTasks();
  } catch (error) {
    console.error("删除任务失败:", error);
    alert("删除任务失败，请检查后端服务是否正常运行");
  }
}

// 应用筛选并渲染任务列表
function applyFiltersAndRender() {
  const statusFilter = filterStatus.value;
  const priorityFilter = filterPriority.value;
  const searchText = searchInput.value.toLowerCase();

  const filteredTasks = allTasks.filter((task) => {
    // 状态筛选
    if (statusFilter !== "all" && task.status !== statusFilter) {
      return false;
    }

    // 优先级筛选
    if (priorityFilter !== "all" && task.priority !== priorityFilter) {
      return false;
    }

    // 搜索筛选
    if (searchText && !task.title.toLowerCase().includes(searchText)) {
      return false;
    }

    return true;
  });

  renderTasks(filteredTasks);
}

// 渲染任务列表
function renderTasks(tasks) {
  taskList.innerHTML = "";

  if (tasks.length === 0) {
    taskList.innerHTML = '<div class="no-tasks">没有符合条件的任务</div>';
    return;
  }

  tasks.forEach((task) => {
    const taskElement = document.createElement("div");
    taskElement.className = `task-item ${task.status}`;
    taskElement.dataset.id = task.id;

    const dueDate = task.due_date
      ? new Date(task.due_date).toLocaleDateString()
      : "无截止日期";

    taskElement.innerHTML = `
            <div class="task-content">
                <input type="checkbox" class="task-checkbox" ${
                  task.status === "completed" ? "checked" : ""
                }>
                <span class="task-title">${task.title}</span>
                <span class="task-priority ${task.priority}">${
      task.priority
    }</span>
                <span class="task-date">${dueDate}</span>
                <button class="delete-btn"><i class="fas fa-trash"></i></button>
            </div>
        `;

    // 添加事件监听器
    const checkbox = taskElement.querySelector(".task-checkbox");
    checkbox.addEventListener("change", () => {
      updateTaskStatus(task.id, checkbox.checked ? "completed" : "active");
    });

    const deleteBtn = taskElement.querySelector(".delete-btn");
    deleteBtn.addEventListener("click", () => {
      if (confirm("确定要删除这个任务吗？")) {
        deleteTask(task.id);
      }
    });

    taskList.appendChild(taskElement);
  });
}

// 更新统计信息
function updateStats() {
  const total = allTasks.length;
  const completed = allTasks.filter(
    (task) => task.status === "completed"
  ).length;
  const active = total - completed;

  totalTasksSpan.textContent = total;
  completedTasksSpan.textContent = completed;
  activeTasksSpan.textContent = active;
}

// 清除已完成的任务
async function clearCompleted() {
  const completedTasks = allTasks.filter((task) => task.status === "completed");
  if (completedTasks.length === 0) {
    alert("没有已完成的任务");
    return;
  }

  if (!confirm("确定要清除所有已完成的任务吗？")) {
    return;
  }

  try {
    for (const task of completedTasks) {
      await deleteTask(task.id);
    }
  } catch (error) {
    console.error("清除已完成任务失败:", error);
  }
}

// 保存到本地
function saveToLocal() {
  const data = JSON.stringify(allTasks);
  const blob = new Blob([data], { type: "application/json" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = `任务列表_${new Date().toLocaleDateString()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// 导出任务为CSV
function exportTasks() {
  let csv = "标题,优先级,截止日期,状态\n";

  allTasks.forEach((task) => {
    const dueDate = task.due_date
      ? new Date(task.due_date).toLocaleDateString()
      : "";
    const status = task.status === "completed" ? "已完成" : "进行中";
    csv += `"${task.title}","${task.priority}","${dueDate}","${status}"\n`;
  });

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = `任务列表_${new Date().toLocaleDateString()}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// 主题切换
function toggleTheme() {
  document.body.classList.toggle("dark-theme");
  localStorage.setItem(
    "theme",
    document.body.classList.contains("dark-theme") ? "dark" : "light"
  );
}

// 事件监听器
addTaskBtn.addEventListener("click", addTask);
filterStatus.addEventListener("change", applyFiltersAndRender);
filterPriority.addEventListener("change", applyFiltersAndRender);
searchInput.addEventListener("input", applyFiltersAndRender);
clearCompletedBtn.addEventListener("click", clearCompleted);
saveToLocalBtn.addEventListener("click", saveToLocal);
exportTasksBtn.addEventListener("click", exportTasks);
document.querySelector(".theme-toggle").addEventListener("click", toggleTheme);

// 初始化
document.addEventListener("DOMContentLoaded", () => {
  // 设置今天的日期为默认日期
  const today = new Date().toISOString().split("T")[0];
  taskDueDate.value = today;

  // 加载主题设置
  const savedTheme = localStorage.getItem("theme");
  if (savedTheme === "dark") {
    document.body.classList.add("dark-theme");
  }

  // 加载任务
  fetchTasks();

  // 添加键盘事件
  taskInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      addTask();
    }
  });
});

// 添加错误处理
window.addEventListener("error", (event) => {
  console.error("全局错误:", event.error);
});
