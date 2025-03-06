# 个人任务管理系统

## 项目介绍

个人任务管理系统是一个全栈 Web 应用程序，用于帮助用户管理日常任务。该系统提供了直观的用户界面，支持任务的添加、删除、状态更新以及多种筛选和导出功能。

## 功能特点

- **任务管理**：添加、删除、标记完成/未完成任务
- **优先级设置**：为任务设置高、中、低优先级
- **截止日期**：为任务设置截止日期
- **多维度筛选**：按状态、优先级筛选任务
- **搜索功能**：根据任务标题搜索特定任务
- **数据统计**：显示总任务数、已完成和进行中的任务数量
- **数据导出**：支持导出任务列表为 JSON 或 CSV 格式
- **主题切换**：支持浅色/深色主题切换

## 技术栈

### 前端

- HTML5
- CSS3
- JavaScript (原生)
- Font Awesome 图标库

### 后端

- Python
- Flask (Web 框架)
- Flask-CORS (跨域资源共享)

### 数据库

- Microsoft SQL Server

## 安装指南

### 前提条件

- Python 3.6+
- Microsoft SQL Server
- 现代浏览器 (Chrome, Firefox, Edge 等)

### 安装步骤

1. **克隆或下载项目**

2. **设置数据库**

   - 确保 SQL Server 已安装并运行
   - 运行以下 SQL 脚本创建数据库和表:

   ```sql
   -- 检查数据库是否存在，如果不存在则创建
   IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'TaskManagement')
   BEGIN
       CREATE DATABASE TaskManagement;
   END
   GO

   USE TaskManagement;
   GO

   -- 检查表是否存在，如果不存在则创建
   IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[Tasks]') AND type in (N'U'))
   BEGIN
       CREATE TABLE Tasks (
           id INT IDENTITY(1,1) PRIMARY KEY,
           title NVARCHAR(255) NOT NULL,
           priority NVARCHAR(50) NOT NULL,
           due_date DATE,
           status NVARCHAR(50) NOT NULL,
           created_at DATETIME DEFAULT GETDATE(),
           updated_at DATETIME DEFAULT GETDATE()
       );
   END
   GO
   ```
