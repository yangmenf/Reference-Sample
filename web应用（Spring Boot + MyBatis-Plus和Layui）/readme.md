# 用户管理系统

## 项目介绍
这是一个基于Spring Boot + MyBatis-Plus + Layui开发的用户管理系统，实现了用户的增删改查等基本功能。

## 技术栈
- 后端
  - Spring Boot 2.5.6
  - MyBatis-Plus 3.4.3.4
  - SQL Server
- 前端
  - Layui
  - jQuery

## 功能特性
- 用户管理
  - 用户列表展示
  - 添加用户
  - 编辑用户
  - 删除用户
  - 分页查询

## 环境要求
- JDK 1.8+
- SQL Server 2012+
- Maven 3.6+

## 快速开始

### 1. 数据库配置
1. 创建数据库
```sql
CREATE DATABASE demo;
```

2. 创建用户表
```sql
USE demo;

CREATE TABLE [sys_user] (
  [id] bigint NOT NULL IDENTITY(1,1),
  [username] varchar(50) NOT NULL,
  [password] varchar(100) NOT NULL,
  [email] varchar(100) NULL,
  [phone] varchar(20) NULL,
  [status] tinyint DEFAULT 1,
  [create_time] datetime DEFAULT GETDATE(),
  [update_time] datetime DEFAULT GETDATE(),
  CONSTRAINT PK_user PRIMARY KEY ([id]),
  CONSTRAINT UK_username UNIQUE ([username])
);
```

### 2. 修改配置
修改 `application.yml` 中的数据库连接信息：
```yaml
spring:
  datasource:
    url: jdbc:sqlserver://localhost:1433;databaseName=demo
    username: 你的用户名
    password: 你的密码
```

### 3. 运行项目
```bash
mvn spring-boot:run
```

### 4. 访问系统
打开浏览器访问：http://localhost:8080

## 项目结构
```
src/main/java/com/example/demo/
├── controller/        // 控制器
├── service/          // 服务层
│   └── impl/        // 服务实现
├── mapper/          // 数据访问层
├── entity/          // 实体类
└── common/          // 公共组件
    ├── config/      // 配置类
    └── result/      // 统一返回结果
```

## 开发说明
- 统一返回格式：使用R类封装返回结果
- 分页查询：使用MyBatis-Plus提供的分页插件
- 前端交互：使用Layui的数据表格组件

## 注意事项
1. 确保数据库服务已启动
2. 检查数据库连接配置是否正确
3. 首次运行需要执行数据库初始化脚本

## 后续优化计划
1. 添加用户认证和权限控制
2. 完善日志记录功能
3. 添加数据导入导出功能
4. 优化界面交互体验