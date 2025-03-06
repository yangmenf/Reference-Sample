package com.example.demo.controller;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.example.demo.common.result.R;
import com.example.demo.entity.User;
import com.example.demo.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.util.StringUtils;

@RestController
@RequestMapping("/user")
public class UserController {
    
    @Autowired
    private UserService userService;

    @GetMapping("/list")
    public R<?> list(@RequestParam(defaultValue = "1") Integer page,
                     @RequestParam(defaultValue = "10") Integer limit,
                     String username) {
        try {
            // 验证分页参数
            if (page == null || page <= 0) {
                return R.error("页码必须大于0");
            }
            if (limit == null || limit <= 0 || limit > 100) {
                return R.error("每页显示条数必须在1-100之间");
            }

            // 构建分页参数
            Page<User> pageParam = new Page<>(page, limit);
            LambdaQueryWrapper<User> wrapper = new LambdaQueryWrapper<>();
            
            // 构建查询条件
            if (StringUtils.hasText(username)) {
                wrapper.like(User::getUsername, username);
            }
            wrapper.orderByDesc(User::getCreateTime);

            // 执行分页查询
            Page<User> resultPage = userService.page(pageParam, wrapper);
            return R.ok(resultPage);
        } catch (Exception e) {
            e.printStackTrace();
            String errorMsg = "查询用户列表失败";
            if (e.getMessage() != null) {
                errorMsg += "：" + e.getMessage();
            }
            return R.error(errorMsg);
        }
    }

    @PostMapping("/add")
    @ResponseBody
    public R<?> add(@RequestBody User user) {
        try {
            // 设置创建时间和更新时间
            if (user.getCreateTime() == null) {
                user.setCreateTime(java.time.LocalDateTime.now());
            }
            if (user.getUpdateTime() == null) {
                user.setUpdateTime(java.time.LocalDateTime.now());
            }
            
            boolean result = userService.save(user);
            return result ? R.ok(null) : R.error("添加用户失败");
        } catch (Exception e) {
            e.printStackTrace();
            return R.error("系统异常：" + e.getMessage());
        }
    }
    @PutMapping("/update")
    public R<?> update(@RequestBody User user) {
        return R.ok(userService.updateById(user));
    }

    @DeleteMapping("/delete/{id}")
    public R<?> delete(@PathVariable Long id) {
        return R.ok(userService.removeById(id));
    }

    @GetMapping("/info/{id}")
    public R<?> info(@PathVariable Long id) {
        return R.ok(userService.getById(id));
    }
}