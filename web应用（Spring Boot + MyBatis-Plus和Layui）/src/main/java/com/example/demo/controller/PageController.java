package com.example.demo.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class PageController {
    
    @GetMapping("/user/list.html")
    public String userList() {
        return "user/list";
    }
    
    @GetMapping("/user/add.html")
    public String userAdd() {
        return "user/add";
    }
    
    @GetMapping("/")
    public String index() {
        return "redirect:/user/list.html";
    }
}