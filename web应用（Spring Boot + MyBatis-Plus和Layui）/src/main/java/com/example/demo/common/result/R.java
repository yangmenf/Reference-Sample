package com.example.demo.common.result;

import lombok.Data;

@Data
public class R<T> {
    private Integer code;
    private String msg;
    private T data;

    public static <T> R<T> ok(T data) {
        R<T> r = new R<>();
        r.setCode(0);
        r.setMsg("success");
        r.setData(data);
        return r;
    }

    public static <T> R<T> error(String msg) {
        R<T> r = new R<>();
        r.setCode(-1);
        r.setMsg(msg);
        return r;
    }
}