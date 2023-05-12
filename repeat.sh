#!/bin/bash

# 循环执行程序 100 次
for i in {1..50}; do
  echo "执行程序，第 $i 次"

  timeout 600 python main.py
  if [ $? -eq 0 ]
  then
    echo "命令已完成"
  else
    echo "命令已超时，删除未完成的数据"
  fi
done