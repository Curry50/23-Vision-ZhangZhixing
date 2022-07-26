# CMake任务

#### 考核标准

1. 是否成功运行程序（50分）
   + 若无法正常编译，此环节记0分，若运行内容异常，此环节记20分
2. CMakeLists.txt编写能力与书写规范（50分）

#### 要求

具体要求参考《华南虎视觉组提前批任务》CMake篇。这里再提一嘴，仅允许使用Ubuntu作为开发平台。

#### 注意事项

考核人员基本情况（防止有同学担心我们会故意刁难，其实不会）：

+ 考核用的电脑环境为Ubuntu20.04 LTS，并满足以下版本号要求：CMake 3.19，OpenCV 4.5.3

+ 考核人员会跳转至 CMake_Test 文件夹下，打开终端键入以下命令行，若不通过即视为失败

  ```bash
  mkdir build
  cd build
  cmake ..
  make -j6
  ./test
  ```

#### 最终参考运行效果

```
M1 construct
I'm M1
I'm A1
I'm A2
I'm A3
M2: I'm A2
size = 1
dis = 28.2843
M1 destruct
```
