# MaxKB上传功能测试

## 测试文件说明

- `test_maxkb_uploader.py` - MaxKB上传模块单元测试
- `test_data_processor_maxkb.py` - DataProcessor MaxKB集成测试
- `run_tests.py` - 运行所有测试的脚本

## 运行测试

### 方法1：从DataProcessor目录运行（推荐）

```bash
cd Datapulse/backend/DataProcessor
python test/run_tests_simple.py
```

### 方法2：使用unittest发现测试

```bash
cd Datapulse/backend/DataProcessor
python -m unittest discover test/ -v
```

### 方法3：运行单个测试文件

```bash
cd Datapulse/backend/DataProcessor
python -m unittest test.test_maxkb_uploader -v
python -m unittest test.test_data_processor_maxkb -v
```

### 方法4：运行单个测试用例

```bash
cd Datapulse/backend/DataProcessor
python -m unittest test.test_maxkb_uploader.TestMaxKBUploader.test_init -v
```

### 方法5：使用pytest（如果已安装）

```bash
cd Datapulse/backend/DataProcessor
pytest test/ -v
```

## 测试内容

### test_maxkb_uploader.py

- ✅ 初始化测试
- ✅ 设置知识库ID测试
- ✅ 手动设置token测试
- ✅ 登录成功测试（token方式）
- ✅ 登录成功测试（Cookie方式）
- ✅ 登录失败测试
- ✅ 未登录时上传失败测试
- ✅ 未设置知识库ID时上传失败测试
- ✅ 文件不存在时上传失败测试
- ✅ 上传成功测试
- ✅ 上传失败测试
- ✅ upload_text_file便捷方法测试

### test_data_processor_maxkb.py

- ✅ 不启用MaxKB时的初始化测试
- ✅ 使用环境变量配置MaxKB测试
- ✅ 使用代码配置MaxKB测试
- ✅ 未提供知识库ID时的处理测试
- ✅ 成功上传到MaxKB测试
- ✅ 登录失败时的处理测试
- ✅ MaxKB模块不可用时的处理测试

## 集成测试

集成测试需要真实的MaxKB服务运行。默认情况下这些测试会被跳过。

要运行集成测试：

1. 确保MaxKB服务正在运行：`http://localhost:8080`
2. 修改测试文件，移除 `@unittest.skip` 装饰器
3. 配置正确的用户名、密码和知识库ID
4. 运行测试

## 测试覆盖率

运行测试覆盖率检查：

```bash
pip install coverage
coverage run -m unittest discover test/
coverage report
coverage html  # 生成HTML报告
```

## 注意事项

- 单元测试使用mock，不需要真实的MaxKB服务
- 集成测试需要真实的MaxKB服务
- 测试会创建临时文件，测试完成后会自动清理

