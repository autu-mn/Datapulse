# 发布 GitPulse 到 PyPI

## 📦 发布步骤

### 1. 安装构建工具

```bash
pip install build twine
```

### 2. 构建分发包

```bash
# 在 GitPulse-Model 目录下
python -m build
```

这会生成 `dist/` 目录，包含：
- `gitpulse-1.0.0.tar.gz` (源码包)
- `gitpulse-1.0.0-py3-none-any.whl` (wheel 包)

### 3. 检查分发包

```bash
# 检查包内容
twine check dist/*
```

### 4. 上传到 PyPI

#### 测试上传（TestPyPI）

```bash
# 先上传到测试环境
twine upload --repository testpypi dist/*

# 测试安装
pip install -i https://test.pypi.org/simple/ gitpulse
```

#### 正式上传

```bash
# 上传到正式 PyPI
twine upload dist/*
```

### 5. 验证安装

```bash
pip install gitpulse
python -c "from gitpulse import GitPulseModel; print('✓ Installed successfully!')"
```

## 🔧 使用方式

安装后，用户可以这样使用：

```python
from gitpulse import GitPulseModel
from transformers import DistilBertTokenizer
import torch

# 从 HuggingFace Hub 加载模型
model = GitPulseModel.from_pretrained("Patronum-ZJ/GitPulse")

# 准备输入
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
text = "A Python library for machine learning"
time_series = torch.randn(1, 128, 16)  # [batch, hist_len, n_vars]

# 预测
predictions = model.predict(
    time_series=time_series,
    text=text,
    tokenizer=tokenizer
)
```

## 📝 注意事项

1. **PyPI 账号**：需要先注册 https://pypi.org/account/register/
2. **API Token**：建议使用 API Token 而不是密码
   - 在 PyPI 设置中创建 Token
   - 使用 `twine upload -u __token__ -p <token>` 上传
3. **版本号**：每次发布需要更新 `setup.py` 中的版本号
4. **模型权重**：模型权重存储在 HuggingFace Hub，不会打包到 PyPI

## 🎯 发布后

发布成功后，用户可以：

```bash
pip install gitpulse
```

然后直接使用：

```python
from gitpulse import GitPulseModel
model = GitPulseModel.from_pretrained()  # 默认从 HuggingFace 下载
```





