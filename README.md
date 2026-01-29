# 🧠 MNIST Neural Network

This project implements a neural network to recognize handwritten digits from the **MNIST dataset**. It includes options for training and manual testing, and automatically sets up a Python virtual environment for isolation.

---
##Fast Installation
```
cd mnist
python setup.py

```
##Rerun after installation
```
source venv/bin/activate

```
and then run the file you want to run


## 🚀 Features

- Train a neural network on the MNIST dataset (`train.py`)
- Test individual images manually (`manual.py`)
- Automatic virtual environment setup
- Easy to use for beginners

---

## 📂 Project Structure

```
mnist-neural-network/
├── venv/           # Virtual environment (auto-created)
├── train.py        # Script to train the neural network
├── manual.py       # Script for manual testing
├── setup.py        # Script to setup venv and run the project
└── README.md       # This file
```

---

## ⚡ Prerequisites

- Python 3.8 or higher
- Git (optional, if cloning repository)
- Internet connection (for installing dependencies and downloading MNIST dataset)

---

## 🏁 How to Run

### 1. Clone the repository (if not already):

```bash
git clone https://github.com/Rakshitsinghhh/mnist.git
cd mnist
```

### 2. Run the setup script:

```bash
python setup.py
```

The script will:
- Create a virtual environment (`venv`) if it doesn't exist
- Install required packages: `numpy`, `matplotlib`, `torch`, `torchvision`
- Ask whether you want to train the model or run manual testing

### 3. Choose an option:

```
What do you want to do?
1️⃣ Train the model (train.py)
2️⃣ Manual testing (manual.py)
Enter choice (1 or 2):
```

- **Train the model**: Enter `1` → runs `train.py`
- **Manual testing**: Enter `2` → runs `manual.py`

> **Note**: The default accuracy is ~97.44%

---

## 🔧 Notes

- Training will download the MNIST dataset automatically (if not already present)
- Manual testing allows you to input images or data to test the trained model
- All scripts run inside the virtual environment, keeping your system Python packages untouched

---

## ✅ Dependencies

The following packages are installed automatically via the setup script:

- `numpy`
- `matplotlib`
- `torch`
- `torchvision`

---

## 📌 Tips

### System Requirements
- Make sure your Python version is compatible (>=3.8)

### Virtual Environment Paths
- **Windows**: `venv\Scripts\python.exe`
- **Linux/Mac**: `venv/bin/python`

### Manual Activation
To manually activate the virtual environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

---

## 📄 License

This project is open source and available for educational purposes.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

## 👨‍💻 Author

Rakshit Singh

---

## ⭐ Show your support

Give a ⭐️ if this project helped you!
