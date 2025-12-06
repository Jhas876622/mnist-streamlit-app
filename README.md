# 🧠 MNIST Streamlit App

An interactive web application built using **Streamlit** that allows users to **draw digits (0–9)** on a canvas and get real-time predictions made by a **Convolutional Neural Network (CNN)** trained on the classic **MNIST handwritten digit dataset**.

👉 **Live App:** [https://mnists.streamlit.app/](https://mnists.streamlit.app/)

---

## ⭐ Overview

This project demonstrates how deep learning models can be deployed seamlessly using **Streamlit Cloud**. The app includes:

* A drawing canvas for real-time digit input.
* Automatic preprocessing: resizing, grayscale conversion, and normalization.
* Digit prediction using a trained TensorFlow/Keras CNN model.
* Clean UI and responsive interaction.

---

## 🚀 Features

* 🎨 Interactive canvas using `streamlit-drawable-canvas`.
* ⚡ Fast prediction with a pretrained CNN model.
* 🔄 Real-time image preprocessing pipeline.
* 📊 Model trained on 60,000 MNIST images.
* 🌐 One-click deployment on Streamlit Cloud.

---

## 📂 Project Structure

```
mnist-streamlit-app/
│
├── mnistapp.py               # Main Streamlit app
├── mnist_cnn_v2.keras        # Trained CNN model (Keras format)
├── mnist_cnn_training.ipynb  # Notebook used for training the model
├── requirements.txt          # Required Python libraries
├── README.md                 # Project documentation
└── LICENSE                   # MIT License
```

---

## 🧠 Model Details

The CNN model used in this project follows this architecture:

* 2× Convolution layers (ReLU activation)
* MaxPooling layer
* Flatten layer
* Dense hidden layer
* Output layer with Softmax activation

The model achieves high accuracy on MNIST and is optimised for fast inference.

---

## 🛠️ Installation Guide

Follow these steps to run the app locally:

### 1️⃣ Clone the Repository

```
git clone https://github.com/Jhas876622/mnist-streamlit-app.git
cd mnist-streamlit-app
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```
streamlit run mnistapp.py
```

The app will open automatically in your browser.

---

## 🧪 How It Works

1. User draws a digit on the canvas.
2. The image is extracted, resized to **28×28** pixels, and converted to grayscale.
3. Preprocessed image is passed into the CNN model.
4. The predicted digit is displayed instantly.

---

## 🌐 Deployment

The app is deployed using **Streamlit Community Cloud**.

Live Site: 👉 **[https://mnists.streamlit.app/](https://mnists.streamlit.app/)**

To deploy your own version:

1. Push repository to GitHub.
2. Open Streamlit Cloud → “Deploy App”.
3. Select repo, branch, and `mnistapp.py` as the entry file.
4. Add `requirements.txt` if needed.
5. Deploy! 🎉

---

## 🤝 Contributing

Contributions are always welcome!
You may:

* Submit issues
* Improve UI/UX
* Add new features
* Enhance the CNN model

To contribute:

```
1. Fork this repo
2. Create a new branch
3. Make changes
4. Submit a pull request
```

---

## 📄 License

This project is licensed under the **MIT License**, allowing free use, modification, and distribution.

---

## ⭐ Support

If you like this project, please:

* ⭐ Star the repository
* 🔁 Share the app
* 💬 Provide suggestions

Made with ❤️ using Streamlit and TensorFlow.
