# Smoking Activity Detection App

A real-time detection system that identifies smoking activity by detecting both faces and cigarettes using a YOLO-based machine learning model.

---

## Features

- Real-time detection of smoking activity via webcam
- Uses YOLO model for accurate detection
- Simple web interface powered by Streamlit

---

## Getting Started

Follow these steps to set up your environment and run the app.

### 1. Clone the Repository

```bash
git clone https://github.com/abhishekkrjha2811/YOLO-Based_Smoking_Activity_Detector.git
cd YOLO-Based_Smoking_Activity_Detector
```

### 2. Create a Virtual Environment

Create a virtual environment to manage dependencies:

```bash
python3 -m venv venv
```

Activate the virtual environment:

- **On Linux/macOS:**
    ```bash
    source venv/bin/activate
    ```
- **On Windows:**
    ```bash
    venv\Scripts\activate
    ```

### 3. Install Requirements

Install all required dependencies:

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt` file, you can install Streamlit directly:

```bash
pip install streamlit
```

### 4. Run the App

Start the Streamlit app:

```bash
streamlit run app.py
```

---

## Usage

- Open the app in your browser (Streamlit will provide a local URL).
- Click the **Start Detection** button to begin.
- The app will use your webcam to detect smoking activity in real time.
- Click **Stop Detection** to end the session.

---

## Contributing

We welcome contributions! If you'd like to improve this project, please fork the repository and submit a pull request.  
For major changes, open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
