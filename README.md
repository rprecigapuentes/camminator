# Intelligent Walker for Visually Impaired Users

An academic embedded systems and artificial intelligence project focused on assisting visually impaired users through real-time obstacle detection, spatial awareness, and voice-based interaction.

This intelligent walker integrates computer vision, LiDAR sensing, and local AI models to enhance user autonomy and safety during navigation.

---

## 📌 Project Overview

The Intelligent Walker is designed to support people with visual impairments by detecting nearby obstacles and providing auditory feedback in real time. The system combines multiple sensing and AI technologies to perceive the environment and interact naturally with the user.

The project explores the integration of embedded systems, artificial intelligence, and human-centered design in an assistive technology context.

---

## 🧠 System Architecture

The system is composed of the following main components:

- **RGB Camera + YOLO (You Only Look Once)**  
  Used for real-time object detection and classification.
  
- **LiDAR Sensor**  
  Provides distance measurements to detect obstacles and estimate proximity.

- **Whisper (Speech-to-Text)**  
  Enables voice command recognition and user interaction.

- **Ollama (Local LLM Inference)**  
  Used to implement an on-device conversational assistant for guidance and assistance.

- **Embedded Platform (Raspberry Pi)**  
  Central processing unit handling sensor fusion, AI inference, and system logic.

---

## ⚙️ Technologies Used

- **Programming Languages:** Python  
- **Computer Vision:** YOLO (CNN-based object detection)  
- **Machine Learning:** Decision Trees, Random Forest (experiments), CNN  
- **Speech Processing:** OpenAI Whisper  
- **LLMs:** Ollama (local deployment)  
- **Hardware:** Raspberry Pi, LiDAR sensor, RGB camera  
- **Operating System:** Linux-based (Raspberry Pi OS)

---

## 🔁 How the System Works

1. The RGB camera captures real-time video frames.
2. YOLO processes each frame to detect and classify objects.
3. LiDAR measures the distance to detected obstacles.
4. Sensor data is fused to determine obstacle relevance and proximity.
5. The system provides auditory feedback to the user.
6. Whisper enables voice commands.
7. Ollama handles natural language responses locally on the device.

---

## 👨‍💻 Contributions

- Integration of computer vision models for real-time object detection.
- Implementation and testing of AI-based decision logic.
- Participation in embedded system deployment and testing.
- System-level integration of vision, LiDAR, and AI modules.

---

## 📁 Repository Structure

camminator/
│
├── src/ # Source code
├── models/ # AI and ML models
├── docs/ # Documentation and diagrams
├── assets/ # Images and resources
└── README.md

(Note: structure may evolve as the project is refined.)

---

## 🚀 Future Improvements

- Improve object detection accuracy in low-light conditions.
- Optimize on-device inference performance.
- Enhance user feedback with haptic signals.
- Expand conversational capabilities of the assistant.
- Conduct usability testing with real users.
- Integrate all the functions in just one file.
---

## 📄 Project Status

This project was developed as an academic prototype and serves as a proof of concept for assistive technology using embedded AI systems. The prototipe was not completely developed due to the end of the academic period.

---

## 📬 Contact

For questions or collaboration, feel free to reach out via GitHub or LinkedIn.
