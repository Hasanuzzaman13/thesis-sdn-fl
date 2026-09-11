# Privacy-Preserving FL Framework for Multi-Layer SDN Intrusion Detection and DDoS Mitigation (FL-SDN)

This repository contains the official implementation of the B.Sc. thesis project: **"Privacy-Preserving FL Framework for Multi-Layer SDN Intrusion Detection and DDoS Mitigation"**. 

The project proposes **FL-SDN**, a novel framework that combines Federated Learning (FL) with Software-Defined Networking (SDN) to detect and mitigate DDoS attacks in real-time while preserving data privacy. Instead of centralizing raw network traffic, each SDN switch trains a local model and shares only model updates, ensuring compliance with data protection principles (e.g., GDPR).

---

## 🌟 Key Features
- **Privacy-Preserving Training**: Raw network traffic never leaves the local client. Only model weights are shared via the Flower framework.
- **Lightweight Detection Model**: A custom fully-connected neural network (29 → 128 → 64 → 2) optimized for fast inference and low communication overhead (~57 KB per update).
- **Real-Time Mitigation**: A custom Ryu controller application that dynamically installs OpenFlow `DROP` rules upon detecting anomalous traffic patterns.
- **Federated Averaging (FedAvg)**: Aggregates local model updates from 4 simulated clients over 20 communication rounds.

---

## 📊 Experimental Results
- **Accuracy**: 96.33%
- **Precision (Macro)**: 96.37%
- **Recall (Macro)**: 91.44%
- **F1-Score (Macro)**: 93.66%
- **Response Time**: 49.56 ms (well within the <100 ms 5G URLLC latency budget)
- **Communication Overhead**: ~8.84 MB total for the entire 20-round training (4 clients).

---

## 🛠️ Tech Stack & Requirements
- **Language**: Python 3.11
- **Deep Learning**: PyTorch, Scikit-learn
- **Federated Learning**: Flower (`flwr`)
- **SDN Emulation**: Mininet, Open vSwitch (OVS)
- **SDN Controller**: Ryu (Python-based)
- **Dataset**: Hybrid dataset (InSDN + CIC-IDS2017), preprocessed to 29 common features.

---

## 📂 Project Structure
```text
thesis-sdn-fl/
├── fl_server.py          # Flower server script for FedAvg aggregation
├── fl_client.py          # Flower client script for local training
├── model.py              # PyTorch IntrusionDetectionNet architecture
├── datasets/             # Partitioned client datasets (sw1.csv to sw4.csv)
├── ryu_apps/
│   ├── ddos_mitigation.py        # Threshold-based real-time mitigation
│   └── ddos_mitigation_ml.py     # ML-ready mitigation skeleton
├── sdn_attack_test.py    # Automated Mininet attack simulation script
├── requirements.txt      # Python dependencies
└── README.md             # This file
