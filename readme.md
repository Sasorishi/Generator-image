## **Setup and Installation**

### **1. Environment Setup**

1. Create and activate the Python virtual environment:

   ```bash
   source myenv/bin/activate
   ```

2. Install the required Python dependencies
   ```bash
   pip install -r requirements.txt
   ```

### **2. Train the GAN**

    1. To train the GAN on the LFW dataset, run:
    ```bash
    python train.py --dataset lfw
    ```

    2. Test the GAN:
    ```bash
    python test_canva_lfw.py
    ```

### **3. Start python server**

    1. Navigate to the Front-End Directory:
    ```bash
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
    ```

### **4. Front-End Setup**

    1. Navigate to the Front-End Directory:
    ```bash
    cd gan-front
    ```

    2. Install Dependencies:
    ```bash
    npm install
    ```

    3. Start the Front-End Application:
    ```bash
    npm start
    ```
