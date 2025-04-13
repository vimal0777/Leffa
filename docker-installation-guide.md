### **Steps**

#### **1. Verify NVIDIA Driver Installation**
Ensure that the NVIDIA drivers are installed and functioning correctly. Run the following command to check:

```bash
nvidia-smi
```

If this command fails or does not display GPU information, you need to install or update the NVIDIA drivers for your system.

---

#### **2. Install NVIDIA Container Toolkit**
The NVIDIA Container Toolkit enables Docker to use GPUs. Follow these steps to install it:

1. **Add the NVIDIA package repository**:
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   ```

2. **Update the package list**:
   ```bash
   sudo apt-get update
   ```

3. **Install NVIDIA Container Toolkit**:
   ```bash
   sudo apt-get install -y nvidia-docker2
   ```

4. **Restart the Docker daemon**:
   ```bash
   sudo systemctl restart docker
   ```

---

#### **3. Test NVIDIA Runtime**
After installing the NVIDIA Container Toolkit, test if the NVIDIA runtime is working correctly by running a simple GPU-enabled container:

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

This command should display the same output as running `nvidia-smi` on your host system. If it works, the NVIDIA runtime is correctly configured.

---

#### **4. Update `docker-compose.yml`**
Ensure that your `docker-compose.yml` file is correctly configured to use the NVIDIA runtime. Here’s an example of a valid configuration:

```yaml
version: '3.8'
services:
  leffa:
    image: leffa-app
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "7860:7860"
    volumes:
      - ./ckpts:/app/ckpts
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

---

#### **4. Build and Run the Application**
After verifying the NVIDIA runtime, rebuild and run the application:

1. **build the Docker image**:
   ```bash
   docker-compose build
   ```

2. **Start the services**:
   ```bash
   docker-compose up
   ```

---

### **Additional Notes**
1. **Check Docker Version**:
   Ensure that your Docker version supports the NVIDIA runtime. You can check your Docker version with:
   ```bash
   docker --version
   ```
   The NVIDIA Container Toolkit requires Docker 19.03 or later.

2. **Fallback to `--gpus` Flag**:
   If the `runtime: nvidia` directive continues to fail, you can replace it with the `--gpus` flag in your `docker-compose.yml` file:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - capabilities: [gpu]
   ```
   Alternatively, run the container manually with:
   ```bash
   docker run --gpus all -p 7860:7860 -v /root/Leffa:/app leffa-app
   ```

---

### **Summary**
To resolve the issue:
1. Verify that the NVIDIA drivers are installed and working (`nvidia-smi`).
2. Install and configure the NVIDIA Container Toolkit.
3. Test the NVIDIA runtime with a GPU-enabled container.
4. Update your `docker-compose.yml` file and rebuild the application.

Once these steps are completed, the container should start successfully with GPU support. Let me know if you encounter further issues!
