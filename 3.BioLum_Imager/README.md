# Imaging System Controller

Biolum system is designed for imaging of bioluminescence in plants.

The system comprises:
- Raspberry Pi 4 + PSU
- Nikon D800 camera (kindly provided by Martin)
- White LED for illumination while focusing and taking "Day" images
- Flask-based local web application

---

## Features

- Live preview for camera focusing
- Acquisition of:
  - day images
  - bioluminescence images
- File preview and download through the web GUI
- Automatic timestamping of images to prevent overwriting
- Currently supports acquisition of:
  - 1 day image
  - 1 bioluminescence image
- Timelapse acquisition is not yet implemented

---



<details> <summary> <H2>🫖 Accessing the GUI </H2> </summary>

To have continous access to the GUI, it has to automatically start and run continuously on the Raspberry Pi:

### 1. Copy the `biolum_controller.service` file to the Raspberry Pi

Example using `scp`:

```bash
scp biolum_controller.service pi@<RPI_IP>:~
```

---

### 2. Install and enable the service on the Raspberry Pi

Run the following commands on the Raspberry Pi:

```bash
sudo cp ~/controller.service /etc/systemd/system/

sudo systemctl daemon-reload

sudo systemctl enable controller

sudo systemctl start controller
```

1. Connect a laptop to the Raspberry Pi hotspot/network.
2. Determine the Raspberry Pi IP address (RPI_IP):

```bash
hostname -I
```

3. Open a web browser and navigate to:

```text
http://<RPI_IP>:5000/
```

Example:

```text
http://10.42.0.1:5000/
```

---
</details>

<details> <summary> <H2>⚙ Restarting the Controller During Development </H2> </summary>

After updating the GUI/controller code, restart the service:

```bash
sudo systemctl stop controller

sleep 2

pkill -f gphoto2

sleep 1

# Copy/update the new controller files here

sudo systemctl start controller
```
</details>
