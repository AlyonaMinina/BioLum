# Bioluminescence measurement
This is a web-based Graphical User Interface (GUI) for quick and convenient quantitative detection of multicolored bioluminescence in plants. The GUI is designed to process images acquired using our Biolum imaging system based on a Nikon D800 camera controlled by a Raspberry Pi 4.

<p align="center">
   <img src="Previews/Measurment%20tab%20preview.png" width="80%" />
   <img src="Previews/Analysis%20tab%20preview.png" width="80%" />
</p>



<details>
<summary><h1>🫖 Installation</h1></summary>

1. Click the green **Code** button in the top-right corner of the repository page and select **Download ZIP**.

2. Extract the downloaded ZIP archive.

3. Download and install Python from:  
   [Python Downloads](https://www.python.org/downloads/?utm_source=chatgpt.com)

4. During installation, make sure to check **"Add Python to PATH"**.
</details>


<details>
<summary><h1>🫖 Starting the GUI </h1></summary>

5. Open **Command Prompt** in Windows:
   - Press `Windows + R`
   - Type `cmd`
   - Press `Enter`

6. In the Command Prompt window, install the required Python packages:

```bash
pip install flask rawpy openpyxl numpy pillow
```

7. Still in the same Command Prompt window, navigate to the extracted repository folder:

```bash
cd "C:\Users\Your_User_Name\Downloads\Bioluminescence-main"
```

8. Still in the same Command Prompt window, start the application:

```bash
python biolum_analysis.py
```

9. The GUI will open in your web browser at:

```text
http://localhost:5001
```

</details>
<details>
<summary><h1> 👣Features and Step-by-Step Instructions</h1></summary>


## Preparing Files

- Prepare a folder containing the files selected for analysis.
- Pair each `.nef` file with a corresponding `.jpg` file.
- Verify that filenames match exactly.

Example:
```text
Sample_01.nef
Sample_01.jpg
```

> **Note:** The system uses light `.jpg` files for previews and RAW `.nef` files for data quantification.

---

## Loading Images

- The GUI opens in the **Measurement** tab.
- Use the folder picker to navigate to the folder containing your files.
- Click a day or bioluminescence file to load it into the image viewer.
- Load the matching day/bioluminescence pair.
- NB! The GUI only allows files with the same sample name to be paired, preventing mix-ups.
- To load a different sample, clear the current images from the viewer by clicking the **X Clear** buttons located in the top-right corner of each image.
---

## Drawing and Managing ROIs

- Select the ROI shape from the dropdown menu below the image viewer.
- Activate the **Draw** button to create ROIs anywhere on the image.
- Use:
  - **Move**
  - **Resize**
  - **Delete**
  - **Clear All**

  to modify ROIs.

- ROI size can be fixed by enabling the **Lock Size** checkbox.
- Use the mouse wheel to zoom in/out.
- Use the **Pan** button to move around the image.
- Click **Fit** to resize the image to fit the viewer window.

---

## Background Measurement

- Click the **BCKG** button to draw ROIs for background signal measurement.

---

## Measuring and Saving Results

- Once all ROIs are drawn, click **Measure ROIs** in the top-right corner.
  - This populates the measurement table.

- Click **Save Results** to save:
  - The measurement table
  - A snapshot of the image with ROIs overlaid

---

# Analysis Tab

- After you clicked **Save Results**, the **Analysis** tab automatically generates box plots for:
  - Mean integrated density values
  - Background-subtracted signals

- If ROIs should be treated as separate samples:
  - Enable the **Rename ROIs** checkbox
  - Rename ROIs as desired

- ROIs sharing the same name are grouped together in the analysis summary.

- The analysis also performs:
  - Tukey’s HSD test
  - Compact Letter Display (CLD) plotting

- Plots can be exported as PDF files.
  - Output files are saved in the `analysis` subfolder.

---

# Re-analysis

- Images can be reopened for additional analysis rounds.
- Previously saved ROIs can be reloaded, adjusted, and re-measured.
</details>
