# OIT 245: Python & Cursor IDE Setup Guide

Welcome to OIT 245! This lecture is the first of two parts where we will get to experience the state-of-the-art workflow with AI and Python.

In today's lecture, our goal is to:
1.  Set up Python on your computer.
2.  Understand how to use a **Virtual Environment**.
3.  Run a simple exercise to verify everything works.

We will be using **Cursor**, a modern code editor that integrates AI directly into the coding process. Follow these steps carefully—no prior technical knowledge is required!

---

## Part 1: Install Cursor IDE

**Cursor** is the program where we will write our code. 
*(Note: If you prefer to use **VS Code**, that is perfectly fine too! See **Appendix A** at the bottom of this document for details.)*

*Concept: An **IDE (Integrated Development Environment)** is a specialized text editor for programming. Unlike Word or Notes, it highlights your code in colors to make it readable and helps catch errors before you run them.*

### For Mac & Windows:
1.  Go to the official website: [cursor.com](https://cursor.com)
2.  Click the **Download** button.
3.  **Mac Users**: 
    - Open the downloaded `.dmg` file.
    - Drag the "Cursor" icon into your "Applications" folder.
    - Open Cursor from your Applications.
4.  **Windows Users**:
    - Open the downloaded installer (`.exe` file).
    - Follow the installation prompts (click "Next", "Install", etc.).
    - Open Cursor once installed.

---

## Part 2: Install Python

We need to install the Python language itself so your computer can understand the code.
*Concept: Python is the "engine" that interprets the text you write and performs the actions. The IDE is just the dashboard; Python is what actually runs the car.*

### For Mac Users:
1.  Go to [python.org/downloads](https://www.python.org/downloads/).
2.  Click the yellow button that says **Download Python 3.x.x** (the latest version).
3.  Open the downloaded package file (`.pkg`).
4.  Follow the "Continue" prompts to install. You may need to enter your Mac password.
5.  Once finished, you can close the installer.

### For Windows Users:
1.  Go to [python.org/downloads](https://www.python.org/downloads/).
2.  Click the yellow button that says **Download Python 3.x.x**.
3.  **IMPORTANT**: When you run the installer, look for a checkbox at the bottom that says **"Add Python to PATH"**. **You MUST check this box** before clicking "Install Now".
    - *Concept: The **PATH** is a list of address books your computer checks to find programs. Checking this ensures that when you type "python", your computer knows exactly where to look to find it.*
4.  Click "Install Now" and wait for it to finish.

---

## Part 3: Setting Up Your Project

Now we will create a folder for your work and set up a "Virtual Environment". 

**What is a Virtual Environment?**  
Think of it like a separate "sandbox" or "container" specifically for this project. It keeps the tools and libraries for this class separate from everything else on your computer, ensuring that updates in one project don't accidentally break another.

### Step 1: Open the Folder in Cursor
Since you have already downloaded the course materials:
1.  Open **Cursor**.
2.  Click on **File** in the top menu bar, then select **Open Folder...**
3.  Navigate to the folder you downloaded and select **AI_With_Python_PartA**.
4.  Click **Open**.

### Step 3: Open the Terminal
We need to type a few commands. We will use the "Terminal" built into Cursor.
*Concept: The **Terminal** (or Command Line) lets you control your computer using text commands instead of a mouse. It gives you direct access to system tools that aren't always available via buttons.*

1.  In Cursor, look at the top menu. Click **Terminal** -> **New Terminal**.
2.  A panel will appear at the bottom of the screen. This is where you type commands.

### Step 4: Create the Virtual Environment
Type the following command into the terminal and press **Enter**:

**For Mac:**
```bash
python3 -m venv venv
```

**For Windows:**
```bash
python -m venv venv
```

*Explanation: This command asks Python to run its "venv" tool to create a new, empty environment inside a folder named `venv`.*

*Note: You might not see anything happen immediately. If a new folder named `venv` appears in your file list on the left, it worked!*

### Step 5: Activate the Virtual Environment
Now we need to "turn on" the environment. Type the command for your system and press **Enter**:

**For Mac:**
```bash
source venv/bin/activate
```

**For Windows:**
```bash
.\venv\Scripts\activate
```

*Explanation: "Activating" tells your terminal to stop looking at the global system for tools and instead look strictly inside your new `venv` folder. This ensures you are using the correct, isolated versions of your tools.*

**How do I know it worked?**  
You should see `(venv)` appear at the very beginning of your command line in the terminal. It will look something like:
`(venv) user@computer AI_With_Python_PartA %`

---

## Part 4: Verify Everything

Let's verify that Python is ready to go inside your virtual environment.

1.  In the same terminal (where you see `(venv)`), type:
    ```bash
    python --version
    ```
    (or `python3 --version` on Mac if the above doesn't work)

2.  You should see a response like `Python 3.12.0` (the numbers might be slightly different).

---

## Part 5: Run a Simple Exercise

Let's write and run your first piece of code to make sure everything is working perfectly.

### Step 1: Create a File
1.  In Cursor, look at the **Explorer** (the file list on the left).
2.  Right-click in the empty space (or click the "New File" icon) and create a new file named `hello.py`.
3.  Copy and paste the following code into that file:

```python
print("Hello, OIT 245!")
print("My Python environment is set up and ready for AI.")
```

4.  Save the file (Cmd+S on Mac, Ctrl+S on Windows).

### Step 2: Run the Code
1.  Go back to your **Terminal** at the bottom (make sure you still see `(venv)`).
2.  Type the following command and press **Enter**:

```bash
python hello.py
```

### Step 3: Success!
If you see the message:
> Hello, OIT 245!
> My Python environment is set up and ready for AI.

Then you are completely done! You have successfully set up your professional Python development environment.

---

## Part 6: Mini-Case: Fitbit Analysis

Now that we have our environment set up, let's solve a real business case using Python!

### Step 1: Understand the Logic
1.  In the **Explorer** on the left, find and click on the file named `fitbit_solver.py`.
2.  Read through the code. You don't need to understand every single character, but look at the comments (lines starting with `#`) and the function names.
3.  **Compare with Excel**:
    *   Think back to Week 1 when you solved this in Excel.
    *   **Logic**: Notice how the `calculate_production_cost` function handles the tiered pricing. In Excel, this likely required a complex nested `IF` formula or a VLOOKUP table. Here, it's a series of logical `if` statements.
    *   **Optimization**: Notice the `for` loops in Part 2 and Part 3. In Excel, you used Data Tables to test different prices. Here, Python "loops" through every possible price to find the best one automatically.

### Step 2: Install Requirements
This script uses powerful tools like `pandas` (for data) and `matplotlib` (for graphs) that don't come with standard Python. We need to install them into our virtual environment.

1.  Make sure your terminal still shows `(venv)` at the start.
2.  Type the following command and press **Enter**:
    ```bash
    pip install -r requirements.txt
    ```
    *Explanation: This tells Python's package installer (`pip`) to read the `requirements.txt` file and install all the tools listed there.*

### Step 3: Run the Analysis
Now, let's run the solver!

1.  In the terminal, type:
    ```bash
    python fitbit_solver.py
    ```
2.  Watch the output! You will see:
    *   The calculated profit for the specific scenario.
    *   The optimal price found by testing hundreds of options.
    *   The global optimal strategy.
3.  **Check the Graphs**: Look in your file explorer. You should see new image files (ending in `.png`) created by the script. Click on them to view the profit curves and heatmaps!

---

## Appendix A: Using VS Code Instead

If you already have **Visual Studio Code (VS Code)** installed or prefer to use it instead of Cursor, you can follow this guide with almost no changes.

**What you need to know:**
1.  **Cursor is built on top of VS Code.** This means 99% of the buttons, menus, and shortcuts are exactly the same.
2.  **Installation:** If you don't have it, download it from [code.visualstudio.com](https://code.visualstudio.com/).
3.  **Opening the Project:** Instead of "Open Cursor", just open "VS Code". The "File -> Open Folder" steps are identical.
4.  **Terminal:** The terminal works exactly the same way.
5.  **Extensions:** You might need to install the "Python" extension by Microsoft if VS Code asks you to (it usually pops up automatically when you open a `.py` file).

**Summary:** Wherever this guide says "Cursor", just read it as "VS Code". Everything else remains the same!

