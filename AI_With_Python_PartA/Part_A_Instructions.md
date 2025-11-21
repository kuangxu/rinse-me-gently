# AI with Python, Part A: Python & Cursor IDE Setup Guide

This lecture is the first of two parts where we will get to experience the state-of-the-art workflow with AI and Python.

In today's lecture, our goal is to:
1.  Set up Python on your computer.
2.  Understand how to use a **Virtual Environment**.
3.  Run simple exercises in Python. 

We will be using **Cursor**, a modern code editor that integrates AI directly into the coding process. That being said, this can be done with any Python code editor such as **VS Code** (See Appendix). 

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

**Shared Environment for Both Parts:**  
We will create the virtual environment at the root level of the project (not inside `AI_With_Python_PartA`). This way, you can use the same environment for both Part A and Part B, which is more efficient and avoids duplicate installations.

### Step 1: Open the Folder in Cursor
Since you have already downloaded the course materials:
1.  Open **Cursor**.
2.  Click on **File** in the top menu bar, then select **Open Folder...**
3.  Navigate to the folder you downloaded and select the **root folder** (the main folder containing both `AI_With_Python_PartA` and `AI_With_Python_PartB`).
   - *Note: We open the root folder so we can create a single virtual environment that works for both Part A and Part B.*
4.  Click **Open**.

### Step 2: Open the Terminal
We need to type a few commands. We will use the "Terminal" built into Cursor.
*Concept: The **Terminal** (or Command Line) lets you control your computer using text commands instead of a mouse. It gives you direct access to system tools that aren't always available via buttons.*

1.  In Cursor, look at the top menu. Click **Terminal** -> **New Terminal**.
2.  A panel will appear at the bottom of the screen. This is where you type commands.

### Step 3: Navigate to the Project Root (if needed)
Make sure you're in the root folder of the project (the one containing both `AI_With_Python_PartA` and `AI_With_Python_PartB`). If you just opened the folder in Cursor, you should already be there. You can verify by checking that your terminal prompt shows the project folder name, or type:
```bash
pwd
```
(This shows your current directory - "print working directory")

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

*Explanation: This command asks Python to run its "venv" tool to create a new, empty environment inside a folder named `venv` at the root of the project. This single virtual environment will be used for both Part A and Part B.*

*Note: You might not see anything happen immediately. If a new folder named `venv` appears in your file list on the left (at the root level, not inside AI_With_Python_PartA), it worked!*

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

*Note: If you encounter an error message like "cannot be loaded because running script is disabled" on Windows, see **Appendix C** below for troubleshooting solutions.*

*Explanation: "Activating" tells your terminal to stop looking at the global system for tools and instead look strictly inside your new `venv` folder. This ensures you are using the correct, isolated versions of your tools.*

**How do I know it worked?**  
You should see `(venv)` appear at the very beginning of your command line in the terminal. It will look something like:
`(venv) user@computer rinse_me_gently-main %`

*Note: The folder name in your prompt might be slightly different, but you should see `(venv)` at the start.*

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

### Step 1: Open the Hello World File
1.  In Cursor, look at the **Explorer** (the file list on the left).
2.  Navigate to the `AI_With_Python_PartA/Python_Scripts` folder.
3.  Click on the file named `hello.py` to open it.
4.  Take a look at the code inside - it's a simple script that prints a greeting message.

### Step 2: Run the Code
1.  Go back to your **Terminal** at the bottom (make sure you still see `(venv)`).
2.  Navigate to the Python_Scripts folder (if not already there):
    ```bash
    cd AI_With_Python_PartA/Python_Scripts
    ```
3.  Type the following command and press **Enter**:

```bash
python hello.py
```

### Step 3: Success!
If you see the message:
> Hello, World!
> My Python environment is set up and ready for AI.

Then you are completely done! You have successfully set up your professional Python development environment.

---

## Part 6: Mini-Case: Fitbit Analysis

Now that we have our environment set up, let's solve the Fitbit case, which you've already seen before in Basic Modeling, but this time with Python. For simplicity, we have already made a solution script. Your goal now is to understand as much of the logic as possible, and be able to execute the code and understand the outputs. 

### Step 1: Understand the Logic
1.  In the **Explorer** on the left, navigate to the `Python_Scripts` folder and find and click on the file named `fitbit_solver.py`. This is the Python solution for the Fitbit case. You can find the raw text of the case in the `Case_Text` folder. 
2.  Read through the code. You don't need to understand every single character, but look at the comments (lines starting with `#`) and the function names.
3.  **Compare with Excel**:
    *   Think back to Basic Modeling when you solved this in Excel.
    *   **Logic**: Notice how the `calculate_production_cost` function handles the tiered pricing. We used an IF logic in the Excel solution. Here, again, it's a series of logical `if` statements.
    *   **Optimization**: Notice the `for` loops in Part 2 and Part 3. In Excel, you used Data Tables to test different prices. Here, Python "loops" through every possible price to find the best one automatically.

### Step 2: Install Requirements
This script uses powerful tools like `pandas` (for data) and `matplotlib` (for graphs) that don't come with standard Python. We need to install them into our virtual environment.

1.  Make sure your terminal still shows `(venv)` at the start.
2.  Navigate to the Part A folder (if not already there):
    ```bash
    cd AI_With_Python_PartA
    ```
3.  Type the following command and press **Enter**:
    ```bash
    pip install -r requirements.txt
    ```
    *Explanation: This tells Python's package installer (`pip`) to read the `requirements.txt` file in the Part A folder and install all the tools listed there.*

### Step 3: Run the Analysis
Now, let's run the solver!

1.  Navigate to the Python_Scripts folder:
    ```bash
    cd Python_Scripts
    ```
2.  In the terminal, type:
    ```bash
    python fitbit_solver.py
    ```
3.  Watch the output! You will see:
    *   The calculated profit for the specific scenario.
    *   The optimal price found by testing hundreds of options.
    *   The global optimal strategy.
4.  **Check the Graphs**: Look in your file explorer. You should see new image files (ending in `.png`) created by the script. Click on them to view the profit curves and heatmaps!

---

## Part 7: Using AI to Solve the JetBlue Case (Optional)

This part is optional and will give you hands-on experience using an AI coding agent, such as Cursor's built-in AI assistant. You'll learn how to effectively communicate with AI to generate Python code for a business problem.

### Step 1: Open Cursor's AI Agent Chat

1.  In Cursor, look for the **AI Chat** or **Agent** feature. This is typically accessible via:
    *   A chat icon in the sidebar
    *   A keyboard shortcut (often `Cmd+L` on Mac or `Ctrl+L` on Windows)
    *   The menu: **View** -> **Chat** or **AI** -> **Chat**
2.  A chat window will appear where you can interact with the AI assistant.
3.  **If you've obtained a student discounted membership with Cursor, make sure you sign in using your student email.**

### Step 2: Prepare the Case Text

1.  Navigate to the `Case_Text` folder in the Explorer.
2.  Open `Jetblue_raw.md` to familiarize yourself with the case.
3.  You'll need to reference this file when creating your prompt for the AI.

### Step 3: Create Your Prompt

**Your Challenge:** Try to create a prompt yourself that asks the AI to:
- Read the JetBlue case text from `Case_Text/Jetblue_raw.md`
- Create a Python script that solves all four questions in the case
- Include terminal output (print statements) showing results
- Generate and save visualization graphs as PNG files when appropriate

**Tips for Writing Effective Prompts:**
- Be specific about what you want (e.g., "create a Python script", "solve all 4 questions")
- Reference the case file location
- Mention the style you want (similar to `fitbit_solver.py`)
- Ask for clear output (terminal prints and saved images)
- Specify the file name for the output script (e.g., `jetblue_solver.py`)

### Step 4: Iterate and Refine

1.  **First Attempt**: Try your initial prompt and see what the AI generates.
2.  **Review the Output**: Check if the script addresses all questions from the case and produces the expected outputs.
3.  **Follow-up Questions**: If the initial attempt doesn't fully solve the problem, ask follow-up questions to refine the solution.

### Step 5: Save and Run Your Solution

1.  Once you're satisfied with the generated script, save it as `jetblue_solver.py` in the `Python_Scripts` folder.
2.  Make sure your virtual environment is activated (you should see `(venv)` in your terminal).
3.  Navigate to the `Python_Scripts` folder:
    ```bash
    cd AI_With_Python_PartA/Python_Scripts
    ```
4.  Run the script:
    ```bash
    python jetblue_solver.py
    ```
5.  Review the terminal output and any generated PNG files.

### Step 6: Reflection

After completing this exercise, think about:
- What made your prompt effective (or ineffective)?
- How many iterations did it take to get a working solution?
- What would you do differently next time?
- How does this compare to writing code manually?

**Hint:** If you really struggle with creating an effective prompt, you can look at **Appendix B** below for an example prompt that successfully generates a JetBlue solver. However, we encourage you to try creating your own first!

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

---

## Appendix B: Example AI Prompt for JetBlue Case

If you struggled with creating an effective prompt in Part 7, here is an example prompt that successfully generates a complete Python solution for the JetBlue case.

**Example Prompt:**

```
Create a Python script `jetblue_solver.py` that reads `AI_With_Python_PartA/Case_Text/Jetblue_raw.md` and solves all questions in that document. Print results to the terminal and save visualizations as PNG files when appropriate.
```

---

## Appendix C: Windows Virtual Environment Activation Troubleshooting

If you encounter an error message like "cannot be loaded because running script is disabled" when trying to activate the virtual environment on Windows, this is due to PowerShell's execution policy security feature. Here are three solutions:

**Solution 1: Use Command Prompt instead of PowerShell (Easiest)**
1. In Cursor, click the dropdown arrow next to the `+` button in the terminal panel (or right-click the terminal tab).
2. Select **Command Prompt** (or **cmd**) instead of PowerShell.
3. Then try the activation command again:
   ```bash
   .\venv\Scripts\activate
   ```

**Solution 2: Change PowerShell Execution Policy (Temporary)**
If you need to use PowerShell, run this command first (you may need to run Cursor as Administrator):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```
Then try activating again:
```bash
.\venv\Scripts\activate
```

**Solution 3: Use the batch file directly**
You can also try:
```bash
venv\Scripts\activate.bat
```


