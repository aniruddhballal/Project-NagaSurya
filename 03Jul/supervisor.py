import subprocess
import time

python_interpreter = r"C:\Users\aniru\pyproj\my_env1\Scripts\python.exe"

while True:
    try:
        result = subprocess.run([python_interpreter, "shortmainscript.py"], check=True)
        if result.returncode == 0:
            print("Iteration completed successfully.")
            time.sleep(1)  # Adjust as needed
    except subprocess.CalledProcessError as e:
        print(f"Iteration failed with error: {e}. Restarting after a brief pause...")
        time.sleep(1)  # Adjust as needed
