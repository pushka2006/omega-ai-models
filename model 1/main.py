"""Top-level forwarding script for Omega GPT-1.

Allows executing 'python main.py <command>' directly from the 'model 1' folder.
"""

import os
import sys

# Target directory: omega-gpt
omega_dir = os.path.join(os.path.dirname(__file__), "omega-gpt")
if os.path.exists(omega_dir):
    os.chdir(omega_dir)
    sys.path.insert(0, omega_dir)
    import main
    if __name__ == "__main__":
        main.main()
else:
    print(f"Error: omega-gpt directory not found at {omega_dir}")

