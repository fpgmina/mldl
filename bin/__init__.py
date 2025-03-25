import sys
import os

# Add the root project directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Now the dataset, training folders etc will be accessible to any script in the 'bin' folder.
