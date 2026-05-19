import os
import glob
import inspect
import importlib.util
from logger import log_info

LOADED_SCRIPTS = []

def load_all_scripts():
    global LOADED_SCRIPTS
    LOADED_SCRIPTS = []
    
    # Scripts folder is in the same directory as this module
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
    if not os.path.exists(scripts_dir):
        os.makedirs(scripts_dir, exist_ok=True)
        log_info(f"Created scripts directory: {scripts_dir}")
        return

    script_files = glob.glob(os.path.join(scripts_dir, "*.py"))
    for script_file in script_files:
        if os.path.basename(script_file) == "__init__.py":
            continue
        try:
            name = os.path.splitext(os.path.basename(script_file))[0]
            spec = importlib.util.spec_from_file_location(name, script_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                LOADED_SCRIPTS.append(module)
                log_info(f"Loaded script: {script_file}")
        except Exception as e:
            log_info(f"Error loading script {script_file}: {e}")

async def run_setup(connection):
    for script in LOADED_SCRIPTS:
        if hasattr(script, "setup"):
            try:
                log_info(f"Running setup in script: {script.__name__}")
                if inspect.iscoroutinefunction(script.setup):
                    await script.setup(connection)
                else:
                    script.setup(connection)
            except Exception as e:
                log_info(f"Error running setup in script {script.__name__}: {e}")

async def run_teardown(connection):
    for script in LOADED_SCRIPTS:
        if hasattr(script, "teardown"):
            try:
                log_info(f"Running teardown in script: {script.__name__}")
                if inspect.iscoroutinefunction(script.teardown):
                    await script.teardown(connection)
                else:
                    script.teardown(connection)
            except Exception as e:
                log_info(f"Error running teardown in script {script.__name__}: {e}")
