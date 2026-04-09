import importlib
import pkgutil
import time

from batchgenerators.utilities.file_and_folder_operations import *


def recursive_find_python_class(folder: str, class_name: str, current_module: str):
    tr = None
    # Retry non-package imports to survive transient import-time errors.
    for attempt in range(3):
        for importer, modname, ispkg in pkgutil.iter_modules([folder]):
            # print(modname, ispkg)
            if not ispkg:
                try:
                    importlib.invalidate_caches()
                    m = importlib.import_module(current_module + "." + modname)
                except Exception:
                    # Some trainer modules depend on optional packages. Skip modules
                    # that cannot be imported so available trainers can still run.
                    continue
                if hasattr(m, class_name):
                    tr = getattr(m, class_name)
                    break
        if tr is not None:
            break
        if attempt < 2:
            time.sleep(0.2 * (attempt + 1))

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules([folder]):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(join(folder, modname), class_name, current_module=next_current_module)
            if tr is not None:
                break
    return tr
