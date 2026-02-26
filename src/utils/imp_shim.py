"""
Robust Shim for Python 3.13 'imp' removal.
Specifically targets speexdsp's SWIG generated import logic which relies on the deprecated 'imp' module.
This shim manually locates shared object files (.so) when standard importlib mechanism fail for SWIG extensions.
"""

import sys
import types
import importlib.util
import importlib.machinery
import os
import logging

logger = logging.getLogger(__name__)

def install_shim():
    """Install the imp shim if 'imp' is missing from sys.modules."""
    if "imp" in sys.modules:
        return

    logger.info("Shimming 'imp' module for Python 3.13 compatibility...")
    imp = types.ModuleType("imp")
    
    # Constants SWIG checks
    imp.C_EXTENSION = 3
    imp.PY_SOURCE = 1
    imp.PKG_DIRECTORY = 5
    
    def get_suffixes():
        return [(s, 'rb', imp.C_EXTENSION) for s in importlib.machinery.EXTENSION_SUFFIXES]

    def find_module(name, path=None):
        # 1. Try standard importlib search
        spec = None
        if path is None:
            spec = importlib.util.find_spec(name)
        else:
            for p in path:
                # importlib.util.find_spec doesn't work well with list of paths for extension modules 
                # that are not proper packages. We use FileFinder logic mostly.
                spec = importlib.util.find_spec(name, [p])
                if spec: break
        
        # 2. Manual fallback for shared objects (SWIG often fails step 1 for _module.so)
        if not spec and path:
            logger.debug(f"Shim searching for C-extension '{name}' in: {path}")
            for dirname in path:
                if not os.path.exists(dirname): continue
                
                # Check for any file starting with name and ending with a valid extension suffix
                try:
                    files = os.listdir(dirname)
                    suffixes = importlib.machinery.EXTENSION_SUFFIXES
                    
                    candidate = None
                    for f in files:
                        if f.startswith(name):
                             for suffix in suffixes:
                                 if f == name + suffix or (f.startswith(name + ".") and f.endswith(suffix)):
                                     candidate = os.path.join(dirname, f)
                                     break
                        if candidate: break
                    
                    if candidate:
                        logger.info(f"✅ Shim found extension '{name}' at: {candidate}")
                        return (open(candidate, 'rb'), candidate, ("", "rb", imp.C_EXTENSION))
                        
                except Exception as e:
                     logger.debug(f"Shim error scanning {dirname}: {e}")
        
        # 3. Last ditch: Check for simple .so match
        if not spec and path:
            for dirname in path:
                    simple_path = os.path.join(dirname, name + ".so")
                    if os.path.exists(simple_path):
                        return (open(simple_path, 'rb'), simple_path, ("", "rb", imp.C_EXTENSION))

        if not spec:
            raise ImportError(f"No module named '{name}'")
        
        # Open the file if it's a file-based module (for load_module compatibility)
        file_obj = None
        if spec.origin:
            try:
                if not spec.origin.endswith('.so') and not spec.origin.endswith('.pyd'):
                    file_obj = open(spec.origin, 'rb')
            except: pass
            
        return (file_obj, spec.origin, ("", "rb", imp.C_EXTENSION))

    def load_module(name, file, pathname, description):
        # Reuse existing module if already loaded (common SWIG pattern)
        if name in sys.modules:
            return sys.modules[name]

        spec = importlib.util.spec_from_file_location(name, pathname)
        if not spec:
             spec = importlib.util.find_spec(name)
        if not spec:
            raise ImportError(f"Cannot load module {name}")
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    imp.find_module = find_module
    imp.load_module = load_module
    imp.get_suffixes = get_suffixes
    imp.load_dynamic = load_module # Map load_dynamic to load_module for some SWIG versions
    
    sys.modules["imp"] = imp
