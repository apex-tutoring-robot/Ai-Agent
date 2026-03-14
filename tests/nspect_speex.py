
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpeexInspect")

try:
    from utils import imp_shim
    imp_shim.install_shim()
    import speexdsp
    
    logger.info(f"SpeexDSP found in: {speexdsp.__file__ if hasattr(speexdsp, '__file__') else 'built-in'}")
    
    # List all members of speexdsp
    members = [m for m in dir(speexdsp) if not m.startswith('_')]
    logger.info(f"SpeexDSP members: {members}")
    
    if hasattr(speexdsp, 'EchoCanceller_create'):
        logger.info("EchoCanceller_create factory found.")
        canceller = speexdsp.EchoCanceller_create(320, 2048, 16000)
        
        c_members = [m for m in dir(canceller) if not m.startswith('_')]
        logger.info(f"EchoCanceller instance members: {c_members}")
        
        # Look for methods that might be AEC process
        candidates = ['process', 'run', 'echo_cancellation', 'cancellation', 'aec']
        for c in candidates:
            if hasattr(canceller, c):
                logger.info(f"FOUND Potential AEC Method: {c}")
            elif hasattr(speexdsp, f"EchoCanceller_{c}"):
                logger.info(f"FOUND Potential Factory Method: EchoCanceller_{c}")
                
    elif hasattr(speexdsp, 'EchoCanceller'):
        logger.info("EchoCanceller class found.")
        try:
            canceller = speexdsp.EchoCanceller(320, 2048, 16000)
            c_members = [m for m in dir(canceller) if not m.startswith('_')]
            logger.info(f"EchoCanceller instance members: {c_members}")
        except Exception as e:
            logger.warning(f"Could not instantiate EchoCanceller directly: {e}")

except Exception as e:
    import traceback
    logger.error(f"Inspection failed: {e}")
    logger.error(traceback.format_exc())
