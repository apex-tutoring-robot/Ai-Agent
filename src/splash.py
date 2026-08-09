#!/usr/bin/env python3
import os
import time
import gi
import signal
import sys
import subprocess
import threading

REPO_PATH = '/home/tutor/Ai-Agent'
PID_FILE = "splash.pid"

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib

def signal_handler(sig, frame):
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    Gtk.main_quit()
    sys.exit(0)

class SplashScreen(Gtk.Window):
    def __init__(self):
        with open(PID_FILE, "w") as f:
             f.write(str(os.getpid()))

        super().__init__(title="Loading")
        self.fullscreen()
        self.set_decorated(False)
        
        # 1. CSS for Black Background and White Text
        screen = Gdk.Screen.get_default()
        provider = Gtk.CssProvider()
        style_context = Gtk.StyleContext()
        style_context.add_provider_for_screen(screen, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        
        css = b"""
        window { background-color: black; }
        label { color: white; font-size: 48px; font-weight: bold; }
        """
        provider.load_from_data(css)

        # 2. Overlay Container
        overlay = Gtk.Overlay()
        self.add(overlay)

        # 3. Load Background Image Safely
        img_path = "/home/tutor/Downloads/JARVIS.png" 
        try:
            pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(
                filename=img_path,
                width=1024,
                height=600,
                preserve_aspect_ratio=True
            )
            background_image = Gtk.Image.new_from_pixbuf(pixbuf)
            overlay.add(background_image)
        except Exception as e:
            print(f"Image load failed: {e}")

        # 4. Foreground UI
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=30)
        vbox.set_valign(Gtk.Align.CENTER)
        vbox.set_halign(Gtk.Align.CENTER)

        self.spinner = Gtk.Spinner()
        self.spinner.set_size_request(100, 100)
        self.spinner.start()

        # Set an initial label
        self.label = Gtk.Label(label="Initializing System...")
        
        vbox.pack_start(self.spinner, False, False, 0)
        vbox.pack_start(self.label, False, False, 0)

        overlay.add_overlay(vbox)
        self.show_all()

        # 5. Start the background tasks in a separate thread
        update_thread = threading.Thread(target=self.run_background_tasks)
        update_thread.daemon = True # Ensures thread dies if main program exits
        update_thread.start()

    # Safely updates the label from the background thread
    def update_label(self, text):
        self.label.set_text(text)
        return False # Required so GLib.idle_add only runs this once per call

    # The background worker
    def run_background_tasks(self):
        env = os.environ.copy()
        env["DEBIAN_FRONTEND"] = "noninteractive"
    
        try:
            GLib.idle_add(self.update_label, "Checking for system updates...")
            subprocess.run(["sudo", "apt-get", "update", "-qq"], check=True, env=env)
            
            sim_result = subprocess.run(["sudo", "apt-get", "-s", "upgrade"], capture_output=True, text=True, env=env)
            needs_update = any(line.startswith("Inst") for line in sim_result.stdout.splitlines())
            
            if needs_update:
                GLib.idle_add(self.update_label, "Installing system updates...")
                subprocess.run(["sudo", "apt-get", "full-upgrade", "-y"], check=True, env=env)
                subprocess.run(["sudo", "apt-get", "dist-upgrade", "-y"], check=True, env=env)
                subprocess.run(["sudo", "apt-get", "auto-remove", "-y"], check=True, env=env)
                GLib.idle_add(self.update_label, "Rebooting to apply system updates...")
                time.sleep(1)
                subprocess.run(["sudo", "reboot"], check=True)
                return # Stop execution here as the system is going down
            else:
                GLib.idle_add(self.update_label, "No system updates found. Checking AI...")
            
            subprocess.run(['git', 'fetch'], cwd=REPO_PATH, check=True)
            status = subprocess.run(
                   ['git', 'status', '-uno'], 
                   cwd=REPO_PATH, 
                   capture_output=True, 
                   text=True
            )
            
            if "Your branch is behind" in status.stdout:
                GLib.idle_add(self.update_label, "Installing AI updates...")
                subprocess.run(['git', 'pull'], cwd=REPO_PATH, check=True)
                GLib.idle_add(self.update_label, "Updates complete. Booting Jarvis...")
            else:
                GLib.idle_add(self.update_label, "No AI updates found. Booting Jarvis...")
                
        except subprocess.CalledProcessError as e:
            # Catch errors so the UI doesn't crash silently
            GLib.idle_add(self.update_label, "An error occurred during boot.")
            print(f"Subprocess failed: {e}")

# Register the signal
signal.signal(signal.SIGTERM, signal_handler)

win = SplashScreen()
win.connect("destroy", Gtk.main_quit)
Gtk.main()

