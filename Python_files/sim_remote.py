import asyncio
import threading
import tkinter as tk
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

class DroneTeleopGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PX4 Manual Teleop Control")
        self.root.geometry("350x450")
        self.root.config(bg="#1e1e1e")

        self.drone = System()
        self.loop = asyncio.new_event_loop()
        
        # Flight state variables
        self.is_connected = False
        self.is_armed = False
        self.is_airborne = False
        self.is_offboard = False

        # Current velocity setpoints (m/s and deg/s)
        self.vx = 0.0  # Forward (+) / Backward (-)
        self.vy = 0.0  # Right (+) / Left (-)
        self.vz = 0.0  # Down (+) / Up (-)
        self.v_yaw = 0.0 # Yaw rate

        self.setup_ui()
        
        # Start background async event loop thread for MAVSDK
        self.thread = threading.Thread(target=self.start_async_loop, daemon=True)
        self.thread.start()

        # Bind keyboard events
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)

    def setup_ui(self):
        # Title Label
        title_lbl = tk.Label(self.root, text="PX4 Teleop Station", fg="#00ffcc", bg="#1e1e1e", font=("Helvetica", 14, "bold"))
        title_lbl.pack(pady=10)

        # Status Label
        self.status_lbl = tk.Label(self.root, text="Status: Disconnected", fg="orange", bg="#1e1e1e", font=("Helvetica", 10))
        self.status_lbl.pack(pady=5)

        # Control Buttons Frame
        btn_frame = tk.Frame(self.root, bg="#1e1e1e")
        btn_frame.pack(pady=15)

        self.connect_btn = tk.Button(btn_frame, text="Connect", bg="#333", fg="white", width=12, command=self.run_coro_safe(self.connect_drone))
        self.connect_btn.grid(row=0, column=0, padx=5, pady=5)

        self.arm_btn = tk.Button(btn_frame, text="Arm / Takeoff", bg="#333", fg="white", width=12, command=self.run_coro_safe(self.arm_and_takeoff), state=tk.DISABLED)
        self.arm_btn.grid(row=0, column=1, padx=5, pady=5)

        self.land_btn = tk.Button(btn_frame, text="Land / Disarm", bg="#333", fg="white", width=12, command=self.run_coro_safe(self.land_drone), state=tk.DISABLED)
        self.land_btn.grid(row=1, column=0, padx=5, pady=5)

        self.offboard_btn = tk.Button(btn_frame, text="Enable Offboard", bg="#333", fg="white", width=12, command=self.run_coro_safe(self.toggle_offboard), state=tk.DISABLED)
        self.offboard_btn.grid(row=1, column=1, padx=5, pady=5)

        # Instructions Box
        instructions = (
            "Keyboard Teleop Instructions:\n\n"
            "• [W] / [S] : Move Forward / Backward\n"
            "• [A] / [D] : Move Left / Right\n"
            "• [R] / [F] : Move Up / Down\n"
            "• [Q] / [E] : Yaw Left / Right\n"
            "• [Space]   : Stop / Hover in Place"
        )
        instr_lbl = tk.Label(self.root, text=instructions, justify=tk.LEFT, fg="#ccc", bg="#2a2a2a", font=("Consolas", 9), relief=tk.SUNKEN, padx=10, pady=10)
        instr_lbl.pack(pady=10, fill=tk.BOTH, padx=20)

    def start_async_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run_coro_safe(self, coro_func):
        def wrapper():
            asyncio.run_coroutine_threadsafe(coro_func(), self.loop)
        return wrapper

    async def connect_drone(self):
        self.update_status("Connecting to drone...", "orange")
        await self.drone.connect(system_address="udpin://0.0.0.0:14540")
        
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                self.is_connected = True
                self.update_status("Connected!", "green")
                self.root.after(0, lambda: self.connect_btn.config(state=tk.DISABLED))
                self.root.after(0, lambda: self.arm_btn.config(state=tk.NORMAL))
                break

    async def arm_and_takeoff(self):
        self.update_status("Arming...", "yellow")
        await self.drone.action.arm()
        self.is_armed = True
        
        self.update_status("Taking off...", "yellow")
        await self.drone.action.takeoff()
        self.is_airborne = True
        
        # Give it a moment to clear the ground
        await asyncio.sleep(5)
        self.update_status("Airborne. Ready for control.", "green")
        self.root.after(0, lambda: self.land_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.offboard_btn.config(state=tk.NORMAL))

    async def toggle_offboard(self):
        try:
            # Set initial setpoint to zero before starting offboard mode
            await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            await self.drone.offboard.start()
            self.is_offboard = True
            self.update_status("Offboard Active - Use Keyboard!", "#00ffcc")
            
            # Start velocity ticker loop
            asyncio.create_task(self.velocity_loop())
        except OffboardError as e:
            self.update_status(f"Offboard failed: {e._result.result}", "red")

    async def velocity_loop(self):
        while self.is_offboard:
            try:
                await self.drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(self.vx, self.vy, self.vz, self.v_yaw)
                )
            except Exception:
                pass
            await asyncio.sleep(0.05) # 20Hz update rate

    async def land_drone(self):
        self.is_offboard = False
        self.update_status("Landing...", "orange")
        try:
            await self.drone.offboard.stop()
        except Exception:
            pass
        await self.drone.action.land()
        self.update_status("Landed.", "gray")

    def update_status(self, text, color):
        self.root.after(0, lambda: self.status_lbl.config(text=f"Status: {text}", fg=color))

    def on_key_press(self, event):
        if not self.is_offboard:
            return
        
        speed = 0.5  # m/s
        yaw_speed = 30.0 # deg/s
        
        key = event.keysym.lower()
        if key == 'w': self.vx = speed
        elif key == 's': self.vx = -speed
        elif key == 'a': self.vy = -speed
        elif key == 'd': self.vy = speed
        elif key == 'r': self.vz = -speed  # Up in NED frame
        elif key == 'f': self.vz = speed   # Down in NED frame
        elif key == 'q': self.v_yaw = -yaw_speed
        elif key == 'e': self.v_yaw = yaw_speed
        elif key == 'space':
            self.vx = 0.0
            self.vy = 0.0
            self.vz = 0.0
            self.v_yaw = 0.0

    def on_key_release(self, event):
        if not self.is_offboard:
            return
        
        key = event.keysym.lower()
        if key in ['w', 's']: self.vx = 0.0
        elif key in ['a', 'd']: self.vy = 0.0
        elif key in ['r', 'f']: self.vz = 0.0
        elif key in ['q', 'e']: self.v_yaw = 0.0


if __name__ == "__main__":
    root = tk.Tk()
    app = DroneTeleopGUI(root)
    root.mainloop()