#!/usr/bin/env python3
"""
SITL Pipeline Proof Script (PC Testing)
Replaces physical RC switch with a PC keyboard trigger.
"""

asyncio = __import__('asyncio')
from mavsdk import System
from mavsdk.telemetry import FlightMode

# ---- SIMULATION CONFIG --------------------------------------------
# Standard MAVSDK port for ArduPilot SITL / QGC integration
SIM_ADDRESS = "udp://:14540"
THROTTLE_PERCENT = 10
SPIN_SECONDS = 2
PAUSE_BETWEEN = 1.5
# ---------------------------------------------------------------------

async def run():
    drone = System()
    print(f"-- Connecting to SITL instance at {SIM_ADDRESS}...")
    await drone.connect(system_address=SIM_ADDRESS)

    print("-- Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to SITL successfully!")
            break

    print("\n" + "=" * 60)
    print(f">>> SIMULATION READY.")
    print(f">>> Press [ENTER] in this terminal to simulate triggering the AUTO sequence.")
    print("=" * 60 + "\n")

    # Block execution until user presses Enter on the PC keyboard
    await asyncio.get_event_loop().run_in_executor(None, input, "Press ENTER to trigger sequence...")

    print("!" * 60)
    print("!! PC TRIGGER ACTIVATED -- SIMULATING CONTROL SEQUENCE !!")
    print("!" * 60)

    try:
        print("-- Attempting to arm drone in simulation...")
        # In SITL, standard arming works cleanly without force_arm
        await drone.action.arm()
        print("-- DRONE ARMED (SITL).")

        # Simulate checking telemetry or performing a state change
        print("-- Verifying telemetry stream (getting current flight mode)...")
        async for mode in drone.telemetry.flight_mode():
            print(f"-- Current SITL Flight Mode: {mode}")
            break

        print("\n-- Simulating task execution window...")
        await asyncio.sleep(3.0)

        print("\n-- Disarming simulation...")
        await drone.action.disarm()
        print("-- DISARMED SAFELY.")

    except Exception as e:
        print(f"-- ERROR during simulation test: {e}")
        try:
            await drone.action.disarm()
        except Exception:
            pass

    print("-- SITL PIPELINE PROOF COMPLETE.")

if __name__ == "__main__":
    asyncio.run(run())