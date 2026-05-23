"""
BLE receiver for XIAO nRF52840 Sense ankle sensors.
Scans for XIAO-LEFT and XIAO-RIGHT, receives IMU data,
and feeds it into the kick detector.
"""
import asyncio
from bleak import BleakScanner, BleakClient

# Nordic UART Service UUIDs
UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_RX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

left_data  = None
right_data = None

def parse(data: bytearray):
    try:
        x, y, z = data.decode().strip().split(",")
        return float(x), float(y), float(z)
    except:
        return None

def left_handler(sender, data):
    global left_data
    parsed = parse(data)
    if parsed:
        left_data = parsed
        x, y, z = parsed
        print(f"LEFT  X:{x:.2f} Y:{y:.2f} Z:{z:.2f}")

def right_handler(sender, data):
    global right_data
    parsed = parse(data)
    if parsed:
        right_data = parsed
        x, y, z = parsed
        print(f"RIGHT X:{x:.2f} Y:{y:.2f} Z:{z:.2f}")

async def main():
    print("Scanning for XIAO sensors...")
    devices = await BleakScanner.discover(timeout=10.0)
    
    left_device  = None
    right_device = None
    
    for d in devices:
        print(f"Found: {d.name} — {d.address}")
        if d.name == "XIAO-LEFT":
            left_device = d
        elif d.name == "XIAO-RIGHT":
            right_device = d

    if not left_device:
        print("XIAO-LEFT not found!")
        return

    print(f"\nConnecting to XIAO-LEFT: {left_device.address}")
    async with BleakClient(left_device.address) as client:
        await client.start_notify(UART_RX_CHAR_UUID, left_handler)
        print("Connected! Receiving data — shake the sensor...")
        await asyncio.sleep(30)  # receive for 30 seconds

asyncio.run(main())
