# Motion Sensor Power Saving Program
    # Below is code to monitor battery levels and enter power-saving deep sleep mode.

# Hardware
    # combined microcontroller MPU: Seeed Studio XIAO nRF52840 Sense
    # battery: Rechargable Qimoo 401230 3.7 V 100mAh LiPo Battery 

# import libraries
import analogio 
import board 
import digitalio 
import time 
import alarm 

# CHECK BATTERY LIFE  -----------------------------------------------------

# use the READ_BATT_ENABLE pin to control the circuit connecting the battery and microcontroller 
read_batt_switch = digitalio.DigitalInOut(board.READ_BATT_ENABLE)
read_batt_switch.direction = digitalio.Direction.OUTPUT

# close circuit (False = ON, True = OFF by default)
read_batt_switch.value = False 

# connect to battery pin
battery_reading_raw_number = analogio.AnalogIn(board.VBATT)

while True:
    # convert analog reading to volts
    voltage = (battery_reading_raw_number.value / 65536) * 3.3 * 3
    # convert voltage to battery percentage
    percent = ((voltage - 3.3) / (4.2 - 3.3)) * 100
    percent = max(0, min(100, percent))
    print(f"Battery: {voltage:.2f}V  {percent:.0f}%")
    
    # ENTER SLEEP MODE --------------------------------------------------------
    # deep sleep mode disables LED indicator on board and pauses all programs to save power 

    # enter deep sleep when battery life is < 20%
    if percent < 20: 
        # enter deep sleep mode
        alarm.exit_and_deep_sleep_until_alarms()

    # check if battery is near full
    elif voltage > 4.15:
        print("WARNING: Battery near full, unplug soon")
        alarm.exit_and_deep_sleep_until_alarms()

    # wait 5 seconds before repeating loop
    time.sleep(5)

# TODOS:
    # ENHANCE POWER SAVING SOFTWARE 
    # detect when MPU is not in use and enter deep sleep mode
        # use ~1-1.5 g (we picked 2g as selective kick power threshold)
    # notify user that battery is 20%
        # write code to calculate remaining battery life
        # transmit via bluetooth
    # SAFETY / HARDWARE
    # medical grade battery 
    # review safety features of the microcontroller board