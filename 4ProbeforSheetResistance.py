import pyvisa
import numpy as np
import matplotlib.pyplot as plt
from pymeasure.instruments.keithley import Keithley2400
from openpyxl import Workbook

keithley = Keithley2400("ASRL6::INSTR")
sample_no = '1NN '
current_start = 0.00000000005  # Starting current in amperes
current_end = 0.0055  # Ending current in amperes
current_step = 0.00005  # Step size in amperes

keithley.write(":SYST:RSEN ON")

# Set the compliance voltage
keithley.write("SENS:VOLT:PROT 10")  # Compliance voltage in volts

# Enable the source
keithley.enable_source()
keithley.apply_current()
keithley.measure_voltage()

wb = Workbook()
ws = wb.active
ws.title = "I-V Data"

ws.append(["Current (A)", "Voltage (V)", "Sheet Resistance (Ohms)"])

current_values = []
voltage_values = []
sheet_resistance_values = []

for current_value in np.arange(current_start, current_end + current_step, current_step):
    keithley.source_current = current_value
    measured_voltage = keithley.voltage
    keithley.write("SENS:VOLT:PROT 10")  # Ensure compliance voltage is set
    sheet_resistance = measured_voltage / current_value * 4.53
    ws.append([current_value, measured_voltage, sheet_resistance])
    current_values.append(current_value)
    voltage_values.append(measured_voltage)
    sheet_resistance_values.append(sheet_resistance)
    print(f"Current: {current_value} A, Voltage: {measured_voltage} V, Sheet Resistance: {sheet_resistance} Ohms")

keithley.shutdown()

excel_filename = sample_no + '.xlsx'
wb.save(excel_filename)

plt.figure(figsize=(10, 5))
plt.plot(current_values, voltage_values, 'o-', label='Voltage')
plt.xlabel('Current (A)')
plt.ylabel('Voltage (V)')
plt.title('I-V Characterization (4-Probe)')
plt.grid(True)
plt.legend()
plt.savefig(sample_no + 'IV_characterization.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(current_values, sheet_resistance_values, 'o-', label='Sheet Resistance')
plt.xlabel('Current (A)')
plt.ylabel('Sheet Resistance (Ohms)')
plt.title('Sheet Resistance vs Current')
plt.grid(True)
plt.legend()
plt.savefig(sample_no + 'Sheet_resistance_vs_current_plot.png')
plt.show()

print(f"Data saved to {excel_filename}")
print(f"Plots saved as 'IV_characterization_4_probe_range_plot.png' and 'Sheet_resistance_vs_current_plot.png'")
