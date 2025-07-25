# Desired Python version
python_version = '3.12.3'

# Write to runtime.txt
with open("runtime.txt", "w") as f:
    f.write(f"python-{python_version}")

print(f"✅ runtime.txt created successfully with Python {python_version}")
