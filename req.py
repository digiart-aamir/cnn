import subprocess

# Run pip freeze and save output to requirements.txt
with open("requirements.txt", "w") as f:
    subprocess.run(["pip", "freeze"], stdout=f)

print("✅ requirements.txt generated successfully.")
