import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter



# file = open("./fullrun.txt")
# lines = file.readlines()
# dfRows = []

# for line in lines:
#     print(line)
#     fileName: str
#     logStr: str = None
#     kernelNum: int
#     numPixels: int
#     time: float
#     log: bool

#     splitLine = line.split()


#     if line.startswith("\t\tCuda completion time:"):
#         time = float(splitLine[3])
#         logStr = "Cuda"

#     elif line.startswith("\t\tLinear completion time:"):
#         time = float(splitLine[3])
#         logStr = "Linear"

#     elif line.startswith("\tKernel:"):
#         kernelNum = int(splitLine[1])

#     elif line.startswith("Filename:"):
#         fileName = splitLine[1][2:]

#     elif line.startswith("Dimensions:"):
#         dims: int = splitLine[1].split("x")
#         numPixels = int(dims[0]) * int(dims[1])

#     if logStr:
#         dfRows.append([fileName, numPixels, kernelNum, logStr, time])

df = pd.read_csv("./secondRun.csv")
# df2 = pd.read_csv("./secondRun.csv")
# df = pd.concat([df1, df2], ignore_index=True)

# df = pd.DataFrame(dfRows, columns=["Filename", "Pixels", "Kernel", "Type", "Time"])


typeColors = {'Cuda': 'darkgreen', 'Linear': 'blue'}
fig, axes = plt.subplots(3, 1, figsize=(6, 18), subplot_kw={"projection": "3d"})

# Plot 1: Cuda only
ax = axes[0]
for kernelVal, kernelGroup in df[df['Type'] == 'Cuda'].groupby("Kernel"):
    kernelGroup = kernelGroup.sort_values("Pixels")
    x = kernelGroup["Pixels"]
    y = kernelGroup["Kernel"]
    z = kernelGroup["Time"]
    
    # Scatter and lines for Cuda type
    ax.scatter(x, y, z, c=[typeColors['Cuda']]*len(x), label="Cuda")
    ax.plot(x, y, z, color=typeColors['Cuda'])
    # ax.plot(x, y, 0, color='grey')

odd_ticks = np.arange(1, 17, 2)  # odd integers from 1 to 15
ax.set_yticks(odd_ticks)
ax.ticklabel_format(style="plain")


ax.set_xlabel('Pixels')
ax.set_ylabel('Kernel Value')
ax.set_zlabel('Time (s)')
ax.set_title('Convolution: Cuda Only')

# Plot 2: Linear only
ax = axes[1]
for kernelVal, kernelGroup in df[df['Type'] == 'Linear'].groupby("Kernel"):
    kernelGroup = kernelGroup.sort_values("Pixels")
    x = kernelGroup["Pixels"]
    y = kernelGroup["Kernel"]
    z = kernelGroup["Time"]
    
    # Scatter and lines for Linear type
    ax.scatter(x, y, z, c=[typeColors['Linear']]*len(x), label="Linear")
    ax.plot(x, y, z, color=typeColors['Linear'])
    # ax.plot(x, y, 0, color='grey')

odd_ticks = np.arange(1, 17, 2)  # odd integers from 1 to 15
ax.set_yticks(odd_ticks)
ax.ticklabel_format(style="plain")

ax.set_xlabel('Pixels')
ax.set_ylabel('Kernel Value')
ax.set_zlabel('Time (s)')
ax.set_title('Convolution: Linear Only')

# Plot 3: Both types overlaid
ax = axes[2]
for kernelVal, kernelGroup in df.groupby("Kernel"):
    for typeVal, typeGroup in kernelGroup.groupby("Type"):
        typeGroup = typeGroup.sort_values("Pixels")
        x = typeGroup["Pixels"]
        y = typeGroup["Kernel"]
        z = typeGroup["Time"]

        # Scatter and lines for both types
        ax.scatter(x, y, z, c=[typeColors[typeVal]]*len(x), label=typeVal)
        ax.plot(x, y, z, color=typeColors[typeVal])
        # ax.plot(x, y, 0, color='grey')

odd_ticks = np.arange(1, 17, 2)  # odd integers from 1 to 15
ax.set_yticks(odd_ticks)
ax.ticklabel_format(style="plain")

ax.set_xlabel('Pixels')
ax.set_ylabel('Kernel Value')
ax.set_zlabel('Time (s)')
ax.set_title('Cuda vs Sequential')

# for kernelVal, kernelGroup in df.groupby("Kernel"):
#     for typeVal, typeGroup in kernelGroup.groupby("Type"):
#         typeGroup = typeGroup[typeGroup["Type"] == "Cuda"]
#         typeGroup = typeGroup.sort_values("Pixels")
#         x = typeGroup["Pixels"]
#         y = typeGroup["Kernel"]
#         z = typeGroup["Time"]

#         # Plotting the 3D scatter plot
#         ax.scatter(x, y, z, c=[typeColors[typeVal]]*len(x), cmap='viridis')
#         ax.plot(x, y, z, color=typeColors[typeVal])
# Show the plot
plt.tight_layout(pad=1.5)
plt.show()

df.to_csv("fullrun.csv")