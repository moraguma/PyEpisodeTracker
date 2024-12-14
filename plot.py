import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def make_lighter(color, saturation_factor=0.2, value_factor=1.2):
    h, s, v = mcolors.rgb_to_hsv(mcolors.to_rgb(color))
    return mcolors.hsv_to_rgb([h, s * saturation_factor, min(v * value_factor, 1.0)])

# Function to compute the running average
def running_average(data, window_size):
    return data.rolling(window=window_size).mean()

# Read CSV files into pandas DataFrames
csv_file1 = 'graphs/3HiddenLayers/data/PongNoFrameskip-v4__PaperDQNETFixedLayeredPong__1__1729564001.csv'
csv_file1_1 = 'graphs/5HiddenLayers/data/PongNoFrameskip-v4__NewPaperDQNETDeeperLayeredPong__1__1729611633.csv'
csv_file2 = 'graphs/3HiddenLayers/data/PongNoFrameskip-v4__PaperDQNNoConvLayeredPong__1__1729564104.csv'
csv_file3 = 'graphs/data/PongNoFrameskip-v4__PaperDQNPong__1__1729454649.csv'

sps_data1 = 'graphs/3HiddenLayers/sps/PongNoFrameskip-v4__PaperDQNETFixedLayeredPong__1__1729564001.csv'
sps_data1_1 = 'graphs/5HiddenLayers/sps/PongNoFrameskip-v4__NewPaperDQNETDeeperLayeredPong__1__1729611633.csv'
sps_data2 = 'graphs/3HiddenLayers/sps/PongNoFrameskip-v4__PaperDQNNoConvLayeredPong__1__1729564104.csv'
sps_data3 = 'graphs/sps/PongNoFrameskip-v4__PaperDQNPong__1__1729454649.csv'

df1 = pd.read_csv(csv_file1)
df1_1 = pd.read_csv(csv_file1_1)
df2 = pd.read_csv(csv_file2)
df3 = pd.read_csv(csv_file3)

spsdf1 = pd.read_csv(sps_data1)
spsdf1_1 = pd.read_csv(sps_data1_1)
spsdf2 = pd.read_csv(sps_data2)
spsdf3 = pd.read_csv(sps_data3)

# Plot raw data from all 3 CSV files in the same graph
fig, axes = plt.subplots(ncols=2)
fig.set_figwidth(10)
fig.set_figheight(4)
axes[0].plot(df1['Step'], df1['Value'], label='Episode Tracker raw', color=make_lighter('orange'))
axes[0].plot(df1_1['Step'], df1_1['Value'], label='Episode Tracker 5 raw', color=make_lighter('red'))
axes[0].plot(df2['Step'], df2['Value'], label='No Convolution raw', color=make_lighter('gray'))
axes[0].plot(df3['Step'], df3['Value'], label='DQN raw', color=make_lighter('blue'))
axes[0].set_ylabel('Episodic return')
axes[0].set_xlabel('Step')

# Plot running averages of all 3 CSV files
window_size = 50  # Adjust as needed
mark_every = 40

a1 = axes[0].plot(df1['Step'], running_average(df1['Value'], window_size), label='Episode Tracker (3 layers)', color='orange', marker='<', markevery=mark_every)[0]
a1_1 = axes[0].plot(df1_1['Step'], running_average(df1_1['Value'], window_size), label='Episode Tracker (5 layers)', color='red', marker='>', markevery=mark_every)[0]
a2 = axes[0].plot(df2['Step'], running_average(df2['Value'], window_size), label='No Convolution', color='gray', marker='o', markevery=mark_every)[0]
a3 = axes[0].plot(df3['Step'], running_average(df3['Value'], window_size), label='DQN', color='blue', marker='^', markevery=mark_every)[0]

axes[1].plot(spsdf1['Step'], spsdf1['Value'], color='orange', marker='<', markevery=mark_every)[0]
axes[1].plot(spsdf1_1['Step'], spsdf1_1['Value'], color='red', marker='>', markevery=mark_every)[0]
axes[1].plot(spsdf2['Step'], spsdf2['Value'], color='gray', marker='o', markevery=mark_every)[0]
axes[1].plot(spsdf3['Step'], spsdf3['Value'], color='blue', marker='^', markevery=mark_every)[0]
axes[1].set_ylabel('Steps per second')
axes[1].set_xlabel('Step')

plt.tight_layout()
plt.legend(handles=[a1, a1_1, a2, a3])
plt.grid(True)
plt.savefig("graphs/pong.png")