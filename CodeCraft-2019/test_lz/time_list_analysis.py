import dill


with open(r'C:\Users\LenFranky\OneDrive\codes\TrafficManagement\CodeCraft-2019\src\time_list.pkl', mode='rb') as file_read:
    time_list = dill.load(file_read)

time_list = sorted(time_list, reverse=True, key=lambda d: d[1])

for item in time_list:
    if item[1] < 0.000001:
        continue
    print("%d - %f" % (item[0], item[1]))
