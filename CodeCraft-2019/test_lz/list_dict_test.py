import time

list_1 = (1, 2)
list_2 = [3, 4]

dict_1 = {}
dict_1[list_1] = True

for i in range(1000):
    for j in range(1000):
        dict_1[(i, j)] = [True, [1, 2, 3, 4, 5, 6, 7, 8]]

time.sleep(100)
print(dict_1[(1, 2)])