t_1 = (1, 3)
t_2 = (2, 3)
t_3 = (1, 3)

list_1 = [t_1, t_2]

print(t_3 in list_1)

print(t_3 == t_2)

list_2 = list(t_1)
list_3 = list(t_3)

print(list_2 == list_3)