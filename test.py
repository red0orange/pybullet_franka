import bisect

sorted_list = [1, 3, 4, 6, 9]
new_value = 5

insert_position = bisect.bisect_left(sorted_list, new_value)
sorted_list.insert(insert_position, new_value)

print("Insert position:", insert_position)  # 输出：Insert position: 3
print("Sorted list:", sorted_list)          # 输出：Sorted list: [1, 3, 4, 5, 6, 9]
