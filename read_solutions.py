f = open('martialjames.txt', 'r')
data = f.readlines()
counts = {}
total = 0
for item in data:
    if item in counts:
        counts[item] += 1
        total += 1
    else:
        counts[item] = 1
        total += 1

for item in sorted(counts, key=counts.get, reverse=True):
    print(item.strip() + "," + str(counts.get(item)))

print("Runs completed: " + str(int(total/9)))