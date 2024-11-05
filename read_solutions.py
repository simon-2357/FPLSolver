def run(scenario, view, cutoff, display, archive):
  if archive == True:
    folder = 'archive'
  else:
    folder = 'output'

  f = open(f'{folder}/{scenario}-{view}.txt', 'r')
  data = f.readlines()
  counts = {}
  total = 0
  simulations = 0
  ft_count = 0
  for item in data:
      if item in counts:
          counts[item] += 1
          total += 1
      else:
          counts[item] = 1
          total += 1

  g = open(f'{folder}/{scenario}-xp.txt', 'r')
  count_data = g.readlines()
  for item in count_data:
     simulations +=1
  if display == 'p':
    for item in sorted(counts, key=counts.get, reverse=True):
        if 100 * counts[item] / simulations >= cutoff:
          print(item.strip() + "," + "{:.1f}".format(100 * counts.get(item) / simulations))
  else:
    for item in sorted(counts, key=counts.get, reverse=True):
      if counts[item] >= cutoff:
        print(item.strip() + "," + str(counts.get(item)))
  h = open(f'{folder}/{scenario}-ftplayer.txt', 'r')
  count_ft = h.readlines()
  for item in count_ft:
     ft_count +=1
  print("FTs Used: " + "{:.2f}".format(ft_count / simulations / 2))

scenario = '12-0'
view = 'squad'
display = 'p'
cutoff = 1
archived=False
run(scenario, view, cutoff, display, archived)