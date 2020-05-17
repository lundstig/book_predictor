
lines = []
while True:
  line = input()
  if len(line) <= 1:
    break
  lines.append(line)

for line in lines:
  name, data = line.split(":")
  print (" & ".join([name] + [x.split("=")[1].replace(',', '') for x in data.split()]) + " \\\\") 