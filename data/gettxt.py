lilist = []
with open('amp_all.csv') as f:
    for line in f:
        for li in line.strip():
            lilist.append(li.strip())
        print(' '.join(lilist))
        lilist = []