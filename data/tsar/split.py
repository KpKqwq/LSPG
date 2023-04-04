# f2=open("tsar2022_en_test_gold.tsv.nosen","w+")

# for line in open("tsar2022_en_test_gold.tsv"):
#     tmp1="|||".join(list(set(line.strip().split("\t")[2:])))
#     f2.write(tmp1.strip()+"\n")
# f2.close()

for line in open("tsar2022_en_test_gold.tsv"):
    print(line.strip().split("\t")[1],"|||"," ".join(list(set(line.strip().split("\t")[2:]))))
    print("-"*95)