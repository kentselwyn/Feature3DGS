import os
s="/home/koki/code/cc/feature_3dgs_2/all_data/scene0708_00/A/outputs"

x=os.listdir(s)

print(x)


x.remove('_3')
x.remove('_7')
x.remove('output')
x.remove('2')
x.remove('6')


for i in x:
    path=f"{s}/{i}/rendering"
    for j in ['trains/ours_10000', "test/ours_10000"]:
        for k in ['score_tensors', 'feature_tensors']:
            p=f"{path}/{j}/{k}"
            print(p)

            all_files=os.listdir(p)

            for f in all_files:
                file_path=f"{p}/{f}"
                new_path=f"{p}/{f[0:10]}.pt"
                os.rename(file_path, new_path)





breakpoint()


