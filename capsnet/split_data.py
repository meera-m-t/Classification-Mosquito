import splitfolders

dir_dataset = ''
dir_current_work = ''
splitfolders.ratio(dir_dataset, output=f"{dir_current_work}/Dataset1", seed=1337, ratio=(.8, 0.2))