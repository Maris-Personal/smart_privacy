from shutil import copy
import csv

with open('fairface_label_train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for line in csv_reader:
        filename = line[0].split('/')
        name = filename[1].split('.')
        if line[2] == 'Male':
            if int(line[1]) < 20:
                copy(line[0], 'male/child/'+name[0]+'train.jpg')
            elif 20 <= int(line[1]) < 50:
                copy(line[0], 'male/adult/'+name[0]+'train.jpg')
            elif int(line[1]) > 50:
                copy(line[0], 'male/old/'+name[0]+'tain.jpg')
        elif line[2] == 'Female':
            if int(line[1]) < 20:
                copy(line[0], 'female/child/'+name[0]+'train.jpg')
            elif 20 <= int(line[1]) < 50:
                copy(line[0], 'female/adult/'+name[0]+'train.jpg')
            elif int(line[1]) > 50:
                copy(line[0], 'female/old/'+name[0]+'train.jpg')
