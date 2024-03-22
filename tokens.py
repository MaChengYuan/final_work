#write
#with open("password.txt",'r+') as file:
#    file.write('ITMO')

secrets = []

with open('password.txt') as f:
    for line in f:
        secrets.append(line.strip())



print(secrets)
#delete
#with open("password.txt",'r+') as file:
#    file.truncate(0)

password = secrets[0]
token = secrets[1]
token2 = secrets[2]

    
