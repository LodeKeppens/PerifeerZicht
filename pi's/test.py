import paramiko


source = r'/home/pi/Documents/SSH_test.py'
dest = r'Documents/SSH_1.py'
hostname = 'raspberrypi'
port = 22 # default port for SSH
username = 'pi'
password = 'qwertyui'

t = paramiko.Transport((hostname, port))
t.connect(username=username, password=password)
sftp = paramiko.SFTPClient.from_transport(t)
sftp.put(source, dest)
t.close()
