cd ..
tar czvf clusternet.tar.gz clusternet
scp -P 8003 clusternet.tar.gz zuochang@s01:/home/zuochang
ssh -p 8003 zuochang@s01 "tar xzvf clusternet.tar.gz"