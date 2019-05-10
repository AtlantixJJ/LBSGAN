rm -r tmp
mkdir tmp
cp -r *.py tmp/
cp -r lib models tmp/ 
find tmp -name "*.pyc" -exec rm -f {} \;  
bash ~/philly-fs.bash -cp -r ../swa //philly/$1/msrlabs/v-xuj/