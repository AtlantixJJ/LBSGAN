tar cvfz - .gitignore *.py *.sh script models lib | ssh xujianjing@211.157.135.154 "cd LBSGAN/; tar xvfz -" && tar cvfz - .gitignore *.py *.sh script models lib | ssh -p 8081 atlantix@166.111.17.31 "cd disk6/LBSGAN/; tar xvfz -"

