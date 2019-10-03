# This script, resamples all files in .../URBAN-SED/audio/{train,test,validate} to 22050 Hz.
# The files are saved in .../URBAN-SED/audio22050/{train,test,validate}

path_original=~/kalman/data_ssd/users/pzinemanas/URBAN-SED/audio/

for folder in train test validate
do
  path=$path_original$folder/*
  echo $path

  for entry in $path
  do
    echo $entry
    if [ -f "$entry" ];then
        file_old=$entry
        file_new=${entry/audio/audio22050}
        sox $file_old -r 22050 $file_new &
        wait $!
   fi
  done
done




