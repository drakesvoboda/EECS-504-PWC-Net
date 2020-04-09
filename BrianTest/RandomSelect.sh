#!/bin/bash

# for entry in "$search_dir"/*
# do
#   echo "$entry"
# done

# for i in {0..0020965}
# pwd
cd ../Data/ChairsSDHom_extended/train

img_0_end="-img_0.png"
img_1_end="-img_1.png"
flo_01_end="-flow_01.flo"

# mb_01_end="-mb_01.png"
# mb_weights_01_end="-mb_weights_01.pfm"
# occ_01_end="-occ_01.png"
# occ_weights_01_end="-occ_weights_01.pfm"
# oids_0_end="-oids_0.png"

filename="ChairsSDHom_extended_subset"

dest="../../${filename}"
num=1000

rnd_sel=($(jot -r $num  0 20965))

echo ${rnd_sel[@]}

mkdir $dest

for i in ${rnd_sel[@]}
do
    filebase=$( printf '%07d' $i )

    img_0="${filebase}${img_0_end}"
    img_1="${filebase}${img_1_end}"
    flo_01="${filebase}${flo_01_end}"
    # mb_01="${filebase}${mb_01_end}"
    # mb_weights_01="${filebase}${mb_weights_01_end}"
    # occ_01="${filebase}${occ_01_end}"
    # occ_weights_01="${filebase}${occ_weights_01_end}"
    # oids_0="${filebase}${oids_0_end}"
    echo "$img_0 $img_1 $flo_01 copied"

    cp $img_0 $img_1 $flo_01 $dest
done

tar -zcvf "../../${filename}.tar.gz" $dest




# for i in {0..0020965}
# do
#     filebase=$( printf '%07d' $i )

#     img_0="${filebase}${img_0_end}"
#     img_1="${filebase}${img_1_end}"
#     flo_01="${filebase}${flo_01_end}"
#     mb_01="${filebase}${mb_01_end}"
#     mb_weights_01="${filebase}${mb_weights_01_end}"
#     occ_01="${filebase}${occ_01_end}"
#     occ_weights_01="${filebase}${occ_weights_01_end}"
#     oids_0="${filebase}${oids_0_end}"

#     if [[ -f $img_0 && -f $img_1 && -f $flo_01 ]]; then
#         echo "$img_0 $img_1 $flo_01 exist"
#         rm $mb_01 $mb_weights_01 $occ_01 $occ_weights_01 $oids_0
#     else 
#         echo "delete $img_0 $img_1 $flo_01"
#         rm $filebase*
#     fi
# done

#  


# if [[ -f <> ]]
# then
#     echo "This file exists on your filesystem."
# fi
# shuf -i 0-20965 -n 1
