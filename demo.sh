#python test.py --content_src data/xia_test/depressed_13_000.bvh --style_src data/xia_test/strutting_01_000.bvh --output_dir demo_results/depressed_13_000@strutting_01_000
#python test.py --content_src data/xia_test/neutral_16_000.bvh --style_src data/xia_test/angry_13_000.bvh --output_dir demo_results/neutral_16_000@angry_13_000

#python test.py --content_src data/xia_test/old_13_000.bvh --style_src data/xia_test/sexy_01_001.bvh --output_dir demo_results/old_13_000@sexy_01_001
#python test.py --content_src data/xia_test/sexy_01_000.bvh --style_src data/xia_test/depressed_18_000.bvh --output_dir demo_results/sexy_01_000@depressed_18_000

python test.py --content_src data/xia_test/neutral_01_000.bvh --style_src data/treadmill/json_inputs/27 --output_dir demo_results/neutral_01_000@27
python test.py --content_src data/xia_test/neutral_01_000.bvh --style_src data/treadmill/json_inputs/95 --output_dir demo_results/neutral_01_000@95
python test.py --content_src data/xia_test/neutral_01_000.bvh --style_src data/treadmill/json_inputs/32 --output_dir demo_results/neutral_01_000@32


#python test.py --content_src data/xia_test/depressed_13_000.bvh --style_src data/xia_test/strutting_01_000.bvh --output_dir demo_results/depressed_13_000.bvh@strutting_01_000.bvh
#python test.py --content_src data/xia_test/neutral_01_000.bvh --style_src data/treadmill/json_inputs/32 --output_dir demo_results/neutral_01_000.bvh@32


#!/usr/bin/env sh
# cd /home/deep-motion-editing/data/xia_test
# DIR=`ls .`
# for dir in ${DIR};do
#     if [ -f ${dir} ];then
# 	    python /home/jxl/deep-motion-editing/test.py --content_src /home/jxl/deep-motion-editing/data/xia_test/neutral_01_000.bvh --style_src /home/jxl/deep-motion-editing/data/xia_test/${dir} --output_dir /home/jxl/deep-motion-editing/test_results/${dir}
#     fi
# done


#content neutral

#walk
python test.py --content_src data/xia_test/neutral_01_000.bvh --style_src data/xia_test/old_01_000.bvh --output_dir demo_results/neutral_01_000@old_01_000
python test.py --content_src data/xia_test/neutral_01_000.bvh --style_src data/xia_test/depressed_01_000.bvh --output_dir demo_results/neutral_01_000@depressed_01_000
python test.py --content_src data/xia_test/neutral_01_000.bvh --style_src data/xia_test/angry_01_000.bvh --output_dir demo_results/neutral_01_000@angry_01_000
python test.py --content_src data/xia_test/neutral_01_000.bvh --style_src data/xia_test/proud_01_000.bvh --output_dir demo_results/neutral_01_000@proud_01_000
python test.py --content_src data/xia_test/neutral_01_000.bvh --style_src data/xia_test/childlike_01_000.bvh --output_dir demo_results/neutral_01_000@childlike_01_000
python test.py --content_src data/xia_test/neutral_01_000.bvh --style_src data/xia_test/sexy_01_000.bvh --output_dir demo_results/neutral_01_000@sexy_01_000
python test.py --content_src data/xia_test/neutral_01_000.bvh --style_src data/xia_test/strutting_01_000.bvh --output_dir demo_results/neutral_01_000@strutting_01_000
#run
python test.py --content_src data/xia_test/neutral_13_000.bvh --style_src data/xia_test/old_01_000.bvh --output_dir demo_results/neutral_13_000@old_01_000
python test.py --content_src data/xia_test/neutral_13_000.bvh --style_src data/xia_test/depressed_01_000.bvh --output_dir demo_results/neutral_13_000@depressed_01_000
python test.py --content_src data/xia_test/neutral_13_000.bvh --style_src data/xia_test/angry_01_000.bvh --output_dir demo_results/neutral_13_000@angry_01_000
python test.py --content_src data/xia_test/neutral_13_000.bvh --style_src data/xia_test/proud_01_000.bvh --output_dir demo_results/neutral_13_000@proud_01_000
python test.py --content_src data/xia_test/neutral_13_000.bvh --style_src data/xia_test/childlike_01_000.bvh --output_dir demo_results/neutral_13_000@childlike_01_000
python test.py --content_src data/xia_test/neutral_13_000.bvh --style_src data/xia_test/sexy_01_000.bvh --output_dir demo_results/neutral_13_000@sexy_01_000
python test.py --content_src data/xia_test/neutral_13_000.bvh --style_src data/xia_test/strutting_01_000.bvh --output_dir demo_results/neutral_13_000@strutting_01_000
#jump
python test.py --content_src data/xia_test/neutral_16_000.bvh --style_src data/xia_test/old_01_000.bvh --output_dir demo_results/neutral_16_000@old_01_000
python test.py --content_src data/xia_test/neutral_16_000.bvh --style_src data/xia_test/depressed_01_000.bvh --output_dir demo_results/neutral_16_000@depressed_01_000
python test.py --content_src data/xia_test/neutral_16_000.bvh --style_src data/xia_test/angry_01_000.bvh --output_dir demo_results/neutral_16_000@angry_01_000
python test.py --content_src data/xia_test/neutral_16_000.bvh --style_src data/xia_test/proud_01_000.bvh --output_dir demo_results/neutral_16_000@proud_01_000
python test.py --content_src data/xia_test/neutral_16_000.bvh --style_src data/xia_test/childlike_01_000.bvh --output_dir demo_results/neutral_16_000@childlike_01_000
python test.py --content_src data/xia_test/neutral_16_000.bvh --style_src data/xia_test/sexy_01_000.bvh --output_dir demo_results/neutral_16_000@sexy_01_000
python test.py --content_src data/xia_test/neutral_16_000.bvh --style_src data/xia_test/strutting_01_000.bvh --output_dir demo_results/neutral_16_000@strutting_01_000
#punch
python test.py --content_src data/xia_test/neutral_18_000.bvh --style_src data/xia_test/old_01_000.bvh --output_dir demo_results/neutral_18_000@old_01_000
python test.py --content_src data/xia_test/neutral_18_000.bvh --style_src data/xia_test/depressed_01_000.bvh --output_dir demo_results/neutral_18_000@depressed_01_000
python test.py --content_src data/xia_test/neutral_18_000.bvh --style_src data/xia_test/angry_01_000.bvh --output_dir demo_results/neutral_18_000@angry_01_000
python test.py --content_src data/xia_test/neutral_18_000.bvh --style_src data/xia_test/proud_01_000.bvh --output_dir demo_results/neutral_18_000@proud_01_000
python test.py --content_src data/xia_test/neutral_18_000.bvh --style_src data/xia_test/childlike_01_000.bvh --output_dir demo_results/neutral_18_000@childlike_01_000
python test.py --content_src data/xia_test/neutral_18_000.bvh --style_src data/xia_test/sexy_01_000.bvh --output_dir demo_results/neutral_18_000@sexy_01_000
python test.py --content_src data/xia_test/neutral_18_000.bvh --style_src data/xia_test/strutting_01_000.bvh --output_dir demo_results/neutral_18_000@strutting_01_000
#kick
python test.py --content_src data/xia_test/neutral_22_000.bvh --style_src data/xia_test/old_01_000.bvh --output_dir demo_results/neutral_22_000@old_01_000
python test.py --content_src data/xia_test/neutral_22_000.bvh --style_src data/xia_test/depressed_01_000.bvh --output_dir demo_results/neutral_22_000@depressed_01_000
python test.py --content_src data/xia_test/neutral_22_000.bvh --style_src data/xia_test/angry_01_000.bvh --output_dir demo_results/neutral_22_000@angry_01_000
python test.py --content_src data/xia_test/neutral_22_000.bvh --style_src data/xia_test/proud_01_000.bvh --output_dir demo_results/neutral_22_000@proud_01_000
python test.py --content_src data/xia_test/neutral_22_000.bvh --style_src data/xia_test/childlike_01_000.bvh --output_dir demo_results/neutral_22_000@childlike_01_000
python test.py --content_src data/xia_test/neutral_22_000.bvh --style_src data/xia_test/sexy_01_000.bvh --output_dir demo_results/neutral_22_000@sexy_01_000
python test.py --content_src data/xia_test/neutral_22_000.bvh --style_src data/xia_test/strutting_01_000.bvh --output_dir demo_results/neutral_22_000@strutting_01_000


#content proud

#walk
python test.py --content_src data/xia_test/proud_01_000.bvh --style_src data/xia_test/old_01_000.bvh --output_dir demo_results/proud_01_000@old_01_000
python test.py --content_src data/xia_test/proud_01_000.bvh --style_src data/xia_test/depressed_01_000.bvh --output_dir demo_results/proud_01_000@depressed_01_000
python test.py --content_src data/xia_test/proud_01_000.bvh --style_src data/xia_test/angry_01_000.bvh --output_dir demo_results/proud_01_000@angry_01_000
python test.py --content_src data/xia_test/proud_01_000.bvh --style_src data/xia_test/childlike_01_000.bvh --output_dir demo_results/proud_01_000@childlike_01_000
python test.py --content_src data/xia_test/proud_01_000.bvh --style_src data/xia_test/sexy_01_000.bvh --output_dir demo_results/proud_01_000@sexy_01_000

#content old

#walk
python test.py --content_src data/xia_test/old_01_000.bvh --style_src data/xia_test/proud_01_000.bvh --output_dir demo_results/old_01_000@proud_01_000
python test.py --content_src data/xia_test/old_01_000.bvh --style_src data/xia_test/depressed_01_000.bvh --output_dir demo_results/old_01_000@depressed_01_000
python test.py --content_src data/xia_test/old_01_000.bvh --style_src data/xia_test/angry_01_000.bvh --output_dir demo_results/old_01_000@angry_01_000
python test.py --content_src data/xia_test/old_01_000.bvh --style_src data/xia_test/childlike_01_000.bvh --output_dir demo_results/old_01_000@childlike_01_000
python test.py --content_src data/xia_test/old_01_000.bvh --style_src data/xia_test/sexy_01_000.bvh --output_dir demo_results/old_01_000@sexy_01_000
