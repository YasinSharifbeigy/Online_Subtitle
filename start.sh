#!/bin/bash
export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
# export LD_LIBRARY_PATH=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
echo  $LD_LIBRARY_PATH
# python3 my_whisper_online.py test/test_per_30s.wav --language en --min-chunk-size 2 --model_dir .cache/whisper/medium-ct2 > out.txt
# python3 my_whisper_online.py test8.wav --language en --min-chunk-size 1 --model_dir .cache/whisper/medium-ct2 --Razi > out.txt
# python3 stream_server.py
# curl -d "username=yasin.sharifbeigy&password=SirQoli1379@B" -X POST "https://net2.sharif.edu/login"
# kill -9 PID