# prod version
docker run -dit --gpus all --shm-size=64g --name kor_med_stt -v $(pwd)/:/workspace/ kor_med_qa_benchmark:latest 

# dev version
docker run -dit --gpus all --shm-size=64g --name kor_med_stt \
  -v $(pwd)/:/workspace/ \
  -v /mnt/sdd/kor_med_opendataset/:/workspace/kor_med_opendataset \
  kor_med_qa_benchmark:latest