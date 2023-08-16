docker volume rm av_exp
docker volume create av_exp
docker run --rm -v /tmp/nvflare/:/source -v av_exp:/destination alpine sh -c 'cp -r /source/cifar10 /destination/'
