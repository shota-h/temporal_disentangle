docker run --rm -it \
--name temp -p 8888:8888 \
--runtime nvidia \
-v $(pwd):/workdir -w /workdir \
pytorch_env \
/bin/bash
