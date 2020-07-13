docker run --rm -it \
--name temp \
-v $(pwd):/workdir -w /workdir \
pytorch_env \
/bin/bash