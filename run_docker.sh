docker run --rm -it \
--name tempral_disentangle \
-v $(pwd):/workdir -w /workdir \
pytorch_env \
/bin/bash
