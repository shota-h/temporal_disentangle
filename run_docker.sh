docker run --rm -it \
--name temp \
-v $(pwd):/workdir -w /workdir \
temp_disentangle \
/bin/bash
