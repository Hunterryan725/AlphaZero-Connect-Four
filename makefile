Drive:
	echo "#!/bin/bash" > Drive
	echo "python3 driver.py \"\$$@\"" >> Drive
	chmod u+x Drive