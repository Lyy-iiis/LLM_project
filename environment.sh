pip install -r requirements.txt 
mkdir -p ./codes/data
ln -s /ssdshare/image/illustration_style ./codes/data/style
mkdir -p ./codes/data/demo
mkdir -p ./codes/data/demo/style
ln -s /ssdshare/image/illustration_style ./codes/data/demo/style
export MKL_SERVICE_FORCE_INTEL=1