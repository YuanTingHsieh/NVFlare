# nvflare-plugin
NVFlare encryption plugin

This plugin is a companion for NVFlare Python based encryption, it processes the data so it can
be properly decoded by Python code running on NVFlare.

The actual encryption is happening on the Python side so this plugin code (c++) has no encrypytion.

# Build Instruction

cd NVFlare/integration/xgboost/encryption_plugins
mkdir build
cd build
cmake ..
make

The library is libxgb_nvflare.so
