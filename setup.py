from setuptools import setup, find_packages

setup(
    name="decbin",
    version="0.0.1",
    packages=find_packages(),
    # 正确写法：父包 mbcclr_utils，匹配内部bin下所有文件
    package_data={
        "decbin.mbcclr_utils": ["bin/*"],
        "decbin.vae_dec": ["hyper_params.json"],
    },
    include_package_data=True,
)

'''
setup(
    name="decbin",
    version="0.0.1",
    packages=find_packages(),
    package_data={
        "decbin.mbcclr_utils.bin": ["*"],
        "decbin.vae_dec": ["hyper_params.json"],
    },
)
'''
