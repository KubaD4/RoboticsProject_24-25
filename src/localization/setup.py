from setuptools import find_packages, setup

package_name = 'localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Alessandro Nardin',
    maintainer_email='ale.daluch@gmail.com',
    description='Package used for Megablock localization',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector = localization.detection_node:main',
            'service = py_srvcli.service_member_function:main',
        ],
    },
)