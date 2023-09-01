from setuptools import setup

package_name = 'image_processor'
bridge = 'image_processor/bridge'


setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, bridge],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ggao22',
    maintainer_email='g.gao@wustl.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_processor = image_processor.image_processor_node:main',
        ],
    },
)
