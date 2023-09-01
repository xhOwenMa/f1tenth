from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    image_process_node = Node(
        package="image_processor",
        executable="image_processor",
        parameters=[
            {'mode': 'cluster_parall'}
        ]
    )

    n = 3

    inv_mux_node = Node(
        package="inv_mux",
        executable="inv_mux",
        parameters=[
            {'num_parall_nodes': n}
        ]
    )

    clustering_node = []
    for i in range(n):
        clustering_node.append(
            Node(
                package="lanenet_postprocess",
                executable="postprocessor",
                parameters=[
                    {'topic': '/parall_'+str(i)}
                ]
            )
        )

    ld.add_action(image_process_node)
    ld.add_action(inv_mux_node)
    for node in clustering_node:
        ld.add_action(node)

    return ld
