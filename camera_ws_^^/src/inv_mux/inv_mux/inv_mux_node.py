#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from lanenet_out.msg import OrderedSegmentation


class InverseMux(Node):

    def __init__(self):
        super().__init__("inv_mux")
        self.declare_parameter('num_parall_nodes')

        if self.get_parameter('num_parall_nodes').value: 
            self.n = int(self.get_parameter('num_parall_nodes').value)
        else: raise ValueError('Number of parallel nodes not provided.')

        self.subscriber_ = self.create_subscription(OrderedSegmentation, "/lanenet_out", self.parall_callback, 10)
        self.publishers_ = []
        for i in range(self.n):
            self.publishers_.append(self.create_publisher(OrderedSegmentation, '/parall_'+str(i), 1))
        
        self.current_count = 0

    def parall_callback(self, data):
        self.publishers_[self.current_count%self.n].publish(data)
        self.current_count += 1


def main(args=None):
    rclpy.init(args=args) 

    inv_mux = InverseMux()
    
    rclpy.spin(inv_mux)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
