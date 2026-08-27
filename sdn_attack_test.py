#!/usr/bin/env python3
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel
import time

def run_sdn_ddos_test():
    setLogLevel('info')
    print("\n🌐 Mininet SDN Network তৈরি করা হচ্ছে...")

    net = Mininet(controller=RemoteController)
    c0 = net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6653)

    s1 = net.addSwitch('s1', protocols='OpenFlow13')
    h1 = net.addHost('h1', ip='10.0.0.1')
    h2 = net.addHost('h2', ip='10.0.0.2')

    net.addLink(h1, s1)
    net.addLink(h2, s1)

    net.start()
    time.sleep(2)

    print("\n🔍 Step 1: স্বাভাবিক পিং টেস্ট...")
    net.pingAll()

    print("\n⚔️ Step 2: DDoS অ্যাটাক শুরু (h1 থেকে h2-এ flood ping)...")
    h1.cmd('ping -f -c 50 10.0.0.2 > /dev/null 2>&1 &')
    time.sleep(8)

    print("\n🛡️ Step 3: অ্যাটাকের পর পিং টেস্ট...")
    net.pingAll()

    print("\n✅ টেস্ট শেষ! বের হতে 'exit' লিখুন।")
    CLI(net)
    net.stop()

if __name__ == '__main__':
    run_sdn_ddos_test()
