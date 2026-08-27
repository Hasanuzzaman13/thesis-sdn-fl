import time
import torch
import numpy as np
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, ipv4, arp

class IntrusionDetectionNet(torch.nn.Module):
    def __init__(self, input_size=29, hidden_size=128, num_classes=2):
        super(IntrusionDetectionNet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.BatchNorm1d(hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

class DDoSMitigationMLApp(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(DDoSMitigationMLApp, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.blocked_ips = set()
        self.ip_stats = {}
        self.PACKET_THRESHOLD = 5
        
        self.logger.info("🧠 Loading Federated Learning Model...")
        self.model = IntrusionDetectionNet()
        try:
            self.model.load_state_dict(torch.load('global_model.pth', map_location='cpu'))
            self.model.eval()
            self.logger.info("✅ Model loaded successfully!")
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth.dst
        src = eth.src
        self.mac_to_port[src] = in_port

        # ARP packets
        if eth.ethertype == ether_types.ETH_TYPE_ARP:
            self.logger.info(f"📡 ARP: {src} -> {dst}")
            if dst in self.mac_to_port:
                out_port = self.mac_to_port[dst]
            else:
                out_port = ofproto.OFPP_FLOOD
            actions = [parser.OFPActionOutput(out_port)]
            out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                       in_port=in_port, actions=actions, data=msg.data)
            datapath.send_msg(out)
            return

        # IPv4 packets
        if eth.ethertype == ether_types.ETH_TYPE_IP:
            ip_pkt = pkt.get_protocol(ipv4.ipv4)
            if ip_pkt:
                src_ip = ip_pkt.src
                dst_ip = ip_pkt.dst
                
                self.logger.info(f"🌐 IPv4: {src_ip} -> {dst_ip}")
                
                if src_ip in self.blocked_ips:
                    self.logger.warning(f"🚫 Blocked IP: {src_ip}")
                    return
                
                if src_ip not in self.ip_stats:
                    self.ip_stats[src_ip] = 0
                self.ip_stats[src_ip] += 1
                
                self.logger.info(f"📊 {src_ip} packet count: {self.ip_stats[src_ip]}/{self.PACKET_THRESHOLD}")
                
                if self.ip_stats[src_ip] >= self.PACKET_THRESHOLD:
                    self.logger.error(f"🚨 DDoS DETECTED from {src_ip}! Packets: {self.ip_stats[src_ip]}")
                    self.blocked_ips.add(src_ip)
                    
                    match_drop = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, ipv4_src=src_ip)
                    self.add_flow(datapath, 10, match_drop, [])
                    self.logger.error(f"✅ Blocked {src_ip} - all traffic will be dropped")
                    return

        # Forward packet (NO flow install - every packet comes to controller)
        if dst in self.mac_to_port:
            out_port = self.mac_to_port[dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                   in_port=in_port, actions=actions, data=msg.data)
        datapath.send_msg(out)
