from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, ipv4, arp

class DDoSMitigationApp(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(DDoSMitigationApp, self).__init__(*args, **kwargs)
        self.packet_counts = {}
        self.blocked_ips = set()
        self.THRESHOLD = 5

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

        src_ip = None
        arp_pkt = pkt.get_protocol(arp.arp)
        ip_pkt = pkt.get_protocol(ipv4.ipv4)

        if arp_pkt:
            src_ip = arp_pkt.src_ip
        elif ip_pkt:
            src_ip = ip_pkt.src

        if src_ip:
            if src_ip in self.blocked_ips:
                return

            if src_ip not in self.packet_counts:
                self.packet_counts[src_ip] = 0
            self.packet_counts[src_ip] += 1

            self.logger.error(f"[COUNT] {src_ip} -> {self.packet_counts[src_ip]} packets")

            if self.packet_counts[src_ip] > self.THRESHOLD:
                self.logger.error(f"🚨 DDoS ATTACK DETECTED from {src_ip}! Blocking IP...")
                self.blocked_ips.add(src_ip)
                match_drop = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, ipv4_src=src_ip)
                self.add_flow(datapath, 10, match_drop, [])
                self.logger.error(f"✅ Flow rule added to DROP all traffic from {src_ip}")
                return

        actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                   in_port=in_port, actions=actions)
        datapath.send_msg(out)
