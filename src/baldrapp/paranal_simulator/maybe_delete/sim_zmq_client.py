import zmq
import json 


def send_mds_command(command: str, address: str = "tcp://127.0.0.1:5555", timeout_ms: int = 2000):
    """Send a ZMQ REQ command to the MDS server and return the response."""
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(address)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)  # Timeout for recv
    socket.setsockopt(zmq.LINGER, 0)

    try:
        print(f">>> {command}")
        socket.send_string(command)
        response = socket.recv_string()
        print(f"<<< {response}")
        return response
    except zmq.ZMQError as e:
        print(f"ZMQ Error: {e}")
        return None
    finally:
        socket.close()
        context.term()



# more specific for baldr rtc server 
class CommanderClient:
    """
    Minimal client for your Commander server over ZMQ REQ/REP.

    Protocol nuances handled:
      - Single parameter: send one JSON value after the command.
      - Multiple parameters: send comma-separated JSON values after the command
        (because your server wraps "everything after the first space" in [] and parses it).
    """
    def __init__(self, addr: str, rcv_timeout_ms=2000, snd_timeout_ms=2000):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, rcv_timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, snd_timeout_ms)
        self.sock.connect(addr)

    def _send(self, msg: str):
        self.sock.send_string(msg)
        return self.sock.recv_string()
        
    def call(self, cmd: str, *args):
        if len(args) == 0:
            # IMPORTANT: add a trailing space so the server sees "args" and builds [].
            msg = cmd + " "
        elif len(args) == 1:
            msg = f"{cmd} {json.dumps(args[0])}"
        else:
            # multiple params -> comma-separated JSON tokens (server will wrap in [])
            msg = f"{cmd} " + ", ".join(json.dumps(a) for a in args)

        self.sock.send_string(msg)
        reply = self.sock.recv_string()
        try:
            return json.loads(reply)
        except json.JSONDecodeError:
            return {"ok": False, "error": "non-JSON reply", "raw": reply}
        
    # Convenience wrappers for your registered functions
    def list_fields(self):
        return self.call("list_rtc_fields")

    def get_field(self, path: str):
        return self.call("get_rtc_field", path)

    def set_field(self, path: str, value):
        return self.call("set_rtc_field", path, value)

    # Telemetry (optional helpers if you want to use them here too)
    def poll_scalar(self, name: str):
        return self.call("poll_telem_scalar", name)

    def poll_vector(self, name: str):
        return self.call("poll_telem_vector", name)

# Example usage
if __name__ == "__main__":
    
    # This is so we can potentially analyse all the RTC fields while running and update manually here in python (.e.g we could updload a different config here and simply update to rtc via zmq)
    # to do is properly review read write permissions for rtc parameters via commander setter.
    
    client = CommanderClient("tcp://127.0.0.1:6662", 2000)
    
    # list the fields available 
    lf = client.list_fields()
    
    thang = 'limits.close_on_strehl_limit'
    
    res = client.get_field(thang) 
    
    newThang = res['value'] #.copy()
    newThang = 20.0 #[0][0] = 0 # set the first element to zero
    
    res = client.set_field(thang, newThang) 
    
    
    ###///////////////// MDS SIMULATION COMMANDS /////////////////////////
    
    # ## MDS SIMULATION BASIC TESTS 
    # print('running')
    # # Send command to move FPM for beam 1 to mask "J1"
    # #send_mds_command("fpm_move 1 J1")
    # for beam in [1,2,3,4]:
    #    send_mds_command(f"moveabs BMX{beam} 0")

    # for beam in [1,2,3,4]:
    #    send_mds_command(f"fpm_move {beam} J3")
    
    
    