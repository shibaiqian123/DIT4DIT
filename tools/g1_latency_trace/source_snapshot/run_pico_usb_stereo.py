"""Robot USB side-by-side camera to PICO; does not use motion interfaces."""
import argparse
import datetime
import os
from pathlib import Path
import signal
import socket
import struct
import threading
import time
from gear_sonic.utils.video import pico_bridge as bridge

def read_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        part = sock.recv(size-len(data))
        if not part: raise ConnectionError('USB source disconnected')
        data.extend(part)
    return bytes(data)

class USBSource:
    mode = 'usb_stereo'
    def __init__(self, host, port, log):
        self.host, self.port, self.log = host, port, log
        self.fps = 30
        self.latest = bridge.LatestFrame()
        self.stop_event = threading.Event()
        self.frames = self.errors = 0
        self.last_rpc_ms = None
        self.last_error = ''
        self.thread = threading.Thread(target=self.run, daemon=True)
    def start(self): self.thread.start()
    def stop(self):
        self.stop_event.set()
        if self.thread.ident is not None: self.thread.join(timeout=3)
    def run(self):
        import cv2
        import numpy as np
        while not self.stop_event.is_set():
            try:
                with socket.create_connection((self.host,self.port),timeout=1) as sock:
                    sock.settimeout(.7)
                    sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                    while not self.stop_event.is_set():
                        start=time.monotonic()
                        sock.sendall(b'N')
                        size=struct.unpack('>I',read_exact(sock,4))[0]
                        if not 4 <= size <= 8*1024*1024: raise ValueError('Invalid JPEG size')
                        data=read_exact(sock,size)
                        frame=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                        if frame is None or frame.shape[:2] != (720,2560):
                            raise ValueError('Expected 2560x720 side-by-side source')
                        self.last_rpc_ms=(time.monotonic()-start)*1000
                        self.latest.put(frame); self.frames+=1; self.last_error=''
                        self.stop_event.wait(max(0,1/self.fps-(time.monotonic()-start)))
            except Exception as exc:
                self.errors+=1; self.last_error=str(exc)
                if self.errors==1 or self.errors%10==0:
                    self.log.emit('usb_source_error',error=str(exc),count=self.errors)
                self.stop_event.wait(.3)

def true_stereo_frame(frame,width,height,label,eye_shift=0,swap=False,image_scale=1.0,vertical_shift=0.0):
    import cv2
    import numpy as np
    if frame is None:
        left=right=None
    else:
        if frame.shape[1]%2: raise ValueError('SBS source width must be even')
        left,right=np.split(frame,2,axis=1)
        if swap: left,right=right,left
    ew=width//2
    left=bridge.mono_frame(left,ew,height,label.replace('RGB MONO','SBS LEFT'))
    right=bridge.mono_frame(right,ew,height,label.replace('RGB MONO','SBS RIGHT'))
    if not .5 <= image_scale <= 1.0:
        raise ValueError('image_scale must be in [0.5,1.0]')
    if not -.25 <= vertical_shift <= .25:
        raise ValueError('vertical_shift must be in [-0.25,0.25]')
    if eye_shift or image_scale != 1.0 or vertical_shift:
        # Scale about each eye's center, then shift in output pixels.
        # Eye-shift remains a fraction of output eye width, independent of scale.
        dx=round(ew*eye_shift)
        tx=(1-image_scale)*(ew-1)/2
        ty=(1-image_scale)*(height-1)/2 + round(height*vertical_shift)
        left=cv2.warpAffine(left,np.float32([[image_scale,0,tx+dx],[0,image_scale,ty]]),(ew,height))
        right=cv2.warpAffine(right,np.float32([[image_scale,0,tx-dx],[0,image_scale,ty]]),(ew,height))
    return np.concatenate((left,right),axis=1)

def preview_frame(frame, received, eye="both", now=None):
    """Raw camera view; never show an expired frame as live."""
    import cv2
    import numpy as np
    now=time.monotonic() if now is None else now
    stale=frame is None or now-received>.75
    if stale:
        canvas=np.zeros((360,1280 if eye=="both" else 640,3),np.uint8)
        canvas[:]=(0,0,60)
        cv2.putText(canvas,'NO LIVE CAMERA IMAGE',(25,180),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
        return canvas
    if eye!="both":
        half=frame.shape[1]//2
        frame=frame[:,:half] if eye=="left" else frame[:,half:]
    scale=min(1.,1280/frame.shape[1],720/frame.shape[0])
    return cv2.resize(frame,(round(frame.shape[1]*scale),round(frame.shape[0]*scale)))

def run_with_preview(app, source, duration, eye):
    """Keep Qt on the main thread; closing preview does not stop PICO streaming."""
    import cv2
    errors=[]
    title='Robot camera - Q/Esc closes preview only'
    def worker():
        try: app.run(duration)
        except Exception as exc: errors.append(exc)
    thread=threading.Thread(target=worker,daemon=True)
    opened=False
    thread.start()
    try:
        cv2.namedWindow(title,cv2.WINDOW_NORMAL)
        opened=True
        cv2.resizeWindow(title,1280,360 if eye=="both" else 720)
        while thread.is_alive():
            if not opened:
                app.stop_event.wait(.05)
                continue
            frame,received,_=source.latest.get()
            cv2.imshow(title,preview_frame(frame,received,eye))
            key=cv2.waitKey(30)&255
            if key in (27,ord('q')) or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:
                cv2.destroyAllWindows();opened=False
    finally:
        app.stop_event.set()
        thread.join()
        if opened: cv2.destroyAllWindows()
    if errors: raise errors[0]

_mono_canvas = bridge.mono_frame

def mono_eye_frame(frame,width,height,label,eye="right"):
    """One physical camera, letterboxed without stretching for the flat view."""
    if frame is not None:
        half=frame.shape[1]//2
        frame=frame[:,:half] if eye=="left" else frame[:,half:]
    return _mono_canvas(frame,width,height,label.replace('RGB MONO',eye.upper()+' MONO'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera-host',default='192.168.123.164')
    p.add_argument('--camera-port',type=int,default=15556)
    p.add_argument('--listen-host',default='192.168.11.224')
    p.add_argument('--listen-port',type=int,default=13579)
    p.add_argument('--headset',default='192.168.11.228')
    p.add_argument('--view',choices=['stereo','left','right'],default='stereo',help='right/left sends a single camera for PICO flat 2D view')
    p.add_argument('--eye-shift',type=float,default=0)
    p.add_argument('--swap-eyes',action='store_true')
    p.add_argument('--image-scale',type=float,default=1.0,
                   help='Per-eye image size, 0.5..1.0; smaller adds black margins. Not virtual screen distance.')
    p.add_argument('--vertical-shift',type=float,default=0.0,
                   help='Both eyes vertical offset as frame-height fraction; positive down, negative up (-0.25..0.25).')
    p.add_argument('--max-bitrate',type=int,default=6000000)
    p.add_argument('--publish-camera-port',type=int,default=None,help='Publish right-eye recording images on localhost (e.g. 5555)')
    p.add_argument('--duration',type=float)
    p.add_argument('--preview',action='store_true',help='Open local raw camera preview; Q closes preview only.')
    p.add_argument('--preview-eye',choices=['both','left','right'],default=None)
    a=p.parse_args()
    if a.view!='stereo' and (a.eye_shift or a.swap_eyes or a.image_scale!=1.0 or a.vertical_shift):
        p.error('2D view: omit eye-shift, swap-eyes, image-scale and vertical-shift')
    if a.preview_eye is None: a.preview_eye='both' if a.view=='stereo' else a.view
    if a.preview and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        p.error('--preview requires a graphical desktop terminal on this computer')
    if not -.2<=a.eye_shift<=.2: p.error('eye-shift must be in [-0.2,0.2]')
    if not .5<=a.image_scale<=1.0: p.error('image-scale must be in [0.5,1.0]')
    if not -.25<=a.vertical_shift<=.25: p.error('vertical-shift must be in [-0.25,0.25]')
    if not 100000<=a.max_bitrate<=8000000: p.error('Invalid bitrate')
    if a.duration is not None and a.duration<=0: p.error('duration must be positive')
    for port in (a.camera_port,a.listen_port,*([a.publish_camera_port] if a.publish_camera_port is not None else [])):
        if not 1024<=port<=65535: p.error('Invalid port')
    # Only this new entry point uses real SBS; the original bridge is untouched.
    bridge.stereo_frame=lambda f,w,h,l,s=0: true_stereo_frame(f,w,h,l,s,a.swap_eyes,a.image_scale,a.vertical_shift)
    if a.view!='stereo':
        bridge.mono_frame=lambda f,w,h,l: mono_eye_frame(f,w,h,l,a.view)
    directory=Path.home()/'pico-video-runs'/('usb-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+f'-{os.getpid()}')
    log=bridge.EventLog(directory)
    log.emit('stereo_display_config',image_scale=a.image_scale,eye_shift=a.eye_shift,swap_eyes=a.swap_eyes,vertical_shift=a.vertical_shift,view=a.view)
    source=USBSource(a.camera_host,a.camera_port,log)
    source.video_bind_host=a.listen_host
    app=bridge.Bridge(a.listen_host,a.listen_port,a.headset,source,log,a.max_bitrate,a.eye_shift,'stereo' if a.view=='stereo' else 'mono')
    signal.signal(signal.SIGTERM,lambda *_: app.stop_event.set())
    publisher = None
    try:
        if a.publish_camera_port is not None:
            from gear_sonic.utils.video.recording_publisher import RecordingPublisher
            publisher = RecordingPublisher(source, a.publish_camera_port)
            publisher.start()
            log.emit('recording_publisher_started',port=a.publish_camera_port,eye='right',width=640,height=480)
        if a.preview: run_with_preview(app,source,a.duration,a.preview_eye)
        else: app.run(a.duration)
    except KeyboardInterrupt: pass
    finally:
        if publisher is not None: publisher.stop()
        log.close()

if __name__=='__main__': main()
